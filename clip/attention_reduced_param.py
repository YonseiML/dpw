import warnings
from typing import Callable, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from torch.nn.parameter import Parameter
from torch.nn.modules import Module
from torch.nn import functional as F

import copy
import math
from torch.distributions import Normal

def _get_clones(module, N):
    return torch.nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class LayerNorm(torch.nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)
    

def _in_projection_packed(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    w: Tensor,
    b: Optional[Tensor] = None,
) -> List[Tensor]:
    E = q.size(-1)
    if k is v:
        if q is k:
            # self-attention
            proj = F.linear(q, w, b)
            # reshape to 3, E and not E, 3 is deliberate for better memory coalescing and keeping same order as chunk()
            proj = proj.unflatten(-1, (3, E)).unsqueeze(0).transpose(0, -2).squeeze(-2).contiguous()
            return proj[0], proj[1], proj[2]
        else:
            # encoder-decoder attention
            w_q, w_kv = w.split([E, E * 2])
            if b is None:
                b_q = b_kv = None
            else:
                b_q, b_kv = b.split([E, E * 2])
            q_proj = F.linear(q, w_q, b_q)
            kv_proj = F.linear(k, w_kv, b_kv)
            # reshape to 2, E and not E, 2 is deliberate for better memory coalescing and keeping same order as chunk()
            kv_proj = kv_proj.unflatten(-1, (2, E)).unsqueeze(0).transpose(0, -2).squeeze(-2).contiguous()
            return (q_proj, kv_proj[0], kv_proj[1])
    else:
        w_q, w_k, w_v = w.chunk(3)
        if b is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = b.chunk(3)
        return F.linear(q, w_q, b_q), F.linear(k, w_k, b_k), F.linear(v, w_v, b_v)


def multi_head_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Optional[Tensor],
    in_proj_bias: Optional[Tensor],
    bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Optional[Tensor],
    training: bool = True,
    need_weights: bool = False,
    attn_mask: Optional[Tensor] = None,
    is_causal: bool = False,
    prefix: Tensor = None,
    batch_weight: Tensor = None,
    custom_attn: Tensor = None,
    prefix_head: int = None,
    custom_value: Tensor = None,
    rt: Tensor = None,
    aux_outputs: dict = None
) -> Tuple[Tensor, Optional[Tensor]]:

    # set up shape vars
    tgt_len, bsz, embed_dim = query.shape  # for visual: [197, bs, 768]; for text: [77, n_cls, 512]
    src_len, _, _ = key.shape
    if prefix is not None:
        _, _, prefix_len, _ = prefix.shape  # [bs, 1, prefix_len, embed_dim]

    assert embed_dim == embed_dim_to_check, \
        f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
    assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"

    # compute in-projection
    q, k, v = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
    if prefix is not None:
        prefix_v = prefix[:, 0, ...].transpose(0, 1).contiguous()  # [prefix_len, bs, embed_dim]
        prefix_head = prefix_head
        prefix_head_dim = embed_dim // prefix_head
        assert prefix_head_dim * prefix_head == embed_dim, f"embed_dim {embed_dim} not divisible by prefix_head {prefix_head}"


    # prep attention mask
    attn_mask = F._canonical_mask(
        mask=attn_mask,
        mask_name="attn_mask",
        other_type=None,
        other_name="",
        target_type=q.dtype,
        check_other=False,
    )

    if attn_mask is not None:
        # ensure attn_mask's dim is 3
        if attn_mask.dim() == 2:
            correct_2d_size = (tgt_len, src_len)
            if attn_mask.shape != correct_2d_size:
                raise RuntimeError(f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
            attn_mask = attn_mask.unsqueeze(0)
        elif attn_mask.dim() == 3:
            correct_3d_size = (bsz * num_heads, tgt_len, src_len)
            if attn_mask.shape != correct_3d_size:
                raise RuntimeError(f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
        else:
            raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

    # reshape q, k, v for multihead attention and make em batch first
    q = q.view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)  # for visual: [bs*12, 197, 64]; for text: [n_cls*8, 77, 64]
    k = k.view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    v = v.view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    if prefix is not None:
        prefix_v = prefix_v.view(prefix_v.size(0), bsz * prefix_head, prefix_head_dim).transpose(0, 1)

    # (deep breath) calculate attention and out projection
    if attn_mask is not None:
        if attn_mask.size(0) == 1 and attn_mask.dim() == 3:
            attn_mask = attn_mask.unsqueeze(0)  # only for text: [1, 1, 77, 77]
        else:
            attn_mask = attn_mask.view(bsz, num_heads, -1, src_len)

    q = q.view(bsz, num_heads, tgt_len, head_dim)  # for visual: [bs, 12, 197, 64]; for text: [n_cls, 8, 77, 64]
    k = k.view(bsz, num_heads, src_len, head_dim)
    v = v.view(bsz, num_heads, src_len, head_dim)
    if prefix is not None:
        prefix_v = prefix_v.view(bsz, prefix_head, prefix_len, prefix_head_dim)
        if custom_value is not None:
            custom_value = custom_value.view(bsz, tgt_len, prefix_head, prefix_head_dim).transpose(1, 2).contiguous()
        

    attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, is_causal)  # for visual: [bs, 12, 197, 64]; for text: [n_cls, 8, 77, 64]
    attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(bsz * tgt_len, embed_dim)  # for visual: [197*bs, 768]; for text: [77*n_cls, 512]
    if prefix is not None:
        if custom_value is not None:
            prompt_value = torch.matmul(custom_attn, prefix_v)
            attn_output_prefix = prompt_value + custom_value * rt

        else:
            attn_output_prefix = torch.matmul(custom_attn, prefix_v)
        
        if batch_weight is not None:
            attn_output_prefix = attn_output_prefix * batch_weight.view(bsz, 1, 1, 1)
        attn_output_prefix = attn_output_prefix.permute(2, 0, 1, 3).contiguous().view(bsz * tgt_len, embed_dim)  # for visual: [197*bs, 768]; for text: [77*n_cls, 512]
        attn_output += attn_output_prefix

    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)  # for visual: [197*bs, 768]; for text: [77*n_cls, 512]
    attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))  # for visual: [197, bs, 768]; for text: [77, n_cls, 512]
    return attn_output, None, aux_outputs


class Adapter(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 bottleneck=None,
                 adapter_scalar=None,
                 lr=0.01,
                 dropout=0.0,
                 in_proj_weight=None):
        super().__init__()
        
        fixed_down = None
        if bottleneck is not None:
            self.bottleneck = True
            down_lora = torch.nn.Linear(in_dim, bottleneck, dtype=torch.float16)
            up_lora = torch.nn.Linear(bottleneck, out_dim, dtype=torch.float16)
            with torch.no_grad():
                if in_proj_weight is not None:
                    v_matrix = in_proj_weight[in_dim*2:].clone().to(torch.float32)
                    U, S, V = torch.svd(v_matrix.t())
                    fixed_down = U[:, :bottleneck].to(torch.float16).cuda()
                    self.register_buffer("fixed_down", fixed_down)
                    torch.nn.init.normal_(up_lora.weight, mean=0.0, std=0.02)
                    torch.nn.init.zeros_(up_lora.bias)
                    
                    self.lora = torch.nn.Sequential(
                        torch.nn.Dropout(p=dropout),
                        up_lora
                    )
                else:
                    torch.nn.init.kaiming_uniform_(down_lora.weight, a=math.sqrt(5))
                    torch.nn.init.zeros_(up_lora.weight)
                    torch.nn.init.zeros_(down_lora.bias)
                    torch.nn.init.zeros_(up_lora.bias)
                
                    self.lora = torch.nn.Sequential(
                            down_lora,
                            torch.nn.ReLU(),
                            torch.nn.Dropout(p=dropout),
                            up_lora
                        )
            
        else:
            self.lora = torch.nn.Linear(in_dim, out_dim, dtype=torch.float16)
            
            with torch.no_grad():
                torch.nn.init.zeros_(self.lora.weight)
                torch.nn.init.zeros_(self.lora.bias)        
        
        if adapter_scalar == "learnable_scalar":
            self.scale = torch.nn.Parameter(torch.empty(in_dim, dtype=torch.float16))
            torch.nn.init.uniform_(self.scale, -1, 1) 
        else:
            self.scale = None
            
        self.lr = lr
        self.dropout = dropout


    def forward(self, x, residual=False):
        if self.fixed_down is not None:
            x = torch.matmul(x.clone(), self.fixed_down)
            
        out = self.lora(x)
        
        if self.scale is not None:
            scale = F.sigmoid(torch.matmul(x, self.scale.unsqueeze(1)))
            out = out * scale

        if residual:
            output = (out + x) * (self.lr)
        else:
            output = out * self.lr
            
        return output


class MultiheadAttention(Module):
    
    def __init__(self, embed_dim, num_heads, layer_index, device=None, dtype=None, prefix_pool_size=0, prefix_len=0) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        
        self.in_proj_weight = Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs))
        self.in_proj_bias = Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
        self.out_proj = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=True, **factory_kwargs)

        xavier_uniform_(self.in_proj_weight)
        constant_(self.in_proj_bias, 0.)
        constant_(self.out_proj.bias, 0.)

        self.add_prefix = False
        if prefix_pool_size > 0:
            self.add_prefix = True
            prefix_shape = (prefix_pool_size, 1, prefix_len, embed_dim)
            prefix_pool = torch.zeros(prefix_shape, dtype=torch.float16)
            self.prefix_pool = Parameter(prefix_pool)  
            
            self.dpw_attn = torch.nn.ParameterList()
            self.dpw_attn_bias = torch.nn.ParameterList()
            self.prefix_head = []
            
            if embed_dim == 768:
                self.text_layer = False
            else:
                self.text_layer = True
            
            if layer_index < 12:
                self.add_adapter = True 
                dpw_adapter = Adapter(in_dim=embed_dim, out_dim=embed_dim, bottleneck=4, adapter_scalar=None, lr=0.05, dropout=0.1, in_proj_weight=self.in_proj_weight)
                self.dpw_adapter = _get_clones(dpw_adapter, prefix_pool_size)
            else:
                self.add_adapter = False
                
            self.means = torch.nn.ParameterList()
            self.vars = torch.nn.ParameterList()
            
            self.threshold = 1.0
            
 
        self.prefix_len = prefix_len
        self.layer_index = layer_index
     
     
    def init_attn(self, fine_grained, tgt_task_id):
        if self.add_prefix:
            with torch.no_grad():                
                num_heads = int(self.num_heads * fine_grained)
                num_heads = self.num_heads // 4
                if num_heads == 0:
                    num_heads = 1
                else:
                    num_heads = int(2 ** round(math.log2(num_heads)))
                    
                self.prefix_head.append(num_heads)
                
                partition = self.in_proj_weight.size(0) // 3
                q_matrix = self.in_proj_weight[0:partition].clone()
                k_matrix = self.in_proj_weight[partition:partition*2].clone()
                v_matrix = self.in_proj_weight[partition*2:].clone().to(torch.float32)
                
                
                prefix_dim = self.embed_dim // num_heads
                random_matrix = torch.randn(num_heads, prefix_dim, prefix_dim).to(torch.float32)
                q, _ = torch.linalg.qr(random_matrix)
                A_matrix = q[:, :, :self.prefix_len].to(torch.float16).to(q_matrix.device)
                self.dpw_attn.append(A_matrix)          # num_heads, prefix_dim, prefix_len
                
                
                if self.embed_dim == 768:
                    attn_bias = torch.full((num_heads, 197, self.prefix_len), -4.0, dtype=torch.float16, device=A_matrix.device)
                    self.dpw_attn_bias.append(attn_bias)
                else:
                    attn_bias = torch.full((num_heads, 77, self.prefix_len), -2.0, dtype=torch.float16, device=A_matrix.device)
                    self.dpw_attn_bias.append(attn_bias)
                    

                self.means.append(torch.zeros(num_heads, self.prefix_len, dtype=torch.float, device=A_matrix.device))
                self.vars.append(torch.ones(num_heads, self.prefix_len, dtype=torch.float, device=A_matrix.device))
                self.stat_sum = torch.zeros(num_heads, self.prefix_len, dtype=torch.float, device=A_matrix.device)
                self.stat_sum_sqr = torch.zeros(num_heads, self.prefix_len, dtype=torch.float, device=A_matrix.device)
                self.stat_total_iter = 0
                    
                
                if tgt_task_id == 0:
                    v_norms = torch.mean(torch.norm(v_matrix, p=2, dim=0))
                    U, S, V = torch.svd(v_matrix)
                    v_matrix = U[:, :self.prefix_len] * v_norms
                    v_matrix = v_matrix.unsqueeze(0).unsqueeze(1).expand(self.prefix_pool.size(0), self.prefix_pool.size(1), -1, -1).contiguous().transpose(2, 3).to(torch.float16)
                    self.prefix_pool.data.copy_(v_matrix)  # prefix_pool_size, 1, prefix_len, embed_dim

        
    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            aux_outputs: dict,
            need_weights: bool = False,
            attn_mask: Optional[Tensor] = None,
            prompt_ids: Tensor = None,
            batch_weight: Tensor = None
            ) -> Tuple[Tensor, Optional[Tensor]]:

        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )
        
        prefix = None
        custom_attn = None
        prefix_head = None
        custom_value = None
        rt = None
        patch_len, bsz, dim = query.shape
        if self.add_prefix:
            assert prompt_ids.size(1) == 1, "Only single prefix for one sample is supported."
            prompt_ids = prompt_ids.squeeze(1)       
            prefix_head = self.prefix_head[prompt_ids]
            attn = self.dpw_attn[prompt_ids].squeeze(0)    
            attn_bias = self.dpw_attn_bias[prompt_ids]
            prefix_head_dim = dim // prefix_head
            
            patch = query.view(patch_len, bsz*prefix_head, prefix_head_dim).transpose(0, 1).contiguous()
            patch = patch.view(bsz, prefix_head, patch_len, prefix_head_dim)   # [bs, num_heads, patch_len, head_dim]

            custom_score = torch.matmul(patch, attn)
            custom_score += attn_bias
            custom_score_sigmoid = F.sigmoid(custom_score)   
            sum_attn = torch.sum(custom_score_sigmoid, dim=-1, keepdim=True)
            conditional_norm = torch.clip(sum_attn / self.threshold, min=torch.tensor(1.0, dtype=torch.float16, device=sum_attn.device))
            custom_attn = custom_score_sigmoid / conditional_norm      # for visual: [bs, 12, 197, prefix_len]; for text: [n_cls, 8, 77, prefix_len]
            

            if batch_weight is not None:
                stat_attn = custom_score[:, :, 0, :].detach().to(torch.float32)    # [bs, 12, prefix_len]; 
                
                if not self.text_layer:
                    dist = Normal(self.means[prompt_ids], torch.sqrt(torch.clamp(self.vars[prompt_ids], min=1e-6)))
                    log_probs = dist.log_prob(stat_attn)
                    probability = torch.sigmoid(log_probs)
                    
                    if torch.mean(batch_weight) < 0.825:
                        uncertainty = (1 - probability.unsqueeze(2))
                        custom_attn = torch.where(custom_attn > uncertainty, custom_attn, torch.zeros_like(custom_attn))
                    
                else:
                    if torch.mean(batch_weight) < 0.825:
                        uncertainty = (1 - batch_weight.unsqueeze(1).unsqueeze(2).unsqueeze(3))
                        custom_attn = torch.where(custom_attn > uncertainty, custom_attn, torch.zeros_like(custom_attn))
                        
            else:
                if not self.text_layer:
                    stat_attn = custom_score[:, :, 0, :].detach().to(torch.float32)    # [bs, 12, prefix_len]; 
                    self.stat_sum += torch.sum(stat_attn, dim=0)                       # [12, prefix_len]; 
                    self.stat_sum_sqr += torch.sum(stat_attn ** 2, dim=0)              # [12, prefix_len]; 
                    self.stat_total_iter += bsz
                
                    if aux_outputs["is_last"]:
                        mean = self.stat_sum / self.stat_total_iter
                        var = self.stat_sum_sqr / self.stat_total_iter - mean ** 2
                        self.means[prompt_ids] = mean
                        self.vars[prompt_ids] = var
                        self.stat_sum = torch.zeros_like(self.stat_sum)
                        self.stat_sum_sqr = torch.zeros_like(self.stat_sum_sqr)
                        self.stat_total_iter = 0
                
            rt = conditional_norm - self.threshold
            
            if self.add_adapter:
                dpw_adapter = self.dpw_adapter[prompt_ids]
                adapter_input = query.transpose(0, 1).contiguous().view(patch_len*bsz, dim).clone()
                custom_value = dpw_adapter(adapter_input).contiguous()   # [bsz, patch_len, embed_dim] 

            if prompt_ids.size(0) == 1:
                prompt_ids = prompt_ids.repeat(query.size(1))
            prefix = self.prefix_pool[prompt_ids]  # [bs, 1, prefix_len, embed_dim]
            
        attn_output, attn_output_weights, aux_outputs = multi_head_attention_forward(
            query, key, value, self.embed_dim, self.num_heads,
            self.in_proj_weight, self.in_proj_bias,
            None, None, False,
            0., self.out_proj.weight, self.out_proj.bias,
            training=self.training,
            need_weights=need_weights,
            attn_mask=attn_mask, 
            prefix=prefix, 
            batch_weight=batch_weight,
            custom_attn=custom_attn,
            prefix_head=prefix_head,
            custom_value=custom_value,
            rt=rt,
            aux_outputs=aux_outputs)    
        return attn_output, aux_outputs