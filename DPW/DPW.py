import os.path as osp
import os
import json
import statistics
import time
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import numpy as np

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from .utils import cosine_loss_3d, cal_MTIL_metrics

from continuum.metrics import Logger
from DPW.utils import build_cosine_scheduler
from DPW.datasets import parse_sample

from torch.distributions.multivariate_normal import MultivariateNormal

from torchinfo import summary
from collections import defaultdict
import copy
from torch.nn import functional as F

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg, with_ori=False):
    backbone_name = cfg.model_backbone_name
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"vision_depth": cfg.DPW.prompt_depth_vision,
                      "language_depth": cfg.DPW.prompt_depth_text, "vision_ctx": cfg.DPW.n_ctx_vision,
                      "language_ctx": cfg.DPW.n_ctx_text,
                      "pool_size": cfg.nb_task}
    train_model = clip.build_model(state_dict or model.state_dict(), design_details)

    if with_ori:
        design_details = {"vision_depth": 0,
                          "language_depth": 0, "vision_ctx": 0,
                          "language_ctx": 0}
        ori_model = clip.build_model(state_dict or model.state_dict(), design_details)
        return train_model, ori_model

    return train_model



def _get_clones(module, N):
    return torch.nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)
    
class TextEncoder(nn.Module):
    def __init__(self, clip_model, cfg):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        
    def init_attn(self, fine_grained, tgt_task_id):  
        self.transformer.init_attn(fine_grained, tgt_task_id)
        
    def forward(self, prompts, tokenized_prompts, indices, batch_weight=None, aux_outputs=None):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x, aux_outputs = self.transformer(x, indices, batch_weight, aux_outputs)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x, aux_outputs


class PromptProcessor(nn.Module):
    def __init__(self, cfg, classnames, templates, clip_model, clip_model_ori):
        super().__init__()

        dtype = clip_model.dtype
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.input_size[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        self.cfg = cfg

        if isinstance(classnames[0], list):
            self.n_cls = 0
            self.class_ids_per_task = []
            self.classnames = []
            for idx, cls_name in enumerate(classnames):
                cur_n = len(cls_name)
                self.class_ids_per_task.append([i for i in range(self.n_cls, self.n_cls+cur_n)])
                cls_name = [templates[idx](name) for name in cls_name]
                self.classnames += cls_name
                self.n_cls += cur_n
        else:
            raise NotImplementedError
        self.cur_n_cls = 0

        self.classnames = [name.replace("_", " ") for name in self.classnames]
        self.all_name_lens = [len(_tokenizer.encode(name)) for name in self.classnames]
        all_prompts = [name for name in self.classnames]
        self.register_buffer("all_tokenized_prompts", torch.cat([clip.tokenize(p) for p in all_prompts]))
        with torch.no_grad():
            self.register_buffer("all_embedding", clip_model.token_embedding(self.all_tokenized_prompts).type(clip_model.dtype))
            self.register_buffer("all_fixed_embeddings", clip_model_ori.encode_text(self.all_tokenized_prompts.cuda()))
            
        # init with all classes, but will be updated before training and testing
        self.register_buffer("token_prefix", self.all_embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", self.all_embedding[:, 1:, :])  # CLS, EOS
        self.register_buffer("tokenized_prompts", self.all_tokenized_prompts.clone())


    def forward(self, indices=None):
        if indices is not None:
            batch_size = indices.size(0)
        else:
            batch_size = 1
        prefix = self.token_prefix.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # [bs, n_cls, 1, ctx_dim]
        suffix = self.token_suffix.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # [bs, n_cls, ..., ctx_dim]
        prompts = torch.cat([prefix, suffix], dim=2)  # [bs, n_cls, 77, ctx_dim]
        prompts = prompts.view(batch_size*self.cur_n_cls, prompts.size(2), prompts.size(3))  # [bs*n_cls, 77, ctx_dim]
        tokenized_prompts = self.tokenized_prompts.unsqueeze(0).repeat(batch_size, 1, 1).view(batch_size*self.cur_n_cls, -1)  # [bs*n_cls, 77, tkn_dim]
        return prompts, tokenized_prompts
    

    def update_classnames(self, task_id):
        class_idx = self.class_ids_per_task[task_id]
        class_idx_tensor = torch.tensor(class_idx, dtype=torch.int, device=self.all_embedding.device)
        self.token_prefix = self.all_embedding[class_idx_tensor, :1, :]
        self.token_suffix = self.all_embedding[class_idx_tensor, 1:, :]
        self.tokenized_prompts = self.all_tokenized_prompts[class_idx_tensor]
        self.name_lens = [self.all_name_lens[idx] for idx in class_idx]
        self.cur_n_cls = len(class_idx)
        
        with torch.no_grad():
            self.fixed_embeddings = self.all_fixed_embeddings[class_idx_tensor]
        

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, templates, clip_model, clip_model_ori=None):
        super().__init__()
        self.prompt_processor = PromptProcessor(cfg, classnames, templates, clip_model, clip_model_ori.cuda())
        self.image_encoder = clip_model.visual
        self.image_encoder_ori = clip_model_ori.visual
        self.text_encoder = TextEncoder(clip_model, cfg)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.vis_dim = clip_model.visual.output_dim
        self.pool_size = cfg.nb_task
        self.visual_prompt = cfg.DPW.prompt_depth_vision > 0
        self.batchwise_prompt = cfg.DPW.batchwise_prompt
        self.is_CIL = cfg.is_CIL


        self.register_buffer("means", torch.empty(self.pool_size, self.vis_dim, dtype=torch.float))
        self.register_buffer("covars", torch.empty(self.pool_size, self.vis_dim, self.vis_dim, dtype=torch.float))
        self.register_buffer("task_learnt", torch.tensor(0, dtype=torch.int))
        
        

    def forward(self, image, task_ids=None, trained_task=True, is_last=False):
        res = {}
        batch_weight = None
        text_batch_weight = None
        
        aux_outputs = {}
        aux_outputs["trained_task"] = trained_task
        aux_outputs["is_last"] = is_last
        
        with torch.no_grad():
            image_features, visual_features_ori = self.image_encoder_ori(image.type(self.dtype))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            res["image_features"] = image_features.detach()
        
        if task_ids is not None:
            task_ids = task_ids.type(torch.int).to(image.device)
            assert (task_ids == task_ids[0]).all()
            indices = task_ids[0:1]
            indices = indices.unsqueeze(1)  # size [1, 1]  

        else:
            dists = [MultivariateNormal(self.means[i], self.covars[i]) for i in range(self.task_learnt.item())]
            log_probs = torch.vstack([dist.log_prob(image_features) for dist in dists]).t()   # [bs, cur_learnt_task_num]
                
            if self.is_CIL and trained_task:      
                probs = torch.sigmoid(log_probs/512-1.0)               # [bs, cur_learnt_task_num]
                batch_weight, best_indices = torch.max(probs, dim=1)
                best_indices = best_indices.unsqueeze(0)                         # [1, selected_prompt_num]
                res["indices"] = best_indices
                image_features, aux_outputs = self.image_encoder(image.type(self.dtype), best_indices, batch_weight, aux_outputs)  # [bs, model_dim]
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                           
                logits = []    
                logit_scale = self.logit_scale.exp()
                for t in range(self.task_learnt.item()):
                    self.prompt_processor.update_classnames(t)
                    task_confidence = probs[:, t].repeat(self.prompt_processor.cur_n_cls)       # [bs]
                    indices = torch.tensor([[t]], device='cuda')
                    prompts, tokenized_prompts = self.prompt_processor(indices)     # [bs*n_cls, 77, ctx_dim]
                    text_feature, aux_outputs = self.text_encoder(prompts, tokenized_prompts, indices, task_confidence, aux_outputs)  # [bs*n_cls, model_dim]
                    text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
                    
                    logit = logit_scale * image_features @ text_feature.t()  # [bs, n_cls]
                    logit *= task_confidence
                    
                    logits.append(logit)
                    
                logits = torch.cat(logits, dim=1)

                res["outputs"] = logits
                res["aux_outputs"] = aux_outputs
                return res
            
            else:
                topk, indices = log_probs.topk(k=1, dim=1)  # [bs, selected_prompt_num]
                exp_part = topk.squeeze(1)/512-1.0
                batch_weight = torch.sigmoid(exp_part)  # [bs]
            
                text_batch_weight = batch_weight.mean(dim=0, keepdim=True).repeat(self.prompt_processor.cur_n_cls)
                res["text_batch_weight"] = text_batch_weight[0].item()
                res["raw_indices"] = indices
                if self.batchwise_prompt:
                    prompt_id, id_counts = torch.unique(indices, return_counts=True, sorted=True)
                    _, major_idx = torch.topk(id_counts, k=1)
                    indices = prompt_id[major_idx]
                    indices = indices.unsqueeze(0)  # [1, selected_prompt_num]
            
        
        res["indices"] = indices

        if self.visual_prompt:
            image_features, aux_outputs = self.image_encoder(image.type(self.dtype), indices, batch_weight, aux_outputs)  # [bs, model_dim]
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
        prompts, tokenized_prompts = self.prompt_processor(indices)  # [bs*n_cls, 77, ctx_dim]
        text_features, aux_outputs = self.text_encoder(prompts, tokenized_prompts, indices, text_batch_weight, aux_outputs)  # [bs*n_cls, model_dim]
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        
        logit_scale = self.logit_scale.exp()
        if indices.size(0) == 1:
            logits = logit_scale * image_features @ text_features.t()  # [bs, n_cls]
        else:
            text_features_resize = text_features.view(image_features.size(0), -1, text_features.size(1))  # [bs, n_cls, model_dim]
            image_features_resize = image_features.unsqueeze(1)  # [bs, 1, model_dim]
            logits = logit_scale * image_features_resize @ text_features_resize.permute(0, 2, 1)  # [bs, 1, n_cls]
            logits = logits.squeeze(1)  # [bs, n_cls]
        res["outputs"] = logits
        
        res["aux_outputs"] = aux_outputs
        
        return res

    
    def update_classnames(self, task_id):
        self.prompt_processor.update_classnames(task_id)


class DPW:
    def __init__(self, cfg, device, classes_names, templates, load_file=None):
        if load_file is not None:
            self.load_file = load_file
        self.build_model(cfg, device, classes_names, templates, load_file)


    def build_model(self, cfg, device, classes_names, templates, load_file=None):

        print(f"Loading CLIP (backbone: {cfg.model_backbone_name})")
        clip_model, clip_model_ori = load_clip_to_cpu(cfg, with_ori=True)

        print("Building custom CLIP")
        model = CustomCLIP(cfg, classes_names, templates, clip_model, clip_model_ori)

        print("Turning off gradients in both the image and the text encoder")
        names_to_update = ["prompt_key", "prefix_pool", "dpw"]

        for name, param in model.named_parameters():
            update_flag = False
            for name_to_update in names_to_update:
                if name_to_update in name:
                    update_flag = True
            if not update_flag:
                param.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        para_log = f"Parameters to be updated: {enabled}"
        print(para_log)
        f = open(osp.join(cfg.log_path, 'output.txt'), 'a')
        f.write(para_log + '\n')
        f.close()

        self.model = model
        self.devices = device
        self.device = device[0]

        if load_file:
            self.load_model(None, None, load_file)
        
        self.model.to(device[0])
        if len(device) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=device)
            
        self.model_wo_dp = self.model.module if len(device) > 1 else self.model


    def save_model(self, cfg, task_id):
        save_dict = {}
        for name, para in self.model.named_parameters():
            if para.requires_grad:
                save_dict[name] = para
        for name, para in self.model.named_buffers():  # for gaussian parameters
            if "means" in name or "covars" in name or "task_learnt" in name:
                save_dict[name] = para
        save_dir = os.path.join(cfg.log_path, 'ckpt')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(save_dict, os.path.join(save_dir, f'task_{task_id}.pt'))
    
    
    def load_model(self, cfg, task_id, load_file=None):
        if load_file is None:
            load_file = os.path.join(cfg.log_path, 'ckpt', f'task_{task_id}.pt')
        if not osp.exists(load_file):
            raise FileNotFoundError('Model not found at "{}"'.format(load_file))

        state_dict = torch.load(load_file, map_location="cpu")

        print(f"Loading weights from {load_file}")
        # set strict=False
        self.model.load_state_dict(state_dict, strict=False)

        return [i for i in state_dict.keys()]


    def inter_class_variance(self, train_loader, task_id, cfg):
        class_features = defaultdict(list)
        
        prompts, tokenized_prompts = self.model_wo_dp.prompt_processor()
        text_features = self.model_wo_dp.prompt_processor.fixed_embeddings
        text_features = text_features.view(1, -1, text_features.size(1))  # [bs, n_cls, model_dim]
        text_features = text_features[0]           # [n_cls, model_dim]
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        with torch.no_grad():
            correct_predictions = 0
            total_samples = 0
            for sample in train_loader:
                inputs, labels, _ = parse_sample(sample, is_train=False, task_id=task_id, cfg=cfg)
                inputs = inputs.type(self.model_wo_dp.dtype).to(self.device)
                labels = labels.to(self.device)

                image_features, _ = self.model_wo_dp.image_encoder_ori(inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                similarity = image_features @ text_features.T  # [batch_size, n_cls]

                predicted_classes = similarity.argmax(dim=1)

                correct_predictions += (predicted_classes == labels).sum().item()
                total_samples += labels.size(0)
                

                for label in torch.unique(labels):
                    mask = labels == label
                    features_of_label = image_features[mask]
                    class_features[label.item()].append(features_of_label.detach())    
            zs_accuracy = correct_predictions / total_samples


            class_means = {}
            intra_class_similarities = []
            
            for label in class_features:
                class_features[label] = torch.cat(class_features[label], dim=0)
                class_means[label] = class_features[label].mean(dim=0)
                class_means[label] = class_means[label] / class_means[label].norm()

                class_similarity_matrix = torch.mm(class_features[label], class_features[label].t())
                num_class_samples = class_features[label].size(0)
                mask = ~torch.eye(num_class_samples, dtype=bool, device=self.device)
                intra_class_similarities.append(class_similarity_matrix[mask].mean().item())

            mean_intra_class_similarity_image = sum(intra_class_similarities) / len(intra_class_similarities)
            intra_class_variance_image = 1 - mean_intra_class_similarity_image

            labels_list = list(class_means.keys())
            mean_features = torch.stack([class_means[label] for label in labels_list], dim=0)
            cosine_sim_matrix_image = torch.mm(mean_features, mean_features.t())

            num_classes = len(labels_list)
            mask = ~torch.eye(num_classes, dtype=bool, device=self.device)
            inter_class_cosine_similarities_image = cosine_sim_matrix_image[mask]

            mean_inter_class_similarity_image = inter_class_cosine_similarities_image.mean().item()
            std_inter_class_similarity_image = inter_class_cosine_similarities_image.std().item()
            inter_class_variance_image = 1 - mean_inter_class_similarity_image


        num_cls = text_features.size(0)
        cosine_sim_matrix_text = torch.mm(text_features, text_features.t())
        inter_class_cosine_similarities_text = cosine_sim_matrix_text[mask]

        mean_inter_class_similarity_text = inter_class_cosine_similarities_text.mean().item()
        inter_class_variance_text = 1 - mean_inter_class_similarity_text
        
        easyness = ((inter_class_variance_image + inter_class_variance_text) / 2) / (intra_class_variance_image + (10 / num_cls))
        fine_grained = easyness / zs_accuracy
        
        return fine_grained
    



    def train_and_eval(self, cfg, datasets):
        acc_list = []
        metric_logger = Logger(list_subsets=["train", "test"])

        metric_writer = open(os.path.join(cfg.log_path, 'metrics.json'), 'w')
        
        if cfg.zero_shot:
            with torch.no_grad():
                for cur_task in tqdm(range(cfg.nb_task)):
                    self.update_classnames(cur_task)
                    eval_loader = self.get_dataloader(cfg, datasets['test'], cur_task, is_train=False)
                    for sample in eval_loader:
                        inputs, targets, task_ids = parse_sample(sample, is_train=False, task_id=cur_task, cfg=cfg)
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                        res = self.model(inputs, task_ids)
                        outputs = res["outputs"]
                        metric_logger.add([outputs.cpu().argmax(dim=1), targets.cpu(), task_ids], subset="test")
                cur_all_task_acc = metric_logger.accuracy_per_task
                acc_list.append(cur_all_task_acc)
                log = {'acc_per_task': [round(100 * acc_t, 2) for acc_t in cur_all_task_acc]}
                metric_writer.write(json.dumps(log) + '\n')
                metric_writer.flush()
                print(log)
                return
         
        if cfg.eval_only:
            for task_id in range(cfg.nb_task):
                self.update_classnames(task_id)
                fine_grained = self.inter_class_variance(self.get_dataloader(cfg, datasets['train'], task_id, is_train=True, drop_last=False), task_id, cfg)
                self.model.image_encoder.init_attn(fine_grained, task_id)
                self.model.text_encoder.init_attn(fine_grained, task_id)
                self.load_model(None, None, self.load_file)
                self.model.task_learnt.fill_(task_id + 1)
                self.eval_all(cfg, datasets, metric_logger, metric_writer, acc_list)
        
        else:
            total_time = 0.0
            total_time_eval = 0.0
            for task_id in range(cfg.nb_task):
                print(f"Training for task {task_id} has started.")

                start_time = time.time()
                self.train_one_task(cfg, task_id, datasets, metric_logger)
                end_time = time.time()
                total_time += (end_time - start_time)

                if datasets['val']:
                    keys = self.load_model(cfg, task_id)
                    log = f"Load best epoch weight (epoch {self.best_epoch}), parameters {keys}."
                    print(log)
                    with open(osp.join(cfg.log_path, 'output.txt'), 'a') as f:
                        f.write(log + '\n')

                print(f"Evaluation for task {task_id} has started.")

                if task_id == (cfg.nb_task - 1):
                    start_time = time.time()
                self.eval_all(cfg, datasets, metric_logger, metric_writer, acc_list, task_id)
                if task_id == (cfg.nb_task - 1):
                    end_time = time.time()
                    total_time_eval += (end_time - start_time)

            hours, remainder = divmod(total_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            f = open(osp.join(cfg.log_path, 'time.txt'), 'a')
            f.write(f"Total training time: {int(hours)}h {int(minutes)}m {seconds:.2f}s" + '\n')

            hours, remainder = divmod(total_time_eval, 3600)
            minutes, seconds = divmod(remainder, 60)
            f.write(f"Total evaluation time: {int(hours)}h {int(minutes)}m {seconds:.2f}s" + '\n')
            f.close()

        res = cal_MTIL_metrics(acc_list)
        metric_writer.write(json.dumps(res["transfer"]) + '\n')
        metric_writer.write(json.dumps(res["avg"]) + '\n')
        metric_writer.write(json.dumps(res["last"]) + '\n')
        metric_writer.write(json.dumps(res["results_mean"]) + '\n')
        metric_writer.flush()


    def train_one_task(self, cfg, task_id, datasets, metric_logger):

        train_dataset, val_dataset, eval_dataset = datasets['train'], datasets['val'], datasets['test']
        train_loader = self.get_dataloader(cfg, train_dataset, task_id, is_train=True, drop_last=True)
        self.update_classnames(task_id)
        self.model.train()

        per_epoch_steps = len(train_loader)
        
        
        with torch.no_grad():
            cpu_rng_state = torch.get_rng_state()
            gpu_rng_state = torch.cuda.get_rng_state_all()
            fine_grained = self.inter_class_variance(self.get_dataloader(cfg, train_dataset, task_id, is_train=True, drop_last=False), task_id, cfg)
            torch.set_rng_state(cpu_rng_state)
            torch.cuda.set_rng_state_all(gpu_rng_state)
            self.model.image_encoder.init_attn(fine_grained, task_id)
            self.model.text_encoder.init_attn(fine_grained, task_id)
        
        
        if cfg.DPW.optim.name == 'SGD':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=cfg.DPW.optim.lr, weight_decay=cfg.DPW.optim.weight_decay)
        else:
            raise NotImplementedError
        
        if cfg.DPW.optim.lr_scheduler == 'cosine':
            scheduler = build_cosine_scheduler(optimizer, lr=cfg.DPW.optim.lr, total_step=cfg.DPW.optim.max_epoch*per_epoch_steps)
        elif cfg.DPW.optim.lr_scheduler == 'no':
            scheduler = None
        else:
            raise NotImplementedError

        self.best_epoch = -1
        self.best_acc = -1

        all_image_features = torch.empty([0, self.model_wo_dp.vis_dim], dtype=self.model_wo_dp.dtype, device=self.device)
        with torch.no_grad():
            for sample in train_loader:
                inputs, _, _ = parse_sample(sample, is_train=False, task_id=task_id, cfg=cfg)
                image_features, _ = self.model_wo_dp.image_encoder_ori(inputs.type(self.model_wo_dp.dtype).to(self.device))
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                all_image_features = torch.cat([all_image_features, image_features.detach()], dim=0)

        all_image_features = all_image_features.type(torch.float)  # to avoid precision problems
        mean = all_image_features.mean(dim=0)
        delta = (all_image_features - mean.unsqueeze(0))
        covar = delta.t() @ delta / (all_image_features.size(0) - 1)
        covar +=  torch.eye(covar.size(0), device=covar.device, dtype=torch.float)*1e-7  # to avoid precision problems
        self.model_wo_dp.means[task_id] = mean
        self.model_wo_dp.covars[task_id] = covar
        self.model_wo_dp.task_learnt += 1
        
        
        with torch.no_grad():                  
            cpu_rng_state = torch.get_rng_state()
            gpu_rng_state = torch.cuda.get_rng_state_all()
            log = str(summary(self.model, input_size=(1, 3, 224, 224), verbose=1))
            f = open(osp.join(cfg.log_path, 'output.txt'), 'a')
            f.write(log + '\n')
            f.close()    
            torch.set_rng_state(cpu_rng_state)
            torch.cuda.set_rng_state_all(gpu_rng_state)
        self.update_classnames(task_id)
            
        for epoch in tqdm(range(cfg.DPW.optim.max_epoch)):
            main_loss_tot = 0
            loss_num = 0
            consistency_loss_tot = 0
            
            for idx, sample in enumerate(train_loader):
                if scheduler:
                    cur_iter_idx = epoch*per_epoch_steps+idx
                    scheduler.step(cur_iter_idx)

                inputs, targets, task_ids = parse_sample(sample, is_train=True, task_id=task_id, cfg=cfg)
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                
                is_last = (idx == len(train_loader) - 1)
                res = self.model(inputs, task_ids, is_last=is_last)
                outputs = res["outputs"]
                loss_main = F.cross_entropy(outputs, targets)
                
                loss = loss_main
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                main_loss_tot += loss_main.item()
                loss_num += 1

                metric_logger.add([outputs.detach().cpu().argmax(dim=1), targets.cpu(), task_ids], subset="train")
            
            log = f"\ntask{task_id}_epoch{epoch}:\n"
            log += f"train acc: {metric_logger.online_accuracy}"
            metric_logger.end_epoch()
            f = open(osp.join(cfg.log_path, 'output.txt'), 'a')
            f.write(log + '\n')
            f.close()

            log = f"avg main loss {round(main_loss_tot/loss_num, 5)}"
            f = open(osp.join(cfg.log_path, 'output.txt'), 'a')
            f.write(log + '\n')
            f.close()

            
            if val_dataset:
                self.model.eval()
                self.update_classnames(task_id)
                val_loader = self.get_dataloader(cfg, val_dataset, task_id, is_train=False)
                cur_right = torch.FloatTensor([0]).to(self.device)
                cur_all = torch.FloatTensor([0]).to(self.device)
                with torch.no_grad():
                    for sample in val_loader:
                        inputs, targets, task_ids = parse_sample(sample, is_train=False, task_id=task_id, cfg=cfg)
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                        res = self.model(inputs, task_ids)
                        outputs = res["outputs"]
                        cur_right += torch.sum((outputs.argmax(dim=1)==targets))
                        cur_all += targets.size(0)
                cur_acc = cur_right/cur_all
                if cur_acc > self.best_acc:
                    self.best_epoch = epoch
                    self.best_acc = cur_acc
                    self.save_model(cfg, task_id)
                self.update_classnames(task_id)
                self.model.train()


    def eval_all(self, cfg, datasets, metric_logger, metric_writer, acc_list, trained_task=-1):
        eval_dataset = datasets['test']
        self.model.eval()
        
        aux_outputs_total = {}
        dataset_name = ["Aircraft", "Caltech101", "CIFAR100", "DTD", "EuroSAT", "Flowers", "Food", "MNIST", "OxfordPet", "Cars", "SUN397"]

        for cur_task in tqdm(range(cfg.nb_task)):
            self.update_classnames(cur_task)
            
            if (cfg.is_CIL and self.model.task_learnt.item() >= cur_task + 1):
                CIL_flag = True
            else:
                CIL_flag = False
            
            eval_loader = self.get_dataloader(cfg, eval_dataset, cur_task, is_train=False, CIL_flag=CIL_flag)
            self.evaluate(cfg, eval_loader, cur_task, metric_logger)
 
        cur_all_task_acc = metric_logger.accuracy_per_task
        acc_list.append(cur_all_task_acc)
        log = {'acc_per_task': [round(100 * acc_t, 2) for acc_t in cur_all_task_acc]}
        metric_writer.write(json.dumps(log) + '\n')
        metric_writer.flush()
        print(log)
        metric_logger.end_task()


    def evaluate(self, cfg, loader, task_id, metric_logger=None):
        class_ids = []
        class_ids.append(0)
        for ids in self.model.prompt_processor.class_ids_per_task:
            class_ids.append(max(ids)+1)

        if self.model.task_learnt.item() < task_id + 1:
            trained_task = False
        else:
            trained_task = True
        
        with torch.no_grad():
            for sample in loader:
                inputs, targets, task_ids = parse_sample(sample, is_train=False, task_id=task_id, cfg=cfg)
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                                
                res = self.model(inputs, trained_task=trained_task)
                outputs = res["outputs"]
                
                if cfg.is_CIL and trained_task:
                    targets += class_ids[task_id]
                
                if metric_logger:
                    metric_logger.add([outputs.cpu().argmax(dim=1), targets.cpu(), task_ids], subset="test")
                    
        return 
    

    def get_dataloader(self, cfg, dataset, task_id, is_train, drop_last=False, CIL_flag=False):
        batch_size = cfg.DPW.optim.batch_size
        if isinstance(dataset, list):
            if cfg.DPW.batchwise_prompt and (not is_train):
                if CIL_flag:
                    batch_size = 1
                else:
                    batch_size = 256
                
            if not is_train:
                drop_last = False
            
            loader = DataLoader(dataset[task_id], batch_size=int(batch_size), shuffle=is_train, num_workers=8, drop_last=drop_last)
        else:
            raise NotImplementedError
        return loader


    def update_classnames(self, task_id):
        if isinstance(self.model, torch.nn.DataParallel):
            self.model.module.update_classnames(task_id)
        else:
            self.model.update_classnames(task_id)
