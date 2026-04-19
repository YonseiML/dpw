import argparse
import os
import sys

from DPW.datasets import get_dataset
from DPW.utils import get_transform, set_random_seed
from DPW.setup_cfg import setup_cfg, print_args
from DPW.DPW import DPW



def run_exp(cfg):
    device = [int(s) for s in cfg.gpu_id.split(',')]

    cfg.use_validation = cfg.use_validation

    train_dataset, classes_names, templates = get_dataset(cfg, split='train', transforms=get_transform(cfg))
    val_dataset, _, _ = get_dataset(cfg, split='val', transforms=get_transform(cfg))
    eval_dataset, _, _ = get_dataset(cfg, split='test', transforms=get_transform(cfg))
    cfg.nb_task = len(eval_dataset)

    load_file = cfg.load_file if cfg.load_file else None
    trainer = DPW(cfg, device, classes_names, templates, load_file=load_file)

    datasets = {'train': train_dataset, 'val': val_dataset, 'test': eval_dataset}
    trainer.train_and_eval(cfg, datasets)


def main(args):
    cfg = setup_cfg(args)
    cfg.command = ' '.join(sys.argv)
    
    if "seed" in args:
        if args.seed:
            cfg.seed = args.seed
    print_args(args, cfg)
    if "log_path" in args:
        if args.log_path:
            cfg.log_path = args.log_path
    if "load_file" in args:
        if args.load_file:
            cfg.load_file = args.load_file
    if "eval_only" in args:
        if args.eval_only:
            cfg.eval_only = args.eval_only
    if "is_CIL" in args:
        if args.is_CIL:
            cfg.is_CIL = args.is_CIL
            
            

    shot = f'-FS' if cfg.num_shots > 0 else ''

    if not os.path.exists(cfg.log_path):
        os.makedirs(cfg.log_path)
    with open(os.path.join(cfg.log_path, 'config.yaml'), 'w') as f: 
        f.write(cfg.dump())
    
    print_args(args, cfg)

    set_random_seed(cfg.seed)
    run_exp(cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, default="configs/MTIL.yaml", help="path to config")
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu id")
    parser.add_argument("--log_path", type=str)
    parser.add_argument("--load_file", type=str)
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument(
        "--seed", type=int, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--is_CIL", action='store_true'
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    main(args)