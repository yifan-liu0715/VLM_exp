# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
import argparse
import os
from omegaconf import OmegaConf
import wandb
from trainer.distillation import Trainer as ScoreDistillationTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--logdir", type=str, default="", help="Path to the directory to save logs")
    parser.add_argument("--wandb-save-dir", type=str, default="", help="Path to the directory to save wandb logs")
    parser.add_argument("--disable-wandb", action="store_true")

    args = parser.parse_args()
    config = OmegaConf.load(args.config_path)
    default_config = OmegaConf.load("configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)

    config_name = os.path.dirname(args.config_path).split("/")[-1]
    config.config_name = config_name
    config.logdir = args.logdir
    config.wandb_save_dir = args.wandb_save_dir
    config.disable_wandb = args.disable_wandb

    if config.trainer == "score_distillation":
        trainer = ScoreDistillationTrainer(config)
    
    trainer.train()

    # TokenBucketSampler starts a daemon preload thread whose _stop_flag upstream
    # never sets. Tearing down the interpreter with it still live aborts with
    # "terminate called without an active exception" (SIGABRT), which masks
    # whatever the real failure was. stage1's train_fsdp.py does the same.
    sampler = getattr(trainer, "batch_sampler", None)
    if sampler is not None:
        sampler._stop_flag = True
        sampler.thread.join(timeout=10)
    import torch.distributed as _dist
    if _dist.is_initialized():
        _dist.destroy_process_group()

    wandb.finish()


if __name__ == "__main__":
    main()
