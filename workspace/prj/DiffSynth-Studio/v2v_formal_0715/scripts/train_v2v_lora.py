#!/usr/bin/env python3
"""Formal Wan2.2 paired V2V LoRA training: source video + reference image -> target video."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path

import pytorch_lightning as L
from pytorch_lightning.callbacks import Callback, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger
from omegaconf import OmegaConf
from peft import LoraConfig
from safetensors.torch import save_file
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from diffsynth.models.wan_video_text_encoder import WanTextEncoder
from diffsynth.models.wan_video_vae import WanVideoVAE38
from models.wan2.attn_process import ConditionAttnProcessor2_0
from models.wan2.transformer_wan import WanTransformer3DModel
from paired_v2v_dataset import PairedV2VDataset
from training_utils import filter_trainable_state_dict, materialize_vae_statistics
from v2v_components import build_timestep_sequence
from wan_flow_scheduler import FlowMatchScheduler


def _load_native_module(module: torch.nn.Module, path: str, prefix: str = "") -> torch.nn.Module:
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if "model_state" in state:
        state = state["model_state"]
    if prefix:
        state = {f"{prefix}{key}": value for key, value in state.items()}
    incompatible = module.load_state_dict(state, strict=False, assign=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"native checkpoint mismatch for {path}: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    return module


def load_text_encoder(path: str) -> WanTextEncoder:
    with torch.device("meta"):
        model = WanTextEncoder()
    return _load_native_module(model, path)


def load_vae(path: str) -> WanVideoVAE38:
    with torch.device("meta"):
        model = WanVideoVAE38()
    model = _load_native_module(model, path, prefix="model.")
    stats_path = Path(__file__).resolve().parents[1] / "configs/vae_config.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    materialize_vae_statistics(model, stats["latents_mean"], stats["latents_std"])
    return model


class FormalPairedV2VSystem(L.LightningModule):
    strict_loading = False

    def __init__(self, config: dict):
        super().__init__()
        self.save_hyperparameters({"config": config})
        self.config_data = config
        self._configured = False

    @property
    def cfg(self):
        return OmegaConf.create(self.config_data)

    def configure_model(self) -> None:
        if self._configured:
            return
        self._configured = True
        cfg = self.cfg
        print("[model] loading tokenizer", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model.tokenizer,
            local_files_only=True,
        )
        print("[model] loading native UMT5 encoder", flush=True)
        self.text_encoder = load_text_encoder(cfg.model.text_encoder).eval().requires_grad_(False)
        print("[model] loading native Wan2.2 VAE38", flush=True)
        self.vae = load_vae(cfg.model.vae).eval().requires_grad_(False)
        print("[model] loading converted Wan transformer", flush=True)
        self.transformer = WanTransformer3DModel.from_pretrained(
            cfg.model.transformer,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
        ).requires_grad_(False)

        lora = LoraConfig(
            r=int(cfg.lora.rank),
            lora_alpha=int(cfg.lora.alpha),
            lora_dropout=float(cfg.lora.dropout),
            init_lora_weights=True,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        )
        self.transformer.add_adapter(lora)
        for block in self.transformer.blocks:
            block.attn1.set_processor(ConditionAttnProcessor2_0())
        self.transformer.patch_embedding_extra = copy.deepcopy(
            self.transformer.patch_embedding
        ).requires_grad_(True)
        if bool(cfg.training.gradient_checkpointing):
            self.transformer.enable_gradient_checkpointing()

        self.flow_scheduler = FlowMatchScheduler(
            shift=float(cfg.training.flow_shift),
            sigma_min=0.0,
            extra_one_step=True,
        )
        self.flow_scheduler.set_timesteps(1000, training=True)
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        total = sum(parameter.numel() for parameter in self.transformer.parameters())
        print(
            f"[model] transformer parameters={total:,}; trainable parameters={trainable:,}",
            flush=True,
        )

    def on_train_epoch_start(self) -> None:
        self.text_encoder.eval()
        self.vae.eval()

    def _encode_video(self, video: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            with torch.autocast(device_type="cuda", enabled=False):
                return self.vae.model.encode(video.to(torch.float32), self.vae.scale)

    def _encode_prompt(self, prompts: list[str]) -> torch.Tensor:
        tokens = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=512,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        ids = tokens.input_ids.to(self.device)
        mask = tokens.attention_mask.to(self.device)
        with torch.no_grad():
            embeddings = self.text_encoder(ids, mask)
        embeddings = embeddings * mask.unsqueeze(-1).to(embeddings.dtype)
        return embeddings

    def _token_count(self, latent: torch.Tensor) -> int:
        patch_t, patch_h, patch_w = self.transformer.config.patch_size
        return (
            latent.shape[2] // patch_t
            * (latent.shape[3] // patch_h)
            * (latent.shape[4] // patch_w)
        )

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        source = self._encode_video(batch["source_video"])
        target = self._encode_video(batch["target_video"])
        reference = self._encode_video(batch["reference_image"].unsqueeze(2))
        prompt_embeddings = self._encode_prompt(list(batch["prompt"]))

        noise = torch.randn_like(target)
        batch_size = target.shape[0]
        timestep_ids = torch.randint(
            0,
            self.flow_scheduler.num_train_timesteps,
            (batch_size,),
        )
        timestep = self.flow_scheduler.timesteps[timestep_ids].to(
            device=target.device,
            dtype=target.dtype,
        )
        noisy_target = self.flow_scheduler.add_noise(target, noise, timestep)
        velocity_target = self.flow_scheduler.training_target(target, noise, timestep)
        timestep_sequence = build_timestep_sequence(
            timestep=timestep,
            main_tokens=self._token_count(noisy_target),
            source_tokens=self._token_count(source),
            reference_tokens=self._token_count(reference),
        )

        prediction = self.transformer(
            hidden_states=noisy_target.to(torch.bfloat16),
            encoder_hidden_states=prompt_embeddings.to(torch.bfloat16),
            timestep=timestep_sequence,
            attention_kwargs={
                "encoder_contion_states": source.to(torch.bfloat16),
                "encoder_first_states": reference.to(torch.bfloat16),
            },
            return_dict=False,
        )[0]
        per_element = torch.nn.functional.mse_loss(
            prediction.float(), velocity_target.float(), reduction="none"
        )
        weight_timestep = timestep.to(self.flow_scheduler.timesteps.device)
        weights = self.flow_scheduler.training_weight(weight_timestep).to(
            device=per_element.device,
            dtype=per_element.dtype,
        )
        loss = (per_element * weights[:, None, None, None, None]).mean()
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=self.trainer.world_size > 1,
            batch_size=batch_size,
        )
        self.log(
            "train/epoch",
            float(self.current_epoch),
            on_step=True,
            logger=True,
            batch_size=batch_size,
        )
        return loss

    def configure_optimizers(self):
        cfg = self.cfg
        parameters = [parameter for parameter in self.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(
            parameters,
            lr=float(cfg.optimizer.learning_rate),
            betas=(float(cfg.optimizer.beta1), float(cfg.optimizer.beta2)),
            weight_decay=float(cfg.optimizer.weight_decay),
            eps=float(cfg.optimizer.eps),
        )
        total_steps = max(int(self.trainer.estimated_stepping_batches), 1)
        warmup_steps = min(int(cfg.optimizer.warmup_steps), total_steps)

        def schedule(step: int) -> float:
            if step < warmup_steps:
                return float(step + 1) / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    def export_trainable_weights(self, output_file: Path) -> dict:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        state = filter_trainable_state_dict(self.state_dict())
        cpu_state = {
            key: value.detach().to(device="cpu").contiguous() for key, value in state.items()
        }
        temporary = output_file.with_name(f".{output_file.name}.tmp")
        save_file(cpu_state, str(temporary))
        os.replace(temporary, output_file)
        metadata = {
            "weights": str(output_file),
            "tensor_count": len(cpu_state),
            "tensor_bytes": sum(value.numel() * value.element_size() for value in cpu_state.values()),
            "base_model": self.cfg.model.transformer,
            "lora_rank": int(self.cfg.lora.rank),
            "lora_alpha": int(self.cfg.lora.alpha),
            "target_modules": ["to_q", "to_k", "to_v", "to_out.0"],
            "condition_branch": "patch_embedding_extra",
            "global_step": int(self.global_step),
            "epoch": int(self.current_epoch),
        }
        output_file.with_suffix(".json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        return metadata

    def on_train_end(self) -> None:
        if not self.trainer.is_global_zero:
            return
        output = Path(self.cfg.output.root) / "final_adapter"
        self.export_trainable_weights(
            output / "v2v_lora_and_source_patch.safetensors"
        )


class LatestAdapterCheckpoint(Callback):
    """Atomically overwrite one compact adapter checkpoint at a fixed step interval."""

    def __init__(self, output_root: Path, every_n_steps: int):
        self.output_file = output_root / "checkpoints/latest_v2v_lora_and_source_patch.safetensors"
        self.every_n_steps = every_n_steps

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if (
            trainer.is_global_zero
            and trainer.global_step > 0
            and trainer.global_step % self.every_n_steps == 0
        ):
            metadata = pl_module.export_trainable_weights(self.output_file)
            print(f"[checkpoint] wrote compact latest: {metadata}", flush=True)


def build_trainer(cfg) -> L.Trainer:
    visible = [gpu for gpu in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if gpu]
    if not visible:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must explicitly list nonzero physical GPUs")
    if "0" in visible:
        raise RuntimeError("GPU0 is forbidden for this formal run")
    if len(visible) > 3:
        raise RuntimeError("at most three GPUs may be used")

    output_root = Path(cfg.output.root)
    logger = CSVLogger(save_dir=str(output_root / "logs"), name="lightning")
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        LatestAdapterCheckpoint(
            output_root=output_root,
            every_n_steps=int(cfg.training.checkpoint_every_n_steps),
        ),
    ]
    return L.Trainer(
        accelerator="gpu",
        devices=len(visible),
        strategy="ddp" if len(visible) > 1 else "auto",
        precision=str(cfg.training.precision),
        max_epochs=int(cfg.training.max_epochs),
        accumulate_grad_batches=int(cfg.training.accumulate_grad_batches),
        gradient_clip_val=float(cfg.training.gradient_clip_val),
        gradient_clip_algorithm="norm",
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        limit_val_batches=0,
        deterministic=bool(cfg.training.deterministic),
        benchmark=not bool(cfg.training.deterministic),
        enable_checkpointing=False,
        limit_train_batches=cfg.training.get("limit_train_batches", 1.0),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--preflight-batches", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_float32_matmul_precision("high")
    cfg = OmegaConf.load(args.config)
    if args.preflight_batches:
        cfg.output.root = str(Path(cfg.output.root) / "preflight")
        cfg.training.limit_train_batches = args.preflight_batches
        cfg.training.max_epochs = 1
        cfg.training.checkpoint_every_n_steps = 1
    output_root = Path(cfg.output.root)
    output_root.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_root / "resolved_train_config.yaml")
    L.seed_everything(int(cfg.training.seed), workers=True)
    dataset = PairedV2VDataset(cfg.data.manifest, sample_frames=int(cfg.data.num_frames))
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.data.batch_size),
        shuffle=True,
        num_workers=int(cfg.data.num_workers),
        pin_memory=True,
        persistent_workers=int(cfg.data.num_workers) > 0,
        drop_last=True,
    )
    system = FormalPairedV2VSystem(OmegaConf.to_container(cfg, resolve=True))
    trainer = build_trainer(cfg)
    trainer.fit(system, train_dataloaders=loader)


if __name__ == "__main__":
    main()
