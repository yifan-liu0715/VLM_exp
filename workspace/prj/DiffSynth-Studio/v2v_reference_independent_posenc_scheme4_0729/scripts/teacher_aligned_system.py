"""Teacher-aligned Wan V2V system with documented native VAE/text loading."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import random
import sys

import pytorch_lightning as lightning
from omegaconf import OmegaConf
from peft import LoraConfig
from safetensors.torch import save_file
import torch
from transformers import AutoTokenizer

from diffsynth.models.wan_video_text_encoder import WanTextEncoder
from diffsynth.models.wan_video_vae import WanVideoVAE38


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
TEACHER_REFERENCE = EXPERIMENT_ROOT / "teacher_reference"
if str(TEACHER_REFERENCE) not in sys.path:
    sys.path.insert(0, str(TEACHER_REFERENCE))

from models.wan2.attn_process import ConditionAttnProcessor2_0
from models.wan2.transformer_wan import WanTransformer3DModel
from tools.my_schedule import FlowMatchScheduler

from install_reference_position_encoding import (
    install_reference_position_encoding,
)
from training_contracts import scaled_learning_rate, teacher_lr_multiplier


def _load_native_module(
    module: torch.nn.Module,
    path: str,
    prefix: str = "",
) -> torch.nn.Module:
    state = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    if "model_state" in state:
        state = state["model_state"]
    if prefix:
        state = {f"{prefix}{key}": value for key, value in state.items()}
    incompatible = module.load_state_dict(state, strict=False, assign=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"native checkpoint mismatch for {path}: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    return module


def load_text_encoder(path: str) -> WanTextEncoder:
    with torch.device("meta"):
        model = WanTextEncoder()
    return _load_native_module(model, path)


def materialize_vae_statistics(model: WanVideoVAE38) -> None:
    stats = json.loads(
        (EXPERIMENT_ROOT / "configs" / "vae_config.json").read_text(
            encoding="utf-8"
        )
    )
    model.mean = torch.tensor(stats["latents_mean"], dtype=torch.float32)
    model.std = torch.tensor(stats["latents_std"], dtype=torch.float32)
    model.scale = [model.mean, 1.0 / model.std]


def load_vae(path: str) -> WanVideoVAE38:
    with torch.device("meta"):
        model = WanVideoVAE38()
    model = _load_native_module(model, path, prefix="model.")
    materialize_vae_statistics(model)
    return model


def build_timestep_sequence(
    timestep: torch.Tensor,
    main_tokens: int,
    source_tokens: int,
    reference_tokens: int,
) -> torch.Tensor:
    batch = timestep.shape[0]
    target = timestep[:, None].repeat(1, main_tokens)
    conditions = torch.zeros(
        batch,
        source_tokens + reference_tokens,
        device=timestep.device,
        dtype=timestep.dtype,
    )
    return torch.cat((target, conditions), dim=1)


def compact_trainable_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key: value
        for key, value in state.items()
        if (
            "lora_" in key
            or "patch_embedding_extra" in key
            or "reference_position_encoder" in key
        )
    }


def assert_reference_position_gradients(
    parameters: dict[str, torch.nn.Parameter],
) -> None:
    for name, parameter in parameters.items():
        gradient = parameter.grad
        if gradient is None:
            raise RuntimeError(
                f"Reference position gradient is missing for {name}"
            )
        if not torch.isfinite(gradient).all():
            raise RuntimeError(
                f"Reference position gradient must be finite for {name}"
            )
        if torch.count_nonzero(gradient) == 0:
            raise RuntimeError(
                f"Reference position gradient must be nonzero for {name}"
            )


class TeacherAlignedV2VSystem(lightning.LightningModule):
    strict_loading = False

    def __init__(self, config: dict):
        super().__init__()
        self.save_hyperparameters({"config": config})
        self.config_data = config
        self._configured = False
        self._reference_gradient_checked = False
        self.last_loss: float | None = None

    @property
    def cfg(self):
        return OmegaConf.create(self.config_data)

    def configure_model(self) -> None:
        if self._configured:
            return
        self._configured = True
        cfg = self.cfg
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model.tokenizer,
            local_files_only=True,
        )
        self.text_encoder = (
            load_text_encoder(cfg.model.text_encoder)
            .eval()
            .requires_grad_(False)
        )
        self.vae = load_vae(cfg.model.vae).eval().requires_grad_(False)
        self.transformer = WanTransformer3DModel.from_pretrained(
            cfg.model.transformer,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
        ).requires_grad_(False)

        self.transformer.add_adapter(
            LoraConfig(
                r=int(cfg.lora.rank),
                lora_alpha=int(cfg.lora.alpha),
                lora_dropout=float(cfg.lora.dropout),
                init_lora_weights=True,
                target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            )
        )
        for block in self.transformer.blocks:
            block.attn1.set_processor(ConditionAttnProcessor2_0())
        self.transformer.patch_embedding_extra = copy.deepcopy(
            self.transformer.patch_embedding
        ).requires_grad_(True)
        install_reference_position_encoding(
            self.transformer,
            cfg,
            trainable=True,
            enabled=True,
        )
        if bool(cfg.training.gradient_checkpointing):
            self.transformer.enable_gradient_checkpointing()

        self.flow_scheduler = FlowMatchScheduler(
            shift=float(cfg.training.flow_shift),
            sigma_min=0.0,
            extra_one_step=True,
        )
        self.flow_scheduler.set_timesteps(1000, training=True)
        trainable = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
        total = sum(
            parameter.numel() for parameter in self.transformer.parameters()
        )
        print(
            f"[model] transformer parameters={total:,}; "
            f"trainable parameters={trainable:,}",
            flush=True,
        )

    def on_train_epoch_start(self) -> None:
        self.text_encoder.eval()
        self.vae.eval()

    def on_after_backward(self) -> None:
        if self._reference_gradient_checked:
            return
        assert_reference_position_gradients(
            dict(self.transformer.reference_position_encoder.named_parameters())
        )
        self._reference_gradient_checked = True

    def _encode_video(self, video: torch.Tensor) -> torch.Tensor:
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
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
        return embeddings * mask.unsqueeze(-1).to(embeddings.dtype)

    def _token_count(self, latent: torch.Tensor) -> int:
        patch_t, patch_h, patch_w = self.transformer.config.patch_size
        return (
            latent.shape[2] // patch_t
            * (latent.shape[3] // patch_h)
            * (latent.shape[4] // patch_w)
        )

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        cfg = self.cfg
        source = self._encode_video(batch["source_video"])
        target = self._encode_video(batch["target_video"])
        reference = self._encode_video(batch["reference_image"].unsqueeze(2))

        text_drop_prob = float(cfg.training.text_drop_prob)
        prompts = [
            prompt if random.random() > text_drop_prob else ""
            for prompt in list(batch["prompt"])
        ]
        reference_drop_prob = float(cfg.training.reference_drop_prob)
        if random.random() < reference_drop_prob:
            reference = torch.zeros_like(reference)
        prompt_embeddings = self._encode_prompt(prompts)

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
            timestep,
            self._token_count(noisy_target),
            self._token_count(source),
            self._token_count(reference),
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
            prediction.float(),
            velocity_target.float(),
            reduction="none",
        )
        weights = self.flow_scheduler.training_weight(
            timestep.to(self.flow_scheduler.timesteps.device)
        ).to(device=per_element.device, dtype=per_element.dtype)
        loss = (per_element * weights[:, None, None, None, None]).mean()
        self.last_loss = float(loss.detach().cpu())
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
            "train/reference_dropped",
            float(torch.count_nonzero(reference) == 0),
            on_step=True,
            logger=True,
            batch_size=batch_size,
        )
        self.log(
            "train/text_drop_fraction",
            sum(prompt == "" for prompt in prompts) / len(prompts),
            on_step=True,
            logger=True,
            batch_size=batch_size,
        )
        return loss

    def configure_optimizers(self):
        cfg = self.cfg
        learning_rate = scaled_learning_rate(
            base_lr=float(cfg.optimizer.learning_rate),
            accumulate_grad_batches=int(cfg.training.accumulate_grad_batches),
            num_gpus=int(self.trainer.num_devices),
            num_nodes=int(cfg.training.num_nodes),
        )
        parameters = [
            parameter
            for parameter in self.transformer.parameters()
            if parameter.requires_grad
        ]
        optimizer = torch.optim.AdamW(
            [{"params": parameters, "lr": learning_rate}],
            betas=(float(cfg.optimizer.beta1), float(cfg.optimizer.beta2)),
            eps=float(cfg.optimizer.eps),
            weight_decay=float(cfg.optimizer.weight_decay),
        )
        warmup_steps = int(cfg.optimizer.warmup_steps)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: teacher_lr_multiplier(step, warmup_steps),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def trainable_tensor_contract(self) -> dict[str, int]:
        keys = compact_trainable_state(self.state_dict())
        return {
            "lora": sum("lora_" in key for key in keys),
            "source_patch": sum("patch_embedding_extra" in key for key in keys),
            "reference_position": sum(
                "reference_position_encoder" in key for key in keys
            ),
            "other": sum(
                "lora_" not in key
                and "patch_embedding_extra" not in key
                and "reference_position_encoder" not in key
                for key in keys
            ),
        }

    def export_trainable_weights(self, output_file: Path) -> dict:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        state = compact_trainable_state(self.state_dict())
        cpu_state = {
            key: value.detach().to(device="cpu").contiguous()
            for key, value in state.items()
        }
        temporary = output_file.with_name(f".{output_file.name}.tmp")
        save_file(cpu_state, str(temporary))
        os.replace(temporary, output_file)
        metadata = {
            "weights": str(output_file),
            "tensor_count": len(cpu_state),
            "tensor_bytes": sum(
                value.numel() * value.element_size() for value in cpu_state.values()
            ),
            "tensor_contract": self.trainable_tensor_contract(),
            "base_transformer": str(self.cfg.model.transformer),
            "text_encoder": str(self.cfg.model.text_encoder),
            "vae": str(self.cfg.model.vae),
            "lora_rank": int(self.cfg.lora.rank),
            "lora_alpha": int(self.cfg.lora.alpha),
            "global_step": int(self.global_step),
            "epoch": int(self.current_epoch),
        }
        output_file.with_suffix(".json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )
        return metadata

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        """Keep only the trainable adapter tensors in the recovery checkpoint."""
        checkpoint["state_dict"] = compact_trainable_state(
            checkpoint["state_dict"]
        )

    def on_train_end(self) -> None:
        if not self.trainer.is_global_zero:
            return
        if bool(self.cfg.training.get("preflight", False)):
            return
        output = (
            Path(self.cfg.output.root)
            / "recommended"
            / "final_adapter"
            / "v2v_lora_source_reference_posenc.safetensors"
        )
        metadata = self.export_trainable_weights(output)
        print(f"[final] wrote adapter: {metadata}", flush=True)
