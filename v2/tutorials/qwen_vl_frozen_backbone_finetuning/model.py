import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import count_parameters


def packed_pixels_to_dense_images(
    packed_pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
    *,
    patch_size: int,
    temporal_patch_size: int,
    channels: int = 3,
) -> torch.Tensor:
    """
    Convert Qwen packed patch tensors back into dense BCHW images.

    In the current Qwen2.5-VL preprocessing path, `pixel_values` may arrive as
    `(num_patches_total, patch_dim)` where `patch_dim` packs
    `temporal_patch_size * channels * patch_size * patch_size`.

    We unpatchify that so the adapter can operate in image space.
    """
    dense_images = []
    offset = 0
    grid_cpu = image_grid_thw.detach().cpu().tolist()
    for grid_t, grid_h, grid_w in grid_cpu:
        patch_count = int(grid_t * grid_h * grid_w)
        sample = packed_pixel_values[offset : offset + patch_count]
        offset += patch_count
        sample = sample.view(
            int(grid_t),
            int(grid_h),
            int(grid_w),
            temporal_patch_size,
            channels,
            patch_size,
            patch_size,
        )
        sample = sample.permute(0, 3, 4, 1, 5, 2, 6).contiguous()
        sample = sample.view(
            int(grid_t) * temporal_patch_size,
            channels,
            int(grid_h) * patch_size,
            int(grid_w) * patch_size,
        )
        # Images are duplicated across the temporal patch dimension for Qwen, so
        # average across that axis to recover a single dense RGB image.
        dense_images.append(sample.mean(dim=0))
    return torch.stack(dense_images, dim=0)


def dense_images_to_packed_pixels(
    dense_images: torch.Tensor,
    image_grid_thw: torch.Tensor,
    *,
    patch_size: int,
    temporal_patch_size: int,
    channels: int = 3,
) -> torch.Tensor:
    packed = []
    grid_cpu = image_grid_thw.detach().cpu().tolist()
    for image, (grid_t, grid_h, grid_w) in zip(dense_images, grid_cpu, strict=False):
        expanded = image.unsqueeze(0).repeat(int(grid_t) * temporal_patch_size, 1, 1, 1)
        expanded = expanded.view(
            int(grid_t),
            temporal_patch_size,
            channels,
            int(grid_h),
            patch_size,
            int(grid_w),
            patch_size,
        )
        expanded = expanded.permute(0, 3, 5, 1, 2, 4, 6).contiguous()
        packed.append(
            expanded.view(
                int(grid_t) * int(grid_h) * int(grid_w),
                temporal_patch_size * channels * patch_size * patch_size,
            )
        )
    return torch.cat(packed, dim=0)


# {{docs-fragment residual-adapter}}
class ResidualOcclusionAdapter(nn.Module):
    def __init__(self, hidden_channels: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 3, kernel_size=1),
            nn.Tanh(),
        )
        self.gate = nn.Parameter(torch.tensor(0.10))

    def forward(
        self, pixel_values: torch.Tensor, occlusion_mask: torch.Tensor
    ) -> torch.Tensor:
        if pixel_values.ndim != 4:
            raise ValueError(
                "ResidualOcclusionAdapter expects dense image tensors with shape "
                f"(B, C, H, W), but received {tuple(pixel_values.shape)}."
            )
        if occlusion_mask.ndim == 3:
            occlusion_mask = occlusion_mask.unsqueeze(1)
        adapter_input = torch.cat(
            [pixel_values, occlusion_mask.to(pixel_values.dtype)],
            dim=1,
        )
        residual = self.net(adapter_input)
        return pixel_values + torch.tanh(self.gate) * residual
# {{/docs-fragment residual-adapter}}


# {{docs-fragment adapter-module-init}}
class QwenVLAdapterModule(L.LightningModule):
    def __init__(
        self,
        model_name: str,
        learning_rate: float,
        weight_decay: float,
        reconstruction_loss_weight: float,
    ):
        super().__init__()
        from transformers import Qwen2_5_VLForConditionalGeneration

        self.save_hyperparameters()
        self.adapter = ResidualOcclusionAdapter()

        self.backbone = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        self.backbone.requires_grad_(False)
        self.backbone.gradient_checkpointing_enable()

        # DeepSpeed checkpoints only persist the trainable adapter weights when
        # `exclude_frozen_parameters=True`. On resume we rebuild the frozen
        # backbone from Hugging Face and load the checkpoint non-strictly.
        self.strict_loading = False

        self.total_params, self.trainable_params = count_parameters(self)
        self.example_input_array = None
        self.vision_patch_size = int(self.backbone.config.vision_config.patch_size)
        self.temporal_patch_size = int(
            getattr(self.backbone.config.vision_config, "temporal_patch_size", 1)
        )
    # {{/docs-fragment adapter-module-init}}

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        state_dict = checkpoint.get("state_dict")
        if not state_dict:
            return

        current_state = self.state_dict()
        for key, value in current_state.items():
            if key.startswith("backbone.") and key not in state_dict:
                state_dict[key] = value

    # {{docs-fragment forward-losses}}
    def _forward_losses(
        self, batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        backbone_dtype = next(self.backbone.parameters()).dtype
        if batch["pixel_values"].ndim == 2:
            if "image_grid_thw" not in batch:
                raise ValueError(
                    "Packed Qwen pixel values require `image_grid_thw` to reconstruct "
                    "dense images for the Conv2d adapter."
                )
            grid_thw = batch["image_grid_thw"]
            dense_pixels = packed_pixels_to_dense_images(
                batch["pixel_values"].to(dtype=backbone_dtype),
                grid_thw,
                patch_size=self.vision_patch_size,
                temporal_patch_size=self.temporal_patch_size,
            )
            clean_pixels = packed_pixels_to_dense_images(
                batch["clean_pixel_values"].to(dtype=backbone_dtype),
                grid_thw,
                patch_size=self.vision_patch_size,
                temporal_patch_size=self.temporal_patch_size,
            )
            adapted_dense = self.adapter(dense_pixels, batch["occlusion_mask"])
            adapted_pixels = dense_images_to_packed_pixels(
                adapted_dense,
                grid_thw,
                patch_size=self.vision_patch_size,
                temporal_patch_size=self.temporal_patch_size,
            )
        else:
            clean_pixels = batch["clean_pixel_values"].to(dtype=backbone_dtype)
            adapted_dense = self.adapter(
                batch["pixel_values"].to(dtype=backbone_dtype),
                batch["occlusion_mask"],
            )
            adapted_pixels = adapted_dense

        forward_kwargs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "pixel_values": adapted_pixels,
            "labels": batch["labels"],
        }
        if "image_grid_thw" in batch:
            forward_kwargs["image_grid_thw"] = batch["image_grid_thw"]
        outputs = self.backbone(**forward_kwargs)

        clean_pixels = clean_pixels.to(
            device=adapted_pixels.device, dtype=backbone_dtype
        )
        occlusion_mask = batch["occlusion_mask"].to(
            device=adapted_pixels.device,
            dtype=backbone_dtype,
        )
        if occlusion_mask.ndim == 3:
            occlusion_mask = occlusion_mask.unsqueeze(1)
        if occlusion_mask.shape[-2:] != adapted_dense.shape[-2:]:
            occlusion_mask = F.interpolate(
                occlusion_mask,
                size=adapted_dense.shape[-2:],
                mode="nearest",
            )

        reconstruction_error = (adapted_dense - clean_pixels).abs() * occlusion_mask
        mask_denominator = (occlusion_mask.sum() * adapted_dense.shape[1]).clamp_min(
            1.0
        )

        reconstruction_loss = reconstruction_error.sum() / mask_denominator
        total_loss = (
            outputs.loss + self.hparams.reconstruction_loss_weight * reconstruction_loss
        )

        return {
            "total_loss": total_loss,
            "lm_loss": outputs.loss,
            "reconstruction_loss": reconstruction_loss,
        }
    # {{/docs-fragment forward-losses}}

    def training_step(self, batch, _batch_idx):
        losses = self._forward_losses(batch)
        batch_size = int(batch["input_ids"].shape[0])
        self.log(
            "train/total_loss",
            losses["total_loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            "train/lm_loss",
            losses["lm_loss"],
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            "train/reconstruction_loss",
            losses["reconstruction_loss"],
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            "train/trainable_param_ratio",
            self.trainable_params / max(1, self.total_params),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        return losses["total_loss"]

    def validation_step(self, batch, _batch_idx):
        losses = self._forward_losses(batch)
        batch_size = int(batch["input_ids"].shape[0])
        self.log(
            "val/total_loss",
            losses["total_loss"],
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            "val_total_loss",
            losses["total_loss"],
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            "val/lm_loss",
            losses["lm_loss"],
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            "val/reconstruction_loss",
            losses["reconstruction_loss"],
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        return losses["total_loss"]

    def configure_optimizers(self):
        decay_params = []
        no_decay_params = []

        for name, param in self.adapter.named_parameters():
            if not param.requires_grad:
                continue
            if name.endswith("bias") or param.ndim == 1:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        return torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": self.hparams.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.hparams.learning_rate,
            betas=(0.9, 0.95),
        )
