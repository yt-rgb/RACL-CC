"""
SAM image encoder LoRA module for region-aware contrastive pretraining.

This mirrors CLIP_LoRA.py at the training-interface level:
- freeze the original pretrained image encoder parameters
- inject trainable LoRA adapters into selected transformer attention q/v branches
- train a lightweight convolutional Adapter plus logit_scale
- expose forward(x) and bi_forward(x1, x2) for train_region_aware.py
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from misc import initialize_weights
from segment_anything import sam_model_registry


class _LoRA_qkv_SAM(nn.Module):
    """Apply LoRA updates to q and v slices of SAM's merged qkv projection."""

    def __init__(self, original_qkv: nn.Linear, r: int):
        super().__init__()
        if original_qkv.out_features % 3 != 0:
            raise ValueError(
                f"SAM qkv projection out_features must be divisible by 3, got {original_qkv.out_features}."
            )

        self.original_qkv = original_qkv
        self.r = r
        self.in_features = original_qkv.in_features
        self.embed_dim = original_qkv.out_features // 3

        self.lora_q_A = nn.Linear(self.in_features, r, bias=False)
        self.lora_q_B = nn.Linear(r, self.embed_dim, bias=False)
        self.lora_v_A = nn.Linear(self.in_features, r, bias=False)
        self.lora_v_B = nn.Linear(r, self.embed_dim, bias=False)

        nn.init.kaiming_uniform_(self.lora_q_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_q_B.weight)
        nn.init.kaiming_uniform_(self.lora_v_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_v_B.weight)

        self.original_qkv.weight.requires_grad = False
        if self.original_qkv.bias is not None:
            self.original_qkv.bias.requires_grad = False

    def forward(self, x: Tensor) -> Tensor:
        qkv = self.original_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q + self.lora_q_B(self.lora_q_A(x))
        v = v + self.lora_v_B(self.lora_v_A(x))
        return torch.cat((q, k, v), dim=-1)


class LoRA_SAM(nn.Module):
    """LoRA wrapper for SAM's ImageEncoderViT."""

    def __init__(self, image_encoder: nn.Module, r: int = 4, lora_layers: Optional[List[int]] = None):
        super().__init__()
        self.image_encoder = image_encoder
        self.r = r

        blocks = self.image_encoder.blocks
        num_layers = len(blocks)
        if lora_layers is None:
            lora_layers = [2, 5, 8, 11] if num_layers == 12 else list(range(max(0, num_layers - 4), num_layers))
        self.lora_layers = lora_layers

        self.lora_A_weights: List[nn.Linear] = []
        self.lora_B_weights: List[nn.Linear] = []

        for param in self.image_encoder.parameters():
            param.requires_grad = False

        print(f"[SAM LoRA] Total encoder blocks: {num_layers}")
        print(f"[SAM LoRA] Applying LoRA to layers: {lora_layers}")

        for layer_idx in lora_layers:
            if layer_idx >= num_layers:
                print(f"[Warning] SAM block {layer_idx} does not exist (max: {num_layers - 1}), skipping")
                continue

            attn = blocks[layer_idx].attn
            qkv_lora = _LoRA_qkv_SAM(attn.qkv, r)
            self.lora_A_weights.extend([qkv_lora.lora_q_A, qkv_lora.lora_v_A])
            self.lora_B_weights.extend([qkv_lora.lora_q_B, qkv_lora.lora_v_B])
            attn.qkv = qkv_lora
            print(f"[SAM LoRA] Applied LoRA to block {layer_idx} (q and v slices of qkv)")

        self.lora_A_list = nn.ModuleList(self.lora_A_weights)
        self.lora_B_list = nn.ModuleList(self.lora_B_weights)

    @staticmethod
    def _resize_pos_embed(pos_embed: Tensor, target_hw: tuple[int, int]) -> Tensor:
        if pos_embed.shape[1:3] == target_hw:
            return pos_embed
        pos_embed = pos_embed.permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=target_hw, mode="bicubic", align_corners=False)
        return pos_embed.permute(0, 2, 3, 1)

    def forward(self, x: Tensor) -> Tensor:
        encoder = self.image_encoder
        x = encoder.patch_embed(x)
        if encoder.pos_embed is not None:
            x = x + self._resize_pos_embed(encoder.pos_embed, x.shape[1:3])

        for block in encoder.blocks:
            x = block(x)

        return encoder.neck(x.permute(0, 3, 1, 2))


class SAM_CD_Encoder(nn.Module):
    """
    Change-detection feature extractor based on SAM image encoder with LoRA.

    Args:
        checkpoint: SAM checkpoint path.
        model_type: one of vit_b, vit_l, vit_h.
        num_embed: adapter output channels.
        lora_r: LoRA rank.
        lora_layers: SAM image encoder block indices to inject LoRA.
        image_size: SAM encoder input size. Official checkpoints use 1024.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        model_type: str = "vit_b",
        num_embed: int = 16,
        lora_r: int = 4,
        lora_layers: Optional[List[int]] = None,
        image_size: int = 1024,
    ):
        super().__init__()
        checkpoint = str(checkpoint)
        print(f"[SAM_CD_Encoder] Loading SAM {model_type} from: {checkpoint}")

        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        self.sam = sam
        self.image_encoder = LoRA_SAM(sam.image_encoder, r=lora_r, lora_layers=lora_layers)
        self.image_size = image_size

        embed_dim = self._get_encoder_output_channels(sam.image_encoder)
        self.hidden_size = embed_dim
        self.patch_size = sam.image_encoder.patch_embed.proj.kernel_size[0]

        print(
            f"[SAM_CD_Encoder] hidden_size={self.hidden_size}, "
            f"patch_size={self.patch_size}, image_size={self.image_size}"
        )

        self.Adapter = nn.Sequential(
            nn.Conv2d(embed_dim, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_embed, kernel_size=1),
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        initialize_weights(self.Adapter)

    @staticmethod
    def _get_encoder_output_channels(image_encoder: nn.Module) -> int:
        for module in reversed(list(image_encoder.neck.modules())):
            if isinstance(module, nn.Conv2d):
                return module.out_channels
        raise AttributeError("Could not infer SAM image encoder output channels from image_encoder.neck.")

    def forward(self, x: Tensor) -> Tensor:
        h, w = x.shape[-2:]
        if h != self.image_size or w != self.image_size:
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)

        feats = self.image_encoder(x)
        return self.Adapter(feats)

    def bi_forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        input_shape = x1.shape[-2:]

        y1 = self.forward(x1)
        y2 = self.forward(x2)

        y1_norm = y1 / (torch.norm(y1, dim=1, keepdim=True) + 1e-6)
        y2_norm = y2 / (torch.norm(y2, dim=1, keepdim=True) + 1e-6)
        sim = torch.sum(y1_norm * y2_norm, dim=1, keepdim=True)
        yc = -sim * self.logit_scale

        return F.interpolate(yc, input_shape, mode="bilinear", align_corners=False)

    def get_trainable_parameters(self):
        params = []
        params.extend(list(self.image_encoder.lora_A_list.parameters()))
        params.extend(list(self.image_encoder.lora_B_list.parameters()))
        params.extend(list(self.Adapter.parameters()))
        params.append(self.logit_scale)
        return params


SAM_CD_LoRA = SAM_CD_Encoder
