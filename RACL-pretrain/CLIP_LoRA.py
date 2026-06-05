import math
import os
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from misc import initialize_weights
from transformers import CLIPVisionModel, CLIPVisionConfig

class _LoRA_qkv_CLIP(nn.Module):
    def __init__(
        self,
        original_linear: nn.Linear,
        r: int,
    ):
        super().__init__()
        self.original_linear = original_linear
        self.r = r
        in_features = original_linear.in_features
        out_features = original_linear.out_features
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.original_linear.weight.requires_grad = False
        if self.original_linear.bias is not None:
            self.original_linear.bias.requires_grad = False

    def forward(self, x: Tensor) -> Tensor:
        return self.original_linear(x) + self.lora_B(self.lora_A(x))

class LoRA_CLIP(nn.Module):
    def __init__(self, clip_model: CLIPVisionModel, r: int = 4, lora_layers: Optional[List[int]] = None):
        super().__init__()
        if lora_layers is None:
            lora_layers = [4, 10, 16, 22]
        self.clip = clip_model
        self.r = r
        self.lora_layers = lora_layers
        self.lora_A_weights: List[nn.Linear] = []
        self.lora_B_weights: List[nn.Linear] = []
        for param in self.clip.parameters():
            param.requires_grad = False
        encoder_layers = self.clip.vision_model.encoder.layers
        num_layers = len(encoder_layers)
        print(f"[CLIP LoRA] Total encoder layers: {num_layers}")
        print(f"[CLIP LoRA] Applying LoRA to layers: {lora_layers}")
        for layer_idx in lora_layers:
            if layer_idx >= num_layers:
                print(f"[Warning] Layer {layer_idx} does not exist (max: {num_layers-1}), skipping")
                continue
            layer = encoder_layers[layer_idx]
            self_attn = layer.self_attn
            q_lora = _LoRA_qkv_CLIP(self_attn.q_proj, r)
            self.lora_A_weights.append(q_lora.lora_A)
            self.lora_B_weights.append(q_lora.lora_B)
            self_attn.q_proj = q_lora
            v_lora = _LoRA_qkv_CLIP(self_attn.v_proj, r)
            self.lora_A_weights.append(v_lora.lora_A)
            self.lora_B_weights.append(v_lora.lora_B)
            self_attn.v_proj = v_lora
            print(f"[CLIP LoRA] Applied LoRA to layer {layer_idx} (q_proj, v_proj)")
        self.lora_A_list = nn.ModuleList(self.lora_A_weights)
        self.lora_B_list = nn.ModuleList(self.lora_B_weights)

    def forward(self, pixel_values: Tensor) -> Tensor:
        outputs = self.clip(pixel_values=pixel_values, output_hidden_states=True)
        return outputs.last_hidden_state

    def get_patch_features(self, pixel_values: Tensor) -> Tensor:
        outputs = self.clip(pixel_values=pixel_values)
        last_hidden = outputs.last_hidden_state
        patch_tokens = last_hidden[:, 1:, :]
        B, num_patches, C = patch_tokens.shape
        h = w = int(math.sqrt(num_patches))
        patch_tokens = patch_tokens.view(B, h, w, C).permute(0, 3, 1, 2).contiguous()
        return patch_tokens

    def get_multi_scale_features(self, pixel_values: Tensor, layers: List[int] = [4, 10, 16, 22]) -> List[Tensor]:
        outputs = self.clip(pixel_values=pixel_values, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        multi_scale_features = []
        for layer_idx in layers:
            if layer_idx >= len(hidden_states):
                continue
            hidden = hidden_states[layer_idx]
            patch_tokens = hidden[:, 1:, :]
            B, num_patches, C = patch_tokens.shape
            h = w = int(math.sqrt(num_patches))
            feat = patch_tokens.view(B, h, w, C).permute(0, 3, 1, 2).contiguous()
            multi_scale_features.append(feat)
        return multi_scale_features

class CLIP_CD_LoRA(nn.Module):
    def __init__(
        self,
        clip_path: str = "autodl-tmp/clip-vit-large-patch14",
        num_embed: int = 16,
        lora_r: int = 4,
        lora_layers: Optional[List[int]] = None,
    ):
        super().__init__()
        if lora_layers is None:
            lora_layers = [4, 10, 16, 22]
        print(f"[CLIP_CD_LoRA] Loading CLIP from: {clip_path}")
        if os.path.exists(clip_path):
            clip_model = CLIPVisionModel.from_pretrained(clip_path)
        else:
            print(f"[Warning] Local path not found, trying to load from HuggingFace...")
            clip_model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
        self.clip = LoRA_CLIP(clip_model, r=lora_r, lora_layers=lora_layers)
        hidden_size = clip_model.config.hidden_size
        self.hidden_size = hidden_size
        self.patch_size = clip_model.config.patch_size
        print(f"[CLIP_CD_LoRA] hidden_size={hidden_size}, patch_size={self.patch_size}")
        self.Adapter = nn.Sequential(
            nn.Conv2d(hidden_size, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_embed, kernel_size=1),
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        initialize_weights(self.Adapter)

    def forward(self, x: Tensor) -> Tensor:
        H, W = x.shape[-2:]
        if H != 224 or W != 224:
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        feats = self.clip.get_patch_features(x)
        y = self.Adapter(feats)
        return y

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
        for param in self.clip.lora_A_list.parameters():
            params.append(param)
        for param in self.clip.lora_B_list.parameters():
            params.append(param)
        for param in self.Adapter.parameters():
            params.append(param)
        params.append(self.logit_scale)
        return params

    def get_multi_scale_features(self, x: Tensor, layers: List[int] = [4, 10, 16, 22]) -> List[Tensor]:
        H, W = x.shape[-2:]
        if H != 224 or W != 224:
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        return self.clip.get_multi_scale_features(x, layers)

Clip_CD_LoRA = CLIP_CD_LoRA
