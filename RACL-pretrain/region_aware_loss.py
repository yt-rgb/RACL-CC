import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional


class PseudoMaskGenerator:
    def __init__(
        self,
        change_threshold: float = 0.3,
        unchanged_threshold: float = 0.7,
        temperature: float = 10.0,
        min_confidence: float = 0.3,
    ):
        self.change_threshold = change_threshold
        self.unchanged_threshold = unchanged_threshold
        self.temperature = temperature
        self.min_confidence = min_confidence

    @torch.no_grad()
    def generate(
        self, feat_a: torch.Tensor, feat_b: torch.Tensor, target_size: Optional[Tuple[int, int]] = None
    ) -> Dict[str, torch.Tensor]:
        feat_a_norm = F.normalize(feat_a, dim=1)
        feat_b_norm = F.normalize(feat_b, dim=1)
        similarity = (feat_a_norm * feat_b_norm).sum(dim=1, keepdim=True)
        change_prob = (1 - similarity) / 2
        change_mask = torch.sigmoid((change_prob - self.change_threshold) * self.temperature)
        unchanged_mask = torch.sigmoid((self.unchanged_threshold - change_prob) * self.temperature)
        confidence = torch.abs(change_prob - 0.5) * 2
        confidence = torch.clamp(confidence, min=self.min_confidence)
        if target_size is not None:
            change_mask = F.interpolate(change_mask, target_size, mode="bilinear", align_corners=False)
            unchanged_mask = F.interpolate(unchanged_mask, target_size, mode="bilinear", align_corners=False)
            confidence = F.interpolate(confidence, target_size, mode="bilinear", align_corners=False)
            similarity = F.interpolate(similarity, target_size, mode="bilinear", align_corners=False)
        return {
            "change_mask": change_mask,
            "unchanged_mask": unchanged_mask,
            "confidence": confidence,
            "similarity": similarity,
            "change_prob": change_prob,
        }


class RegionAwareContrastiveLoss(nn.Module):
    def __init__(
        self, temperature: float = 0.07, pull_margin: float = 0.2, push_margin: float = 0.5, patch_size: int = 8
    ):
        super().__init__()
        self.temperature = temperature
        self.pull_margin = pull_margin
        self.push_margin = push_margin
        self.patch_size = patch_size
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))
        self.region_pool = nn.AdaptiveAvgPool2d((patch_size, patch_size))

    def forward(
        self,
        feat_a: torch.Tensor,
        feat_b: torch.Tensor,
        change_mask: torch.Tensor,
        unchanged_mask: torch.Tensor,
        confidence: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if confidence is not None:
            change_mask = change_mask * confidence
            unchanged_mask = unchanged_mask * confidence
        feat_a_norm = F.normalize(feat_a, dim=1)
        feat_b_norm = F.normalize(feat_b, dim=1)
        pixel_similarity = (feat_a_norm * feat_b_norm).sum(dim=1, keepdim=True)
        pull_loss = self._compute_pull_loss(pixel_similarity, unchanged_mask)
        push_loss = self._compute_push_loss(pixel_similarity, change_mask)
        total_loss = pull_loss + push_loss
        return {"total": total_loss, "pull": pull_loss, "push": push_loss}

    def _compute_pull_loss(self, similarity: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        distance = 1 - similarity
        loss = F.relu(distance - self.pull_margin) * mask
        mask_sum = mask.sum() + 1e-6
        return loss.sum() / mask_sum

    def _compute_push_loss(self, similarity: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        loss = F.relu(similarity + self.push_margin) * mask
        mask_sum = mask.sum() + 1e-6
        return loss.sum() / mask_sum

    def _compute_region_contrastive_loss(
        self, feat_a: torch.Tensor, feat_b: torch.Tensor, unchanged_mask: torch.Tensor
    ) -> torch.Tensor:
        B, C, H, W = feat_a.shape
        masked_a = feat_a * unchanged_mask
        masked_b = feat_b * unchanged_mask
        pooled_a = self.region_pool(masked_a)
        pooled_b = self.region_pool(masked_b)
        pooled_a = pooled_a.flatten(2)
        pooled_b = pooled_b.flatten(2)
        pooled_a = F.normalize(pooled_a, dim=1)
        pooled_b = F.normalize(pooled_b, dim=1)
        sim_matrix = torch.bmm(pooled_a.transpose(1, 2), pooled_b) * self.logit_scale.exp()
        labels = torch.arange(sim_matrix.size(1), device=sim_matrix.device).unsqueeze(0).expand(B, -1)
        loss = F.cross_entropy(sim_matrix.view(-1, sim_matrix.size(-1)), labels.view(-1), label_smoothing=0.1)
        return loss


class RegionAwareLossModule(nn.Module):
    def __init__(
        self,
        change_threshold: float = 0.3,
        unchanged_threshold: float = 0.7,
        temperature: float = 0.07,
        pull_margin: float = 0.2,
        push_margin: float = 0.5,
        pull_weight: float = 1.0,
        push_weight: float = 1.0,
        warmup_epochs: int = 5,
        min_region_ratio: float = 0.02,
    ):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.min_region_ratio = min_region_ratio
        self.change_threshold = change_threshold
        self.unchanged_threshold = unchanged_threshold
        self.pull_weight = pull_weight
        self.push_weight = push_weight
        self.mask_generator = PseudoMaskGenerator(
            change_threshold=change_threshold, unchanged_threshold=unchanged_threshold
        )
        self.contrastive_loss = RegionAwareContrastiveLoss(
            temperature=temperature, pull_margin=pull_margin, push_margin=push_margin
        )
        self._model: Optional[nn.Module] = None

    def init_model(self, model: nn.Module) -> None:
        self._model = model

    @torch.no_grad()
    def _generate_pseudo_mask_v2(
        self, img_a: torch.Tensor, img_b: torch.Tensor, target_size: Tuple[int, int]
    ) -> Dict[str, torch.Tensor]:
        if self._model is None:
            raise RuntimeError("The model has not been initialised; please call `init_model()` first.")
        was_training = self._model.training
        self._model.eval()
        module = self._model.module if hasattr(self._model, "module") else self._model
        feat_a = module(img_a)
        feat_b = module(img_b)
        if was_training:
            self._model.train()
        return self.mask_generator.generate(feat_a, feat_b, target_size)

    def forward(
        self, feat_a: torch.Tensor, feat_b: torch.Tensor, img_a: torch.Tensor, img_b: torch.Tensor, epoch: int = 0
    ) -> Dict[str, torch.Tensor]:
        if self._model is not None:
            masks = self._generate_pseudo_mask_v2(img_a, img_b, target_size=feat_a.shape[-2:])
        else:
            masks = self.mask_generator.generate(feat_a.detach(), feat_b.detach(), target_size=feat_a.shape[-2:])
        change_mask = masks["change_mask"]
        unchanged_mask = masks["unchanged_mask"]
        confidence = masks["confidence"]
        change_ratio = change_mask.mean().item()
        unchanged_ratio = unchanged_mask.mean().item()
        if change_ratio < self.min_region_ratio and unchanged_ratio < self.min_region_ratio:
            zero_loss = torch.tensor(0.0, device=feat_a.device, requires_grad=True)
            return {
                "loss": zero_loss,
                "pull_loss": zero_loss,
                "push_loss": zero_loss,
                "change_ratio": change_ratio,
                "weight": 0.0,
            }
        loss_dict = self.contrastive_loss(feat_a, feat_b, change_mask, unchanged_mask, confidence)
        pull_loss = loss_dict["pull"] * self.pull_weight
        push_loss = loss_dict["push"] * self.push_weight
        total_loss = pull_loss + push_loss
        if epoch < self.warmup_epochs:
            warmup_weight = epoch / self.warmup_epochs
        else:
            warmup_weight = 1.0
        weighted_loss = total_loss * warmup_weight
        return {
            "loss": weighted_loss,
            "pull_loss": loss_dict["pull"],
            "push_loss": loss_dict["push"],
            "pull_loss_weighted": pull_loss,
            "push_loss_weighted": push_loss,
            "change_ratio": change_ratio,
            "unchanged_ratio": unchanged_ratio,
            "warmup_weight": warmup_weight,
        }
