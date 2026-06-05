import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple

class CDPseudoLabelGenerator:

    def __init__(
        self,
        high_thresh: float = 0.65,   # High-confidence change threshold
        low_thresh: float = 0.35,    # High-confidence unchanged threshold
        temperature: float = 12.0,   # sigmoid sharpening temperature
        min_conf: float = 0.2,       # Truncation at the lower confidence limit
    ):
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.temperature = temperature
        self.min_conf = min_conf

    @torch.no_grad()
    def generate(
        self,
        feat_a: torch.Tensor,
        feat_b: torch.Tensor,
        img_a: Optional[torch.Tensor] = None,
        img_b: Optional[torch.Tensor] = None,
        gt_mask: Optional[torch.Tensor] = None,
        target_size: Optional[Tuple[int, int]] = None,
        pixel_alpha: float = 0.6,
    ) -> Dict[str, torch.Tensor]:

        fa = F.normalize(feat_a, dim=1)
        fb = F.normalize(feat_b, dim=1)
        similarity = (fa * fb).sum(dim=1, keepdim=True)  # [-1, 1]

        
        feat_change_prob = (1.0 - similarity) / 2.0

        
        if img_a is not None and img_b is not None:
            # Calculate the absolute difference in pixels and take the mean across the channel dimension
            pixel_diff = (img_a - img_b).abs().mean(dim=1, keepdim=True)  # [B, 1, H, W]

            # Downsampling to the size of the feature map (feature maps are usually smaller than the original image)
            feat_h, feat_w = feat_change_prob.shape[-2:]
            pixel_diff_ds = F.interpolate(
                pixel_diff, size=(feat_h, feat_w),
                mode='bilinear', align_corners=False
            )

            # Normalise to [0, 1]: Normalise each sample independently to avoid scale differences within a batch
            B = pixel_diff_ds.size(0)
            pixel_diff_flat = pixel_diff_ds.view(B, -1)
            p_min = pixel_diff_flat.min(dim=1)[0].view(B, 1, 1, 1)
            p_max = pixel_diff_flat.max(dim=1)[0].view(B, 1, 1, 1)
            pixel_diff_norm = (pixel_diff_ds - p_min) / (p_max - p_min + 1e-6)

            # Weighted blending: `pixel_alpha` controls the weighting of pixel differences
            change_prob = pixel_alpha * pixel_diff_norm + (1.0 - pixel_alpha) * feat_change_prob
        else:

            change_prob = feat_change_prob

        
        if gt_mask is not None:
            gt_resized = F.interpolate(
                gt_mask.float(), size=change_prob.shape[-2:],
                mode='bilinear', align_corners=False
            )
            # Weighted fusion: 0.7 weight for the true label, 0.3 weight for the mixing probability
            change_prob = 0.7 * gt_resized + 0.3 * change_prob

        
        change_mask = torch.sigmoid(
            (change_prob - self.high_thresh) * self.temperature
        )
        unchanged_mask = torch.sigmoid(
            (self.low_thresh - change_prob) * self.temperature
        )
        uncertain_mask = 1.0 - change_mask - unchanged_mask
        uncertain_mask = torch.clamp(uncertain_mask, min=0.0)

        
        confidence = (torch.abs(change_prob - 0.5) * 2.0).clamp(min=self.min_conf)

        
        if target_size is not None:
            def up(x):
                return F.interpolate(x, target_size, mode='bilinear', align_corners=False)
            change_mask = up(change_mask)
            unchanged_mask = up(unchanged_mask)
            uncertain_mask = up(uncertain_mask)
            confidence = up(confidence)
            change_prob = up(change_prob)

        return {
            'change_mask': change_mask,
            'unchanged_mask': unchanged_mask,
            'uncertain_mask': uncertain_mask,
            'confidence': confidence,
            'change_prob': change_prob,
        }

class L12_RegionAwareQuadrupletLoss(nn.Module):

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.tau = nn.Parameter(torch.tensor(np.log(1.0 / temperature)))

    def _masked_pool(
        self,
        feat: torch.Tensor,   # [B, C, H, W]
        mask: torch.Tensor,   # [B, 1, H, W]
    ) -> torch.Tensor:
        # Normalise the mask weights so that their sum across all spatial dimensions equals 1
        weight = mask / (mask.sum(dim=[2, 3], keepdim=True) + 1e-6)  # [B, 1, H, W]
        pooled = (feat * weight).sum(dim=[2, 3])  # [B, C]
        return F.normalize(pooled, dim=-1)

    def forward(
        self,
        feat_a: torch.Tensor,        # Scenario 1: Original Features [B, C, H, W]
        feat_a_aug: torch.Tensor,    # Time-series 1 feature enhancement [B, C, H, W]
        feat_b: torch.Tensor,        # Time Series 2: Original Features [B, C, H, W]
        feat_b_aug: torch.Tensor,    # Time-Series 2 Enhanced Features [B, C, H, W]
        change_mask: torch.Tensor,   # Change Area Mask  [B, 1, H, W]
        unchanged_mask: torch.Tensor,# Unchanged region mask [B, 1, H, W]
        confidence: torch.Tensor,    # Confidence plot      [B, 1, H, W]
    ) -> torch.Tensor:
        B = feat_a.size(0)
        scale = self.tau.exp().clamp(max=100.0)

        
        conf_ch = change_mask * confidence       # [B, 1, H, W]
        conf_unch = unchanged_mask * confidence  # [B, 1, H, W]

        # Vector of unchanged regions
        z_a_unch    = self._masked_pool(feat_a,     conf_unch)  # [B, C]
        z_a_aug_unch = self._masked_pool(feat_a_aug, conf_unch)  # [B, C]
        z_b_unch    = self._masked_pool(feat_b,     conf_unch)  # [B, C]

        # Vector of change areas
        z_a_ch      = self._masked_pool(feat_a,     conf_ch)    # [B, C]
        z_a_aug_ch  = self._masked_pool(feat_a_aug, conf_ch)    # [B, C]
        z_b_ch      = self._masked_pool(feat_b,     conf_ch)    # [B, C]

        
        all_keys = torch.cat([
            z_a_unch, z_a_aug_unch, z_b_unch,
            z_a_ch, z_a_aug_ch, z_b_ch
        ], dim=0)  # [6B, C]

        # Fixed perspective: Calculate the similarity between the anchor and all keys
        sim_unch = (z_a_unch @ all_keys.T) * scale  # [B, 6B]

        # Positive sample mask: z_a_aug_unch (offset B) and z_b_unch (offset 2B) from the same batch
        idx = torch.arange(B, device=feat_a.device)
        pos_mask_unch = torch.zeros(B, 6 * B, device=feat_a.device)
        pos_mask_unch[idx, idx + B]     = 1.0  # z_a_aug_unch
        pos_mask_unch[idx, idx + 2 * B] = 1.0  # z_b_unch

        # Exclude self (anchor=z_a_unch at position idx)
        sim_unch[idx, idx] = -1e9

        # InfoNCE: Multi-positive version
        log_prob_unch = F.log_softmax(sim_unch, dim=-1)  # [B, 6B]
        loss_unch = -(log_prob_unch * pos_mask_unch).sum(dim=-1) / \
                     pos_mask_unch.sum(dim=-1).clamp(min=1.0)  # [B]

      
        sim_ch = (z_a_ch @ all_keys.T) * scale  # [B, 6B]

        
        pos_mask_ch = torch.zeros(B, 6 * B, device=feat_a.device)
        pos_mask_ch[idx, idx + 4 * B] = 1.0  # z_a_aug_ch

        
        sim_ch[idx, idx + 3 * B] = -1e9

        log_prob_ch = F.log_softmax(sim_ch, dim=-1)  # [B, 6B]
        loss_ch = -(log_prob_ch * pos_mask_ch).sum(dim=-1) / \
                   pos_mask_ch.sum(dim=-1).clamp(min=1.0)  # [B]

        
        return (loss_unch.mean() + loss_ch.mean()) / 2.0



class L34_SpatioTemporalContrastLoss(nn.Module):

    def __init__(self, temperature: float = 0.1, patch_size: int = 8):
        super().__init__()
        self.tau = nn.Parameter(torch.tensor(np.log(1.0 / temperature)))
        self.pool = nn.AdaptiveAvgPool2d(patch_size)
        self.P = patch_size

    def forward(
        self,
        feat_a: torch.Tensor,         # Scenario 1: Original Features [B, C, H, W]
        feat_b: torch.Tensor,         # Scenario 2: Original Features [B, C, H, W]
        change_mask: torch.Tensor,    # Change Area Mask  [B, 1, H, W]
        unchanged_mask: torch.Tensor, # Unchanged region mask [B, 1, H, W]
        confidence: torch.Tensor,     # Confidence plot      [B, 1, H, W]
    ) -> torch.Tensor:
        B, C, H, W = feat_a.shape
        N = self.P * self.P
        scale = self.tau.exp().clamp(max=100.0)

        fa = F.normalize(self.pool(feat_a), dim=1)  # [B, C, P, P]
        fb = F.normalize(self.pool(feat_b), dim=1)  # [B, C, P, P]

        unch_w = self.pool(unchanged_mask * confidence)  # [B, 1, P, P]
        ch_w   = self.pool(change_mask * confidence)     # [B, 1, P, P]

        # Flatten to a sequence: [B, N, C]
        fa_seq = fa.flatten(2).permute(0, 2, 1)       # [B, N, C]
        fb_seq = fb.flatten(2).permute(0, 2, 1)       # [B, N, C]
        unch_seq = unch_w.flatten(2).squeeze(1)        # [B, N]
        ch_seq   = ch_w.flatten(2).squeeze(1)          # [B, N]

        all_keys = torch.cat([
            fa_seq.reshape(B * N, C),   # [BN, C]
            fb_seq.reshape(B * N, C),   # [BN, C]
        ], dim=0)  # [2BN, C]

        total_loss = torch.zeros(1, device=feat_a.device)
        valid_anchors = 0

        for b in range(B):
            unch_idx = (unch_seq[b] > 0.3).nonzero(as_tuple=True)[0]  # [K_u]
            if len(unch_idx) < 2:
                continue
            anchors = fa_seq[b][unch_idx]  # [K_u, C]
            K_u = anchors.size(0)
            positives = fb_seq[b][unch_idx]  # [K_u, C]
            sim_all = (anchors @ all_keys.T) * scale  # [K_u, 2BN]
            pos_global_idx = B * N + b * N + unch_idx  # [K_u]
            self_global_idx = b * N + unch_idx  # [K_u]
            pos_label_matrix = torch.zeros(K_u, 2 * B * N, device=feat_a.device)
            k_idx = torch.arange(K_u, device=feat_a.device)
            pos_label_matrix[k_idx, pos_global_idx] = 1.0
            sim_all[k_idx, self_global_idx] = -1e9
            log_prob = F.log_softmax(sim_all, dim=-1)  # [K_u, 2BN]
            loss_b = -(log_prob * pos_label_matrix).sum(dim=-1).mean()  # scalar
            conf_weights = unch_seq[b][unch_idx]  # [K_u]
            conf_weights = conf_weights / conf_weights.sum().clamp(min=1e-6)
            loss_b_weighted = (
                -(log_prob * pos_label_matrix).sum(dim=-1) * conf_weights
            ).sum()
            total_loss = total_loss + loss_b_weighted
            valid_anchors += 1
        if valid_anchors == 0:
            return torch.zeros(1, device=feat_a.device, requires_grad=True).squeeze()
        return (total_loss / valid_anchors).squeeze()

class L5_ChangeSparsityRegularization(nn.Module):

    def __init__(
        self,
        global_thresh: float = 0.35,  # Expected upper limit on the proportion of global changes
        local_margin: float = 0.1,    # Local sparsity margin
        patch_size: int = 8,
    ):
        super().__init__()
        self.global_thresh = global_thresh
        self.local_margin = local_margin
        self.pool = nn.AdaptiveAvgPool2d(patch_size)

    def forward(self, change_prob: torch.Tensor) -> torch.Tensor:
  
        global_loss = F.relu(change_prob.mean() - self.global_thresh)

        patch_prob = self.pool(change_prob)  # [B, 1, P, P]
        patch_flat = patch_prob.flatten()
        
        k = int(len(patch_flat) * (1.0 - self.global_thresh))
        if k > 0:
            small_vals, _ = torch.topk(patch_flat, k, largest=False)
            local_loss = F.relu(small_vals - self.local_margin).mean()
        else:
            local_loss = torch.tensor(0.0, device=change_prob.device)

        return global_loss + local_loss

class CDCLoss(nn.Module):

    def __init__(
        self,
        # Loss weights
        w_l12: float = 1.5,        
        w_l34: float = 1.0,        
        w_sparsity: float = 2.0,  
        # Pseudo label configuration
        high_thresh: float = 0.65,
        low_thresh: float = 0.35,
        # Training configuration
        warmup_epochs: int = 5,
        # Component configuration
        temperature_l12: float = 0.07,
        temperature_l34: float = 0.1,
        patch_size_l34: int = 8,
        global_sparsity_thresh: float = 0.35,
    ):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.w_l12 = w_l12
        self.w_l34 = w_l34
        self.w_sparsity = w_sparsity

        self.pseudo_gen = CDPseudoLabelGenerator(
            high_thresh=high_thresh,
            low_thresh=low_thresh,
        )

        self.L12 = L12_RegionAwareQuadrupletLoss(temperature=temperature_l12)
        self.L34 = L34_SpatioTemporalContrastLoss(
            temperature=temperature_l34,
            patch_size=patch_size_l34,
        )
        self.L5 = L5_ChangeSparsityRegularization(
            global_thresh=global_sparsity_thresh,
        )

    def _warmup_weight(self, epoch: int) -> float:
        if epoch >= self.warmup_epochs:
            return 1.0
        return (epoch + 1) / self.warmup_epochs

    @torch.no_grad()
    def _generate_masks(
        self,
        feat_a: torch.Tensor,
        feat_b: torch.Tensor,
        img_a: Optional[torch.Tensor],
        img_b: Optional[torch.Tensor],
        gt_mask: Optional[torch.Tensor],
        target_size: Tuple[int, int],
    ) -> Dict[str, torch.Tensor]:
        
        return self.pseudo_gen.generate(
            feat_a.detach(), feat_b.detach(),
            img_a=img_a,
            img_b=img_b,
            gt_mask=gt_mask,
            target_size=target_size,
        )

    def forward(
        self,
        feat_a: torch.Tensor,      # Original features of time series 1 [B, C, H, W]
        feat_b: torch.Tensor,      # Original features of time series 2 [B, C, H, W]
        feat_a_aug: torch.Tensor,  # Enhanced features of time series 1 [B, C, H, W]
        feat_b_aug: torch.Tensor,  # Enhanced features of time series 2 [B, C, H, W]
        epoch: int = 0,
        gt_mask: Optional[torch.Tensor] = None,  # Ground truth mask [B,1,H,W] (optional)
        img_a: Optional[torch.Tensor] = None,    # Original image of time series 1 [B,3,H,W] (optional)
        img_b: Optional[torch.Tensor] = None,    # Original image of time series 2 [B,3,H,W] (optional)
    ) -> Dict[str, torch.Tensor]:
        H, W = feat_a.shape[-2:]
        wup = self._warmup_weight(epoch)

        masks = self._generate_masks(feat_a, feat_b, img_a, img_b, gt_mask, (H, W))
        change_mask = masks['change_mask']
        unchanged_mask = masks['unchanged_mask']
        confidence = masks['confidence']
        change_prob = masks['change_prob']

        l12 = self.L12(
            feat_a, feat_a_aug, feat_b, feat_b_aug,
            change_mask, unchanged_mask, confidence
        )

        l34 = self.L34(feat_a, feat_b, change_mask, unchanged_mask, confidence)

        l5 = self.L5(change_prob)

        total = (
            wup * self.w_l12 * l12
            + wup * self.w_l34 * l34
            + self.w_sparsity * l5
        )

        return {
            'total': total,
            'L12_quadruplet': l12.detach(),
            'L34_spatiotemporal': l34.detach(),
            'L5_sparsity': l5.detach(),
            'warmup_weight': wup,
            'change_ratio': change_mask.mean().item(),
            'unchanged_ratio': unchanged_mask.mean().item(),
        }


def build_cdcl_loss(cfg: dict = None) -> CDCLoss:
    if cfg is None:
        cfg = {}
    return CDCLoss(
        w_l12=cfg.get('w_l12', 1.5),
        w_l34=cfg.get('w_l34', 1.0),
        w_sparsity=cfg.get('w_sparsity', 2.0),
        high_thresh=cfg.get('high_thresh', 0.65),
        low_thresh=cfg.get('low_thresh', 0.35),
        warmup_epochs=cfg.get('warmup_epochs', 5),
        temperature_l12=cfg.get('temperature_l12', 0.07),
        temperature_l34=cfg.get('temperature_l34', 0.1),
        patch_size_l34=cfg.get('patch_size_l34', 8),
        global_sparsity_thresh=cfg.get('global_sparsity_thresh', 0.35),
    )
