"""
TGCFA: Text-Guided Attention Pooling + Auxiliary NCE Alignment

重写后的 TGCFA 不再修改视觉变化特征本身，而是仅在训练阶段计算一个
辅助语义对齐损失：

    z_t = TextPool(F_t)
    z_v^aux = AttnPool(F_c | z_t)
    L_nce = -log exp(sim(z_v^i, z_t^i) / tau) / sum_j exp(sim(z_v^i, z_t^j) / tau)

推理阶段没有 GT 文本时，直接返回 0 损失，不影响视觉主干路径。
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextGuidedAttentionPooling(nn.Module):
    """
    使用句级文本语义对变化 token 做注意力池化。

    Args:
        visual_dim: 视觉变化特征维度 C_v
        text_dim  : 文本特征维度 C_t
    """

    def __init__(self, visual_dim: int, text_dim: int):
        super().__init__()
        self.visual_dim = visual_dim
        self.query_proj = nn.Linear(text_dim, visual_dim)
        self.key_proj = nn.Linear(visual_dim, visual_dim)
        self.visual_proj = nn.Linear(visual_dim, visual_dim)
        self.text_proj = nn.Linear(text_dim, visual_dim)
        self.visual_norm = nn.LayerNorm(visual_dim)
        self.text_norm = nn.LayerNorm(visual_dim)

    def text_pool(self, text_feat: torch.Tensor) -> torch.Tensor:
        """
        文本池化：对 token 维取均值，得到句级语义向量。

        Args:
            text_feat: [B, L, C_t]
        Returns:
            z_t: [B, C_t]
        """
        return text_feat.mean(dim=1)

    def forward(self, visual_feat: torch.Tensor, text_feat: torch.Tensor):
        """
        Args:
            visual_feat: [B, N, C_v]
            text_feat  : [B, L, C_t]
        Returns:
            z_v: [B, C_v] 文本引导池化后的视觉向量
            z_t: [B, C_v] 投影后的句级文本向量
            attn: [B, N]  注意力权重
        """
        z_t_raw = self.text_pool(text_feat)
        q_t = self.query_proj(z_t_raw)                     # [B, C_v]
        k_v = self.key_proj(visual_feat)                   # [B, N, C_v]

        scores = torch.einsum("bc,bnc->bn", q_t, k_v) / (self.visual_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        z_v = torch.einsum("bn,bnc->bc", attn, visual_feat)

        z_v = self.visual_norm(self.visual_proj(z_v))
        z_t = self.text_norm(self.text_proj(z_t_raw))
        return z_v, z_t, attn


class TextGuidedChangeAlignment(nn.Module):
    """
    新版 TGCFA：仅计算训练期辅助对齐损失，不改动视觉变化特征。

    Args:
        visual_dim       : 视觉变化特征维度
        text_dim         : 文本特征维度
        temperature      : InfoNCE 温度系数
    """

    def __init__(
        self,
        visual_dim: int,
        text_dim: int,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.pooler = TextGuidedAttentionPooling(visual_dim=visual_dim, text_dim=text_dim)
        self.temperature = temperature

    def _info_nce(self, visual_embed: torch.Tensor, text_embed: torch.Tensor) -> torch.Tensor:
        visual_embed = F.normalize(visual_embed, dim=-1)
        text_embed = F.normalize(text_embed, dim=-1)

        logits = torch.matmul(visual_embed, text_embed.transpose(0, 1)) / self.temperature
        targets = torch.arange(logits.size(0), device=logits.device)
        return F.cross_entropy(logits, targets)

    def forward(
        self,
        visual_feat: torch.Tensor,
        text_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            visual_feat: [B, N, C_v]
            text_feat  : [B, L, C_t]，训练时传入，推理时为 None
        Returns:
            alignment_loss: 标量；无文本时返回 0
        """
        if text_feat is None:
            return visual_feat.new_zeros(())

        z_v, z_t, _ = self.pooler(visual_feat, text_feat.to(visual_feat.dtype))
        return self._info_nce(z_v, z_t)


def build_tgcfa(config) -> Optional["TextGuidedChangeAlignment"]:
    if not getattr(config, "use_tgcfa", False):
        return None

    return TextGuidedChangeAlignment(
        visual_dim=getattr(config, "mm_hidden_size", 1024),
        text_dim=getattr(config, "tgcfa_text_dim", 768),
        temperature=getattr(config, "tgcfa_temperature", 0.07),
    )
