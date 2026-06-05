"""llava_trainer_tgcfa.py

在 LLaVATrainer 基础上添加 TGCFA 文本引导支持。

核心修改：
  compute_loss() 中，若 self.args.tgcfa_text_feat_fn 不为 None，
  则从当前 batch 的 labels 中解码出 caption 文本，
  经冻结的 CLIP 文本编码器得到 text_feat，
  再以 inputs['text_feat'] 形式传入模型前向。

文本编码器完全冻结，不参与反向传播，纯作语义引导。
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from constants import IGNORE_INDEX
from utils import rank0_print

# 导入原版 LLaVATrainer（从正式 llava_trainer.py）
from llava_trainer import LLaVATrainer


class LLaVATrainerTGCFA(LLaVATrainer):
    """
    在原版 LLaVATrainer 基础上，于 compute_loss 中注入 TGCFA 文本特征。

    额外属性（由 training_args 提供）：
      self.args.tgcfa_text_feat_fn : Callable[[List[str]], Tensor] | None
          接收 caption 字符串列表，返回 [B, L, C_t] text_feat。
          由 train_tgcfa.py 中的 TGCFATextEncoder.encode 包装而成。
    """

    def _decode_labels_to_captions(self, labels: torch.Tensor) -> list:
        """
        从 batch labels 中解码 caption 文本。

        labels: [B, seq_len]，IGNORE_INDEX(-100) 处为 prompt/padding，
                其余位置是 caption token id。
        返回: List[str]，长度 = B，每条为对应样本的 caption 字符串。
        """
        captions = []
        for row in labels:
            # 取有效 token（非 IGNORE_INDEX 部分）
            valid_ids = row[row != IGNORE_INDEX]
            if len(valid_ids) == 0:
                captions.append("")
            else:
                # 解码时跳过特殊 token
                text = self.tokenizer.decode(
                    valid_ids.tolist(),
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                captions.append(text.strip())
        return captions

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        在原版 DWA compute_loss 基础上，注入 text_feat 到 inputs。

        流程:
          1. 若 tgcfa_text_feat_fn 存在，优先使用 inputs["captions"]，否则从 labels 解码 caption
          2. 调用冻结的 CLIP 文本编码器，得到 text_feat [B, L, C_t]
          3. 将 text_feat 存入 inputs["text_feat"]，随 model(**inputs) 传入
          4. captions 仅供 trainer 使用，在调用模型前移除
          5. 其余 DWA 逻辑与原版完全一致
        """
        captions = inputs.pop("captions", None)

        # ── 注入 text_feat ───────────────────────────────────────────────
        tgcfa_fn = getattr(self.args, "tgcfa_text_feat_fn", None)
        if tgcfa_fn is not None:
            try:
                if captions is None and "labels" in inputs:
                    captions = self._decode_labels_to_captions(inputs["labels"])
                if captions is not None:
                    with torch.no_grad():
                        text_feat = tgcfa_fn(captions)  # [B, L, C_t]
                    inputs["text_feat"] = text_feat
            except Exception as e:
                # 文本编码失败时静默降级，不中断训练
                rank0_print(f"[TGCFA] WARNING: text encoding failed: {e}. Skipping text_feat.")
        # ────────────────────────────────────────────────────────────────

        # 原版 DWA 逻辑（caption/seg 保留，align_loss 不参与 DWA）
        outputs = model(**inputs)
        losses = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        if not isinstance(losses, (list, tuple)):
            losses = [losses]

        align_loss = None
        dwa_losses = list(losses)
        if len(dwa_losses) > self.task_num:
            align_loss = dwa_losses[self.task_num]
            dwa_losses = dwa_losses[:self.task_num]

        align_loss_weight = getattr(self.args, "tgcfa_align_weight", 0.2)

        if self.train_loss_buffer is None:
            self.train_loss_buffer = torch.zeros((self.task_num, 2)).to(self.args.device)

        if self.epoch > 1:
            epsilon = 1e-8
            w_i = torch.Tensor(
                self.train_loss_buffer[:, 1] / (self.train_loss_buffer[:, 0] + epsilon)
            ).to(self.args.device)
            batch_weight = self.task_num * F.softmax(w_i / self.T, dim=-1)
        else:
            batch_weight = torch.ones(self.task_num).to(self.args.device)

        self.train_loss_buffer[:, 0] = self.train_loss_buffer[:, 1].clone()
        self.train_loss_buffer[:, 1] = torch.Tensor(
            [loss.item() for loss in dwa_losses]
        ).to(self.args.device)

        loss = torch.mul(torch.stack(dwa_losses), batch_weight).sum()
        if align_loss is not None:
            loss = loss + align_loss * align_loss_weight

        if return_outputs:
            return (loss, outputs)
        return loss
