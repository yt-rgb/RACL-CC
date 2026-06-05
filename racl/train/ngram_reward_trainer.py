"""
N-gram Reward Trainer: 使用多层次 n-gram 奖励提升 BLEU-2~4

核心设计理念：
1. 分层奖励：BLEU-2, BLEU-3, BLEU-4 分别计算奖励
2. 自适应权重：根据当前性能动态调整各层权重
3. 保护机制：确保不降低 BLEU-1, CIDEr, ROUGE-L

损失函数：
Total_Loss = α × CE_Loss + β × N-gram_Reward_Loss + γ × Diversity_Loss

其中：
- CE_Loss: 标准交叉熵损失（保证基础性能）
- N-gram_Reward_Loss: 多层次 n-gram 奖励（提升 BLEU-2~4）
- Diversity_Loss: 多样性惩罚（保护 CIDEr 和 ROUGE-L）

作者: RACL Team
日期: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from collections import Counter
import numpy as np


class NGramRewardLoss(nn.Module):
    """
    多层次 N-gram 奖励损失
    
    设计特点：
    1. 分层计算 BLEU-2, BLEU-3, BLEU-4
    2. 自适应权重：根据当前性能调整
    3. 平滑函数：避免梯度消失
    """
    
    def __init__(
        self,
        bleu2_weight: float = 0.3,
        bleu3_weight: float = 0.4,
        bleu4_weight: float = 0.3,
        epsilon: float = 1e-6,
        use_adaptive_weights: bool = True
    ):
        """
        Args:
            bleu2_weight: BLEU-2 的初始权重
            bleu3_weight: BLEU-3 的初始权重
            bleu4_weight: BLEU-4 的初始权重
            epsilon: 数值稳定性参数
            use_adaptive_weights: 是否使用自适应权重
        """
        super().__init__()
        self.bleu2_weight = bleu2_weight
        self.bleu3_weight = bleu3_weight
        self.bleu4_weight = bleu4_weight
        self.epsilon = epsilon
        self.use_adaptive_weights = use_adaptive_weights
        
        # 记录历史性能（用于自适应权重）
        self.register_buffer('bleu2_history', torch.tensor([0.5]))
        self.register_buffer('bleu3_history', torch.tensor([0.3]))
        self.register_buffer('bleu4_history', torch.tensor([0.2]))
    
    def extract_ngrams(self, tokens: List[str], n: int) -> Counter:
        """
        提取 n-gram
        
        Args:
            tokens: token 列表
            n: n-gram 的 n
        
        Returns:
            n-gram 计数器
        """
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            ngrams.append(ngram)
        return Counter(ngrams)
    
    def compute_ngram_precision(
        self, 
        pred_tokens: List[str], 
        ref_tokens: List[str], 
        n: int
    ) -> float:
        """
        计算 n-gram 精确率（带裁剪）
        
        Args:
            pred_tokens: 预测的 token 列表
            ref_tokens: 参考的 token 列表
            n: n-gram 的 n
        
        Returns:
            n-gram 精确率 [0, 1]
        """
        if len(pred_tokens) < n or len(ref_tokens) < n:
            return 0.0
        
        pred_ngrams = self.extract_ngrams(pred_tokens, n)
        ref_ngrams = self.extract_ngrams(ref_tokens, n)
        
        # 裁剪计数（每个 n-gram 最多计数参考中的出现次数）
        clipped_count = 0
        total_count = 0
        
        for ngram, count in pred_ngrams.items():
            clipped_count += min(count, ref_ngrams.get(ngram, 0))
            total_count += count
        
        if total_count == 0:
            return 0.0
        
        precision = clipped_count / total_count
        return precision
    
    def compute_brevity_penalty(
        self, 
        pred_length: int, 
        ref_length: int
    ) -> float:
        """
        计算长度惩罚（Brevity Penalty）
        
        Args:
            pred_length: 预测长度
            ref_length: 参考长度
        
        Returns:
            BP [0, 1]
        """
        if pred_length >= ref_length:
            return 1.0
        else:
            return np.exp(1 - ref_length / (pred_length + self.epsilon))
    
    def compute_bleu_n(
        self, 
        pred_tokens: List[str], 
        ref_tokens: List[str], 
        n: int
    ) -> float:
        """
        计算 BLEU-n 分数
        
        Args:
            pred_tokens: 预测的 token 列表
            ref_tokens: 参考的 token 列表
            n: n-gram 的 n
        
        Returns:
            BLEU-n 分数 [0, 1]
        """
        # 计算 1-gram 到 n-gram 的精确率
        precisions = []
        for i in range(1, n + 1):
            p = self.compute_ngram_precision(pred_tokens, ref_tokens, i)
            precisions.append(p)
        
        # 几何平均
        if any(p == 0 for p in precisions):
            # 使用平滑（加 epsilon）
            precisions = [p + self.epsilon for p in precisions]
        
        geo_mean = np.exp(np.mean([np.log(p) for p in precisions]))
        
        # 长度惩罚
        bp = self.compute_brevity_penalty(len(pred_tokens), len(ref_tokens))
        
        bleu = bp * geo_mean
        return bleu
    
    def update_adaptive_weights(
        self, 
        current_bleu2: float, 
        current_bleu3: float, 
        current_bleu4: float
    ):
        """
        根据当前性能更新自适应权重
        
        策略：性能越低的指标，权重越高
        
        Args:
            current_bleu2: 当前 BLEU-2 分数
            current_bleu3: 当前 BLEU-3 分数
            current_bleu4: 当前 BLEU-4 分数
        """
        if not self.use_adaptive_weights:
            return
        
        # 更新历史
        self.bleu2_history = torch.cat([
            self.bleu2_history, 
            torch.tensor([current_bleu2])
        ])[-100:]  # 保留最近 100 个
        
        self.bleu3_history = torch.cat([
            self.bleu3_history, 
            torch.tensor([current_bleu3])
        ])[-100:]
        
        self.bleu4_history = torch.cat([
            self.bleu4_history, 
            torch.tensor([current_bleu4])
        ])[-100:]
        
        # 计算平均性能
        avg_bleu2 = self.bleu2_history.mean().item()
        avg_bleu3 = self.bleu3_history.mean().item()
        avg_bleu4 = self.bleu4_history.mean().item()
        
        # 计算权重：性能越低，权重越高
        # 使用 softmax 归一化
        scores = torch.tensor([
            1.0 / (avg_bleu2 + self.epsilon),
            1.0 / (avg_bleu3 + self.epsilon),
            1.0 / (avg_bleu4 + self.epsilon)
        ])
        weights = F.softmax(scores, dim=0)
        
        self.bleu2_weight = weights[0].item()
        self.bleu3_weight = weights[1].item()
        self.bleu4_weight = weights[2].item()
    
    def forward(
        self, 
        pred_ids: torch.Tensor,      # [B, L]
        ref_ids: torch.Tensor,       # [B, L]
        tokenizer,
        pad_token_id: int = 0
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算 N-gram 奖励损失
        
        Args:
            pred_ids: 预测的 token IDs [batch_size, seq_len]
            ref_ids: 参考的 token IDs [batch_size, seq_len]
            tokenizer: 用于解码的 tokenizer
            pad_token_id: padding token 的 ID
        
        Returns:
            (loss, metrics): 损失和各项指标
        """
        batch_size = pred_ids.shape[0]
        
        bleu2_scores = []
        bleu3_scores = []
        bleu4_scores = []
        
        # 是否打印样本（每100步打印一次）
        should_print_samples = hasattr(self, '_step_counter')
        if not hasattr(self, '_step_counter'):
            self._step_counter = 0
        self._step_counter += 1
        print_samples = (self._step_counter % 100 == 1)  # 每100步打印一次
        
        for i in range(batch_size):
            # ⚠️ 关键修复：确保 pred_ids 和 ref_ids 长度一致
            # 取两者的最小长度，避免索引越界
            min_len = min(pred_ids.shape[1], ref_ids.shape[1])
            pred_ids_i = pred_ids[i, :min_len]
            ref_ids_i = ref_ids[i, :min_len]
            
            # ref_ids 中 -100 表示输入部分（不参与损失计算），>=0 表示输出部分
            output_mask = ref_ids_i >= 0
            
            # 从 pred_ids 和 ref_ids 中提取输出部分
            pred_tokens_ids = pred_ids_i[output_mask]
            ref_tokens_ids = ref_ids_i[output_mask]
            
            # 进一步移除 padding token
            valid_mask = (pred_tokens_ids != pad_token_id) & (ref_tokens_ids != pad_token_id)
            pred_tokens_ids = pred_tokens_ids[valid_mask]
            ref_tokens_ids = ref_tokens_ids[valid_mask]
            
            # 转换为 Python list 以避免类型转换问题
            pred_tokens_ids_list = pred_tokens_ids.tolist()
            ref_tokens_ids_list = ref_tokens_ids.tolist()
            
            # 过滤特殊 token（Qwen2 的特殊 token 范围通常是 151643-151655）
            # 同时过滤常见的特殊 token
            special_token_ids = set(range(151643, 151656))  # Qwen2 特殊 tokens
            if hasattr(tokenizer, 'all_special_ids'):
                special_token_ids.update(tokenizer.all_special_ids)
            
            pred_tokens_ids_list = [tid for tid in pred_tokens_ids_list if tid not in special_token_ids]
            ref_tokens_ids_list = [tid for tid in ref_tokens_ids_list if tid not in special_token_ids]
            
            # 解码为文本
            pred_text = tokenizer.decode(pred_tokens_ids_list, skip_special_tokens=True)
            ref_text = tokenizer.decode(ref_tokens_ids_list, skip_special_tokens=True)
            
            # 打印前几个样本（用于调试）
            if print_samples and i < 2:  # 只打印前2个样本
                print(f"\n{'='*80}")
                print(f"样本 {i+1}:")
                print(f"{'='*80}")
                print(f"预测 token IDs (前20个): {pred_tokens_ids_list[:20]}")
                print(f"参考 token IDs (前20个): {ref_tokens_ids_list[:20]}")
                print(f"预测文本: {pred_text}")
                print(f"参考文本: {ref_text}")
                print(f"{'='*80}\n")
            
            # 分词
            pred_tokens = pred_text.lower().split()
            ref_tokens = ref_text.lower().split()
            
            # 计算 BLEU-2, BLEU-3, BLEU-4
            bleu2 = self.compute_bleu_n(pred_tokens, ref_tokens, 2)
            bleu3 = self.compute_bleu_n(pred_tokens, ref_tokens, 3)
            bleu4 = self.compute_bleu_n(pred_tokens, ref_tokens, 4)
            
            bleu2_scores.append(bleu2)
            bleu3_scores.append(bleu3)
            bleu4_scores.append(bleu4)
        
        # 转换为 tensor
        bleu2_tensor = torch.tensor(bleu2_scores, device=pred_ids.device)
        bleu3_tensor = torch.tensor(bleu3_scores, device=pred_ids.device)
        bleu4_tensor = torch.tensor(bleu4_scores, device=pred_ids.device)
        
        # 计算损失：-log(BLEU + ε)
        # 使用 log 使得梯度更稳定
        loss2 = -torch.log(bleu2_tensor + self.epsilon).mean()
        loss3 = -torch.log(bleu3_tensor + self.epsilon).mean()
        loss4 = -torch.log(bleu4_tensor + self.epsilon).mean()
        
        # 加权组合
        total_loss = (
            self.bleu2_weight * loss2 +
            self.bleu3_weight * loss3 +
            self.bleu4_weight * loss4
        )
        
        # 更新自适应权重
        with torch.no_grad():
            self.update_adaptive_weights(
                bleu2_tensor.mean().item(),
                bleu3_tensor.mean().item(),
                bleu4_tensor.mean().item()
            )
        
        # 返回指标（用于监控）
        metrics = {
            'bleu2': bleu2_tensor.mean().item(),
            'bleu3': bleu3_tensor.mean().item(),
            'bleu4': bleu4_tensor.mean().item(),
            'bleu2_weight': self.bleu2_weight,
            'bleu3_weight': self.bleu3_weight,
            'bleu4_weight': self.bleu4_weight,
        }
        
        return total_loss, metrics


class DiversityLoss(nn.Module):
    """
    多样性损失：保护 CIDEr 和 ROUGE-L
    
    设计理念：
    1. 惩罚重复的 n-gram（提升多样性）
    2. 鼓励使用不同的词汇（提升 CIDEr）
    3. 保持句子结构的流畅性（保护 ROUGE-L）
    """
    
    def __init__(
        self,
        repetition_penalty: float = 0.1,
        vocab_diversity_weight: float = 0.05
    ):
        """
        Args:
            repetition_penalty: 重复惩罚权重
            vocab_diversity_weight: 词汇多样性权重
        """
        super().__init__()
        self.repetition_penalty = repetition_penalty
        self.vocab_diversity_weight = vocab_diversity_weight
    
    def compute_repetition_penalty(
        self, 
        pred_tokens: List[str]
    ) -> float:
        """
        计算重复惩罚
        
        策略：惩罚重复的 2-gram 和 3-gram
        
        Args:
            pred_tokens: 预测的 token 列表
        
        Returns:
            重复惩罚分数 [0, 1]，越高表示重复越多
        """
        if len(pred_tokens) < 2:
            return 0.0
        
        # 2-gram 重复率
        bigrams = [tuple(pred_tokens[i:i+2]) for i in range(len(pred_tokens)-1)]
        bigram_counter = Counter(bigrams)
        bigram_repetition = sum(c - 1 for c in bigram_counter.values() if c > 1)
        bigram_repetition_rate = bigram_repetition / max(len(bigrams), 1)
        
        # 3-gram 重复率
        if len(pred_tokens) >= 3:
            trigrams = [tuple(pred_tokens[i:i+3]) for i in range(len(pred_tokens)-2)]
            trigram_counter = Counter(trigrams)
            trigram_repetition = sum(c - 1 for c in trigram_counter.values() if c > 1)
            trigram_repetition_rate = trigram_repetition / max(len(trigrams), 1)
        else:
            trigram_repetition_rate = 0.0
        
        # 综合重复率
        repetition_score = (bigram_repetition_rate + trigram_repetition_rate) / 2
        return repetition_score
    
    def compute_vocab_diversity(
        self, 
        pred_tokens: List[str]
    ) -> float:
        """
        计算词汇多样性
        
        策略：Type-Token Ratio (TTR)
        
        Args:
            pred_tokens: 预测的 token 列表
        
        Returns:
            词汇多样性分数 [0, 1]，越高表示多样性越好
        """
        if len(pred_tokens) == 0:
            return 0.0
        
        unique_tokens = len(set(pred_tokens))
        total_tokens = len(pred_tokens)
        
        ttr = unique_tokens / total_tokens
        return ttr
    
    def forward(
        self, 
        pred_ids: torch.Tensor,      # [B, L]
        ref_ids: torch.Tensor,       # [B, L] - 用于确定输出位置
        tokenizer,
        pad_token_id: int = 0
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算多样性损失
        
        Args:
            pred_ids: 预测的 token IDs [batch_size, seq_len]
            ref_ids: 参考的 token IDs [batch_size, seq_len] - 用于确定输出位置
            tokenizer: 用于解码的 tokenizer
            pad_token_id: padding token 的 ID
        
        Returns:
            (loss, metrics): 损失和各项指标
        """
        batch_size = pred_ids.shape[0]
        
        repetition_scores = []
        diversity_scores = []
        
        for i in range(batch_size):
            # ⚠️ 关键修复：确保 pred_ids 和 ref_ids 长度一致
            min_len = min(pred_ids.shape[1], ref_ids.shape[1])
            pred_ids_i = pred_ids[i, :min_len]
            ref_ids_i = ref_ids[i, :min_len]
            
            # 只使用 ref_ids 中非 -100 的位置（输出部分）
            output_mask = ref_ids_i >= 0
            pred_tokens_ids = pred_ids_i[output_mask]
            
            # 进一步移除 padding token
            valid_mask = pred_tokens_ids != pad_token_id
            pred_tokens_ids = pred_tokens_ids[valid_mask]
            
            # 转换为 Python list 以避免类型转换问题
            pred_tokens_ids_list = pred_tokens_ids.tolist()
            
            # 过滤特殊 token（Qwen2 的特殊 token 范围通常是 151643-151655）
            special_token_ids = set(range(151643, 151656))
            if hasattr(tokenizer, 'all_special_ids'):
                special_token_ids.update(tokenizer.all_special_ids)
            
            pred_tokens_ids_list = [tid for tid in pred_tokens_ids_list if tid not in special_token_ids]
            
            # 解码为文本
            pred_text = tokenizer.decode(pred_tokens_ids_list, skip_special_tokens=True)
            pred_tokens = pred_text.lower().split()
            
            # 计算重复惩罚
            rep_score = self.compute_repetition_penalty(pred_tokens)
            repetition_scores.append(rep_score)
            
            # 计算词汇多样性
            div_score = self.compute_vocab_diversity(pred_tokens)
            diversity_scores.append(div_score)
        
        # 转换为 tensor
        rep_tensor = torch.tensor(repetition_scores, device=pred_ids.device)
        div_tensor = torch.tensor(diversity_scores, device=pred_ids.device)
        
        # 损失：惩罚重复，鼓励多样性
        repetition_loss = rep_tensor.mean()
        diversity_loss = -torch.log(div_tensor.mean() + 1e-6)
        
        total_loss = (
            self.repetition_penalty * repetition_loss +
            self.vocab_diversity_weight * diversity_loss
        )
        
        # 返回指标
        metrics = {
            'repetition_rate': rep_tensor.mean().item(),
            'vocab_diversity': div_tensor.mean().item(),
        }
        
        return total_loss, metrics


class MultiObjectiveCaptionLoss(nn.Module):
    """
    多目标描述生成损失
    
    组合：
    1. 交叉熵损失（基础性能）
    2. N-gram 奖励损失（提升 BLEU-2~4）
    3. 多样性损失（保护 CIDEr 和 ROUGE-L）
    
    自适应权重调整：
    - 监控各项指标的变化
    - 动态调整损失权重
    """
    
    def __init__(
        self,
        ce_weight: float = 0.7,
        ngram_weight: float = 0.2,
        diversity_weight: float = 0.1,
        use_adaptive_weights: bool = True
    ):
        """
        Args:
            ce_weight: 交叉熵损失权重
            ngram_weight: N-gram 奖励损失权重
            diversity_weight: 多样性损失权重
            use_adaptive_weights: 是否使用自适应权重
        """
        super().__init__()
        self.ce_weight = ce_weight
        self.ngram_weight = ngram_weight
        self.diversity_weight = diversity_weight
        self.use_adaptive_weights = use_adaptive_weights
        
        # 子损失
        self.ngram_loss = NGramRewardLoss(use_adaptive_weights=use_adaptive_weights)
        self.diversity_loss = DiversityLoss()
        
        # 记录历史指标（用于自适应权重）
        self.register_buffer('bleu1_history', torch.tensor([0.7]))
        self.register_buffer('cider_history', torch.tensor([0.5]))
        self.register_buffer('rougel_history', torch.tensor([0.6]))
    
    def update_adaptive_weights(
        self,
        current_bleu1: float,
        current_cider: float,
        current_rougel: float
    ):
        """
        根据保护指标的变化调整权重
        
        策略：
        - 如果 BLEU-1, CIDEr, ROUGE-L 下降，降低 ngram_weight
        - 如果它们稳定或提升，可以增加 ngram_weight
        
        Args:
            current_bleu1: 当前 BLEU-1 分数
            current_cider: 当前 CIDEr 分数
            current_rougel: 当前 ROUGE-L 分数
        """
        if not self.use_adaptive_weights:
            return
        
        # 更新历史
        self.bleu1_history = torch.cat([
            self.bleu1_history,
            torch.tensor([current_bleu1])
        ])[-50:]
        
        self.cider_history = torch.cat([
            self.cider_history,
            torch.tensor([current_cider])
        ])[-50:]
        
        self.rougel_history = torch.cat([
            self.rougel_history,
            torch.tensor([current_rougel])
        ])[-50:]
        
        # 计算趋势（最近 10 个 vs 之前 10 个）
        if len(self.bleu1_history) >= 20:
            bleu1_trend = (
                self.bleu1_history[-10:].mean() - 
                self.bleu1_history[-20:-10].mean()
            ).item()
            
            cider_trend = (
                self.cider_history[-10:].mean() - 
                self.cider_history[-20:-10].mean()
            ).item()
            
            rougel_trend = (
                self.rougel_history[-10:].mean() - 
                self.rougel_history[-20:-10].mean()
            ).item()
            
            # 如果保护指标下降，降低 ngram_weight
            if bleu1_trend < -0.01 or cider_trend < -0.01 or rougel_trend < -0.01:
                self.ngram_weight = max(0.1, self.ngram_weight * 0.9)
                self.diversity_weight = min(0.2, self.diversity_weight * 1.1)
                print(f"⚠️  保护指标下降，调整权重：ngram={self.ngram_weight:.3f}, diversity={self.diversity_weight:.3f}")
            
            # 如果保护指标稳定，可以增加 ngram_weight
            elif bleu1_trend > 0 and cider_trend > 0 and rougel_trend > 0:
                self.ngram_weight = min(0.3, self.ngram_weight * 1.05)
                print(f"✅ 保护指标提升，增加 ngram_weight={self.ngram_weight:.3f}")
    
    def forward(
        self,
        ce_loss: torch.Tensor,       # 交叉熵损失
        pred_ids: torch.Tensor,      # [B, L]
        ref_ids: torch.Tensor,       # [B, L]
        tokenizer,
        pad_token_id: int = 0,
        current_metrics: Optional[Dict[str, float]] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算多目标损失
        
        Args:
            ce_loss: 交叉熵损失
            pred_ids: 预测的 token IDs
            ref_ids: 参考的 token IDs
            tokenizer: tokenizer
            pad_token_id: padding token ID
            current_metrics: 当前的评估指标（用于自适应权重）
        
        Returns:
            (total_loss, all_metrics): 总损失和所有指标
        """
        # 计算 N-gram 奖励损失
        ngram_loss, ngram_metrics = self.ngram_loss(
            pred_ids, ref_ids, tokenizer, pad_token_id
        )
        
        # 计算多样性损失（传入 ref_ids 用于确定输出位置）
        div_loss, div_metrics = self.diversity_loss(
            pred_ids, ref_ids, tokenizer, pad_token_id
        )
        
        # 组合损失
        total_loss = (
            self.ce_weight * ce_loss +
            self.ngram_weight * ngram_loss +
            self.diversity_weight * div_loss
        )
        
        # 更新自适应权重
        if current_metrics is not None:
            with torch.no_grad():
                self.update_adaptive_weights(
                    current_metrics.get('bleu1', 0.7),
                    current_metrics.get('cider', 0.5),
                    current_metrics.get('rougel', 0.6)
                )
        
        # 合并所有指标
        all_metrics = {
            'ce_loss': ce_loss.item(),
            'ngram_loss': ngram_loss.item(),
            'diversity_loss': div_loss.item(),
            'total_loss': total_loss.item(),
            'ce_weight': self.ce_weight,
            'ngram_weight': self.ngram_weight,
            'diversity_weight': self.diversity_weight,
            **ngram_metrics,
            **div_metrics,
        }
        
        return total_loss, all_metrics
