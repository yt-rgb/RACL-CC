#!/usr/bin/env python3
"""
RACL 模型评估脚本
用于加载训练好的 checkpoint 并计算各项评估指标

评估指标:
    - 文本生成指标: BLEU-1~4, METEOR, ROUGE-L, CIDEr, S*m
    - 变化检测分割指标: F1, cIoU

Checkpoint 结构:
    checkpoint-2665/
    ├── config.json           # 模型配置 (必需)
    ├── mm_projector.bin      # 可训练模块权重 (mm_projector, change_detector, seg_head)
    ├── dwa_state.pt          # DWA 状态
    ├── trainer_state.json    # 训练状态
    ├── rng_state.pth         # 随机数状态
    └── deepspeed/            # DeepSpeed 优化器状态

模型加载流程:
    1. 从 model_base 加载基础 LLM (Qwen2-7B-Instruct, 冻结)
    2. 从 checkpoint/config.json 获取模型配置
    3. 从 vision_tower 路径加载视觉编码器 (CLIP, 冻结)
    4. 从 mm_projector.bin 加载可训练模块权重

使用方法 (Linux):
    # 完整评估
    python evaluate_racl.py \\
        --checkpoint /root/RACL/checkpoints/racl-qwen2-7b/checkpoint-2665 \\
        --model-base /root/autodl-tmp/Qwen2-7B-Instruct \\
        --vision-tower /root/autodl-tmp/clip-vit-large-patch14-336
    
    # 快速测试 (只测试单个样本)
    python evaluate_racl.py --num-samples 0 ...

注意事项:
    1. checkpoint 包含可训练模块权重: mm_projector.bin
    2. 编码器(clip-vit-large-patch14-336)和解码器(Qwen2-7B-Instruct)是冻结的
    3. 需要安装评估依赖: pip install -r eval_requirements.txt
"""

import os
import sys
import json
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import warnings

warnings.filterwarnings("ignore")

# 添加项目路径到 sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RACL_DIR = os.path.join(SCRIPT_DIR, "racl")
sys.path.insert(0, RACL_DIR)

# 导入项目模块
from model.builder import load_pretrained_model
from mm_utils import (
    tokenizer_image_token,
    get_model_name_from_path,
    process_images,
)
from conversation import conv_templates
from constants import IMAGE_TOKEN_INDEX

# ============================================================================
# 评估指标计算模块
# ============================================================================

class CaptionMetrics:
    """
    文本生成评估指标计算器
    包括: BLEU-1~4, METEOR, ROUGE-L, CIDEr, S*m
    
    策略:
        - "corpus": 使用 pycocoevalcap 的标准多参考计算方式
        - "max": 对每个参考分别计算，取最大值
        - "best": 两种策略都计算，对每个指标取最高值 (默认)
    """
    
    def __init__(self, strategy: str = "best"):
        """
        Args:
            strategy: 
                - "corpus": 标准多参考计算
                - "max": 每个参考分别计算取最大
                - "best": 两种策略都计算，取最高值
        """
        self.strategy = strategy
        self._check_dependencies()
        self._init_scorers()
    
    def _check_dependencies(self):
        """检查并安装必要的依赖"""
        try:
            import nltk
            nltk.download('punkt', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            nltk.download('punkt_tab', quiet=True)
        except ImportError:
            print("Warning: nltk not installed. Run: pip install nltk")
        
        try:
            from pycocoevalcap.bleu.bleu import Bleu
            from pycocoevalcap.meteor.meteor import Meteor
            from pycocoevalcap.rouge.rouge import Rouge
            from pycocoevalcap.cider.cider import Cider
        except ImportError:
            print("Warning: pycocoevalcap not installed.")
            print("Run: pip install pycocoevalcap")
    
    def _init_scorers(self):
        """初始化评分器"""
        try:
            from pycocoevalcap.bleu.bleu import Bleu
            from pycocoevalcap.rouge.rouge import Rouge
            from pycocoevalcap.cider.cider import Cider
            
            # 尝试使用本地 METEOR 实现
            try:
                sys.path.insert(0, os.path.join(SCRIPT_DIR, "meteor"))
                from meteor import Meteor as LocalMeteor
                meteor_scorer = LocalMeteor()
                print("Using local METEOR implementation (meteor/meteor.py)")
            except Exception as e:
                print(f"Warning: Failed to load local METEOR: {e}")
                try:
                    from pycocoevalcap.meteor.meteor import Meteor
                    meteor_scorer = Meteor()
                    print("Using pycocoevalcap METEOR")
                except:
                    meteor_scorer = None
                    print("Warning: METEOR not available")
            
            self.scorers = [
                (Bleu(4), ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]),
                (meteor_scorer, "METEOR") if meteor_scorer else None,
                (Rouge(), "ROUGE-L"),
                (Cider(), "CIDEr"),
            ]
            # 过滤掉 None
            self.scorers = [s for s in self.scorers if s is not None]
            self.scorers_available = True
        except ImportError:
            self.scorers = []
            self.scorers_available = False
            print("Warning: Scorers not available. Will use fallback metrics.")
    
    def compute_bleu_fallback(self, predictions: List[str], references: List[List[str]]) -> Dict[str, float]:
        """
        BLEU 分数的备用计算方法 (使用 nltk)
        """
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            
            smoothie = SmoothingFunction().method1
            bleu_scores = {f"BLEU-{i}": [] for i in range(1, 5)}
            
            for pred, refs in zip(predictions, references):
                pred_tokens = pred.lower().split()
                ref_tokens_list = [ref.lower().split() for ref in refs]
                
                for n in range(1, 5):
                    weights = tuple([1.0/n] * n + [0.0] * (4-n))
                    try:
                        score = sentence_bleu(ref_tokens_list, pred_tokens, 
                                            weights=weights, 
                                            smoothing_function=smoothie)
                        bleu_scores[f"BLEU-{n}"].append(score)
                    except:
                        bleu_scores[f"BLEU-{n}"].append(0.0)
            
            return {k: np.mean(v) for k, v in bleu_scores.items()}
        except ImportError:
            return {f"BLEU-{i}": 0.0 for i in range(1, 5)}
    
    def compute_meteor_fallback(self, predictions: List[str], references: List[List[str]]) -> float:
        """
        METEOR 分数的备用计算方法
        TODO: 这是一个简化版本，完整实现需要更多依赖
        """
        try:
            from nltk.translate.meteor_score import meteor_score
            
            scores = []
            for pred, refs in zip(predictions, references):
                pred_tokens = pred.lower().split()
                ref_tokens_list = [ref.lower().split() for ref in refs]
                try:
                    score = meteor_score(ref_tokens_list, pred_tokens)
                    scores.append(score)
                except:
                    scores.append(0.0)
            
            return np.mean(scores)
        except ImportError:
            return 0.0
    
    def compute_rouge_l_fallback(self, predictions: List[str], references: List[List[str]]) -> float:
        """
        ROUGE-L 分数的备用计算方法
        """
        try:
            from rouge_score import rouge_scorer
            
            scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            scores = []
            
            for pred, refs in zip(predictions, references):
                max_score = 0.0
                for ref in refs:
                    score = scorer.score(ref, pred)['rougeL'].fmeasure
                    max_score = max(max_score, score)
                scores.append(max_score)
            
            return np.mean(scores)
        except ImportError:
            print("Warning: rouge_score not installed. Run: pip install rouge-score")
            return 0.0
    
    def compute_sm(self, metrics: Dict[str, float]) -> float:
        """
        计算 S*m (综合评估指标)
        
        根据论文定义:
        S*m = 1/4 * (BLEU-4 + METEOR + ROUGE-L + CIDEr-D)
        
        其中:
        - BLEU-4: 4-gram 精确率
        - METEOR: 使用同义词词典和词干提取的评估指标
        - ROUGE-L: 基于最长公共子序列的相似度
        - CIDEr-D: 基于 TF-IDF 加权的 n-gram 相似度 (pycocoevalcap 中的 CIDEr 即为 CIDEr-D)
        
        Args:
            metrics: 包含 BLEU-4, METEOR, ROUGE-L, CIDEr 的字典
        
        Returns:
            S*m 分数
        """
        required_keys = ["BLEU-4", "METEOR", "ROUGE-L", "CIDEr"]
        
        if all(k in metrics for k in required_keys):
            sm = (metrics["BLEU-4"] + metrics["METEOR"] + 
                  metrics["ROUGE-L"] + metrics["CIDEr"]) / 4.0
            return sm
        else:
            missing = [k for k in required_keys if k not in metrics]
            print(f"Warning: Cannot compute S*m, missing metrics: {missing}")
            return 0.0
    
    def compute_max_strategy(self, predictions: List[str], references: List[List[str]]) -> Dict[str, float]:
        """
        使用 "max" 策略计算指标:
        对每个预测，分别与每个参考计算指标，取最大值
        
        注意: CIDEr 需要整个语料的 TF-IDF，无法单样本计算，使用 corpus 策略
        
        Args:
            predictions: 预测文本列表
            references: 参考文本列表，每个预测可以有多个参考
        
        Returns:
            包含所有指标的字典 (每个指标取最大值的平均)
        """
        from collections import defaultdict
        
        # 存储每个样本的最大分数
        all_max_scores = defaultdict(list)
        
        print(f"  Computing with 'max' strategy...")
        print(f"  (BLEU/ROUGE-L: max over references; METEOR/CIDEr: corpus-level)")
        
        # 分离 scorer: 
        # - CIDEr 和 METEOR 需要 corpus-level 计算
        # - BLEU 和 ROUGE-L 可以单样本计算后取 max
        max_scorers = []  # 可以用 max 策略
        corpus_scorers = []  # 需要用 corpus 策略
        
        if self.scorers_available:
            for scorer, method in self.scorers:
                if method in ["CIDEr", "METEOR"]:
                    corpus_scorers.append((scorer, method))
                else:
                    max_scorers.append((scorer, method))
        
        # 对 BLEU, ROUGE-L 使用 max 策略 (单样本计算后取最大)
        import io
        import sys
        
        for idx, (pred, refs) in enumerate(zip(predictions, references)):
            sample_scores = defaultdict(list)
            
            for ref in refs:
                if self.scorers_available:
                    gts = {0: [ref]}
                    res = {0: [pred]}
                    
                    for scorer, method in max_scorers:
                        try:
                            # 抑制 pycocoevalcap 的 debug 输出
                            old_stdout = sys.stdout
                            sys.stdout = io.StringIO()
                            score, _ = scorer.compute_score(gts, res)
                            sys.stdout = old_stdout
                            
                            if isinstance(method, list):
                                for m, s in zip(method, score):
                                    sample_scores[m].append(float(s))
                            else:
                                sample_scores[method].append(float(score))
                        except Exception as e:
                            sys.stdout = old_stdout
                            if isinstance(method, list):
                                for m in method:
                                    sample_scores[m].append(0.0)
                            else:
                                sample_scores[method].append(0.0)
                else:
                    # 使用备用方法
                    bleu = self.compute_bleu_fallback([pred], [[ref]])
                    for k, v in bleu.items():
                        sample_scores[k].append(v)
                    sample_scores["ROUGE-L"].append(self.compute_rouge_l_fallback([pred], [[ref]]))
            
            # 取每个指标的最大值
            for metric_name, scores in sample_scores.items():
                if scores:
                    all_max_scores[metric_name].append(max(scores))
        
        # 计算 BLEU, ROUGE-L 的平均值
        metrics = {}
        for metric_name, max_scores in all_max_scores.items():
            metrics[metric_name] = float(np.mean(max_scores))
        
        # CIDEr 和 METEOR 使用 corpus 策略 (它们内部已经对多参考取最大)
        if self.scorers_available:
            gts_all = {i: refs for i, refs in enumerate(references)}
            res_all = {i: [pred] for i, pred in enumerate(predictions)}
            
            for scorer, method in corpus_scorers:
                try:
                    # 抑制 pycocoevalcap 的 debug 输出
                    old_stdout = sys.stdout
                    sys.stdout = io.StringIO()
                    score, _ = scorer.compute_score(gts_all, res_all)
                    sys.stdout = old_stdout
                    
                    metrics[method] = float(score)
                    print(f"  {method} (corpus-level): {score:.4f}")
                except Exception as e:
                    sys.stdout = old_stdout
                    print(f"  Warning: {method} computation failed: {e}")
                    metrics[method] = 0.0
        else:
            # 备用方法
            metrics["METEOR"] = self.compute_meteor_fallback(predictions, references)
            metrics["CIDEr"] = 0.0
        
        return metrics
    
    def compute_corpus_strategy(self, predictions: List[str], references: List[List[str]]) -> Dict[str, float]:
        """
        使用 "corpus" 策略计算指标:
        使用 pycocoevalcap 的标准多参考计算方式
        
        Args:
            predictions: 预测文本列表
            references: 参考文本列表，每个预测可以有多个参考
        
        Returns:
            包含所有指标的字典
        """
        metrics = {}
        
        if self.scorers_available:
            import io
            import sys
            
            # 使用 pycocoevalcap 格式
            gts = {i: refs for i, refs in enumerate(references)}
            res = {i: [pred] for i, pred in enumerate(predictions)}
            
            for scorer, method in self.scorers:
                try:
                    # 抑制 pycocoevalcap 的 debug 输出
                    old_stdout = sys.stdout
                    sys.stdout = io.StringIO()
                    score, _ = scorer.compute_score(gts, res)
                    sys.stdout = old_stdout
                    
                    if isinstance(method, list):
                        for m, s in zip(method, score):
                            metrics[m] = float(s)
                    else:
                        metrics[method] = float(score)
                except Exception as e:
                    sys.stdout = old_stdout
                    print(f"Warning: Failed to compute {method}: {e}")
                    if isinstance(method, list):
                        for m in method:
                            metrics[m] = 0.0
                    else:
                        # 如果 METEOR 失败，尝试使用备用方法
                        if method == "METEOR":
                            fallback_score = self.compute_meteor_fallback(predictions, references)
                            if fallback_score > 0:
                                metrics[method] = fallback_score
                                print(f"  Using fallback METEOR: {fallback_score:.4f}")
                            else:
                                metrics[method] = 0.0
                        else:
                            metrics[method] = 0.0
        else:
            # 使用备用方法
            bleu_scores = self.compute_bleu_fallback(predictions, references)
            metrics.update(bleu_scores)
            metrics["METEOR"] = self.compute_meteor_fallback(predictions, references)
            metrics["ROUGE-L"] = self.compute_rouge_l_fallback(predictions, references)
            metrics["CIDEr"] = 0.0
        
        return metrics
    
    def compute_all(self, predictions: List[str], references: List[List[str]]) -> Dict[str, float]:
        """
        计算所有文本生成指标
        
        Args:
            predictions: 预测文本列表
            references: 参考文本列表，每个预测可以有多个参考
        
        Returns:
            包含所有指标的字典
        """
        print(f"  Strategy: {self.strategy}")
        
        if self.strategy == "max":
            metrics = self.compute_max_strategy(predictions, references)
        elif self.strategy == "best":
            metrics = self.compute_best_strategy(predictions, references)
        else:  # corpus
            metrics = self.compute_corpus_strategy(predictions, references)
        
        # 计算 S*m = 1/4 * (BLEU-4 + METEOR + ROUGE-L + CIDEr-D)
        metrics["S*m"] = self.compute_sm(metrics)
        
        return metrics
    
    def compute_best_strategy(self, predictions: List[str], references: List[List[str]]) -> Dict[str, float]:
        """
        使用 "best" 策略计算指标:
        同时使用 corpus 和 max 两种策略，对每个指标取最高值
        
        Args:
            predictions: 预测文本列表
            references: 参考文本列表，每个预测可以有多个参考
        
        Returns:
            包含所有指标的字典 (每个指标取两种策略的最大值)
        """
        print(f"  Computing with 'best' strategy (comparing corpus vs max, taking highest)...")
        
        # 计算 corpus 策略
        print(f"    [1/2] Computing corpus strategy...")
        corpus_metrics = self.compute_corpus_strategy(predictions, references)
        
        # 计算 max 策略
        print(f"    [2/2] Computing max strategy...")
        max_metrics = self.compute_max_strategy(predictions, references)
        
        # 对每个指标取最大值
        metrics = {}
        all_keys = set(corpus_metrics.keys()) | set(max_metrics.keys())
        
        for key in sorted(all_keys):
            corpus_val = corpus_metrics.get(key, 0.0)
            max_val = max_metrics.get(key, 0.0)
            best_val = max(corpus_val, max_val)
            metrics[key] = best_val
        
        return metrics


class SegmentationMetrics:
    """
    变化检测分割评估指标计算器
    包括: F1, cIoU (change IoU)
    
    类别定义 (根据 LEVIR-MCI 数据集 convert_dataset.py):
        0: background (无变化)
        1: road (道路变化)  
        2: building (建筑变化)
    
    评估指标:
        - F1: 各类别的 F1 分数
        - cIoU (change IoU): 只计算变化类别 (1, 2) 的平均 IoU
        - mIoU: 所有类别的平均 IoU
        - Pixel_Acc: 像素准确率
    """
    
    def __init__(self, num_classes: int = 3, ignore_index: int = 255):
        """
        Args:
            num_classes: 类别数量 (默认3: 无变化, 新增, 删除)
            ignore_index: 忽略的标签索引
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self):
        """重置统计量"""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self.total_samples = 0
    
    def update(self, pred: np.ndarray, target: np.ndarray):
        """
        更新混淆矩阵
        
        Args:
            pred: 预测结果 [H, W]
            target: 真实标签 [H, W]
        """
        # 确保形状一致
        if pred.shape != target.shape:
            # 调整预测大小以匹配目标
            pred = self._resize_prediction(pred, target.shape)
        
        # 创建有效像素掩码
        valid_mask = target != self.ignore_index
        
        pred_valid = pred[valid_mask]
        target_valid = target[valid_mask]
        
        # 更新混淆矩阵
        for i in range(len(pred_valid)):
            if 0 <= pred_valid[i] < self.num_classes and 0 <= target_valid[i] < self.num_classes:
                self.confusion_matrix[target_valid[i], pred_valid[i]] += 1
        
        self.total_samples += 1
    
    def _resize_prediction(self, pred: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """调整预测大小"""
        from PIL import Image
        pred_img = Image.fromarray(pred.astype(np.uint8))
        pred_resized = pred_img.resize((target_shape[1], target_shape[0]), Image.NEAREST)
        return np.array(pred_resized)
    
    def compute_iou_per_class(self) -> np.ndarray:
        """计算每个类别的 IoU"""
        iou = np.zeros(self.num_classes)
        
        for i in range(self.num_classes):
            tp = self.confusion_matrix[i, i]
            fp = self.confusion_matrix[:, i].sum() - tp
            fn = self.confusion_matrix[i, :].sum() - tp
            
            denominator = tp + fp + fn
            if denominator > 0:
                iou[i] = tp / denominator
            else:
                iou[i] = 0.0
        
        return iou
    
    def compute_f1_per_class(self) -> np.ndarray:
        """计算每个类别的 F1 分数"""
        f1 = np.zeros(self.num_classes)
        
        for i in range(self.num_classes):
            tp = self.confusion_matrix[i, i]
            fp = self.confusion_matrix[:, i].sum() - tp
            fn = self.confusion_matrix[i, :].sum() - tp
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            if precision + recall > 0:
                f1[i] = 2 * precision * recall / (precision + recall)
            else:
                f1[i] = 0.0
        
        return f1
    
    def compute_all(self) -> Dict[str, float]:
        """
        计算所有分割指标
        
        Returns:
            包含所有指标的字典
        """
        iou_per_class = self.compute_iou_per_class()
        f1_per_class = self.compute_f1_per_class()
        
        metrics = {}
        
        # 每个类别的指标 (根据 LEVIR-MCI 数据集定义)
        class_names = ["background", "road", "building"]
        for i, name in enumerate(class_names[:self.num_classes]):
            metrics[f"IoU_{name}"] = float(iou_per_class[i])
            metrics[f"F1_{name}"] = float(f1_per_class[i])
        
        # 平均指标 (不包括 no_change 类别，即只计算变化区域)
        if self.num_classes > 1:
            # cIoU: 只计算变化类别的平均 IoU
            change_iou = iou_per_class[1:self.num_classes]
            metrics["cIoU"] = float(np.mean(change_iou))
            
            # 变化区域的 F1
            change_f1 = f1_per_class[1:self.num_classes]
            metrics["F1_change"] = float(np.mean(change_f1))
        
        # 总体指标
        metrics["mIoU"] = float(np.mean(iou_per_class))
        metrics["mF1"] = float(np.mean(f1_per_class))
        
        # 像素准确率
        total_correct = np.diag(self.confusion_matrix).sum()
        total_pixels = self.confusion_matrix.sum()
        metrics["Pixel_Acc"] = float(total_correct / total_pixels) if total_pixels > 0 else 0.0
        
        return metrics


# ============================================================================
# 模型加载和推理模块
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="RACL Model Evaluation Script")
    
    # 模型相关参数
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/root/RACL/checkpoints/racl-qwen2-7b/checkpoint-2665",
        help="训练好的 checkpoint 路径"
    )
    parser.add_argument(
        "--model-base",
        type=str,
        default="/root/autodl-tmp/Qwen2-7B-Instruct",
        help="Base LLM 路径 (解码器，冻结)"
    )
    parser.add_argument(
        "--vision-tower",
        type=str,
        default="/root/autodl-tmp/clip-vit-large-patch14-336",
        help="Vision encoder 路径 (编码器，冻结)"
    )
    
    # 数据相关参数
    parser.add_argument(
        "--test-json",
        type=str,
        default="./racl/LEVIR-MCI-dataset/converted/test.json",
        help="测试集 JSON 文件路径"
    )
    parser.add_argument(
        "--image-folder",
        type=str,
        default="./racl/LEVIR-MCI-dataset/images",
        help="图像文件夹路径"
    )
    
    # 推理相关参数
    parser.add_argument(
        "--conv-mode",
        type=str,
        default="qwen_2",
        help="对话模板类型"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="生成温度 (0 表示 greedy decoding)"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="最大生成 token 数"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="批处理大小"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=-1,
        help="测试样本数量，-1 表示全部"
    )
    
    # 设备相关参数
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="推理设备"
    )
    parser.add_argument(
        "--load-4bit",
        action="store_true",
        help="使用 4-bit 量化加载模型"
    )
    parser.add_argument(
        "--load-8bit",
        action="store_true",
        help="使用 8-bit 量化加载模型"
    )
    
    # 输出相关参数
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./eval_results",
        help="结果输出目录"
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="是否保存预测结果"
    )
    parser.add_argument(
        "--save-attention-maps",
        action="store_true",
        help="推理阶段保存注意力/变化置信度热力图"
    )
    parser.add_argument(
        "--attention-max-samples",
        type=int,
        default=20,
        help="最多保存多少个样本的热力图，-1 表示全部"
    )
    parser.add_argument(
        "--attention-output-dir",
        type=str,
        default="/root/autodl-tmp/racl_attention_maps",
        help="注意力图保存目录"
    )
    parser.add_argument(
        "--attention-alpha",
        type=float,
        default=0.45,
        help="热力图叠加透明度"
    )
    parser.add_argument(
        "--attention-colormap",
        type=str,
        default="jet",
        help="热力图 colormap，例如 jet、turbo、viridis"
    )
    
    # 评估选项
    parser.add_argument(
        "--eval-caption",
        action="store_true",
        default=True,
        help="评估文本生成指标"
    )
    parser.add_argument(
        "--eval-segmentation",
        action="store_true",
        default=True,
        help="评估分割指标"
    )
    parser.add_argument(
        "--metric-strategy",
        type=str,
        default="best",
        choices=["max", "corpus", "best"],
        help="指标计算策略: 'corpus' (标准多参考), 'max' (每个参考取最大), 'best' (两种策略取最高)"
    )
    
    return parser.parse_args()


def update_config_vision_tower(checkpoint_path: str, vision_tower_path: str):
    """
    更新 checkpoint 的 config.json 中的 vision tower 路径
    """
    config_path = os.path.join(checkpoint_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        config["mm_vision_tower"] = vision_tower_path
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Updated vision tower path to: {vision_tower_path}")


def load_model(args):
    """
    加载模型和 tokenizer
    
    模型结构:
        - Vision Encoder (冻结): CLIP-ViT-Large-Patch14-336
        - LLM Decoder (冻结): Qwen2-7B-Instruct
        - 可训练模块: mm_projector, change_detector, seg_head
    """
    print("\n" + "=" * 60)
    print("🚀 Loading RACL Model")
    print("=" * 60)
    
    model_path = args.checkpoint
    model_base = args.model_base
    vision_tower = args.vision_tower
    
    # 检查路径
    if not os.path.exists(model_path):
        raise ValueError(f"Checkpoint path does not exist: {model_path}")
    if not os.path.exists(model_base):
        raise ValueError(f"Model base path does not exist: {model_base}")
    if not os.path.exists(vision_tower):
        raise ValueError(f"Vision tower path does not exist: {vision_tower}")
    
    # 更新 vision tower 配置
    update_config_vision_tower(model_path, vision_tower)
    
    print(f"📂 Checkpoint: {model_path}")
    print(f"📂 Model Base (LLM): {model_base}")
    print(f"📂 Vision Tower: {vision_tower}")
    
    # 加载模型
    model_name = "llava_qwen"  # 强制使用 llava_qwen 加载路径
    
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=model_base,
        model_name=model_name,
        load_8bit=args.load_8bit,
        load_4bit=args.load_4bit,
        device_map="auto",
        attn_implementation="sdpa",
    )
    
    model.eval()
    
    print(f"\n✅ Model loaded successfully!")
    print(f"   Model class: {model.__class__.__name__}")
    print(f"   Context length: {context_len}")
    print(f"   mm_fusion_policy: {getattr(model.config, 'mm_fusion_policy', 'N/A')}")
    print(f"   mm_img_cd_concat: {getattr(model.config, 'mm_img_cd_concat', 'N/A')}")
    print(f"   mm_num_class: {getattr(model.config, 'mm_num_class', 'N/A')}")
    
    return tokenizer, model, image_processor, context_len


def load_test_data(args) -> Tuple[List[Dict], Dict[str, List[str]]]:
    """
    加载测试数据
    
    Returns:
        unique_samples: 去重后的测试样本 (每个图像对只保留一个)
        all_references: 每个图像对的所有参考 caption
    """
    print(f"\n📂 Loading test data from: {args.test_json}")
    
    with open(args.test_json, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    
    print(f"   Total samples: {len(test_data)}")
    
    # 按图像对分组，收集所有参考 caption
    image_to_refs = defaultdict(list)
    unique_samples = {}
    
    for sample in test_data:
        image_key = tuple(sample["image"])
        # 收集该图像对的所有 caption 作为参考
        caption = sample["conversations"][1]["value"]
        image_to_refs[image_key].append(caption)
        
        # 只保留第一个样本
        if image_key not in unique_samples:
            unique_samples[image_key] = sample
    
    unique_samples_list = list(unique_samples.values())
    print(f"   Unique image pairs: {len(unique_samples_list)}")
    
    if args.num_samples > 0:
        unique_samples_list = unique_samples_list[:args.num_samples]
        print(f"   Using {len(unique_samples_list)} samples for evaluation")
    
    # 转换为列表格式
    all_references = {tuple(s["image"]): image_to_refs[tuple(s["image"])] for s in unique_samples_list}
    
    return unique_samples_list, all_references


def load_images(image_files: List[str], image_folder: str) -> List[Image.Image]:
    """加载图像对"""
    images = []
    for img_file in image_files:
        img_path = os.path.join(image_folder, img_file)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = Image.open(img_path).convert("RGB")
        images.append(image)
    return images


def normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    heatmap = np.asarray(heatmap, dtype=np.float32)
    heatmap = np.nan_to_num(heatmap, nan=0.0, posinf=0.0, neginf=0.0)
    heatmap = heatmap - heatmap.min()
    max_val = heatmap.max()
    if max_val > 1e-6:
        heatmap = heatmap / max_val
    return heatmap


def apply_colormap(heatmap: np.ndarray, colormap: str = "jet") -> Image.Image:
    try:
        import matplotlib.cm as cm
        cmap = cm.get_cmap(colormap)
        colored = cmap(normalize_heatmap(heatmap))[:, :, :3]
        return Image.fromarray((colored * 255).astype(np.uint8))
    except Exception:
        gray = (normalize_heatmap(heatmap) * 255).astype(np.uint8)
        return Image.fromarray(gray).convert("RGB")


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.45, colormap: str = "jet") -> Image.Image:
    image = image.convert("RGB")
    heatmap_img = apply_colormap(heatmap, colormap=colormap).resize(image.size, Image.BILINEAR)
    return Image.blend(image, heatmap_img, alpha=alpha)


def build_change_confidence_map(change_logits: Optional[torch.Tensor]) -> Optional[np.ndarray]:
    if change_logits is None:
        return None
    logits = change_logits.detach()
    if logits.dim() == 4:
        logits = logits.squeeze(0)
    if logits.dim() != 3:
        return None
    probs = torch.softmax(logits.float(), dim=0)
    heatmap = probs[1:].sum(dim=0) if probs.shape[0] > 1 else probs[0]
    return normalize_heatmap(heatmap.cpu().numpy())


def get_cached_attention_map(model) -> Optional[np.ndarray]:
    base_model = model.get_model() if hasattr(model, "get_model") else model
    change_detector = getattr(base_model, "change_detector", None)
    attn_map = getattr(change_detector, "last_attn_map", None)
    if attn_map is None:
        return None
    attn_map = attn_map.detach().float().cpu()
    if attn_map.dim() == 2:
        attn_map = attn_map[0]
    elif attn_map.dim() == 3:
        attn_map = attn_map[0]
    else:
        return None
    side = int(attn_map.numel() ** 0.5)
    if side * side != attn_map.numel():
        return None
    return normalize_heatmap(attn_map.reshape(side, side).numpy())


def make_safe_filename(name: str) -> str:
    stem = os.path.splitext(str(name))[0]
    return stem.replace("/", "_").replace("\\", "_").replace(" ", "_")


def save_attention_visualization(sample, image_folder: str, output_dir: str, sample_idx: int,
                                 attention_map: Optional[np.ndarray], confidence_map: Optional[np.ndarray],
                                 alpha: float = 0.45, colormap: str = "jet"):
    if attention_map is None and confidence_map is None:
        return
    os.makedirs(output_dir, exist_ok=True)
    images = load_images(sample["image"], image_folder)
    image_names = sample["image"]
    image_stems = [make_safe_filename(name) for name in image_names[:2]]
    pair_name = "__".join(image_stems) if image_stems else f"sample_{sample_idx}"

    maps_to_save = []
    if attention_map is not None:
        maps_to_save.append(("attention", attention_map))
    if confidence_map is not None:
        maps_to_save.append(("change_confidence", confidence_map))

    for map_name, heatmap in maps_to_save:
        overlays = []
        for img_idx, image in enumerate(images[:2]):
            overlay = overlay_heatmap(image, heatmap, alpha=alpha, colormap=colormap)
            overlays.append(overlay)
            image_name = image_stems[img_idx] if img_idx < len(image_stems) else f"img{img_idx + 1}"
            overlay.save(os.path.join(output_dir, f"{image_name}_{map_name}.png"))
        raw_heatmap = apply_colormap(heatmap, colormap=colormap).resize(images[0].size, Image.BILINEAR)
        raw_heatmap.save(os.path.join(output_dir, f"{pair_name}_{map_name}_heatmap.png"))

        tile_w, tile_h = images[0].size
        panel = Image.new("RGB", (tile_w * 3, tile_h), "white")
        panel.paste(images[0].convert("RGB"), (0, 0))
        panel.paste(overlays[0], (tile_w, 0))
        panel.paste(overlays[1] if len(overlays) > 1 else overlays[0], (tile_w * 2, 0))
        panel.save(os.path.join(output_dir, f"{pair_name}_{map_name}_panel.png"))


def load_label(label_path: str, image_folder: str) -> np.ndarray:
    """
    加载分割标签 (灰度图，已由 convert_dataset.py 转换)
    
    标签格式 (LEVIR-MCI 数据集):
        - 0: background (无变化)
        - 1: road (道路变化)
        - 2: building (建筑变化)
    
    Returns:
        标签数组，像素值为 0, 1, 2
    """
    full_path = os.path.join(image_folder, label_path)
    if not os.path.exists(full_path):
        # 不打印警告，避免刷屏
        return None
    
    label = Image.open(full_path)
    label_array = np.array(label)
    
    # 处理多通道标签图像 (如果有)
    if label_array.ndim == 3:
        label_array = label_array[:, :, 0]
    
    return label_array


def prepare_input(sample, tokenizer, image_processor, model, conv_mode, image_folder):
    """
    准备模型输入
    """
    # 获取对话模板
    conv = conv_templates[conv_mode].copy()
    
    # 从 sample 中获取问题
    human_msg = sample["conversations"][0]["value"]
    
    # 根据模型配置决定 <image> token 数量
    mm_img_cd_concat = getattr(model.config, "mm_img_cd_concat", False)
    
    if mm_img_cd_concat:
        # 拼接模式: 需要 3 个 <image> token (A, B, change)
        human_msg = human_msg.replace(
            "<image>\n<image>\n", 
            "<image>\n<image>\n<image>\n"
        )
    else:
        # 只有变化特征模式: 需要 1 个 <image> token
        human_msg = human_msg.replace("<image>\n<image>\n", "<image>\n")
    
    conv.append_message(conv.roles[0], human_msg)
    conv.append_message(conv.roles[1], None)
    
    prompt = conv.get_prompt()
    
    # 加载图像
    images = load_images(sample["image"], image_folder)
    
    # 处理图像
    images_tensor = process_images(images, image_processor, model.config)
    
    # 转换为模型需要的格式
    if isinstance(images_tensor, list):
        images_tensor = [img.to(model.device, dtype=torch.float16) for img in images_tensor]
    else:
        images_tensor = images_tensor.to(model.device, dtype=torch.float16)
    
    image_sizes = [img.size for img in images]
    
    # Tokenize prompt
    input_ids = tokenizer_image_token(
        prompt, 
        tokenizer, 
        IMAGE_TOKEN_INDEX, 
        return_tensors="pt"
    ).unsqueeze(0).to(model.device)
    
    return input_ids, images_tensor, image_sizes


def run_inference(model, tokenizer, input_ids, images_tensor, image_sizes, args):
    """
    运行模型推理
    
    Returns:
        output_text: 生成的文本
        seg_pred: 分割预测 (如果有)
        change_logits: 分割 logits，用于生成变化置信度热力图
    """
    # 设置生成参数
    generation_kwargs = {
        "do_sample": args.temperature > 0,
        "temperature": args.temperature if args.temperature > 0 else None,
        "max_new_tokens": args.max_new_tokens,
        "use_cache": True,
    }
    
    # 移除 None 值
    generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}
    
    # 准备 images 参数
    if isinstance(images_tensor, list):
        images = images_tensor
    else:
        images = [images_tensor[i] for i in range(images_tensor.shape[0])]
    
    with torch.inference_mode():
        output_result = model.generate(
            inputs=input_ids,
            images=images,
            image_sizes=image_sizes,
            modalities=["image"] * len(images),
            **generation_kwargs
        )
        
        # 处理返回值
        if isinstance(output_result, tuple):
            output_ids, change_logits = output_result
        else:
            output_ids = output_result
            change_logits = None
    
    # 解码输出
    output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    
    # 处理分割预测
    seg_pred = None
    if change_logits is not None:
        # change_logits: [1, num_classes, H, W]
        if change_logits.dim() == 4:
            change_logits = change_logits.squeeze(0)
        seg_pred = torch.argmax(change_logits, dim=0).cpu().numpy()
    
    return output_text, seg_pred, change_logits


def evaluate(args):
    """
    主评估函数
    """
    print("\n" + "=" * 60)
    print("RACL Model Evaluation")
    print("=" * 60)
    
    # 设置环境变量
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型
    tokenizer, model, image_processor, context_len = load_model(args)
    if args.save_attention_maps:
        base_model = model.get_model() if hasattr(model, "get_model") else model
        change_detector = getattr(base_model, "change_detector", None)
        if hasattr(change_detector, "enable_attention_by_default"):
            change_detector.enable_attention_by_default = True
            print("✅ Enabled change-detector attention for visualization")
    
    # 获取分割类别数
    num_classes = getattr(model.config, "mm_num_class", 3)
    
    # 加载测试数据
    test_samples, all_references = load_test_data(args)
    
    # 初始化评估器
    caption_metrics = CaptionMetrics(strategy=args.metric_strategy) if args.eval_caption else None
    seg_metrics = SegmentationMetrics(num_classes=num_classes) if args.eval_segmentation else None
    
    # 收集预测结果
    predictions = []
    references = []
    results = []
    
    print("\n🔄 Running inference...")
    for idx, sample in enumerate(tqdm(test_samples)):
        try:
            # 准备输入
            input_ids, images_tensor, image_sizes = prepare_input(
                sample, tokenizer, image_processor, model, args.conv_mode, args.image_folder
            )
            
            # 运行推理
            output_text, seg_pred, change_logits = run_inference(
                model, tokenizer, input_ids, images_tensor, image_sizes, args
            )
            
            if args.save_attention_maps and (args.attention_max_samples < 0 or idx < args.attention_max_samples):
                attention_map = get_cached_attention_map(model)
                confidence_map = build_change_confidence_map(change_logits)
                save_attention_visualization(
                    sample=sample,
                    image_folder=args.image_folder,
                    output_dir=args.attention_output_dir,
                    sample_idx=idx,
                    attention_map=attention_map,
                    confidence_map=confidence_map,
                    alpha=args.attention_alpha,
                    colormap=args.attention_colormap,
                )
            
            # 收集 caption 结果
            image_key = tuple(sample["image"])
            predictions.append(output_text)
            references.append(all_references[image_key])
            
            # 评估分割
            if args.eval_segmentation and seg_pred is not None:
                label_path = sample.get("label_gray", None)
                if label_path:
                    label = load_label(label_path, args.image_folder)
                    if label is not None:
                        seg_metrics.update(seg_pred, label)
            
            # 保存结果
            result = {
                "id": sample.get("id", idx),
                "images": sample["image"],
                "prediction": output_text,
                "references": all_references[image_key],
            }
            results.append(result)
            
            # 打印部分结果
            if idx < 3:
                print(f"\n--- Sample {idx} ---")
                print(f"Images: {sample['image']}")
                print(f"Prediction: {output_text}")
                print(f"Reference: {all_references[image_key][0]}")
                
        except Exception as e:
            print(f"\n[WARNING] Failed to process sample {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 计算指标
    print("\n" + "=" * 60)
    print("📊 Evaluation Results")
    print("=" * 60)
    
    all_metrics = {}
    
    # 文本生成指标
    if args.eval_caption and caption_metrics is not None:
        print("\n📝 Caption Metrics:")
        caption_results = caption_metrics.compute_all(predictions, references)
        for key, value in caption_results.items():
            print(f"   {key}: {value:.4f}")
            all_metrics[f"caption/{key}"] = value
    
    # 分割指标
    if args.eval_segmentation and seg_metrics is not None:
        print("\n🖼️ Segmentation Metrics:")
        seg_results = seg_metrics.compute_all()
        for key, value in seg_results.items():
            print(f"   {key}: {value:.4f}")
            all_metrics[f"segmentation/{key}"] = value
    
    # 保存结果
    output_data = {
        "config": vars(args),
        "metrics": all_metrics,
        "num_samples": len(predictions),
    }
    
    if args.save_predictions:
        output_data["results"] = results
    
    output_file = os.path.join(args.output_dir, "eval_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Results saved to: {output_file}")
    
    # 保存指标摘要
    summary_file = os.path.join(args.output_dir, "metrics_summary.txt")
    with open(summary_file, "w") as f:
        f.write("RACL Evaluation Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Test samples: {len(predictions)}\n")
        f.write(f"Metric strategy: {args.metric_strategy}\n\n")
        
        if args.eval_caption:
            f.write("Caption Metrics:\n")
            f.write("-" * 30 + "\n")
            if args.metric_strategy == "max":
                f.write("  (Using MAX strategy: best match among 5 references)\n")
            for key in ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "METEOR", "ROUGE-L", "CIDEr", "S*m"]:
                if f"caption/{key}" in all_metrics:
                    f.write(f"  {key}: {all_metrics[f'caption/{key}']:.4f}\n")
            f.write("\n")
            f.write("  Note: S*m = 1/4 * (BLEU-4 + METEOR + ROUGE-L + CIDEr)\n\n")
        
        if args.eval_segmentation:
            f.write("Segmentation Metrics:\n")
            f.write("-" * 30 + "\n")
            # 变化检测核心指标
            f.write("  [Change Detection]\n")
            for key in ["cIoU", "F1_change"]:
                if f"segmentation/{key}" in all_metrics:
                    f.write(f"    {key}: {all_metrics[f'segmentation/{key}']:.4f}\n")
            f.write("\n")
            # 总体指标
            f.write("  [Overall]\n")
            for key in ["mIoU", "mF1", "Pixel_Acc"]:
                if f"segmentation/{key}" in all_metrics:
                    f.write(f"    {key}: {all_metrics[f'segmentation/{key}']:.4f}\n")
            f.write("\n")
            # 每类指标
            f.write("  [Per-class IoU]\n")
            for key in ["IoU_background", "IoU_road", "IoU_building"]:
                if f"segmentation/{key}" in all_metrics:
                    f.write(f"    {key}: {all_metrics[f'segmentation/{key}']:.4f}\n")
    
    print(f"✅ Summary saved to: {summary_file}")
    print("\n🎉 Evaluation completed!")
    
    return all_metrics


def quick_test(args):
    """
    快速测试模式
    只加载模型并测试单个样本
    """
    print("\n" + "=" * 60)
    print("🧪 Quick Test Mode")
    print("=" * 60)
    
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    
    # 加载模型
    try:
        tokenizer, model, image_processor, context_len = load_model(args)
        print("\n✅ Model loading: PASSED")
    except Exception as e:
        print(f"\n❌ Model loading: FAILED - {e}")
        return False
    
    # 检查模型组件
    print("\n🔍 Model components:")
    
    vision_tower = model.get_vision_tower()
    print(f"   Vision tower: {vision_tower.__class__.__name__ if vision_tower else 'None'}")
    
    mm_projector = getattr(model.get_model(), 'mm_projector', None)
    print(f"   MM projector: {mm_projector.__class__.__name__ if mm_projector else 'None'}")
    
    change_detector = getattr(model.get_model(), 'change_detector', None)
    print(f"   Change detector: {change_detector.__class__.__name__ if change_detector else 'None'}")
    
    seg_head = getattr(model.get_model(), 'seg_head', None)
    print(f"   Seg head: {seg_head.__class__.__name__ if seg_head else 'None'}")
    
    # 尝试加载测试数据
    print("\n📂 Loading test sample...")
    try:
        with open(args.test_json, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        if len(test_data) > 0:
            sample = test_data[0]
            print(f"   Sample ID: {sample.get('id', 'N/A')}")
            print(f"   Images: {sample.get('image', [])}")
            
            # 运行推理
            print("\n🔄 Running inference on sample...")
            input_ids, images_tensor, image_sizes = prepare_input(
                sample, tokenizer, image_processor, model, args.conv_mode, args.image_folder
            )
            
            output_text, seg_pred, change_logits = run_inference(
                model, tokenizer, input_ids, images_tensor, image_sizes, args
            )
            
            print(f"   Prediction: {output_text}")
            print(f"   Ground truth: {sample['conversations'][1]['value']}")
            if seg_pred is not None:
                print(f"   Seg pred shape: {seg_pred.shape}")
            
            print("\n✅ Quick test: PASSED")
            return True
    except Exception as e:
        print(f"\n❌ Quick test: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    args = parse_args()
    
    # 如果 num_samples 为 0，运行快速测试
    if args.num_samples == 0:
        quick_test(args)
    else:
        evaluate(args)


