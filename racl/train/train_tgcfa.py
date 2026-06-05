# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

"""
train_tgcfa.py -- 在 train.py 基础上集成 TGCFA 文本引导变化特征对齐

主要改动:
  1. ModelArguments 新增 6 个 TGCFA 参数
  2. safe_save_model_for_hf_trainer 将 tgcfa 纳入保存 key
  3. train() 中 mm_tunable_parts 解析新增 mm_tgcfa 分支
  4. train() 中构建 CLIP 文本特征（text_feat）并注入 LLaVATrainer

待确认事项（标注 TODO）:
  [TODO-A] CLIP 文本编码器路径：当前从 vision_tower 路径推断，请确认是否正确
  [TODO-B] text_feat 注入方式：当前通过 training_args.tgcfa_text_feat_fn 传递一个
           回调函数，LLaVATrainer 需要在 compute_loss 中调用该回调获取 text_feat
           并传入 prepare_inputs_labels_for_multimodal。如果 LLaVATrainer 不支持
           此接口，需要同步修改 llava_trainer.py
  [TODO-C] CLIP 模型加载：当前使用 transformers CLIPTextModel，如果项目使用的是
           自定义 EVA-CLIP，需要替换加载方式
  [TODO-D] text_feat 的 batch 对齐：当前假设每个图像对只有一条 caption，
           如果一张图有多条 caption，需要调整对齐逻辑
"""

import ast
import os
from dataclasses import dataclass, field
import logging
import pathlib
from typing import Dict, Optional, Callable
from PIL import ImageFile
import re
import torch
import torch.nn as nn
import transformers
import deepspeed
from transformers import AutoConfig
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import make_supervised_data_module
from llava_trainer_tgcfa import LLaVATrainerTGCFA as LLaVATrainer
import conversation as conversation_lib
from model import *
from utils import rank0_print

torch.multiprocessing.set_sharing_strategy("file_system")

ImageFile.LOAD_TRUNCATED_IMAGES = True
local_rank = None


# ---------------------------------------------------------------------------
# ModelArguments  （新增 TGCFA 参数，其余与原版完全相同）
# ---------------------------------------------------------------------------

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    model_class_name: Optional[str] = field(default=None, metadata={"help": "Used to init model class, format is XXXXForCausalLM."})

    mm_tunable_parts: Optional[str] = field(
        default=None,
        metadata={"help": 'e.g. "mm_mlp_adapter,mm_change_detector,mm_seg_head,mm_tgcfa"'}
    )

    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    tune_mm_change_detector: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    vision_tower_pretrained: Optional[str] = field(default=None)

    unfreeze_mm_vision_tower: bool = field(default=False)
    unfreeze_language_model: bool = field(default=False)
    mm_vision_select_layer: Optional[int] = field(default=-1)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default="linear")
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default="flat")
    mm_vision_select_feature: Optional[str] = field(default="patch, slicefour_patch, slice_m25811_f6_patch")

    mm_change_detector_type: Optional[str] = field(default="dafm")
    mm_fusion_policy: Optional[str] = field(default=None)
    mm_img_cd_concat: Optional[bool] = field(default=False)
    mm_seg_head_type: Optional[str] = field(default="conv")
    mm_num_class: Optional[int] = field(default=2)
    proc_crop_size: Optional[int] = field(default=336)

    # KCPM_Attention parameters（与原版相同）
    num_attention_layers: Optional[int] = field(default=3)
    attention_num_heads: Optional[int] = field(default=4)
    attention_window_size: Optional[int] = field(default=7)
    use_gate: Optional[bool] = field(default=True)
    use_attention: Optional[bool] = field(default=False)

    # Loss weight parameters（与原版相同）
    detection_loss_weight: Optional[float] = field(default=1.0)
    caption_loss_weight: Optional[float] = field(default=1.0)

    s2: Optional[bool] = field(default=False)
    s2_scales: Optional[str] = field(default="336,672,1008")
    use_pos_skipping: Optional[bool] = field(default=False)
    pos_skipping_range: Optional[int] = field(default=4096)
    mm_newline_position: Optional[str] = field(default="one_token")

    # ── TGCFA 参数（新增）──────────────────────────────────────────────
    use_tgcfa: bool = field(
        default=False,
        metadata={"help": "Whether to enable Text-Guided Change Feature Alignment module."}
    )
    tgcfa_text_dim: int = field(
        default=768,
        metadata={"help": "CLIP text encoder output dim. CLIP ViT-L/14=768, ViT-B/32=512."}
    )
    tgcfa_temperature: float = field(
        default=0.07,
        metadata={"help": "InfoNCE temperature for TGCFA auxiliary alignment loss."}
    )
    tgcfa_align_weight: float = field(
        default=0.2,
        metadata={"help": "Weight of TGCFA auxiliary alignment loss."}
    )
    tgcfa_clip_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a full CLIP model (must contain both text encoder and tokenizer). "
                          "e.g. /root/autodl-tmp/openai/clip-vit-large-patch14. "
                          "If None, falls back to openai/clip-vit-large-patch14 (requires internet or local cache). "
                          "NOTE: vision_tower path is vision-only and cannot be used here."}
    )
    # ──────────────────────────────────────────────────────────────────


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    early_mix_text: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: Optional[str] = field(default="")
    image_grid_pinpoints: Optional[str] = field(default=None)
    image_crop_resolution: Optional[int] = field(default=None)
    image_split_resolution: Optional[int] = field(default=None)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    freeze_mm_change_detector: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(default=4096)
    double_quant: bool = field(default=True)
    quant_type: str = field(default="nf4")
    bits: int = field(default=16)
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    mm_vision_tower_lr: Optional[float] = None
    group_by_varlen: bool = field(default=False)
    group_by_modality_length: bool = field(default=False)
    group_by_modality_length_auto: bool = field(default=False)
    auto_find_batch_size: bool = field(default=False)
    gradient_checkpointing: bool = field(default=True)
    verbose_logging: bool = field(default=False)
    attn_implementation: str = field(default="sdpa")
    ignore_data_skip: bool = field(default=False)
    use_dwa: bool = field(default=True)
    dwa_temperature: float = field(default=2.0)
    use_ngram_reward: bool = field(default=False)
    ngram_ce_weight: float = field(default=0.7)
    ngram_reward_weight: float = field(default=0.2)
    ngram_diversity_weight: float = field(default=0.1)
    ngram_adaptive_weights: bool = field(default=True)


# ---------------------------------------------------------------------------
# 工具函数（与原版完全相同）
# ---------------------------------------------------------------------------

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if k in lora_bias_names:
                to_return[k] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ["mm_projector", "vision_tower", "change_detector"]
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if "lm_head" in lora_module_names:
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk. TGCFA 权重纳入 adapter 保存."""
    if hasattr(trainer.args, "tune_mm_mlp_adapter") and trainer.args.tune_mm_mlp_adapter:
        check_only_save_mm_adapter_tunnable = True
    elif hasattr(trainer.args, "mm_tunable_parts") and (trainer.args.mm_tunable_parts) and (
        any(p in trainer.args.mm_tunable_parts for p in
            ["mm_mlp_adapter","mm_change_detector","mm_seg_head","mm_tgcfa"])
        and "mm_language_model" not in trainer.args.mm_tunable_parts
    ):
        check_only_save_mm_adapter_tunnable = True
    else:
        check_only_save_mm_adapter_tunnable = False

    trainer.accelerator.wait_for_everyone()
    torch.cuda.synchronize()
    rank0_print(f"Only save projectors: {check_only_save_mm_adapter_tunnable}")
    if check_only_save_mm_adapter_tunnable:
        keys_to_match = ["vision_tower","mm_projector","change_detector","seg_head","tgcfa"]
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(["embed_tokens","embed_in"])
        weight_to_save = get_mm_adapter_state_maybe_zero_3(
            trainer.model.named_parameters(), keys_to_match
        )
        trainer.model.config.save_pretrained(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            torch.save(weight_to_save, os.path.join(output_dir, "mm_projector.bin"))
        return
    if trainer.deepspeed:
        trainer.save_model(output_dir)
        return
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {k: v.cpu() for k, v in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


def smart_tokenizer_and_embedding_resize(special_tokens_dict, tokenizer, model):
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    if num_new_tokens > 0:
        ie = model.get_input_embeddings().weight.data
        oe = model.get_output_embeddings().weight.data
        ie[-num_new_tokens:] = ie[:-num_new_tokens].mean(dim=0, keepdim=True)
        oe[-num_new_tokens:] = oe[:-num_new_tokens].mean(dim=0, keepdim=True)


def get_model(model_args, training_args, bnb_model_from_pretrained_args):
    assert training_args.attn_implementation
    if training_args.attn_implementation == "sdpa" and torch.__version__ < "2.1.2":
        raise ValueError("sdpa requires torch >= 2.1.2")
    customized_kwargs = dict()
    customized_kwargs.update(bnb_model_from_pretrained_args)
    if model_args.model_class_name is not None:
        actual_model_class_name = f"{model_args.model_class_name}ForCausalLM"
        model_class = getattr(transformers, actual_model_class_name)
        model = model_class.from_pretrained(
            model_args.model_name_or_path, cache_dir=training_args.cache_dir,
            attn_implementation=training_args.attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            low_cpu_mem_usage=False, **customized_kwargs,
        )
    elif model_args.vision_tower is not None:
        if "qwen" in model_args.model_name_or_path.lower():
            if "moe" in model_args.model_name_or_path.lower() or "A14B" in model_args.model_name_or_path:
                model = LlavaQwenMoeForCausalLM.from_pretrained(
                    model_args.model_name_or_path, cache_dir=training_args.cache_dir,
                    attn_implementation=training_args.attn_implementation,
                    torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                    low_cpu_mem_usage=False, **customized_kwargs,
                )
                from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeSparseMoeBlock
                deepspeed.utils.set_z3_leaf_modules(model, [Qwen2MoeSparseMoeBlock])
            else:
                model = LlavaQwenForCausalLM.from_pretrained(
                    model_args.model_name_or_path, cache_dir=training_args.cache_dir,
                    attn_implementation=training_args.attn_implementation,
                    torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                    low_cpu_mem_usage=False, **customized_kwargs,
                )
        else:
            raise ValueError(f"Unknown model class {model_args}")
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path, cache_dir=training_args.cache_dir,
            attn_implementation=training_args.attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            low_cpu_mem_usage=False, **customized_kwargs,
        )
    return model


class TGCFATextEncoder:
    """
    封装 CLIP 文本编码器，提供 encode(captions) -> text_feat 接口。

    加载方式:
      - 使用 transformers CLIPModel 从完整 CLIP checkpoint 中单独提取文本编码器
      - clip_model_path 必须是包含完整 CLIP 权重的目录（视觉+文本），
        例如 openai/clip-vit-large-patch14 或其本地镜像路径
      - vision_tower 路径（CLIP_RegionAware_merged）仅含视觉权重，不可用于此处
    """
    def __init__(self, clip_model_path: str, device: str, dtype):
        from transformers import CLIPTokenizer, CLIPModel
        rank0_print(f"[TGCFA] Loading CLIP text encoder from: {clip_model_path}")
        try:
            full_clip       = CLIPModel.from_pretrained(clip_model_path)
            self.text_model = full_clip.text_model      # CLIPTextTransformer
            self.tokenizer  = CLIPTokenizer.from_pretrained(clip_model_path)
            del full_clip   # 释放视觉部分，节省显存
        except Exception as e:
            rank0_print(f"[TGCFA] ERROR: Failed to load CLIP text encoder from {clip_model_path}")
            rank0_print("[TGCFA] clip_model_path must be a FULL CLIP checkpoint, not vision-only.")
            rank0_print("[TGCFA] Recommended: openai/clip-vit-large-patch14 or its local mirror.")
            raise RuntimeError(f"TGCFA text encoder load failed: {e}") from e
        self.text_model = self.text_model.to(device=device, dtype=dtype)
        self.text_model.eval()
        for p in self.text_model.parameters():
            p.requires_grad = False
        self.device = device
        self.dtype  = dtype
        rank0_print(f"[TGCFA] Text encoder loaded and frozen. "
                    f"hidden_size={self.text_model.config.hidden_size}")

    @torch.no_grad()
    def encode(self, captions: list, max_length: int = 77):
        """
        Args:
            captions  : List[str], len = batch_size // 2
        Returns:
            text_feat : [B//2, L, C_t]  C_t = hidden_size (ViT-L/14 = 768)
        注意: 返回 last_hidden_state（token 级），供 TGCFA 交叉注意力使用。
        """
        inputs = self.tokenizer(
            captions, padding="max_length", truncation=True,
            max_length=max_length, return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out = self.text_model(**inputs)
        return out.last_hidden_state.to(self.dtype)

def train():
    global local_rank
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if training_args.verbose_logging:
        rank0_print(f"model_args = {vars(model_args)}")
    local_rank    = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16
                     else (torch.bfloat16 if training_args.bf16 else torch.float32))
    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector", "change_detector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type,
            ),
        ))
    model = get_model(model_args, training_args, bnb_model_from_pretrained_args)
    model.config.use_cache = False
    if model_args.freeze_backbone:
        model.model.requires_grad_(False)
    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype = (torch.float32 if training_args.fp16
                                    else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r, lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias, task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16: model.to(torch.bfloat16)
            if training_args.fp16: model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)
    if "qwen" in model_args.model_name_or_path.lower():
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length, padding_side="right"
        )
    rank0_print(f"Prompt version: {model_args.version}")
    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(dict(pad_token="[PAD]"), tokenizer, model)
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    if model_args.vision_tower is not None:
        import re, ast as _ast
        model.config.proc_crop_size = model_args.proc_crop_size
        model.get_model().initialize_vision_modules(model_args=model_args, fsdp=training_args.fsdp)
        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)
        data_args.image_processor    = vision_tower.image_processor
        data_args.is_multimodal      = True
        model.config.image_aspect_ratio       = data_args.image_aspect_ratio
        model.config.image_crop_resolution    = data_args.image_crop_resolution
        model.config.image_split_resolution   = data_args.image_split_resolution
        model.config.tokenizer_padding_side    = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length
        model.config.mm_newline_position       = model_args.mm_newline_position
        if data_args.image_grid_pinpoints is not None:
            if isinstance(data_args.image_grid_pinpoints, str) and "x" in data_args.image_grid_pinpoints:
                try:
                    patch_size = data_args.image_processor.size[0]
                except Exception:
                    patch_size = data_args.image_processor.size["shortest_edge"]
                assert patch_size in [224,336,384,448,512]
                matches     = re.findall(r"\((\d+)x(\d+)\)", data_args.image_grid_pinpoints)
                range_start = tuple(map(int, matches[0]))
                range_end   = tuple(map(int, matches[-1]))
                grid_pinpoints = [(i,j) for i in range(range_start[0], range_end[0]+1)
                                        for j in range(range_start[1], range_end[1]+1)]
                data_args.image_grid_pinpoints = [[d*patch_size for d in p] for p in grid_pinpoints]
            elif isinstance(data_args.image_grid_pinpoints, str):
                data_args.image_grid_pinpoints = _ast.literal_eval(data_args.image_grid_pinpoints)
        model.config.image_grid_pinpoints = data_args.image_grid_pinpoints
        # ------ 可训练部分 ------
        if model_args.mm_tunable_parts is None:
            model.config.tune_mm_mlp_adapter     = training_args.tune_mm_mlp_adapter     = model_args.tune_mm_mlp_adapter
            model.config.tune_mm_change_detector = training_args.tune_mm_change_detector = model_args.tune_mm_change_detector
            if model_args.tune_mm_mlp_adapter or model_args.tune_mm_change_detector:
                model.requires_grad_(False)
            if model_args.tune_mm_mlp_adapter:
                for p in model.get_model().mm_projector.parameters(): p.requires_grad = True
            if model_args.tune_mm_change_detector:
                for p in model.get_model().change_detector.parameters(): p.requires_grad = True
            model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
            if training_args.freeze_mm_mlp_adapter:
                for p in model.get_model().mm_projector.parameters(): p.requires_grad = False
            model.config.freeze_mm_change_detector = training_args.freeze_mm_change_detector
            if training_args.freeze_mm_change_detector:
                for p in model.get_model().change_detector.parameters(): p.requires_grad = False
            model.config.unfreeze_mm_vision_tower = model_args.unfreeze_mm_vision_tower
            vision_tower.requires_grad_(model_args.unfreeze_mm_vision_tower)
        else:
            rank0_print(f"Using mm_tunable_parts: {model_args.mm_tunable_parts}")
            model.config.mm_tunable_parts = training_args.mm_tunable_parts = model_args.mm_tunable_parts
            model.requires_grad_(False)
            vision_tower.requires_grad_(False)
            model.get_model().mm_projector.requires_grad_(False)
            model.get_model().change_detector.requires_grad_(False)
            model.get_model().seg_head.requires_grad_(False)
            if hasattr(model.get_model(), "tgcfa") and model.get_model().tgcfa is not None:
                model.get_model().tgcfa.requires_grad_(False)
            tunable_parts = model_args.mm_tunable_parts.split(",")
            if "mm_mlp_adapter"     in tunable_parts:
                for p in model.get_model().mm_projector.parameters():   p.requires_grad = True
            if "mm_change_detector" in tunable_parts:
                for p in model.get_model().change_detector.parameters(): p.requires_grad = True
            if "mm_seg_head"        in tunable_parts:
                for p in model.get_model().seg_head.parameters():        p.requires_grad = True
            # TGCFA 解冻（新增）
            if "mm_tgcfa" in tunable_parts:
                tgcfa_mod = getattr(model.get_model(), "tgcfa", None)
                if tgcfa_mod is not None:
                    for p in tgcfa_mod.parameters(): p.requires_grad = True
                    rank0_print("[TGCFA] tgcfa parameters unfrozen.")
                else:
                    rank0_print("[TGCFA] WARNING: mm_tgcfa in tunable_parts but model.tgcfa is None.")
            if "mm_vision_tower" in tunable_parts:
                for n, p in model.named_parameters():
                    if "vision_tower" in n: p.requires_grad_(True)
            if "mm_language_model" in tunable_parts:
                for n, p in model.named_parameters():
                    if "vision_tower" not in n and "mm_projector" not in n and "change_detector" not in n:
                        p.requires_grad_(True)
        total_p    = sum(p.ds_numel if hasattr(p,"ds_numel") else p.numel() for p in model.parameters())
        trainable_p = sum(p.ds_numel if hasattr(p,"ds_numel") else p.numel() for p in model.parameters() if p.requires_grad)
        rank0_print(f"Total: ~{total_p/1e6:.2f} M   Trainable: ~{trainable_p/1e6:.2f} M")
        if training_args.bits in [4,8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)
        model.config.mm_use_im_start_end   = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr       = training_args.mm_projector_lr
        model.config.mm_vision_tower_lr    = training_args.mm_vision_tower_lr
        training_args.use_im_start_end     = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)
    if training_args.bits in [4,8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer) and training_args.bf16:
                module = module.to(torch.bfloat16)
            if "norm" in name: module = module.to(torch.float32)
            if ("lm_head" in name or "embed_tokens" in name) and hasattr(module,"weight"):
                if training_args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
    # TGCFA: 构建文本编码器并注入 trainer
    # [TODO-B] 通过 training_args.tgcfa_text_feat_fn 传递回调给 LLaVATrainer。
    # LLaVATrainer 需在 compute_loss 中调用此回调获取 text_feat，
    # 并将其传入 prepare_inputs_labels_for_multimodal。
    # 若 LLaVATrainer 不支持，需同步修改 llava_trainer.py。
    training_args.tgcfa_text_feat_fn = None
    if model_args.use_tgcfa:
        tgcfa_mod = getattr(model.get_model(), 'tgcfa', None)
        if tgcfa_mod is None:
            rank0_print('[TGCFA] WARNING: use_tgcfa=True but model.tgcfa is None.')
        else:
            # clip_path 必须指向完整 CLIP checkpoint（含文本编码器）
            # vision_tower 路径仅含视觉权重，不可直接用于文本编码器加载
            # 优先使用 --tgcfa_clip_path 参数；未指定时回退到 openai/clip-vit-large-patch14
            clip_path = getattr(model_args, 'tgcfa_clip_path', None) or 'openai/clip-vit-large-patch14'
            rank0_print(f'[TGCFA] Loading text encoder from: {clip_path}')
            try:
                _te = TGCFATextEncoder(clip_path, training_args.device, compute_dtype)
                def _fn(captions):
                    # [TODO-D] captions: List[str], len = batch_size // 2
                    return _te.encode(captions)
                training_args.tgcfa_text_feat_fn = _fn
                rank0_print('[TGCFA] text_feat_fn registered.')
            except Exception as e:
                rank0_print(f'[TGCFA] WARNING: text encoder load failed: {e}. Falling back to no-text path.')

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    training_args.detection_loss_weight  = model_args.detection_loss_weight
    training_args.caption_loss_weight    = model_args.caption_loss_weight
    training_args.use_dwa                = getattr(training_args, 'use_dwa', False)
    training_args.dwa_temperature        = getattr(training_args, 'dwa_temperature', 2.0)
    training_args.use_ngram_reward       = getattr(training_args, 'use_ngram_reward', False)
    training_args.ngram_ce_weight        = getattr(training_args, 'ngram_ce_weight', 0.7)
    training_args.ngram_reward_weight    = getattr(training_args, 'ngram_reward_weight', 0.2)
    training_args.ngram_diversity_weight = getattr(training_args, 'ngram_diversity_weight', 0.1)
    training_args.ngram_adaptive_weights = getattr(training_args, 'ngram_adaptive_weights', True)
    trainer = LLaVATrainer(model=model, tokenizer=tokenizer, args=training_args, task_num=2, **data_module)

    # 断点续训（与原版相同）
    checkpoint_list = list(pathlib.Path(training_args.output_dir).glob('checkpoint-*'))
    resume_from_checkpoint = None
    if checkpoint_list:
        latest_checkpoint = max(checkpoint_list, key=lambda x: int(x.name.split('-')[-1]))
        rank0_print(f'Found checkpoint: {latest_checkpoint}')
        mm_projector_path = latest_checkpoint / 'mm_projector.bin'
        model_safetensors = latest_checkpoint / 'model.safetensors'
        pytorch_model     = latest_checkpoint / 'pytorch_model.bin'
        trainer_state_path = latest_checkpoint / 'trainer_state.json'
        is_adapter_only = (mm_projector_path.exists()
                           and not model_safetensors.exists()
                           and not pytorch_model.exists())
        if is_adapter_only:
            rank0_print('Adapter-only checkpoint - manual resume')
            aw = torch.load(mm_projector_path, map_location='cpu')
            missing, _ = model.load_state_dict(aw, strict=False)
            rank0_print(f'  Loaded adapter weights (missing: {len(missing)})')
            deepspeed_dir = latest_checkpoint / 'deepspeed'
            if deepspeed_dir.exists():
                lf = deepspeed_dir / 'latest'
                if lf.exists():
                    step_dir = open(lf).read().strip()
                    op = deepspeed_dir / step_dir / 'bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt'
                    if op.exists():
                        try:
                            os_dict = torch.load(op, map_location='cpu', weights_only=False)
                            trainer._pending_optimizer_state = os_dict.get('optimizer_state_dict')
                            rank0_print('  Optimizer state pending.')
                        except Exception as e:
                            rank0_print(f'  WARNING optimizer: {e}')
            if trainer_state_path.exists():
                import json
                ts = json.load(open(trainer_state_path))
                trainer.state.global_step = ts.get('global_step', 0)
                trainer.state.epoch       = ts.get('epoch', 0)
                trainer.state.total_flos  = ts.get('total_flos', 0)
                rank0_print(f'  Step={trainer.state.global_step} epoch={trainer.state.epoch:.2f}')
            rng_p = latest_checkpoint / 'rng_state.pth'
            if rng_p.exists():
                try:
                    rng = torch.load(rng_p, map_location='cpu', weights_only=False)
                    if 'python' in rng and isinstance(rng['python'], torch.Tensor): torch.random.set_rng_state(rng['python'].cpu())
                    if 'cpu' in rng: torch.set_rng_state(rng['cpu'].cpu())
                    if 'cuda' in rng and torch.cuda.is_available(): torch.cuda.set_rng_state_all(rng['cuda'])
                    if 'numpy' in rng: import numpy as np; np.random.set_state(rng['numpy'])
                    rank0_print('  RNG states restored.')
                except Exception as e:
                    rank0_print(f'  WARNING rng: {e}')
            dwa_p = latest_checkpoint / 'dwa_state.pt'
            if dwa_p.exists():
                try:
                    dwa = torch.load(dwa_p, map_location='cpu')
                    if hasattr(trainer,'train_loss_buffer') and dwa.get('train_loss_buffer') is not None:
                        trainer.train_loss_buffer = dwa['train_loss_buffer'].to(training_args.device)
                    rank0_print('  DWA state restored.')
                except Exception as e:
                    rank0_print(f'  WARNING dwa: {e}')
            resume_from_checkpoint = str(latest_checkpoint)
        else:
            rank0_print('Full checkpoint - DeepSpeed resume')
            resume_from_checkpoint = str(latest_checkpoint)

    if resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    else:
        trainer.train()
    trainer.save_state()
    model.config.use_cache = True
    if training_args.lora_enable:
        sd = get_peft_state_maybe_zero_3(model.named_parameters(), training_args.lora_bias)
        non_lora = get_peft_state_non_lora_maybe_zero_3(model.named_parameters())
        if training_args.local_rank in [0, -1]:
            if hasattr(model,'config'): model.config.save_pretrained(training_args.output_dir)
            if hasattr(model,'generation_config'): model.generation_config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=sd)
            torch.save(non_lora, os.path.join(training_args.output_dir,'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    rank0_print(f'Model saved to {training_args.output_dir}')


if __name__ == '__main__':
    train()
    # TGCFA: 构建文本编码器并注入 trainer
    # [TODO-B] 通过 training_args.tgcfa_text_feat_fn 传递回调给 LLaVATrainer。
    # LLaVATrainer 需在 compute_loss 中调用此回调获取 text_feat，
    # 并将其传入 prepare_inputs_labels_for_multimodal。
    # 若 LLaVATrainer 不支持，需同步修改 llava_trainer.py。
    training_args.tgcfa_text_feat_fn = None
    if model_args.use_tgcfa:
        tgcfa_mod = getattr(model.get_model(), 'tgcfa', None)
        if tgcfa_mod is None:
            rank0_print('[TGCFA] WARNING: use_tgcfa=True but model.tgcfa is None.')
        else:
            # clip_path 必须指向完整 CLIP checkpoint（含文本编码器）
            # vision_tower 路径仅含视觉权重，不可直接用于文本编码器加载
            # 优先使用 --tgcfa_clip_path 参数；未指定时回退到 openai/clip-vit-large-patch14
            clip_path = getattr(model_args, 'tgcfa_clip_path', None) or 'openai/clip-vit-large-patch14'
            rank0_print(f'[TGCFA] Loading text encoder from: {clip_path}')
            try:
                _te = TGCFATextEncoder(clip_path, training_args.device, compute_dtype)
                def _fn(captions):
                    # [TODO-D] captions: List[str], len = batch_size // 2
                    return _te.encode(captions)
                training_args.tgcfa_text_feat_fn = _fn
                rank0_print('[TGCFA] text_feat_fn registered.')
            except Exception as e:
                rank0_print(f'[TGCFA] WARNING: text encoder load failed: {e}. Falling back to no-text path.')

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    training_args.detection_loss_weight  = model_args.detection_loss_weight
    training_args.caption_loss_weight    = model_args.caption_loss_weight
    training_args.use_dwa                = getattr(training_args, 'use_dwa', False)
    training_args.dwa_temperature        = getattr(training_args, 'dwa_temperature', 2.0)
    training_args.use_ngram_reward       = getattr(training_args, 'use_ngram_reward', False)
    training_args.ngram_ce_weight        = getattr(training_args, 'ngram_ce_weight', 0.7)
    training_args.ngram_reward_weight    = getattr(training_args, 'ngram_reward_weight', 0.2)
    training_args.ngram_diversity_weight = getattr(training_args, 'ngram_diversity_weight', 0.1)
    training_args.ngram_adaptive_weights = getattr(training_args, 'ngram_adaptive_weights', True)
    trainer = LLaVATrainer(model=model, tokenizer=tokenizer, args=training_args, task_num=2, **data_module)

    # 断点续训（与原版相同）
    checkpoint_list = list(pathlib.Path(training_args.output_dir).glob('checkpoint-*'))
    resume_from_checkpoint = None
    if checkpoint_list:
        latest_checkpoint = max(checkpoint_list, key=lambda x: int(x.name.split('-')[-1]))
        rank0_print(f'Found checkpoint: {latest_checkpoint}')
        mm_projector_path = latest_checkpoint / 'mm_projector.bin'
        model_safetensors = latest_checkpoint / 'model.safetensors'
        pytorch_model     = latest_checkpoint / 'pytorch_model.bin'
        trainer_state_path = latest_checkpoint / 'trainer_state.json'
        is_adapter_only = (mm_projector_path.exists()
                           and not model_safetensors.exists()
                           and not pytorch_model.exists())
        if is_adapter_only:
            rank0_print('Adapter-only checkpoint - manual resume')
            aw = torch.load(mm_projector_path, map_location='cpu')
            missing, _ = model.load_state_dict(aw, strict=False)
            rank0_print(f'  Loaded adapter weights (missing: {len(missing)})')
            deepspeed_dir = latest_checkpoint / 'deepspeed'
            if deepspeed_dir.exists():
                lf = deepspeed_dir / 'latest'
                if lf.exists():
                    step_dir = open(lf).read().strip()
                    op = deepspeed_dir / step_dir / 'bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt'
                    if op.exists():
                        try:
                            os_dict = torch.load(op, map_location='cpu', weights_only=False)
                            trainer._pending_optimizer_state = os_dict.get('optimizer_state_dict')
                            rank0_print('  Optimizer state pending.')
                        except Exception as e:
                            rank0_print(f'  WARNING optimizer: {e}')
            if trainer_state_path.exists():
                import json
                ts = json.load(open(trainer_state_path))
                trainer.state.global_step = ts.get('global_step', 0)
                trainer.state.epoch       = ts.get('epoch', 0)
                trainer.state.total_flos  = ts.get('total_flos', 0)
                rank0_print(f'  Step={trainer.state.global_step} epoch={trainer.state.epoch:.2f}')
            rng_p = latest_checkpoint / 'rng_state.pth'
            if rng_p.exists():
                try:
                    rng = torch.load(rng_p, map_location='cpu', weights_only=False)
                    if 'python' in rng and isinstance(rng['python'], torch.Tensor): torch.random.set_rng_state(rng['python'].cpu())
                    if 'cpu' in rng: torch.set_rng_state(rng['cpu'].cpu())
                    if 'cuda' in rng and torch.cuda.is_available(): torch.cuda.set_rng_state_all(rng['cuda'])
                    if 'numpy' in rng: import numpy as np; np.random.set_state(rng['numpy'])
                    rank0_print('  RNG states restored.')
                except Exception as e:
                    rank0_print(f'  WARNING rng: {e}')
            dwa_p = latest_checkpoint / 'dwa_state.pt'
            if dwa_p.exists():
                try:
                    dwa = torch.load(dwa_p, map_location='cpu')
                    if hasattr(trainer,'train_loss_buffer') and dwa.get('train_loss_buffer') is not None:
                        trainer.train_loss_buffer = dwa['train_loss_buffer'].to(training_args.device)
                    rank0_print('  DWA state restored.')
                except Exception as e:
                    rank0_print(f'  WARNING dwa: {e}')
            resume_from_checkpoint = str(latest_checkpoint)
        else:
            rank0_print('Full checkpoint - DeepSpeed resume')
            resume_from_checkpoint = str(latest_checkpoint)

    if resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    else:
        trainer.train()
    trainer.save_state()
    model.config.use_cache = True
    if training_args.lora_enable:
        sd = get_peft_state_maybe_zero_3(model.named_parameters(), training_args.lora_bias)
        non_lora = get_peft_state_non_lora_maybe_zero_3(model.named_parameters())
        if training_args.local_rank in [0, -1]:
            if hasattr(model,'config'): model.config.save_pretrained(training_args.output_dir)
            if hasattr(model,'generation_config'): model.generation_config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=sd)
            torch.save(non_lora, os.path.join(training_args.output_dir,'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    rank0_print(f'Model saved to {training_args.output_dir}')


if __name__ == '__main__':
    train()
