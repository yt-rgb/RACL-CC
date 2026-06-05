#    Copyright 2023 Haotian Liu
#    Licensed under the Apache License, Version 2.0 (the "License").

"""
llava_arch_tgcfa.py -- llava_arch.py + TGCFA 文本引导变化特征对齐
新增: tgcfa 子模块注册; encode_images 在 Projector 前插入 TGCFA;
prepare_inputs_labels_for_multimodal 透传 text_feat.
"""

from abc import ABC, abstractmethod
import os, random
import torch
import torch.nn as nn
from .encoder.builder import build_vision_tower
from .fusion.builder import build_change_detector
from .fusion.tgcfa import build_tgcfa
from .projector.builder import build_vision_projector
from .seg_head.builder import build_seg_head
from constants import (
    IGNORE_INDEX, IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN,
)
from mm_utils import get_anyres_image_grid_shape
from utils import rank0_print, rank_print
import pdb


class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)
        if hasattr(config, "mm_vision_tower"):
            delay_load = getattr(config, "delay_load", False)
            self.vision_tower    = build_vision_tower(config, delay_load=delay_load)
            self.change_detector = build_change_detector(config)
            self.seg_head        = build_seg_head(config)
            self.mm_projector    = build_vision_projector(config, vision_cfg=self.vision_tower.config)
            tgcfa = build_tgcfa(config)
            if tgcfa is not None:
                self.tgcfa = tgcfa
            if "unpad" in getattr(config, "mm_patch_merge_type", ""):
                self.image_newline = nn.Parameter(torch.empty(config.hidden_size, dtype=self.dtype))

    def get_vision_tower(self):
        vt = getattr(self, "vision_tower", None)
        return vt[0] if type(vt) is list else vt

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower             = model_args.vision_tower
        mm_vision_select_layer   = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter  = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type      = model_args.mm_patch_merge_type
        self.config.mm_vision_tower         = vision_tower
        self.config.vision_tower_pretrained  = getattr(model_args, "vision_tower_pretrained", "")

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)
            self.vision_tower = [vision_tower] if fsdp else vision_tower
        else:
            vision_tower = self.vision_tower[0] if fsdp else self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj              = True
        self.config.use_mm_cd                = True
        self.config.mm_projector_type        = getattr(model_args, "mm_projector_type", "linear")
        self.config.mm_hidden_size           = vision_tower.hidden_size
        self.config.mm_vision_select_layer   = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type      = mm_patch_merge_type
        self.config.mm_num_patches_per_side  = vision_tower.num_patches_per_side
        self.config.mm_change_detector_type  = getattr(model_args, "mm_change_detector_type", "fdfa")
        self.config.mm_fusion_policy         = model_args.mm_fusion_policy
        self.config.mm_img_cd_concat         = model_args.mm_img_cd_concat
        self.config.mm_seg_head_type         = model_args.mm_seg_head_type
        self.config.mm_num_class             = model_args.mm_num_class
        self.config.attention_num_heads      = getattr(model_args, "attention_num_heads", 4)
        self.config.attention_window_size    = getattr(model_args, "attention_window_size", 7)
        self.config.use_gate                 = getattr(model_args, "use_gate", True)
        self.config.use_attention            = getattr(model_args, "use_attention", False)
        self.config.use_tgcfa          = getattr(model_args, "use_tgcfa", False)
        self.config.tgcfa_text_dim     = getattr(model_args, "tgcfa_text_dim", 768)
        self.config.tgcfa_temperature  = getattr(model_args, "tgcfa_temperature", 0.07)

        if getattr(self, "change_detector", None) is None:
            self.change_detector = build_change_detector(self.config)
        else:
            for p in self.change_detector.parameters(): p.requires_grad = True

        if getattr(self, "seg_head", None) is None:
            self.seg_head = build_seg_head(self.config)
        else:
            for p in self.seg_head.parameters(): p.requires_grad = True

        if getattr(self, "mm_projector", None) is None:
            self.mm_projector = build_vision_projector(self.config, vision_cfg=vision_tower.config)
            if "unpad" in mm_patch_merge_type:
                std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(torch.randn(self.config.hidden_size, dtype=self.dtype) * std)
        else:
            for p in self.mm_projector.parameters(): p.requires_grad = True

        if self.config.use_tgcfa:
            if getattr(self, "tgcfa", None) is None:
                self.tgcfa = build_tgcfa(self.config)
                rank0_print("[TGCFA] initialized.")
            else:
                for p in self.tgcfa.parameters(): p.requires_grad = True
                rank0_print("[TGCFA] unfrozen.")

        if pretrain_mm_mlp_adapter is not None:
            if not os.path.exists(pretrain_mm_mlp_adapter):
                raise FileNotFoundError(f"pretrain_mm_mlp_adapter not found: {pretrain_mm_mlp_adapter}")
            rank0_print(f"Loading adapter weights from {pretrain_mm_mlp_adapter}")
            try:
                weights = torch.load(pretrain_mm_mlp_adapter, map_location="cpu")
            except Exception as e:
                raise RuntimeError(f"Failed to load: {e}")
            dtype = self.dtype if hasattr(self, "dtype") else torch.float16
            weights = {k: v.to(dtype) for k, v in weights.items()}
            prefixes = ["mm_projector", "change_detector", "seg_head", "vision_tower", "tgcfa"]
            ckpt_mods = set()
            for k in weights:
                for p in prefixes:
                    if k.startswith(p): ckpt_mods.add(p); break
            rank0_print(f"  Detected: {ckpt_mods}")
            missing, _ = self.load_state_dict(weights, strict=False)
            buf_kws = ["_index","_mask","running_mean","running_var","num_batches_tracked"]
            bad = [k for k in missing if any(p in k for p in ckpt_mods) and not any(b in k for b in buf_kws)]
            if bad:
                rank0_print(f"  Warning: {len(bad)} weights not loaded; first 5: {bad[:5]}")
            else:
                rank0_print("  All adapter weights loaded.")
            for mod in ckpt_mods:
                mk = [k for k in weights if k.startswith(mod)]
                lk = [k for k in mk if k not in missing]
                rank0_print(f"  {mod}: {len(lk)}/{len(mk)}")


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self): pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def generate_indices(self, N):
        assert N % 2 == 0
        return torch.cat([torch.arange(0, N, 2), torch.arange(1, N, 2)])

    def encode_images(self, images, text_feat=None):
        """
        text_feat: [B//2, L, C_t] CLIP文本特征，训练时传入，推理时可为None。
        TGCFA 不再改动 change_features，仅计算辅助语义对齐损失。
          F_out = LayerNorm(F_c + FFN(concat(F_c', F_t')))
        """
        if self.config.mm_fusion_policy is not None:
            image_features = self.get_model().get_vision_tower()(images)
            B, N, _ = image_features.shape
            indices          = self.generate_indices(B)
            original_indices = torch.argsort(indices)
            image_features   = image_features[indices]
            pre  = image_features[:B // 2]
            post = image_features[B // 2:]
            change_features = self.get_model().change_detector(pre, post, self.config.mm_fusion_policy)
            change_logits   = self.get_model().seg_head(change_features)

            align_loss = change_features.new_zeros(())
            tgcfa = getattr(self.get_model(), "tgcfa", None)
            if tgcfa is not None:
                align_loss = tgcfa(change_features, text_feat)

            if self.config.mm_img_cd_concat:
                image_features       = image_features[original_indices]
                image_features       = self.get_model().mm_projector(image_features)
                change_features_proj = self.get_model().mm_projector(change_features)
                _, _, C = image_features.shape
                image_features       = image_features.view(B // 2, 2, N, C)
                change_features_proj = change_features_proj.view(B // 2, 1, N, C)
                image_features = torch.cat([image_features, change_features_proj], dim=1).view(-1, N, C)
                return image_features, change_logits, align_loss
            else:
                return self.get_model().mm_projector(change_features), change_logits, align_loss
        else:
            return self.get_model().mm_projector(self.get_model().get_vision_tower()(images))


    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values,
        labels, images, modalities=["image"], image_sizes=None, text_feat=None,
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels, None, None

        align_loss = None
        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            images_list   = [img if img.ndim == 4 else img.unsqueeze(0) for img in images]
            concat_images = torch.cat(images_list, dim=0)
            split_sizes   = [img.shape[0] for img in images_list]
            if self.config.mm_fusion_policy is not None:
                if self.config.mm_img_cd_concat:
                    split_sizes = split_sizes[:len(split_sizes)//2] * 3
                else:
                    split_sizes = [1] * (len(split_sizes) // 2)
            if self.config.mm_fusion_policy is not None:
                encoded_image_features, change_logits, align_loss = self.encode_images(
                    concat_images, text_feat=text_feat
                )
            else:
                encoded_image_features = self.encode_images(concat_images)
                change_logits = None
            image_features = list(torch.split(encoded_image_features, split_sizes))
            mm_pmt = getattr(self.config, "mm_patch_merge_type", "flat")
            if mm_pmt == "flat":
                image_features = [x.flatten(0, 1) for x in image_features]
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            image_features = self.encode_images(images)
            change_logits = None

        if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(self.config, "mm_use_im_start_end", False):
            raise NotImplementedError

        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        attention_mask = attention_mask.bool() if attention_mask is not None else torch.ones_like(input_ids, dtype=torch.bool)
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        input_ids = [cur[mask] for cur, mask in zip(input_ids, attention_mask)]
        labels    = [cur[mask] for cur, mask in zip(labels, attention_mask)]

        new_input_embeds, new_labels = [], []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_feats    = image_features[cur_image_idx]
                cur_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                new_input_embeds.append(torch.cat([cur_embeds_1, cur_feats[0:0]], dim=0))
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue
            tok_idx = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            ids_noim, labs_noim = [], []
            cur_labs = labels[batch_idx]
            for i in range(len(tok_idx) - 1):
                ids_noim.append(cur_input_ids[tok_idx[i]+1:tok_idx[i+1]])
                labs_noim.append(cur_labs[tok_idx[i]+1:tok_idx[i+1]])
            split_sz     = [x.shape[0] for x in labs_noim]
            embeds       = self.get_model().embed_tokens(torch.cat(ids_noim))
            embeds_no_im = torch.split(embeds, split_sz, dim=0)
            cur_new_embeds, cur_new_labs = [], []
            for i in range(num_images + 1):
                cur_new_embeds.append(embeds_no_im[i])
                cur_new_labs.append(labs_noim[i])
                if i < num_images:
                    try:    cur_feats = image_features[cur_image_idx]
                    except: cur_feats = image_features[cur_image_idx - 1]
                    cur_image_idx += 1
                    cur_new_embeds.append(cur_feats)
                    cur_new_labs.append(torch.full(
                        (cur_feats.shape[0],), IGNORE_INDEX,
                        device=cur_labs.device, dtype=cur_labs.dtype
                    ))
            cur_new_embeds = [x.to(self.device) for x in cur_new_embeds]
            new_input_embeds.append(torch.cat(cur_new_embeds))
            new_labels.append(torch.cat(cur_new_labs))

        tml = getattr(self.config, "tokenizer_model_max_length", None)
        new_input_embeds = [x[:tml] for x, _ in zip(new_input_embeds, modalities)]
        new_labels       = [x[:tml] for x, _ in zip(new_labels, modalities)]
        max_len    = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)
        new_input_embeds_padded = []
        new_labels_padded = torch.full(
            (batch_size, max_len), IGNORE_INDEX,
            dtype=new_labels[0].dtype, device=new_labels[0].device
        )
        attn_mask_out = torch.zeros((batch_size, max_len), dtype=torch.bool, device=new_labels[0].device)
        pos_ids_out   = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
        for i, (emb, lab) in enumerate(zip(new_input_embeds, new_labels)):
            L   = emb.shape[0]
            pad = torch.zeros((max_len - L, emb.shape[1]), dtype=emb.dtype, device=emb.device)
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(torch.cat([pad, emb], dim=0))
                if L > 0:
                    new_labels_padded[i, -L:] = lab
                    attn_mask_out[i, -L:]     = True
                    pos_ids_out[i, -L:]       = torch.arange(0, L, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat([emb, pad], dim=0))
                if L > 0:
                    new_labels_padded[i, :L] = lab
                    attn_mask_out[i, :L]     = True
                    pos_ids_out[i, :L]       = torch.arange(0, L, dtype=position_ids.dtype, device=position_ids.device)
        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
        new_labels       = None if _labels is None else new_labels_padded
        attention_mask   = None if _attention_mask is None else attn_mask_out.to(dtype=_attention_mask.dtype)
        position_ids     = None if _position_ids is None else pos_ids_out
        if getattr(self.config, "use_pos_skipping", False) and self.training:
            position_ids = torch.arange(
                new_input_embeds.size(1), device=new_input_embeds.device
            ).unsqueeze(0).to(new_input_embeds.device)
            sp   = random.randint(0, new_input_embeds.size(1))
            ladd = random.randint(0, self.config.pos_skipping_range)
            radd = random.randint(ladd, self.config.pos_skipping_range)
            position_ids[:, :sp] += ladd
            position_ids[:, sp:] += radd
        if self.config.mm_fusion_policy is not None:
            return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, change_logits, align_loss
        else:
            return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, None, None

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))
        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
            )
            self.resize_token_embeddings(len(tokenizer))
            if num_new_tokens > 0:
                ie = self.get_input_embeddings().weight.data
                oe = self.get_output_embeddings().weight.data
                ie[-num_new_tokens:] = ie[:-num_new_tokens].mean(dim=0, keepdim=True)
                oe[-num_new_tokens:] = oe[:-num_new_tokens].mean(dim=0, keepdim=True)
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():  p.requires_grad = True
                for p in self.get_output_embeddings().parameters(): p.requires_grad = False
            if model_args.pretrain_mm_mlp_adapter:
                try:
                    w  = torch.load(model_args.pretrain_mm_mlp_adapter, map_location="cpu")
                    ie = self.get_input_embeddings().weight.data
                    if "model.embed_tokens.weight" in w:
                        etw = w["model.embed_tokens.weight"]
                        assert num_new_tokens == 2
                        if ie.shape == etw.shape:
                            ie[-num_new_tokens:] = etw[-num_new_tokens:]
                            rank0_print(f"  Loaded embed_tokens for {num_new_tokens} special tokens")
                        elif etw.shape[0] == num_new_tokens:
                            ie[-num_new_tokens:] = etw
                            rank0_print(f"  Loaded embed_tokens for {num_new_tokens} special tokens")
                        else:
                            rank0_print("  Warning: embed_tokens shape mismatch, skipping")
                    else:
                        rank0_print(f"  No embed_tokens in {model_args.pretrain_mm_mlp_adapter}")
                except Exception as e:
                    rank0_print(f"  Warning: failed to load embed_tokens: {e}")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():  p.requires_grad = False
                for p in self.get_output_embeddings().parameters(): p.requires_grad = False
