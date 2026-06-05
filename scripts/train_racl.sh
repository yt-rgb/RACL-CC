#!/bin/bash

# usage:  bash scripts/train_racl.sh

# Switch to the root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
echo "Working directory: $(pwd)"

# LLM path
MODEL_PATH="/root/autodl-tmp/Qwen2-7B-Instruct"

# Vision Encoder 
VISION_TOWER="/root/autodl-tmp/checkpoints/CLIP_RegionAware_merged"

# Dataset path
DATA_PATH="./racl/LEVIR-MCI-dataset/converted/train.json"
IMAGE_FOLDER="./racl/LEVIR-MCI-dataset/images"

# Output directory
OUTPUT_DIR="./checkpoints/racl-qwen2-7b-stacked-attention"

DEEPSPEED_CONFIG="./scripts/zero2.json"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
NUM_GPUS=${NUM_GPUS:-2}

deepspeed --num_gpus=${NUM_GPUS} ./racl/train/train.py \
    --deepspeed ${DEEPSPEED_CONFIG} \
    --model_name_or_path ${MODEL_PATH} \
    --version qwen_2 \
    --data_path ${DATA_PATH} \
    --image_folder ${IMAGE_FOLDER} \
    --vision_tower ${VISION_TOWER} \
    --mm_projector_type mlp2x_gelu \
    --mm_change_detector_type TA_attention_stacked \
    --mm_fusion_policy abs_diff \
    --mm_img_cd_concat False \
    --mm_seg_head_type conv \
    --mm_num_class 3 \
    --proc_crop_size 224 \
    --mm_tunable_parts mm_mlp_adapter,mm_change_detector,mm_seg_head \
    --mm_vision_select_layer -2 \
    --mm_vision_select_feature "slicefour_patch" \
    --mm_patch_merge_type flat \
    --num_attention_layers 3 \
    --attention_num_heads 4 \
    --attention_window_size 7 \
    --use_gate True \
    --use_attention True \
    --use_cross_layer_connection True \
    --use_gradient_checkpointing True \
    --bf16 True \
    --attn_implementation sdpa \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 20 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --optim adamw_torch \
    --learning_rate 1e-4 \
    --mm_projector_lr 1e-5 \
    --weight_decay 0.0005 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --save_strategy "epoch" \
    --save_total_limit 50 \
    --logging_steps 10 \
    --dataloader_num_workers 8 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --ignore_data_skip True \
    --report_to tensorboard
