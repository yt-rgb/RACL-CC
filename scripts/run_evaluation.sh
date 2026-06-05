#!/bin/bash
# RACL 模型评估脚本
# 
# 使用方法:
#   bash scripts/run_evaluation.sh
#
# 或者指定参数:
#   bash scripts/run_evaluation.sh --num-samples 100

# 切换到项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
echo "Working directory: $(pwd)"

# 设置路径 (根据实际环境修改)
# CHECKPOINT_PATH="/root/RACL/checkpoints/racl-qwen2-7b-attention/checkpoint-1599"
CHECKPOINT_PATH="/root/autodl-tmp/checkpoints-new/stage1-joint-warmup/checkpoint-7462"
MODEL_BASE="/root/autodl-tmp/Qwen2-7B-Instruct"
# VISION_TOWER="/root/autodl-tmp/clip-vit-large-patch14-336"
VISION_TOWER="/root/autodl-tmp/checkpoints/CLIP_RegionAware_merged-2"
TEST_JSON="./racl/LEVIR-MCI-dataset/converted/test.json"
IMAGE_FOLDER="./racl/LEVIR-MCI-dataset/images"
OUTPUT_DIR="./eval_results"

# 设置离线模式
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# 运行评估
python evaluate_racl.py \
    --checkpoint ${CHECKPOINT_PATH} \
    --model-base ${MODEL_BASE} \
    --vision-tower ${VISION_TOWER} \
    --test-json ${TEST_JSON} \
    --image-folder ${IMAGE_FOLDER} \
    --output-dir ${OUTPUT_DIR} \
    --conv-mode qwen_2 \
    --temperature 0.0 \
    --max-new-tokens 256 \
    --eval-caption \
    --eval-segmentation \
    --save-predictions \
    --save-attention-maps \
    --attention-max-samples 100 \
    --attention-output-dir /root/autodl-tmp/attention_maps \
    "$@"

echo "Evaluation completed! Results saved to ${OUTPUT_DIR}"
