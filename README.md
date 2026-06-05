# RACL-CC

This repo is the official implementation for RACL-CC: Bridging Domain and Semantic Gaps in Remote Sensing Change Captioning

## Project Structure

```text
RACL-CC/
├── racl/                         # Main RACL-CC model, dataset, training and evaluation code
│   ├── dataset/                   # Dataset loading code
│   ├── model/                     # Multimodal model, encoder, fusion, projector and segmentation head
│   ├── train/                     # Training scripts
│   ├── meteor/                    # METEOR evaluation files
│   └── LEVIR-MCI-dataset/         # Dataset annotations and conversion scripts
├── RACL-pretrain/                 # Region-aware CLIP pretraining code
├── scripts/                       # Training, evaluation and conversion scripts
├── evaluate_racl.py               # Main evaluation entry
└── .gitignore
```

## Environment Create

The recommended environment is:

- Ubuntu 22.04
- Python 3.12
- PyTorch 2.7.0
- CUDA 12.8

Create and activate a conda environment:

```bash
conda create -n raclcc python=3.12 -y
conda activate raclcc
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset

RACL-CC uses [LEVIR-MCI](https://github.com/Chen-Yang-Liu/LEVIR-CC-Dataset) for training and evaluation. The region-aware CLIP pretraining stage supports LEVIR-MCI, [CLCD](https://github.com/liumency/CropLand-CD), and [SECOND](https://captain-whu.github.io/SCD/).

```text
racl/LEVIR-MCI-dataset/
├── converted/
│   ├── train.json
│   ├── val.json
│   └── test.json
└── images/
    ├── train/
    ├── val/
    ├── test/
    ├── label_gray/
    └── label_rgb/
```

Pretraining dataset settings are configured in `RACL-pretrain/clip_region_aware.yaml`:

```yaml
dataset:
  module: LEVIR_MCI_aug
  root: /path/to/dataset
```

Use `LEVIR_MCI_aug`, `CLCD_aug`, or `SECOND_aug` according to the selected dataset. 

## Training

The training pipeline contains two stages: region-aware CLIP pretraining and RACL-CC multimodal training.

### Region-Aware CLIP Pretraining

The pretraining code is located in `RACL-pretrain/`.

Update the dataset path and checkpoint path in `RACL-pretrain/clip_region_aware.yaml`, then run:

```bash
cd RACL-pretrain
python train_region_aware.py --config clip_region_aware.yaml
```

After pretraining, merge the LoRA weights into HuggingFace CLIP format:

```bash
python scripts/convert_clip_lora.py \
  --input /path/to/best_model.pth \
  --base-model /path/to/clip-vit-large-patch14 \
  --output /path/to/CLIP_RegionAware_merged
```

### RACL-CC Training

Run training:

```bash
bash scripts/train_racl.sh
```

The default script trains the multimodal projector, change detector, and segmentation head with DeepSpeed.
