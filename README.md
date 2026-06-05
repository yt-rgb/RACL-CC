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


Data Preparation
The project uses LEVIR-MCI style bi-temporal remote-sensing data.

Expected paths used by the default scripts:
racl/LEVIR-MCI-dataset/converted/train.json
racl/LEVIR-MCI-dataset/converted/test.json
racl/LEVIR-MCI-dataset/images/

The image folder is usually large and is ignored by Git. Please prepare the image data locally before training or evaluation.

Region-Aware CLIP Pretraining
The RACL-pretrain/ directory contains the region-aware CLIP pretraining code.
