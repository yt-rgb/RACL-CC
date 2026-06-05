# RACL-CC

RACL-CC is a remote-sensing change captioning project for generating natural-language descriptions of changes between bi-temporal images. The project combines a Qwen2-based multimodal language model with a CLIP visual encoder and change-aware fusion modules, and also supports change segmentation evaluation.

## Features

- Bi-temporal remote-sensing image change captioning
- Region-aware CLIP pretraining with LoRA
- Change-aware multimodal fusion modules
- Joint captioning and segmentation evaluation
- Support for LEVIR-MCI formatted datasets
- DeepSpeed training scripts

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

EnvironmentCreate a Python environment and install the required deep learning packages according to your CUDA version.
Main dependencies include:
torch
transformers
deepspeed
accelerate
numpy
Pillow
tqdm
nltk
pycocoevalcap

For evaluation metrics, install:
pip install nltk pycocoevalcap

Data Preparation
The project uses LEVIR-MCI style bi-temporal remote-sensing data.

Expected paths used by the default scripts:
racl/LEVIR-MCI-dataset/converted/train.json
racl/LEVIR-MCI-dataset/converted/test.json
racl/LEVIR-MCI-dataset/images/

The image folder is usually large and is ignored by Git. Please prepare the image data locally before training or evaluation.

Region-Aware CLIP Pretraining
The RACL-pretrain/ directory contains the region-aware CLIP pretraining code.
