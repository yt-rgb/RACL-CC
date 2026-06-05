"""
LEVIR-MCI 数据集转换脚本
将原始 LevirCCcaptions.json 转换为 RACL 项目所需格式

原始数据结构:
- LevirCCcaptions.json: {"images": [...]}
- images/{split}/A/xxx.png (前时相)
- images/{split}/B/xxx.png (后时相)
- images/{split}/label/xxx.png (RGB格式标签)

目标数据结构:
- train.json, val.json, test.json
- 统一的图像目录结构
- 灰度标签 (0, 1, 2)

⚠️ 注意事项:
1. 原始label是RGB格式，需要转换为灰度 (0, 1, 2)
2. 原始每个样本有5个caption，这里使用第一个（可根据需要修改）
3. 生成的JSON路径相对于 --image_folder 参数
"""

import json
import os
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm


def convert_label_to_gray(label_path, output_path):
    """
    将RGB标签转换为灰度标签
    
    原始RGB值 (根据readme.txt):
    - [0, 0, 0] -> 0 (background)
    - [128, 128, 128] -> 1 (road)
    - [255, 255, 255] -> 2 (building)
    """
    img = Image.open(label_path)
    arr = np.array(img)
    
    # 创建灰度标签
    gray = np.zeros((arr.shape[0], arr.shape[1]), dtype=np.uint8)
    
    # RGB [128,128,128] -> 1
    mask_road = np.all(arr == [128, 128, 128], axis=-1)
    gray[mask_road] = 1
    
    # RGB [255,255,255] -> 2
    mask_building = np.all(arr == [255, 255, 255], axis=-1)
    gray[mask_building] = 2
    
    # 保存灰度图
    gray_img = Image.fromarray(gray, mode='L')
    gray_img.save(output_path)
    
    return gray


def convert_to_racl_format(original_json_path, images_dir, output_dir):
    """
    转换为RACL项目所需的数据格式
    
    Args:
        original_json_path: LevirCCcaptions.json 路径
        images_dir: images/ 目录路径
        output_dir: 输出目录
    """
    # 读取原始JSON
    print(f"Loading {original_json_path}...")
    with open(original_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建灰度label输出目录
    label_gray_dir = Path(images_dir) / "label_gray"
    for split in ['train', 'val', 'test']:
        (label_gray_dir / split).mkdir(parents=True, exist_ok=True)
    
    # 按split分组
    split_data = {'train': [], 'val': [], 'test': []}
    
    print("Converting dataset...")
    for item in tqdm(data['images']):
        split = item['split']  # train, val, test
        filename = item['filename']  # e.g., train_000001.png
        
        # 构建路径 (相对于 --image_folder)
        # 项目代码: os.path.join(image_folder, image_file)
        # 所以路径应该是: {split}/A/xxx.png
        img_a_path = f"{split}/A/{filename}"
        img_b_path = f"{split}/B/{filename}"
        label_gray_path = f"label_gray/{split}/{filename}"
        
        # 检查原始文件是否存在
        orig_label_path = Path(images_dir) / split / "label" / filename
        if not orig_label_path.exists():
            print(f"Warning: Label not found: {orig_label_path}")
            continue
            
        # 转换label为灰度格式
        gray_output_path = label_gray_dir / split / filename
        if not gray_output_path.exists():
            try:
                convert_label_to_gray(orig_label_path, gray_output_path)
            except Exception as e:
                print(f"Error converting {orig_label_path}: {e}")
                continue
        
        # 5种不同的指令格式，用于引导LLM
        INSTRUCTIONS = [
            "<image> This represents the change features of geographic targets extracted from remote sensing images. Does this feature contain any information about changes in the geographic targets? If so, please describe the change information.",
            "<image> These are the change characteristics of geographic targets derived from remote sensing images. Do these features reflect any changes in the geographic targets? If yes, please provide details on the changes.",
            "<image> These are the geographic target change features extracted from remote sensing images. Do these features indicate any changes in the geographic targets? If so, please describe the changes.",
            "<image> This is the change information of geographic targets extracted from remote sensing images. Does this information reveal any changes in the geographic targets? If so, please describe the changes.",
            "<image> Here are the change features of geographic targets pulled from remote sensing images. Do these features suggest any changes in the geographic targets? If they do, please describe the nature of those changes."
        ]
        
        # 使用全部5条caption，每条caption单独作为一个训练样本，配对不同的指令
        for sent_idx, sentence in enumerate(item['sentences']):
            caption = sentence['raw'].strip()
            
            # 构建对话格式，使用对应的指令格式
            sample = {
                "id": f"{split}_{item['imgid']}_{sent_idx}",
                "image": [img_a_path, img_b_path],
                "label_gray": label_gray_path,
                "conversations": [
                    {
                        "from": "human",
                        "value": INSTRUCTIONS[sent_idx % len(INSTRUCTIONS)]
                    },
                    {
                        "from": "gpt",
                        "value": caption
                    }
                ]
            }
            
            split_data[split].append(sample)
    
    # 保存转换后的JSON文件
    for split, samples in split_data.items():
        output_path = output_dir / f"{split}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(samples)} samples to {output_path}")
    
    print("\n转换完成！")
    print(f"灰度标签保存在: {label_gray_dir}")
    print(f"JSON文件保存在: {output_dir}")
    
    return split_data


def verify_dataset(images_dir, json_path, num_samples=3):
    """
    验证转换后的数据集格式是否正确
    """
    print(f"\n验证数据集: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"总样本数: {len(data)}")
    
    for i, sample in enumerate(data[:num_samples]):
        print(f"\n--- Sample {i+1} ---")
        print(f"ID: {sample['id']}")
        print(f"Images: {sample['image']}")
        print(f"Label: {sample['label_gray']}")
        
        # 检查文件是否存在
        for img_path in sample['image']:
            full_path = Path(images_dir) / img_path
            exists = "✓" if full_path.exists() else "✗ NOT FOUND"
            print(f"  {img_path}: {exists}")
        
        label_path = Path(images_dir) / sample['label_gray']
        if label_path.exists():
            img = Image.open(label_path)
            arr = np.array(img)
            print(f"  Label: ✓ (mode={img.mode}, unique={np.unique(arr)})")
        else:
            print(f"  Label: ✗ NOT FOUND")
        
        print(f"Caption: {sample['conversations'][1]['value'][:100]}...")


if __name__ == "__main__":
    # 配置路径 - 使用相对路径，兼容 Windows 和 Linux
    BASE_DIR = Path(__file__).parent
    
    ORIGINAL_JSON = BASE_DIR / "LevirCCcaptions.json"
    IMAGES_DIR = BASE_DIR / "images"
    OUTPUT_DIR = BASE_DIR / "converted"
    
    # 执行转换
    convert_to_racl_format(ORIGINAL_JSON, IMAGES_DIR, OUTPUT_DIR)
    
    # 验证结果
    verify_dataset(IMAGES_DIR, OUTPUT_DIR / "train.json")
