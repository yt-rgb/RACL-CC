"""
LEVIR-MCI 数据集加载器

数据集结构:
LEVIR-MCI-dataset/
├── images/
│   ├── train/
│   │   ├── A/          # 时相1图像
│   │   ├── B/          # 时相2图像
│   │   ├── label/      # 变化标签
│   │   └── label_rgb/  # RGB格式标签 (可选)
│   ├── val/
│   │   ├── A/
│   │   ├── B/
│   │   └── label/
│   └── test/
│       ├── A/
│       ├── B/
│       └── label/
"""

import os
import math
import random
import numpy as np
from skimage import io, exposure
from torch.utils import data
from torchvision.transforms import functional as F
import warnings

import albumentations as A

warnings.filterwarnings(
    "ignore",
    message="ShiftScaleRotate is a special case of Affine transform",
    module="albumentations",
)

num_classes = 1
root = 'LEVIR-MCI-dataset/images'


def set_root(new_root: str) -> None:
    """设置数据集根目录"""
    global root
    root = new_root


def Color2Index(ColorLabel):
    """将标签转换为二值索引"""
    IndexMap = ColorLabel.clip(max=1)
    return IndexMap


def tensor2color(img_tensor):
    img = img_tensor.cpu().detach().numpy()
    img = exposure.rescale_intensity(img, out_range=np.uint8)
    return img


def Index2Color(pred):
    """将预测结果转换为可视化图像"""
    pred = pred * 255
    return pred.astype(np.uint8)


def _normalize_image_channels(img):
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.ndim == 3 and img.shape[-1] == 4:
        img = img[:, :, :3]
    return img


def sliding_crop_CD(imgs1, imgs2, labels, size, names=None):
    """滑动窗口裁剪"""
    crop_imgs1 = []
    crop_imgs2 = []
    crop_labels = []
    crop_names = [] if names is not None else None
    label_dims = len(labels[0].shape)
    iterator = zip(imgs1, imgs2, labels, names) if names is not None else zip(imgs1, imgs2, labels)

    for items in iterator:
        if names is not None:
            img1, img2, label, name = items
        else:
            img1, img2, label = items
            name = None
        h = img1.shape[0]
        w = img1.shape[1]
        c_h = size[0]
        c_w = size[1]

        if h < c_h or w < c_w:
            crop_imgs1.append(img1)
            crop_imgs2.append(img2)
            crop_labels.append(label)
            if crop_names is not None:
                crop_names.append(name)
            continue

        h_rate = h / c_h
        w_rate = w / c_w
        h_times = math.ceil(h_rate)
        w_times = math.ceil(w_rate)

        stride_h = 0 if h_times == 1 else math.ceil(c_h * (h_times - h_rate) / (h_times - 1))
        stride_w = 0 if w_times == 1 else math.ceil(c_w * (w_times - w_rate) / (w_times - 1))

        for j in range(h_times):
            for i in range(w_times):
                s_h = int(j * c_h - j * stride_h)
                if j == (h_times - 1):
                    s_h = h - c_h
                e_h = s_h + c_h

                s_w = int(i * c_w - i * stride_w)
                if i == (w_times - 1):
                    s_w = w - c_w
                e_w = s_w + c_w

                crop_imgs1.append(img1[s_h:e_h, s_w:e_w, :])
                crop_imgs2.append(img2[s_h:e_h, s_w:e_w, :])

                if label_dims == 2:
                    crop_labels.append(label[s_h:e_h, s_w:e_w])
                else:
                    crop_labels.append(label[s_h:e_h, s_w:e_w, :])

                if crop_names is not None:
                    crop_names.append(name)

    print(f'Sliding crop finished. {len(crop_imgs1)} pairs created.')
    if crop_names is not None:
        return crop_imgs1, crop_imgs2, crop_labels, crop_names
    return crop_imgs1, crop_imgs2, crop_labels


def read_RSimages(mode, read_list=False):
    """
    读取遥感图像对和标签

    Args:
        mode: 'train', 'val', 或 'test'
        read_list: 是否从列表文件读取 (未使用)

    Returns:
        data_A: 时相1图像列表
        data_B: 时相2图像列表
        labels: 标签列表
        filenames: 文件名列表
    """
    del read_list

    img_A_dir = os.path.join(root, mode, 'A')
    img_B_dir = os.path.join(root, mode, 'B')
    label_dir = os.path.join(root, mode, 'label')

    # 检查目录是否存在
    if not os.path.exists(img_A_dir):
        raise FileNotFoundError(f"Directory not found: {img_A_dir}")

    valid_exts = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')
    data_list = [it for it in os.listdir(img_A_dir) if it.lower().endswith(valid_exts)]

    data_A, data_B, labels, names = [], [], [], []
    for idx, it in enumerate(data_list):
        img_A_path = os.path.join(img_A_dir, it)
        img_B_path = os.path.join(img_B_dir, it)
        label_path = os.path.join(label_dir, it)

        if not os.path.exists(img_B_path):
            print(f"Warning: missing B image for {it}, skipped.")
            continue

        # 如果标签文件扩展名不同，尝试查找
        if not os.path.exists(label_path):
            base_name = os.path.splitext(it)[0]
            matched = False
            for ext in valid_exts:
                alt_path = os.path.join(label_dir, base_name + ext)
                if os.path.exists(alt_path):
                    label_path = alt_path
                    matched = True
                    break
            if not matched:
                print(f"Warning: missing label for {it}, skipped.")
                continue

        img_A = _normalize_image_channels(io.imread(img_A_path))
        img_B = _normalize_image_channels(io.imread(img_B_path))
        label = io.imread(label_path)

        if len(label.shape) == 3:
            label = label[:, :, 0]

        data_A.append(img_A)
        data_B.append(img_B)
        labels.append(Color2Index(label))
        names.append(it)

        if idx % 50 == 0:
            print(f'{idx}/{len(data_list)} images loaded.')

    print(f'{len(data_A)} {mode} images loaded.')
    if len(data_A) > 0:
        print(f'Image shape: {data_A[0].shape}')

    return data_A, data_B, labels, names


def weak_aug(img1, img2, mask):
    """弱增强: 随机裁剪和翻转"""
    h, w, _ = img1.shape
    aug = A.Compose([
        A.RandomResizedCrop(size=(h, w), scale=(0.75, 1.0), p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ], p=1., additional_targets={'image2': 'image', 'mask': 'mask'})

    tf_sample = aug(image=img1, image2=img2, mask=mask)
    return tf_sample['image'], tf_sample['image2'], tf_sample['mask']


def strong_aug(img, img_ref):
    """强增强: 颜色变换、模糊、空间变换"""
    aug = A.Compose([
        # 颜色变换 (移除 PixelDistributionAdaptation，新版本不兼容)
        A.RGBShift(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.5),
        # 模糊和噪声
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        # 空间变换
        A.ShiftScaleRotate(shift_limit=0.03, scale_limit=0.0, rotate_limit=0, p=0.5),
    ], p=1.)

    aug_result = aug(image=img)
    return aug_result['image']


class RS(data.Dataset):
    """
    LEVIR-MCI 遥感变化检测数据集

    Args:
        mode: 'train', 'val', 或 'test'
        random_crop: 是否随机裁剪
        crop_nums: 每张图像裁剪次数
        sliding_crop: 是否滑动窗口裁剪
        crop_size: 裁剪尺寸
        random_flip: 是否随机翻转
        return_filename: 是否在验证/测试时返回文件名
    """

    def __init__(
        self,
        mode,
        random_crop=False,
        crop_nums=6,
        sliding_crop=False,
        crop_size=512,
        random_flip=False,
        return_filename=False,
    ):
        self.mode = mode
        self.random_flip = random_flip
        self.random_crop = random_crop
        self.crop_nums = crop_nums
        self.crop_size = crop_size
        self.return_filename = return_filename

        data_A, data_B, labels, names = read_RSimages(mode, read_list=False)

        if sliding_crop:
            data_A, data_B, labels, names = sliding_crop_CD(
                data_A, data_B, labels, [self.crop_size, self.crop_size], names=names
            )

        self.data_A, self.data_B, self.labels, self.names = data_A, data_B, labels, names

        if self.random_crop:
            self.len = crop_nums * len(self.data_A)
        else:
            self.len = len(self.data_A)

    def __getitem__(self, idx):
        if self.random_crop:
            idx = idx // self.crop_nums

        data_A = self.data_A[idx]
        data_B = self.data_B[idx]
        label = self.labels[idx]
        name = self.names[idx]

        if self.mode == 'train':
            # 弱增强
            data_A, data_B, label = weak_aug(data_A, data_B, label)
            # 强增强
            data_A_aug = strong_aug(data_A, data_B)
            data_B_aug = strong_aug(data_B, data_A)
            return F.to_tensor(data_A), F.to_tensor(data_B), F.to_tensor(data_A_aug), F.to_tensor(data_B_aug)
        else:
            if self.return_filename:
                return F.to_tensor(data_A), F.to_tensor(data_B), label, name
            return F.to_tensor(data_A), F.to_tensor(data_B), label

    def __len__(self):
        return self.len
