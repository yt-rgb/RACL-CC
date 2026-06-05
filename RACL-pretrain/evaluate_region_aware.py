"""
usage:
    python evaluate_region_aware.py --config clip_region_aware.yaml
"""

from __future__ import annotations

import argparse
import inspect
import time
from collections import defaultdict
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from skimage import io
from torch.utils.data import DataLoader
import os

from config import PROJECT_ROOT, load_config
from utils import AverageMeter, binary_accuracy as accuracy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Region-Aware pretraining checkpoint")
    parser.add_argument(
        "--config",
        type=str,
        default="sam_region_aware.yaml",
        help="YAML config path (relative to pretrain/ or absolute).",
    )
    parser.add_argument(
        "--options",
        nargs="*",
        default=None,
        help="Override config fields via key=value pairs.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/root/autodl-tmp/checkpoints/SAM_LoRA_RegionAware/epoch_26.pth",
        help=(
            "Path to a .pth state_dict. Default: /root/autodl-tmp/checkpoints/CLIP_RegionAware/best_model.pth "
            "(override with --checkpoint)."
        ),
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=("val", "test"),
        help="Which dataset split to evaluate. NOTE: This script is intended to run on test to match your request.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override config training.val_batch_size (default: follow config).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override config training.num_workers (default: follow config).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--tta",
        dest="tta",
        action="store_true",
        default=None,
        help="Apply flip TTA, overriding training.tta from the config.",
    )
    parser.add_argument(
        "--no-tta",
        dest="tta",
        action="store_false",
        help="Disable flip TTA, overriding training.tta from the config.",
    )
    parser.add_argument(
        "--save-pred",
        dest="save_pred",
        action="store_true",
        help="Save prediction probability maps (same format as train_region_aware.py saved images).",
    )
    parser.add_argument(
        "--no-save-pred",
        dest="save_pred",
        action="store_false",
        help="Disable saving prediction images.",
    )
    parser.add_argument(
        "--save-max",
        type=int,
        default=100,
        help="Max number of predictions to save when save_pred is enabled. Default=100 (first 100 image pairs).",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="/root/autodl-tmp/results/test",
        help=(
            "Output directory for saved predictions. If omitted, uses cfg.paths.results. "
            "Saved images are probability maps (0..1) multiplied by 255."
        ),
    )
    parser.add_argument(
        "--save-binary",
        action="store_true",
        help="Also save binary change masks (0/255 PNG) using a fixed threshold.",
    )
    parser.add_argument(
        "--binary-threshold",
        type=float,
        default=0.5,
        help="Threshold for binary masks (pred >= threshold). Default=0.5.",
    )
    parser.set_defaults(save_pred=True)
    return parser.parse_args()


def _get_device(device_str: str) -> torch.device:
    if device_str.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_str)
    return torch.device("cpu")


def _override_dataset_root(module, dataset_cfg: Dict[str, Any], project_root: Path | None) -> None:
    root_override = dataset_cfg.get("root")
    if not root_override:
        return
    root_path = Path(root_override)
    if not root_path.is_absolute():
        base = project_root or Path.cwd()
        root_path = base / root_path
    root_str = str(root_path)
    setter = getattr(module, "set_root", None)
    if callable(setter):
        setter(root_str)
    elif hasattr(module, "root"):
        setattr(module, "root", root_str)
    else:
        raise AttributeError(f"Dataset module {module.__name__} does not support root override.")


def _build_dataset_and_module(cfg: Dict[str, Any], split: str):
    dataset_cfg = cfg["dataset"]
    module = import_module(dataset_cfg["module"])
    dataset_cls = getattr(module, dataset_cfg.get("class_name", "RS"))
    _override_dataset_root(module, dataset_cfg, cfg.get("project_root"))

    split_cfg = dataset_cfg.get(split)
    if not split_cfg:
        raise ValueError(f"Missing dataset split config: {split}")
    kwargs = split_cfg.get("kwargs", {}) or {}
    split_name = split_cfg.get("split", split)

    # Prefer to get original filenames for saving prediction images.
    try:
        sig = inspect.signature(dataset_cls.__init__)
        supports_return_filename = "return_filename" in sig.parameters
    except (TypeError, ValueError):
        supports_return_filename = False

    if supports_return_filename:
        kwargs.setdefault("return_filename", True)

    try:
        dataset = dataset_cls(split_name, **kwargs)
    except TypeError as e:
        # Fallback for older dataset implementations.
        if "return_filename" in str(e):
            kwargs.pop("return_filename", None)
            dataset = dataset_cls(split_name, **kwargs)
        else:
            raise
    return module, dataset


def _list_split_filenames(dataset_root: Path, split_name: str) -> Optional[list[str]]:
    img_a_dir = dataset_root / split_name / "A"
    if not img_a_dir.exists():
        return None
    names = []
    for it in os.listdir(img_a_dir):
        low = it.lower()
        if low.endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
            names.append(it)
    return names


def _resolve_dataset_root(cfg: Dict[str, Any]) -> Path:
    dataset_root = Path(cfg["dataset"].get("root"))
    if not dataset_root.is_absolute():
        dataset_root = (PROJECT_ROOT / dataset_root).resolve()
    return dataset_root


def _sanitize_filename(name: str) -> str:
    # Windows forbids characters: <>:"/\\|?*
    # Also keep it short-ish to avoid path issues.
    name = name.replace("/", "_").replace("\\", "_")
    for ch in ['<', '>', ':', '"', '|', '?', '*']:
        name = name.replace(ch, "_")
    name = name.strip().strip(".")
    return name[:180] if len(name) > 180 else name


def _resolve_sample_name(dataset, names, batch_index: int, sample_index_in_batch: int, fallback_index: int) -> str:
    raw_name = None
    try:
        raw_name = names[sample_index_in_batch]
    except Exception:
        raw_name = None

    if raw_name is not None:
        return str(raw_name)

    dataset_index = batch_index + sample_index_in_batch
    sample_indices = getattr(dataset, "sample_indices", None)
    if sample_indices is not None and 0 <= dataset_index < len(sample_indices):
        dataset_index = sample_indices[dataset_index]

    dataset_names = getattr(dataset, "names", None)
    if dataset_names is not None and 0 <= dataset_index < len(dataset_names):
        candidate = dataset_names[dataset_index]
        if candidate is not None:
            return str(candidate)

    return f"sample-{fallback_index:06d}"


def _extract_tp_tn_fp_fn_map(pred: np.ndarray, label: np.ndarray, threshold: float) -> np.ndarray:
    pred_bin = pred >= threshold
    label_bin = label >= 0.5

    result = np.zeros((*pred.shape, 3), dtype=np.uint8)
    tn = (~pred_bin) & (~label_bin)
    tp = pred_bin & label_bin
    fp = pred_bin & (~label_bin)
    fn = (~pred_bin) & label_bin

    result[tn] = [0, 0, 0]
    result[tp] = [255, 255, 255]
    result[fp] = [255, 0, 0]
    result[fn] = [0, 255, 0]
    return result


def _build_model(cfg: Dict[str, Any]) -> torch.nn.Module:
    model_cfg = cfg["model"]
    module = import_module(model_cfg["module"])
    cls = getattr(module, model_cfg["class_name"])
    kwargs = model_cfg.get("args", {}) or {}
    return cls(**kwargs)


def _wrap_model(model: torch.nn.Module, train_cfg: Dict[str, Any]) -> torch.nn.Module:
    multi_gpu = train_cfg.get("multi_gpu")
    if not multi_gpu:
        return model

    device_ids = (
        [int(i) for i in multi_gpu]
        if isinstance(multi_gpu, (list, tuple))
        else [int(i) for i in str(multi_gpu).split(",")]
    )
    return torch.nn.DataParallel(model, device_ids=device_ids)


def _bi_forward(model, *args, **kwargs):
    module = model.module if hasattr(model, "module") else model
    return module.bi_forward(*args, **kwargs)


def _tta_aggregate(net, imgs_A, imgs_B, base_pred):
    preds = base_pred.clone()
    for dims in ([2], [3], [2, 3]):
        imgs_A_flip = torch.flip(imgs_A, dims)
        imgs_B_flip = torch.flip(imgs_B, dims)
        yc_flip = _bi_forward(net, imgs_A_flip, imgs_B_flip)
        yc_flip = torch.flip(yc_flip, dims)
        preds += torch.sigmoid(yc_flip)
    preds = preds / 4.0
    return preds


def _resolve_checkpoint_path(cfg: Dict[str, Any], cli_checkpoint: Optional[str]) -> Path:
    if cli_checkpoint:
        return Path(cli_checkpoint)

    load_path = cfg.get("training", {}).get("load_path")
    if load_path:
        return Path(load_path)

    ckpt_dir = cfg.get("paths", {}).get("checkpoints")
    if ckpt_dir:
        return Path(ckpt_dir) / "best_model.pth"

    return Path("/root/autodl-tmp/checkpoints/CLIP_RegionAware/best_model.pth")


def evaluate(cfg: Dict[str, Any], args: argparse.Namespace) -> None:
    device = _get_device(args.device)

    dataset_module, dataset = _build_dataset_and_module(cfg, args.split)

    split_cfg = cfg["dataset"].get(args.split, {}) or {}
    result_map_dir = Path("/root/autodl-tmp/CLCD_result_pictures")
    # result_map_dir = Path("/root/autodl-tmp/Levir_result_pictures")
    # result_map_dir = Path("/root/autodl-tmp/SECOND_result_pictures")
    # result_map_dir = Path("/root/autodl-tmp/SAM_LoRA_result_pictures")
    result_map_dir.mkdir(parents=True, exist_ok=True)

    train_cfg = cfg.get("training", {})
    batch_size = int(args.batch_size or train_cfg.get("val_batch_size", 1))
    num_workers = int(args.num_workers or train_cfg.get("num_workers", 4))

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = _build_model(cfg)
    checkpoint_path = _resolve_checkpoint_path(cfg, args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = (PROJECT_ROOT / checkpoint_path).resolve()

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    model = _wrap_model(model, train_cfg)
    model.to(device=device)
    model.eval()

    save_dir = Path(args.save_dir) if args.save_dir else cfg.get("paths", {}).get("results")
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    val_loss = AverageMeter()
    F1_meter = AverageMeter()
    IoU_meter = AverageMeter()
    Acc_meter = AverageMeter()
    Pre_meter = AverageMeter()
    Rec_meter = AverageMeter()

    n_images = 0
    global_tp = 0.0
    global_fp = 0.0
    global_fn = 0.0
    global_tn = 0.0
    global_total = 0.0

    saved = 0
    saved_bin = 0
    result_name_counter: dict[str, int] = defaultdict(int)
    start = time.time()

    with torch.no_grad():
        for vi, data in enumerate(loader):
            if isinstance(data, (list, tuple)) and len(data) == 4:
                imgs_A, imgs_B, labels, names = data
            else:
                imgs_A, imgs_B, labels = data
                names = [None] * int(imgs_A.shape[0])
            imgs_A = imgs_A.to(device).float()
            imgs_B = imgs_B.to(device).float()

            labels = labels.to(device).float().unsqueeze(1)

            logits = _bi_forward(model, imgs_A, imgs_B)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            yc = torch.sigmoid(logits)
            if args.tta:
                yc = _tta_aggregate(model, imgs_A, imgs_B, yc)

            val_loss.update(loss.detach().cpu().item())

            preds = yc.detach().cpu().numpy()  # [B,1,H,W]
            labels_np = labels.detach().cpu().numpy()  # [B,1,H,W]

            for b in range(preds.shape[0]):
                pred = preds[b].squeeze()
                lab = labels_np[b].squeeze()
                pred_bin = pred >= 0.5
                lab_bin = lab >= 0.5
                global_tp += float((pred_bin & lab_bin).sum())
                global_fp += float((pred_bin & ~lab_bin).sum())
                global_fn += float((~pred_bin & lab_bin).sum())
                global_tn += float((~pred_bin & ~lab_bin).sum())
                global_total += float(pred_bin.size)
                acc, precision, recall, f1, iou = accuracy(pred, lab)
                Acc_meter.update(acc)
                Pre_meter.update(precision)
                Rec_meter.update(recall)
                F1_meter.update(f1)
                IoU_meter.update(iou)
                n_images += 1

                sample_name = _resolve_sample_name(
                    dataset=dataset,
                    names=names,
                    batch_index=vi * batch_size,
                    sample_index_in_batch=b,
                    fallback_index=n_images - 1,
                )
                base_name = _sanitize_filename(Path(os.path.basename(sample_name)).stem)

                if args.save_pred and save_dir is not None and saved < int(args.save_max):
                    pred_gray = dataset_module.Index2Color(pred)
                    out_path = save_dir / f"{cfg['experiment']['name']}_{args.split}_{n_images:06d}_{base_name}.png"
                    if args.save_binary:
                        thr = float(args.binary_threshold)
                        pred_bin = (pred >= thr).astype(np.uint8) * 255
                        io.imsave(out_path, pred_bin)
                        saved_bin += 1
                    else:
                        io.imsave(out_path, pred_gray)
                    saved += 1

                result_name_counter[base_name] += 1
                result_name = f"{base_name}-result-{result_name_counter[base_name]:04d}"
                result_map = _extract_tp_tn_fp_fn_map(pred, lab, float(args.binary_threshold))
                io.imsave(result_map_dir / f"{result_name}.png", result_map)

    elapsed = time.time() - start

    if n_images == 0:
        raise RuntimeError("No samples were evaluated. Check your dataset split and paths.")

    global_precision = global_tp / (global_tp + global_fp + 1e-10)
    global_recall = global_tp / (global_tp + global_fn + 1e-10)
    global_f1 = 2 * global_precision * global_recall / (global_precision + global_recall + 1e-10)
    global_iou = global_tp / (global_tp + global_fp + global_fn + 1e-10)
    global_acc = (global_tp + global_tn) / (global_total + 1e-10)

    print("=" * 60)
    print(f"Config: {cfg.get('config_path')}")
    print(f"Split: {args.split}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"TTA: {bool(args.tta)}")
    print(f"Num images: {n_images}")
    print(f"Elapsed: {elapsed:.1f}s")
    if args.save_pred and result_map_dir is not None:
        print(f"Saved TP/TN/FP/FN result maps -> {result_map_dir}")
    if missing:
        print(f"NOTE: missing keys when loading state_dict (strict=False): {len(missing)}")
    if unexpected:
        print(f"NOTE: unexpected keys when loading state_dict (strict=False): {len(unexpected)}")
    print("-" * 60)
    print(f"Loss: {val_loss.average():.4f}")
    print(f"Acc: {Acc_meter.average():.6f}")
    print(f"Precision: {Pre_meter.average():.6f}")
    print(f"Recall: {Rec_meter.average():.6f}")
    print(f"F1: {F1_meter.average():.6f}")
    print(f"IoU: {IoU_meter.average():.6f}")
    print("-" * 60)
    print("Global metrics computed from accumulated TP/FP/FN/TN")
    print(f"Global TP: {global_tp:.0f}")
    print(f"Global FP: {global_fp:.0f}")
    print(f"Global FN: {global_fn:.0f}")
    print(f"Global TN: {global_tn:.0f}")
    print(f"Global Acc: {global_acc:.6f}")
    print(f"Global Precision: {global_precision:.6f}")
    print(f"Global Recall: {global_recall:.6f}")
    print(f"Global F1: {global_f1:.6f}")
    print(f"Global IoU: {global_iou:.6f}")
    print("=" * 60)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    cfg = load_config(config_path, args.options)
    if args.tta is None:
        args.tta = bool(cfg.get("training", {}).get("tta", False))
    evaluate(cfg, args)


if __name__ == "__main__":
    main()
