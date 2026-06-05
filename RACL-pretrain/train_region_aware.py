from __future__ import annotations

import argparse
import copy
import time
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from scipy import stats
from skimage import io

from config import load_config, PROJECT_ROOT
from loss import TripletLoss, InfoNCE, loss_change_sparsity
from region_aware_loss import RegionAwareLossModule
from utils import AverageMeter, binary_accuracy as accuracy
from data_parallel import BalancedDataParallel


FIXED_TRIPLET_WEIGHT = 1.0
FIXED_INFONCE_WEIGHT = 0.2
FIXED_CHANGE_SPARSITY_WEIGHT = 2.0
FIXED_CHANGE_SPARSITY_THRESHOLD = 0.2
QUADRUPLET_LOSS_ALIASES = {'quadruplet', 'quadruplet_loss', 'triplet', 'region_aware'}
INFONCE_LOSS_ALIASES = {'infoNCE', 'InfoNCE', 'infonce'}
CHANGE_SPARSITY_LOSS_ALIASES = {'change_sparsity', 'change_sparsity_loss'}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Region-Aware Contrastive Learning Training')
    parser.add_argument(
        '--config',
        type=str,
        default='clip_region_aware.yaml',
        help='Path to the config file',
    )
    parser.add_argument(
        '--options',
        nargs='*',
        default=None,
        help='Override config fields via key=value pairs',
    )
    return parser.parse_args()


def run_training(cfg: Dict[str, Any]) -> None:
    """Main training programme"""
    _prepare_directories(cfg)
    device = _get_device(cfg['training'])
    writer = SummaryWriter(str(cfg['paths']['logs']))
    
    # Create a text log file
    log_file = cfg['paths']['logs'] / 'training_log.txt'
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"Training started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Config: {cfg['config_path']}\n")
        f.write("=" * 60 + "\n")

    # Building datasets and data loaders
    dataset_module, datasets = _build_datasets(cfg)
    loaders = _build_dataloaders(datasets, cfg['training'])

    # Build a model
    model = _build_model(cfg['model'])
    load_path = cfg['training'].get('load_path')
    if load_path:
        state_dict = torch.load(load_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        print(f'Loaded pretrained weights from: {load_path}')
    
    model = _wrap_model(model, cfg['training'])
    model.to(device=device)

    # Integrity check
    _sanity_check(model, device)

    # Training loop
    best_model = _train_loop(
        loaders,
        model,
        cfg,
        writer,
        device,
        dataset_module,
        log_file,
    )
    writer.close()
    
    # Save best model
    if best_model is not None:
        save_path = cfg['paths']['checkpoints'] / 'best_model.pth'
        # Get the actual model (handle DataParallel wrapping)
        model_to_save = best_model.module if hasattr(best_model, 'module') else best_model
        torch.save(model_to_save.state_dict(), save_path)
        print(f'Best model saved to: {save_path}')
        
        # Write to log file
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"Best model saved to: {save_path}\n")
    
    # Write training end information
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write(f"Training finished at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    print('Training finished.')


def _train_loop(loaders, net, cfg, writer, device, dataset_module, log_file):
    """
    Training loop - integrate original S2C loss and region-aware contrastive learning
    """
    train_cfg = cfg['training']
    epochs = train_cfg['epochs']

    bestF = 0.0
    bestacc = 0.0
    bestIoU = 0.0
    net_reserved = None

    curr_epoch = 0
    begin_time = time.time()
    all_iters = float(len(loaders['train']) * epochs)

    # ============ Loss functions ============
    loss_triplet = TripletLoss()
    loss_infonce = InfoNCE()
    
    # Region-aware loss module 
    region_aware_cfg = train_cfg.get('region_aware', {})
    region_aware_loss = RegionAwareLossModule(
        change_threshold=region_aware_cfg.get('change_threshold', 0.3),
        unchanged_threshold=region_aware_cfg.get('unchanged_threshold', 0.7),
        warmup_epochs=region_aware_cfg.get('warmup_epochs', 5),
        pull_margin=region_aware_cfg.get('pull_margin', 0.2),
        push_margin=region_aware_cfg.get('push_margin', 0.5),
        pull_weight=region_aware_cfg.get('pull_weight', 0.5),  # Unchanged region weight
        push_weight=region_aware_cfg.get('push_weight', 1.0),  # Changed region weight
    )
    
    # Initialize model reference 
    use_region_aware = train_cfg.get('use_region_aware', True)
    if use_region_aware:
        region_aware_loss.init_model(net)
        region_aware_loss.to(device)  # Move learnable parameters to device

    # ============ Optimizer ============
    params = list(filter(lambda p: p.requires_grad, net.parameters()))
    # Add learnable parameters for region-aware loss
    if use_region_aware:
        params += list(region_aware_loss.parameters())
    
    optimizer = optim.SGD(
        params,
        train_cfg['lr'],
        weight_decay=5e-4,
        momentum=0.9,
        nesterov=True
    )

    # ============ Fixed loss weights ============
    triplet_weight = FIXED_TRIPLET_WEIGHT
    infonce_weight = FIXED_INFONCE_WEIGHT
    change_sparsity_weight = FIXED_CHANGE_SPARSITY_WEIGHT
    change_sparsity_thred = FIXED_CHANGE_SPARSITY_THRESHOLD

    enabled_losses = set(train_cfg.get('loss_terms', ['quadruplet', 'infoNCE', 'change_sparsity']))
    use_quadruplet_loss = bool(enabled_losses & QUADRUPLET_LOSS_ALIASES)
    use_infonce_loss = bool(enabled_losses & INFONCE_LOSS_ALIASES)
    use_change_sparsity_loss = bool(enabled_losses & CHANGE_SPARSITY_LOSS_ALIASES)
    if not enabled_losses:
        raise ValueError('`loss_terms` must contain at least one entry.')
    if not (use_quadruplet_loss or use_infonce_loss or use_change_sparsity_loss):
        raise ValueError(f'No supported loss terms found in `loss_terms`: {sorted(enabled_losses)}')

    # ============ Training loop ================
    while curr_epoch < epochs:
        _clear_cuda_cache(device)
        net.train()
        start = time.time()
        
        # Loss meters
        loss_triplet_meter = AverageMeter()
        loss_quadruplet_meter = AverageMeter()
        loss_infonce_meter = AverageMeter()
        loss_change_sparsity_meter = AverageMeter()
        loss_region_meter = AverageMeter()
        loss_pull_meter = AverageMeter()
        loss_push_meter = AverageMeter()

        curr_iter = curr_epoch * len(loaders['train'])
        
        for i, data in enumerate(loaders['train']):
            running_iter = curr_iter + i + 1
            _adjust_lr(optimizer, running_iter, all_iters, train_cfg)
            
            imgs_A, imgs_B, imgs_A_aug, imgs_B_aug = data
            imgs_A = imgs_A.to(device).float()
            imgs_B = imgs_B.to(device).float()
            imgs_A_aug = imgs_A_aug.to(device).float()
            imgs_B_aug = imgs_B_aug.to(device).float()

            optimizer.zero_grad()
            
            # ============ Forward propagation ============
            y1 = net(imgs_A)        # Phase 1 original features
            y2 = net(imgs_B)        # Phase 2 original features
            y1_ = net(imgs_A_aug)   # Phase 1 augmented features
            y2_ = net(imgs_B_aug)   # Phase 2 augmented features

            # ============ Calculate losses ============
            total_loss = torch.tensor(0.0, device=device)
            loss_tri = torch.tensor(0.0, device=device)
            loss_region = torch.tensor(0.0, device=device)
            region_result = None

            # 1. Quadruplet loss = TripletLoss * 1.0 + RegionAwareLoss
            if use_quadruplet_loss:
                loss_tri = loss_triplet(y1, y1_, y2, y2_)
                loss_triplet_meter.update(loss_tri.detach().cpu().item())

                if use_region_aware:
                    region_result = region_aware_loss(
                        feat_a=y1_,
                        feat_b=y2_,
                        img_a=imgs_A,
                        img_b=imgs_B,
                        epoch=curr_epoch
                    )
                    loss_region = region_result['loss']
                    loss_region_meter.update(loss_region.detach().cpu().item())
                    loss_pull_meter.update(region_result['pull_loss'].detach().cpu().item())
                    loss_push_meter.update(region_result['push_loss'].detach().cpu().item())

                loss_quadruplet = loss_tri * triplet_weight + loss_region
                total_loss = total_loss + loss_quadruplet
                loss_quadruplet_meter.update(loss_quadruplet.detach().cpu().item())
            
            # 2. InfoNCE loss (cross-phase location correspondence)
            if use_infonce_loss:
                loss_nce = loss_infonce(y1, y2)
                total_loss = total_loss + loss_nce * infonce_weight
                loss_infonce_meter.update(loss_nce.detach().cpu().item())
            
            # 3. Change sparsity loss
            if use_change_sparsity_loss:
                # Need change detection output, use bi_forward
                yc = _bi_forward(net, imgs_A, imgs_B)
                change_sparsity_loss = loss_change_sparsity(yc, T=change_sparsity_thred)
                total_loss = total_loss + change_sparsity_loss * change_sparsity_weight
                loss_change_sparsity_meter.update(change_sparsity_loss.detach().cpu().item())

            # ============ Backward propagation ============
            total_loss.backward()
            optimizer.step()

            # ============ Log ============
            curr_time = time.time() - start
            if (i + 1) % train_cfg['print_freq'] == 0:
                log_str = (
                    f'[epoch {curr_epoch}] [iter {i + 1}/{len(loaders["train"])} {curr_time:.1f}s] '
                    f'[lr {optimizer.param_groups[0]["lr"]:.6f}] '
                )

                if use_quadruplet_loss:
                    log_str += f'[quadruplet {loss_quadruplet_meter.val:.4f}]'
                
                if use_infonce_loss:
                    log_str += f' [infoNCE {loss_infonce_meter.val:.4f}]'
                
                if use_change_sparsity_loss:
                    log_str += f' [change_sparsity {loss_change_sparsity_meter.val:.4f}]'
                
                print(log_str)
                
                # Write to txt log
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(log_str + '\n')
                
                # TensorBoard
                if use_quadruplet_loss:
                    writer.add_scalar('loss/quadruplet', loss_quadruplet_meter.val, running_iter)
                writer.add_scalar('lr', optimizer.param_groups[0]['lr'], running_iter)
                
                if use_infonce_loss:
                    writer.add_scalar('loss/infoNCE', loss_infonce_meter.val, running_iter)
                
                if use_change_sparsity_loss:
                    writer.add_scalar('loss/change_sparsity', loss_change_sparsity_meter.val, running_iter)
                
        # ============ Validation ============
        val_F, val_acc, val_IoU, val_log = _validate(
            loaders['val'],
            net,
            cfg,
            device,
            dataset_module,
            writer,
            curr_epoch,
            TTA=train_cfg.get('tta', False),
        )
        
        # Write validation log
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(val_log + '\n')

        save_path_epoch = cfg['paths']['checkpoints'] / f'epoch_{curr_epoch}.pth'
        model_to_save = net.module if hasattr(net, 'module') else net
        torch.save(model_to_save.state_dict(), save_path_epoch)
        print(f'Epoch model saved to: {save_path_epoch}')
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f'Epoch model saved to: {save_path_epoch}\n')
        
        if val_F > bestF:
            bestF = val_F
            bestacc = val_acc
            bestIoU = val_IoU
            net_reserved = copy.deepcopy(net)
            
                # Save best model
            save_path = cfg['paths']['checkpoints'] / 'best_model.pth'
            model_to_save = net.module if hasattr(net, 'module') else net
            torch.save(model_to_save.state_dict(), save_path)
            print(f'Best model saved to: {save_path}')
            
            # Write to log file
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f'Best model saved to: {save_path}\n')

        epoch_summary = (f'[epoch {curr_epoch}/{epochs} {time.time() - begin_time:.1f}s] '
                         f'Best val acc {bestacc * 100:.2f} F1 {bestF * 100:.2f} IoU {bestIoU * 100:.2f}')
        print(epoch_summary)
        
        # Write to txt log file
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(epoch_summary + '\n')
            f.write("-" * 60 + "\n")
        
        curr_epoch += 1

    return net_reserved


# ============ Helper functions (consistent with original engine.py) ============

def _prepare_directories(cfg: Dict[str, Any]) -> None:
    for key in ('checkpoints', 'results', 'logs'):
        path: Path = cfg['paths'][key]
        path.mkdir(parents=True, exist_ok=True)


def _get_device(train_cfg: Dict[str, Any]) -> torch.device:
    if train_cfg.get('gpu', True) and torch.cuda.is_available():
        return torch.device('cuda', int(train_cfg.get('dev_id', 0)))
    return torch.device('cpu')


def _build_model(model_cfg: Dict[str, Any]) -> nn.Module:
    module = import_module(model_cfg['module'])
    cls = getattr(module, model_cfg['class_name'])
    kwargs = model_cfg.get('args', {}) or {}
    return cls(**kwargs)


def _build_datasets(cfg: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    dataset_cfg = cfg['dataset']
    module = import_module(dataset_cfg['module'])
    dataset_cls = getattr(module, dataset_cfg.get('class_name', 'RS'))
    _override_dataset_root(module, dataset_cfg, cfg.get('project_root'))

    datasets = {}
    for split in ('train', 'val', 'test'):
        split_cfg = dataset_cfg.get(split)
        if not split_cfg:
            raise ValueError(f'Missing dataset split config: {split}')
        kwargs = split_cfg.get('kwargs', {}) or {}
        datasets[split] = dataset_cls(split_cfg.get('split', split), **kwargs)

    return module, datasets


def _build_dataloaders(datasets: Dict[str, Any], train_cfg: Dict[str, Any]) -> Dict[str, DataLoader]:
    num_workers = int(train_cfg.get('num_workers', 4))
    pin_memory = bool(train_cfg.get('gpu', True) and torch.cuda.is_available())
    loaders = {
        'train': DataLoader(
            datasets['train'],
            batch_size=train_cfg['train_batch_size'],
            num_workers=num_workers,
            shuffle=True,
            pin_memory=pin_memory,
        ),
        'val': DataLoader(
            datasets['val'],
            batch_size=train_cfg['val_batch_size'],
            num_workers=num_workers,
            shuffle=False,
            pin_memory=pin_memory,
        ),
        'test': DataLoader(
            datasets['test'],
            batch_size=train_cfg['val_batch_size'],
            num_workers=num_workers,
            shuffle=False,
            pin_memory=pin_memory,
        ),
    }
    return loaders


def _wrap_model(model: nn.Module, train_cfg: Dict[str, Any]) -> nn.Module:
    multi_gpu = train_cfg.get('multi_gpu')
    if not multi_gpu:
        return model

    device_ids = [int(i) for i in multi_gpu] if isinstance(multi_gpu, (list, tuple)) else [int(i) for i in str(multi_gpu).split(',')]
    if train_cfg.get('use_balanced_dp', False):
        model = BalancedDataParallel(0, model, device_ids=device_ids)
    else:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    return model


def _sanity_check(model: nn.Module, device: torch.device) -> None:
    model.eval()
    dsize = (1, 3, 512, 512)
    x1 = torch.randn(dsize, device=device)
    x2 = torch.randn(dsize, device=device)
    with torch.no_grad():
        model(x1)
        model(x2)
        _bi_forward(model, x1, x2)
    _clear_cuda_cache(device)


def _bi_forward(model, *args, **kwargs):
    module = model.module if hasattr(model, 'module') else model
    return module.bi_forward(*args, **kwargs)


def _adjust_lr(optimizer, curr_iter, all_iter, train_cfg):
    scale_running_lr = ((1.0 - float(curr_iter) / all_iter) ** 1.5)
    running_lr = train_cfg['lr'] * scale_running_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = running_lr


def _clear_cuda_cache(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.cuda.empty_cache()


def _override_dataset_root(module, dataset_cfg: Dict[str, Any], project_root: Path | None) -> None:
    root_override = dataset_cfg.get('root')
    if not root_override:
        return
    root_path = Path(root_override)
    if not root_path.is_absolute():
        base = project_root or Path.cwd()
        root_path = base / root_path
    root_str = str(root_path)
    setter = getattr(module, 'set_root', None)
    if callable(setter):
        setter(root_str)
    elif hasattr(module, 'root'):
        setattr(module, 'root', root_str)
    else:
        raise AttributeError(
            f'Dataset module {module.__name__} does not support root override.'
        )


def _validate(val_loader, net, cfg, device, dataset_module, writer, curr_epoch=0, TTA=False):
    """Validation function"""
    net.eval()
    _clear_cuda_cache(device)
    start = time.time()
    paths = cfg['paths']

    val_loss = AverageMeter()
    F1_meter = AverageMeter()
    IoU_meter = AverageMeter()
    Acc_meter = AverageMeter()
    Pre_meter = AverageMeter()
    Rec_meter = AverageMeter()

    for vi, data in enumerate(val_loader):
        imgs_A, imgs_B, labels = data
        imgs_A = imgs_A.to(device).float()
        imgs_B = imgs_B.to(device).float()
        labels = labels.to(device).float().unsqueeze(1)

        with torch.no_grad():
            yc = _bi_forward(net, imgs_A, imgs_B)
            yc = F.sigmoid(yc)
            if TTA:
                yc = _tta_aggregate(net, imgs_A, imgs_B, yc)
            loss = F.binary_cross_entropy_with_logits(yc, labels)
        val_loss.update(loss.detach().cpu().item())

        preds = yc.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        for (pred, label) in zip(preds, labels_np):
            acc, precision, recall, F1, IoU = accuracy(pred, label)
            F1_meter.update(F1)
            Acc_meter.update(acc)
            IoU_meter.update(IoU)
            Pre_meter.update(precision)
            Rec_meter.update(recall)

        if vi == 0:
            pred_color = dataset_module.Index2Color(preds[0].squeeze())
            save_path = paths['results'] / f"{cfg['experiment']['name']}_e{curr_epoch}.png"
            io.imsave(save_path, pred_color)

    curr_time = time.time() - start
    val_log = (f'{curr_time:.1f}s Val loss {val_loss.average():.2f} '
               f'Acc {Acc_meter.average() * 100:.2f} F {F1_meter.average() * 100:.2f} '
               f'Pre {Pre_meter.average() * 100:.2f} Rec {Rec_meter.average() * 100:.2f}')
    print(val_log)

    writer.add_scalar('val_loss', val_loss.average(), curr_epoch)
    writer.add_scalar('val_Accuracy', Acc_meter.average(), curr_epoch)
    writer.add_scalar('val_F1', F1_meter.average(), curr_epoch)
    return F1_meter.avg, Acc_meter.avg, IoU_meter.avg, val_log


def _evaluate(test_loader, net, cfg, device, dataset_module, TTA=False):
    """Test evaluation function"""
    net.eval()
    _clear_cuda_cache(device)

    TP_total, TN_total, FP_total, FN_total = 0, 0, 0, 0
    
    for vi, data in enumerate(test_loader):
        imgs_A, imgs_B, labels = data
        imgs_A = imgs_A.to(device).float()
        imgs_B = imgs_B.to(device).float()
        labels = labels.to(device).float().unsqueeze(1)

        with torch.no_grad():
            yc = _bi_forward(net, imgs_A, imgs_B)
            yc = F.sigmoid(yc)
            if TTA:
                yc = _tta_aggregate(net, imgs_A, imgs_B, yc)

        preds = yc.detach().cpu().numpy() > 0.5
        labels_np = labels.detach().cpu().numpy()
        
        for (pred, GT) in zip(preds, labels_np):
            pred = (pred >= 0.5)
            label = (GT >= 0.5)
            TP_total += (pred * label).sum()
            FP_total += (pred * (~label)).sum()
            FN_total += ((~pred) * label).sum()
            TN_total += ((~pred) * (~label)).sum()

    precision = TP_total / (TP_total + FP_total + 1e-10)
    recall = TP_total / (TP_total + FN_total + 1e-10)
    IoU = TP_total / (FP_total + TP_total + FN_total + 1e-10)
    acc = (TP_total + TN_total) / (TP_total + FP_total + FN_total + TN_total + 1e-10)
    F1 = stats.hmean([precision, recall])
    
    print(f'Test eval: Acc {acc * 100:.2f}, Pre {precision * 100:.2f}, '
          f'Rec {recall * 100:.2f}, F1 {F1 * 100:.2f}, IoU {IoU * 100:.2f}')
    return F1, acc, IoU


def _tta_aggregate(net, imgs_A, imgs_B, base_pred):
    """Test time augmentation aggregation (TTA)"""
    preds = base_pred.clone()
    for dims in ([2], [3], [2, 3]):
        imgs_A_flip = torch.flip(imgs_A, dims)
        imgs_B_flip = torch.flip(imgs_B, dims)
        yc_flip = _bi_forward(net, imgs_A_flip, imgs_B_flip)
        yc_flip = torch.flip(yc_flip, dims)
        preds += F.sigmoid(yc_flip)
    preds = preds / 4.0
    return preds


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    
    cfg = load_config(config_path, args.options)
    run_training(cfg)


if __name__ == '__main__':
    main()
