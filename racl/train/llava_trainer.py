import os
import json
import numpy as np
import torch
import torch.nn as nn
import datetime
from functools import partial

from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs, GradientAccumulationPlugin
from torch.utils.data import Dataset, Sampler, DataLoader
import torch.nn.functional as F

from transformers import Trainer
from transformers.trainer import is_sagemaker_mp_enabled, get_parameter_names, has_length, logger, is_accelerate_available, is_datasets_available
try:
    from transformers.trainer import ALL_LAYERNORM_LAYERS
except ImportError:
    from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_utils import seed_worker, PREFIX_CHECKPOINT_DIR
from transformers.trainer_pt_utils import get_length_grouped_indices as get_length_grouped_indices_hf
from transformers.trainer_pt_utils import AcceleratorConfig
from typing import Dict, List, Optional
from datetime import timedelta

if is_accelerate_available():
    from accelerate import skip_first_batches, InitProcessGroupKwargs

if is_datasets_available():
    import datasets

import pdb
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import rank0_print


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, "no ignore status")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match) and t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_variable_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)
    megabatch_size = world_size * batch_size
    megabatches = [sorted_indices[i : i + megabatch_size] for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: indices[i], reverse=True) for megabatch in megabatches]
    shuffled_indices = [i for megabatch in megabatches for i in megabatch]
    world_batch_size = world_size * batch_size
    batches = [shuffled_indices[i : i + world_batch_size] for i in range(0, len(lengths), world_batch_size)]
    batch_indices = torch.randperm(len(batches), generator=generator)
    batches = [batches[i] for i in batch_indices]

    return [i for batch in batches for i in batch]


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    """
    Return a list of indices so that each slice of `batch_size` consecutive indices correspond to elements of similar
    lengths. To do this, the indices are:

    - randomly permuted
    - grouped in mega-batches of size `mega_batch_mult * batch_size`
    - reorder by length in each mega-batch

    The result is the concatenation of all mega-batches, with the batch of `batch_size` containing the element of
    maximum length placed first, so that an OOM happens sooner rather than later.
    """

    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices_hf(lengths, batch_size * world_size, generator=generator)

    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    """
    Return a list of indices so that each slice of `batch_size` consecutive indices correspond to elements of similar
    lengths. To do this, the indices are:

    - randomly permuted
    - grouped in mega-batches of size `mega_batch_mult * batch_size`
    - reorder by length in each mega-batch

    The result is the concatenation of all mega-batches, with the batch of `batch_size` containing the element of
    maximum length placed first, so that an OOM happens sooner rather than later.
    """

    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


def get_length_grouped_indices_auto_single(lengths, batch_size, world_size, generator=None):
    indices = get_length_grouped_indices_hf(lengths, batch_size * world_size, generator=generator)

    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size] for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    batch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in batch_indices]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


def get_modality_length_grouped_indices_auto(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices_auto_single(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices_auto_single(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices_auto_single(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    # FIXME: Hard code to avoid last batch mixed with different modalities
    # if len(additional_batch) > 0:
    #     megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        variable_length: bool = False,
        group_by_modality: bool = False,
        group_by_modality_auto: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.variable_length = variable_length
        self.group_by_modality = group_by_modality
        self.group_by_modality_auto = group_by_modality_auto

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.variable_length:
            assert not self.group_by_modality, "Variable length grouping is not supported with modality grouping."
            indices = get_variable_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            if self.group_by_modality:
                indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
            elif self.group_by_modality_auto:
                indices = get_modality_length_grouped_indices_auto(self.lengths, self.batch_size, self.world_size, generator=self.generator)
            else:
                indices = get_length_grouped_indices_auto_single(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


class LLaVATrainer(Trainer):
    def __init__(self, *args, T=2.0, task_num=2, **kwargs):
        # transformers >= 4.45 renamed `tokenizer` to `processing_class`
        if "tokenizer" in kwargs and "processing_class" not in kwargs:
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        super().__init__(*args, **kwargs)
        self.T = T  # Softmax温度参数
        self.task_num = task_num  # 任务数量
        self.train_loss_buffer = None  # 存储每个任务的历史损失
        self.epoch = 0  # 训练的epoch计数器

    def create_accelerator_and_postprocess(self):
        grad_acc_kwargs = {"num_steps": self.args.gradient_accumulation_steps}
        grad_acc_kwargs["sync_with_dataloader"] = False
        gradient_accumulation_plugin = GradientAccumulationPlugin(**grad_acc_kwargs)

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        rank0_print("Setting NCCL timeout to INF to avoid running errors.")

        # create accelerator object (compatible with new accelerate versions)
        deepspeed_plugin = getattr(self.args, 'deepspeed_plugin', None)
        
        self.accelerator = Accelerator(
            deepspeed_plugin=deepspeed_plugin, 
            gradient_accumulation_plugin=gradient_accumulation_plugin, 
            kwargs_handlers=[accelerator_kwargs]
        )
        # some Trainer classes need to use `gather` instead of `gather_for_metrics`, thus we store a flag
        self.gather_function = self.accelerator.gather_for_metrics

        # deepspeed and accelerate flags covering both trainer args and accelerate launcher
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None
        self.is_tp_enabled = False  # Tensor Parallelism not used

        # post accelerator creation setup
        if self.is_fsdp_enabled:
            fsdp_plugin = self.accelerator.state.fsdp_plugin
            fsdp_plugin.limit_all_gathers = self.args.fsdp_config.get("limit_all_gathers", fsdp_plugin.limit_all_gathers)
            if is_accelerate_available("0.23.0"):
                fsdp_plugin.activation_checkpointing = self.args.fsdp_config.get("activation_checkpointing", fsdp_plugin.activation_checkpointing)
                if fsdp_plugin.activation_checkpointing and self.args.gradient_checkpointing:
                    raise ValueError("The activation_checkpointing in FSDP config and the gradient_checkpointing in training arg " "can't be set to True simultaneously. Please use FSDP's activation_checkpointing logic " "when using FSDP.")

        if self.is_deepspeed_enabled and getattr(self.args, "hf_deepspeed_config", None) is None:
            self.propagate_args_to_deepspeed()

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_length:
            lengths = self.train_dataset.lengths
            return LengthGroupedSampler(
                # self.args.train_batch_size * self.args.gradient_accumulation_steps, # TODO: seems that we should not have gradient_accumulation_steps
                self.args.train_batch_size,
                # world_size=self.args.world_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,  # TODO: seems that this may work?
                lengths=lengths,
            )
        elif self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                # self.args.train_batch_size * self.args.gradient_accumulation_steps, # TODO: seems that we should not have gradient_accumulation_steps
                self.args.train_batch_size,
                # world_size=self.args.world_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,  # TODO: seems that this may work?
                lengths=lengths,
                group_by_modality=True,
            )
        elif self.args.group_by_modality_length_auto:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                # self.args.train_batch_size * self.args.gradient_accumulation_steps, # TODO: seems that we should not have gradient_accumulation_steps
                self.args.train_batch_size,
                # world_size=self.args.world_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,  # TODO: seems that this may work?
                lengths=lengths,
                group_by_modality_auto=True,
            )
        elif self.args.group_by_varlen:
            lengths = self.train_dataset.lengths
            return LengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                # self.args.train_batch_size, # TODO: seems that we should have gradient_accumulation_steps
                # world_size=self.args.world_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,  # TODO: seems that this may work?
                lengths=lengths,
                variable_length=True,
            )
        else:
            return super()._get_train_sampler()

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = partial(
                seed_worker, 
                num_workers=self.args.dataloader_num_workers, 
                rank=self.args.process_index
            )
            dataloader_params["prefetch_factor"] = self.args.dataloader_num_workers * 2 if self.args.dataloader_num_workers != 0 else None

        dataloader = self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

        return dataloader

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            lr_mapper = {}
            if self.args.mm_projector_lr is not None:
                lr_mapper["mm_projector"] = self.args.mm_projector_lr
            if self.args.mm_vision_tower_lr is not None:
                lr_mapper["mm_vision_tower"] = self.args.mm_vision_tower_lr
            if len(lr_mapper) > 0:
                special_lr_parameters = [name for name, _ in opt_model.named_parameters() if any(module_keyword in name for module_keyword in lr_mapper)]
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in special_lr_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in special_lr_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                ]
                for module_keyword, lr in lr_mapper.items():
                    module_parameters = [name for name, _ in opt_model.named_parameters() if module_keyword in name]
                    optimizer_grouped_parameters.extend(
                        [
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in module_parameters and p.requires_grad)],
                                "weight_decay": self.args.weight_decay,
                                "lr": lr,
                            },
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in module_parameters and p.requires_grad)],
                                "weight_decay": 0.0,
                                "lr": lr,
                            },
                        ]
                    )
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def _save_checkpoint(self, model, trial, metrics=None):
        """保存checkpoint，支持完整断点重训"""
        # 检查是否只训练adapter部分 (不训练完整模型)
        adapter_only_parts = {"mm_mlp_adapter", "mm_change_detector", "mm_seg_head", "mm_vision_tower", "mm_tgcfa"}
        is_adapter_only = getattr(self.args, "tune_mm_mlp_adapter", False)
        if not is_adapter_only and hasattr(self.args, "mm_tunable_parts") and self.args.mm_tunable_parts:
            tunable_parts = set(self.args.mm_tunable_parts.split(","))
            # 如果所有可训练部分都是adapter，则为adapter-only模式
            is_adapter_only = tunable_parts.issubset(adapter_only_parts)
        
        if is_adapter_only:
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)
            os.makedirs(output_dir, exist_ok=True)

            # 保存 Adapter 权重 (只保存可训练部分)
            # 注意: vision_tower 只在 mm_vision_tower 在 tunable_parts 中时才保存
            keys_to_match = ["mm_projector", "change_detector", "seg_head", "tgcfa"]
            if hasattr(self.args, "mm_tunable_parts") and "mm_vision_tower" in self.args.mm_tunable_parts:
                keys_to_match.append("vision_tower")
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(["embed_tokens", "embed_in"])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, "mm_projector.bin"))
                
                # 保存训练状态 (trainer_state.json)
                self.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))
                
                # 保存 DWA 状态
                dwa_state = {
                    "train_loss_buffer": self.train_loss_buffer.cpu() if self.train_loss_buffer is not None else None,
                    "epoch": self.epoch,
                    "T": self.T,
                    "task_num": self.task_num,
                }
                torch.save(dwa_state, os.path.join(output_dir, "dwa_state.pt"))
                
                # 保存 RNG 状态
                rng_state = {
                    "python": torch.random.get_rng_state(),
                    "numpy": np.random.get_state(),
                    "cpu": torch.get_rng_state(),
                }
                if torch.cuda.is_available():
                    rng_state["cuda"] = torch.cuda.get_rng_state_all()
                torch.save(rng_state, os.path.join(output_dir, "rng_state.pth"))
            
            # 保存 optimizer 和 scheduler 状态
            if not self.is_deepspeed_enabled:
                if self.optimizer is not None:
                    torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                if self.lr_scheduler is not None:
                    torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
            else:
                # DeepSpeed: 仅保存优化器状态，不保存完整模型 (adapter权重已单独保存)
                # 使用 save_zero_checkpoint 仅保存优化器状态，避免保存完整模型
                try:
                    # 创建 DeepSpeed checkpoint 子目录
                    ds_checkpoint_dir = os.path.join(output_dir, "deepspeed")
                    os.makedirs(ds_checkpoint_dir, exist_ok=True)
                    # 仅保存优化器和训练状态，排除模型参数
                    self.model_wrapped.save_checkpoint(
                        ds_checkpoint_dir,
                        exclude_frozen_parameters=True  # 排除冻结参数，仅保存训练参数
                    )
                except Exception as e:
                    rank0_print(f"Warning: Failed to save DeepSpeed optimizer state: {e}")
                
            rank0_print(f"Checkpoint saved to {output_dir}")
        else:
            # transformers 版本兼容: 部分版本 _save_checkpoint(self, model, trial)
            # 旧版本可能含 metrics 参数
            try:
                super(LLaVATrainer, self)._save_checkpoint(model, trial)
            except TypeError:
                super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)
            # 额外保存 DWA 状态
            self._save_dwa_state(trial)
    
    def _save_dwa_state(self, trial=None):
        """单独保存 DWA 状态"""
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        
        if self.args.local_rank == 0 or self.args.local_rank == -1:
            os.makedirs(output_dir, exist_ok=True)  # 确保目录存在
            dwa_state = {
                "train_loss_buffer": self.train_loss_buffer.cpu() if self.train_loss_buffer is not None else None,
                "epoch": self.epoch,
                "T": self.T,
                "task_num": self.task_num,
            }
            torch.save(dwa_state, os.path.join(output_dir, "dwa_state.pt"))
            
            # 同时保存 RNG 状态
            rng_state = {
                "python": torch.random.get_rng_state(),
                "numpy": np.random.get_state(),
                "cpu": torch.get_rng_state(),
            }
            if torch.cuda.is_available():
                rng_state["cuda"] = torch.cuda.get_rng_state_all()
            torch.save(rng_state, os.path.join(output_dir, "rng_state.pth"))
    
    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        """从checkpoint恢复训练，包括DWA状态"""
        # 调用父类方法恢复模型和训练状态
        super()._load_from_checkpoint(resume_from_checkpoint, model)
        
        # 恢复 DWA 状态
        self._load_dwa_state(resume_from_checkpoint)
        
        # 恢复 RNG 状态
        self._load_rng_state(resume_from_checkpoint)
        
        # 恢复 optimizer 和 scheduler (非DeepSpeed模式)
        self._load_optimizer_and_scheduler(resume_from_checkpoint)
    
    def _load_dwa_state(self, checkpoint_path):
        """从checkpoint恢复DWA状态"""
        dwa_state_file = os.path.join(checkpoint_path, "dwa_state.pt")
        if os.path.exists(dwa_state_file):
            dwa_state = torch.load(dwa_state_file, map_location="cpu")
            if dwa_state["train_loss_buffer"] is not None:
                self.train_loss_buffer = dwa_state["train_loss_buffer"].to(self.args.device)
            else:
                self.train_loss_buffer = None
            self.epoch = dwa_state.get("epoch", 0)
            self.T = dwa_state.get("T", self.T)
            self.task_num = dwa_state.get("task_num", self.task_num)
            rank0_print(f"DWA state restored: epoch={self.epoch}, train_loss_buffer shape={self.train_loss_buffer.shape if self.train_loss_buffer is not None else None}")
        else:
            rank0_print(f"No DWA state found at {dwa_state_file}, starting fresh.")
    
    def _load_rng_state(self, checkpoint_path):
        """从checkpoint恢复RNG状态"""
        rng_state_file = os.path.join(checkpoint_path, "rng_state.pth")
        if os.path.exists(rng_state_file):
            rng_state = torch.load(rng_state_file, map_location="cpu")
            torch.random.set_rng_state(rng_state["python"])
            np.random.set_state(rng_state["numpy"])
            torch.set_rng_state(rng_state["cpu"])
            if torch.cuda.is_available() and "cuda" in rng_state:
                torch.cuda.set_rng_state_all(rng_state["cuda"])
            rank0_print(f"RNG state restored from {rng_state_file}")
    
    def _load_optimizer_and_scheduler(self, checkpoint_path):
        """从checkpoint恢复optimizer和scheduler"""
        # 对于非DeepSpeed模式，手动恢复
        if not self.is_deepspeed_enabled:
            optimizer_file = os.path.join(checkpoint_path, "optimizer.pt")
            scheduler_file = os.path.join(checkpoint_path, "scheduler.pt")
            
            if os.path.exists(optimizer_file) and self.optimizer is not None:
                self.optimizer.load_state_dict(torch.load(optimizer_file, map_location="cpu"))
                rank0_print(f"Optimizer state restored from {optimizer_file}")
            
            if os.path.exists(scheduler_file) and self.lr_scheduler is not None:
                self.lr_scheduler.load_state_dict(torch.load(scheduler_file, map_location="cpu"))
                rank0_print(f"Scheduler state restored from {scheduler_file}")
    

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        重写 Trainer 的 compute_loss 方法，添加 DWA 逻辑。
        """

        # captions 仅供 TGCFA trainer 使用；基础 trainer 需移除避免 forward 参数报错
        inputs.pop("captions", None)

        # 获取多个任务的损失值 (假设损失函数已经返回一个包含多个任务损失的字典)
        outputs = model(**inputs)
        losses = outputs["loss"] if isinstance(outputs, dict) else outputs[0] # 假设模型返回的是一个包含多个任务损失的字典
        if isinstance(losses, torch.Tensor):
            losses = [losses]
        else:
            losses = list(losses)
        actual_task_num = len(losses)
        if actual_task_num != self.task_num:
            rank0_print(f"DWA: adjusting task_num from {self.task_num} to {actual_task_num} based on model losses")
            self.task_num = actual_task_num
            self.train_loss_buffer = None
        
        # 第一次初始化 loss buffer
        if self.train_loss_buffer is None or self.train_loss_buffer.shape[0] != self.task_num:
            self.train_loss_buffer = torch.zeros((self.task_num, 2), device=self.args.device)
        
        # 计算每个任务的动态权重
        if self.epoch > 1:
            epsilon = 1e-8  # 防止除以0的情况
            w_i = self.train_loss_buffer[:, 1] / (self.train_loss_buffer[:, 0] + epsilon)
            # print(f"w_i values at step {self.epoch}: {w_i}")  # Debug
            batch_weight = self.task_num * F.softmax(w_i / self.T, dim=-1)
        else:
            # 在第一轮训练时，设置所有任务权重相等
            batch_weight = torch.ones(self.task_num).to(self.args.device)

        # 更新损失缓冲区
        self.train_loss_buffer[:, 0] = self.train_loss_buffer[:, 1].clone()  # 保存前一轮的损失
        self.train_loss_buffer[:, 1] = torch.tensor([loss.detach().float().item() for loss in losses], device=self.args.device)  # 更新当前损失

        # print(f"Losses at step {self.epoch}: {[loss.item() for loss in losses]}")  # Debug
        # print(f"Batch weights at step {self.epoch}: {batch_weight}")  # Debug

        # 计算加权损失
        loss = torch.mul(torch.stack(losses), batch_weight).sum()

        if return_outputs:
            return (loss, outputs)
        return loss
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        return super().training_step(model, inputs, num_items_in_batch)
    
    def on_epoch_begin(self):
        """在每个epoch开始时更新epoch计数器"""
        self.epoch = int(self.state.epoch) + 1
        rank0_print(f"DWA: Starting epoch {self.epoch}")