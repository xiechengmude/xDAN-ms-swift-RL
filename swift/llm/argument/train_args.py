# Copyright (c) Alibaba, Inc. and its affiliates.
import inspect
import json
import math
import os
import platform
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Union

import torch
import torch.distributed as dist
import transformers
from packaging import version
from transformers import Seq2SeqTrainingArguments
from transformers.utils import is_torch_npu_available
from transformers.utils.versions import require_version

from swift.trainers import LOSS_MAPPING, TrainerFactory
from swift.utils import (add_version_to_work_dir, get_dist_setting, get_pai_tensorboard_dir, is_dist,
                         is_local_master, is_mp, is_pai_training_job, use_torchacc)
from .data_args import TemplateArguments, DataArguments
from .model_args import ModelArguments, QuantizeArguments, GenerationArguments, ArgumentsBase
from .tuner_args import TunerArguments


class Seq2SeqTrainingOverrideArguments(Seq2SeqTrainingArguments):

    output_dir: str = 'output'
    gradient_checkpointing: Optional[bool] = None

    num_train_epochs: int = 1
    # if max_steps >= 0, override num_train_epochs

    save_steps: Optional[int] = None
    save_total_limit: int = 2  # save last and best. -1: all checkpoints
    logging_steps: int = 5
    adam_beta2: float = 0.95
    learning_rate: Optional[float] = None
    weight_decay: float = 0.1
    gradient_accumulation_steps: Optional[int] = None
    lr_scheduler_type: str = 'cosine'
    lr_scheduler_kwargs: Optional[str] = None  # json
    warmup_ratio: float = 0.05
    dataloader_num_workers: Optional[int] = None
    report_to: List[str] = field(default_factory=lambda: ['tensorboard'])
    evaluation_strategy: Literal['steps', 'epoch', 'no'] = 'steps'


@dataclass
class SftArguments(ArgumentsBase, Seq2SeqTrainingOverrideArguments, ModelArguments, TunerArguments, TemplateArguments, QuantizeArguments, GenerationArguments, DataArguments):
    freeze_parameters: List[str] = field(default_factory=list)
    freeze_vit: bool = False
    freeze_parameters_ratio: float = 0.  # 0 ~ 1
    additional_trainable_parameters: List[str] = field(default_factory=list)

    add_output_dir_suffix: Optional[bool] = None
    resume_only_model: bool = False

    packing: bool = False
    # megatron
    train_backend: Literal['transformers', 'megatron'] = 'transformers'
    tp: int = 1
    pp: int = 1
    min_lr: Optional[float] = None
    sequence_parallel: bool = False

    # multimodal
    loss_name: Optional[str] = field(default=None, metadata={'help': f'loss_func choices: {list(LOSS_MAPPING.keys())}'})

    use_loss_scale: bool = False  # for agent
    loss_scale_config_path: str = 'DEFAULT'

    # streaming dataset
    streaming: bool = False
    streaming_val_size: int = 0
    streaming_buffer_size: int = 16384
    batch_size: int = 1
    eval_batch_size: Optional[int] = None
    acc_steps: int = 1

    # other
    test_oom_error: bool = field(
        default=False,
        metadata={
            'help':
            'If set to True, the train_dataset will be sorted in descending order based on max_length, '
            'enabling faster detection of OOM (Out of Memory) errors.'
        })
    lazy_tokenize: Optional[bool] = None
    preprocess_num_proc: int = 1
    use_flash_attn: Optional[bool] = None
    ignore_args_error: bool = False  # True: notebook compatibility
    check_model_is_latest: bool = True

    acc_strategy: Literal['token', 'sentence'] = 'token'
    gpu_memory_fraction: Optional[float] = None

    sequence_parallel_size: int = 1
    # for torchacc
    model_layer_cls_name: Optional[str] = field(
        default=None,
        metadata={'help': "Decoder Class name of model, e.g. 'QWenBlock' for QWen, 'LlamaDecoderLayer' for LLama"})
    metric_warmup_step: Optional[float] = 0
    fsdp_num: int = 1

    def _prepare_target_modules(self, target_modules) -> Union[List[str], str]:
        if isinstance(target_modules, str):
            target_modules = [target_modules]
        if len(target_modules) == 0:
            return target_modules
        elif len(target_modules) == 1:
            if ',' in target_modules[0]:
                target_modules = target_modules[0].split(',')
        if 'AUTO' in target_modules:
            target_modules.remove('AUTO')
            target_modules.append('DEFAULT')
        if 'DEFAULT' in target_modules:
            target_modules.remove('DEFAULT')
            default_lora_tm = get_default_lora_target_modules(self.model_type)
            if isinstance(default_lora_tm, str):
                return default_lora_tm
            target_modules += default_lora_tm
        if 'EMBEDDING' in target_modules:
            self.lora_use_embedding = True
        if 'ALL' in target_modules:
            self.lora_use_all = True
        return target_modules

    def handle_lr_scheduler_kwargs(self):
        if self.lr_scheduler_kwargs is None:
            self.lr_scheduler_kwargs = {}
        elif isinstance(self.lr_scheduler_kwargs, str):
            self.lr_scheduler_kwargs = json.loads(self.lr_scheduler_kwargs)

    def _prepare_modules_to_save(self, modules_to_save) -> List[str]:
        if isinstance(modules_to_save, str):
            modules_to_save = [modules_to_save]
        if len(modules_to_save) == 0:
            return modules_to_save
        if 'EMBEDDING' in modules_to_save:
            modules_to_save.remove('EMBEDDING')
            self.lora_m2s_use_embedding = True
        if 'LN' in modules_to_save:
            modules_to_save.remove('LN')
            self.lora_m2s_use_ln = True
        return modules_to_save

    def __post_init__(self) -> None:
        super().__post_init__()
        self.handle_compatibility()
        if self.preprocess_num_proc and self.preprocess_num_proc > 1:
            os.environ['DATASET_MAP_NPROC'] = str(self.preprocess_num_proc)
        if len(self.val_dataset) > 0:
            self.dataset_test_ratio = 0.0
            logger.info('Using val_dataset, ignoring dataset_test_ratio')
        if is_pai_training_job():
            self._handle_pai_compat()
        ds_config_folder = os.path.abspath(os.path.join(__file__, '..', '..', 'ds_config'))
        deepspeed_mapping = {
            'default-zero2': 'zero2.json',
            'default-zero3': 'zero3.json',
            'zero2-offload': 'zero2_offload.json',
            'zero3-offload': 'zero3_offload.json',
        }
        for ds_name, ds_config in deepspeed_mapping.items():
            if self.deepspeed == ds_name:
                self.deepspeed = os.path.join(ds_config_folder, ds_config)
                break
        if self.loss_scale_config_path:
            if self.loss_scale_config_path == 'DEFAULT':
                self.loss_scale_config_path = os.path.abspath(
                    os.path.join(__file__, '..', '..', 'agent', 'default_loss_scale_config.json'))
            elif self.loss_scale_config_path == 'alpha-umi':  # https://arxiv.org/pdf/2401.07324
                self.loss_scale_config_path = os.path.abspath(
                    os.path.join(__file__, '..', '..', 'agent', 'alpha_umi_loss_scale_config.json'))
            elif self.loss_scale_config_path == 'agent-flan':  # https://arxiv.org/abs/2403.12881
                self.loss_scale_config_path = os.path.abspath(
                    os.path.join(__file__, '..', '..', 'agent', 'agentflan.json'))
        if self.train_backend == 'megatron' and self.resume_from_checkpoint is None:
            self.resume_from_checkpoint = f'{self.model_type}-tp{self.tp}-pp{self.pp}'
        self.handle_path()
        self._handle_dataset_sample()
        self._register_self_cognition()
        self.handle_custom_register()
        self.handle_custom_dataset_info()
        if self.resume_from_checkpoint:
            self.load_from_ckpt_dir(True)
            if self.sft_type == 'full' or self.train_backend == 'megatron':
                self.model_id_or_path = self.resume_from_checkpoint

        if self.rope_scaling:
            logger.info(f'rope_scaling is set to {self.rope_scaling}, please remember to set max_length')

        if self.dataset_seed is None:
            self.dataset_seed = self.seed
        self.set_model_type()
        self.check_flash_attn()
        self.handle_lr_scheduler_kwargs()
        self.is_multimodal = self._is_multimodal(self.model_type)
        self.is_vision = self._is_vision(self.model_type)

        self.lora_use_embedding = False
        self.lora_use_all = False
        self.lora_m2s_use_embedding = False
        self.lora_m2s_use_ln = False
        self.target_modules = self._prepare_target_modules(self.target_modules)
        self.modules_to_save = self._prepare_modules_to_save(self.modules_to_save)
        if self.use_self_cognition and self.sft_type == 'lora' and not self.lora_use_all:
            logger.warning('Due to knowledge editing involved, it is recommended to add LoRA on MLP. '
                           'For example: `--lora_target_modules ALL`. '
                           'If you have already added LoRA on MLP, please ignore this warning.')

        if self.sft_type in {'adalora', 'ia3'} and self.lora_use_embedding:
            raise ValueError('`adalora` and `ia3` do not support setting embedding as target_modules.')

        self.torch_dtype, self.fp16, self.bf16 = self.select_dtype()
        self.rank, self.local_rank, self.world_size, self.local_world_size = get_dist_setting()
        if is_dist():
            if is_torch_npu_available():
                torch.npu.set_device(self.local_rank)
            else:
                torch.cuda.set_device(self.local_rank)
            self.seed += self.rank  # Avoid the same dropout
            if self.ddp_backend is None:
                self.ddp_backend = 'nccl'
            if self.ddp_backend == 'gloo' and self.quantization_bit != 0:
                raise ValueError('not supported, please use `nccl`')

        if self.train_backend == 'megatron' and self.sft_type == 'lora':
            logger.warning('Currently, only full parameter is supported. Setting args.sft_type: "full"')
            self.sft_type = 'full'

        model_info = MODEL_MAPPING[self.model_type]
        if is_adapter(self.sft_type):
            assert self.freeze_parameters_ratio == 0., (
                'lora does not support `freeze_parameters_ratio`, please set `--sft_type full`')
            assert len(self.additional_trainable_parameters) == 0, (
                'lora does not support `additional_trainable_parameters`, please set `--sft_type full`')
            if is_quant_model(self.model_type):
                assert self.quantization_bit == 0, (
                    f'{self.model_type} is already a quantized model and does not need to be quantized again.')
            if self.learning_rate is None:
                self.learning_rate = 1e-4
            if self.eval_steps is None:
                self.eval_steps = 50
        elif self.sft_type == 'full':
            if self.freeze_vit:
                from swift.utils.module_mapping import MODEL_KEYS_MAPPING
                lora_target_modules = model_info.get('lora_target_modules')
                vision_tower = None
                if isinstance(lora_target_modules, str):
                    vision_tower = MODEL_KEYS_MAPPING[lora_target_modules].vision_tower
                if vision_tower:
                    self.freeze_parameters += vision_tower
            assert 0 <= self.freeze_parameters_ratio <= 1
            assert self.quantization_bit == 0, 'Full parameter fine-tuning does not support quantization.'
            assert self.dtype != 'fp16', ("Fine-tuning with dtype=='fp16' can lead to NaN issues. "
                                          'Please use fp32+AMP or bf16 to perform full parameter fine-tuning.')
            if isinstance(self.additional_trainable_parameters, str):
                self.additional_trainable_parameters = [self.additional_trainable_parameters]
            if self.learning_rate is None:
                self.learning_rate = 1e-5
            if self.eval_steps is None:
                self.eval_steps = 200
        else:
            raise ValueError(f'sft_type: {self.sft_type}')

        self.prepare_template()
        if len(self.dataset) == 0:
            raise ValueError(f'self.dataset: {self.dataset}, Please input the training dataset.')

        if self.save_steps is None:
            self.save_steps = self.eval_steps

        if self.use_liger:
            assert is_liger_available(), 'use_liger requires liger_kernels, try `pip install liger-kernel`'
            if self.use_loss_scale:
                logger.warn('use_liger is not compatible with `use_loss_scale`, setting to False...')
                self.use_loss_scale = False

        # compatibility
        if self.quantization_bit > 0 and self.quant_method is None:
            if self.quantization_bit == 4 or self.quantization_bit == 8:
                logger.info('Since you have specified quantization_bit as greater than 0 '
                            "and have not designated a quant_method, quant_method will be set to 'bnb'.")
                self.quant_method = 'bnb'
            else:
                self.quant_method = 'hqq'
                logger.info('Since you have specified quantization_bit as greater than 0 '
                            "and have not designated a quant_method, quant_method will be set to 'hqq'.")

        self.bnb_4bit_compute_dtype, self.load_in_4bit, self.load_in_8bit = self.select_bnb()

        if self.neftune_backend is None:
            self.neftune_backend = 'swift' if version.parse(transformers.__version__) < version.parse('4.35') \
                else 'transformers'

        self.prepare_ms_hub()
        self.train_sampler_random = not self.test_oom_error
        if self.eval_batch_size is None:
            if self.predict_with_generate:
                self.eval_batch_size = 1
            else:
                self.eval_batch_size = self.batch_size
        if self.save_total_limit == -1:
            self.save_total_limit = None

        if self.deepspeed is not None:
            if is_mp():
                raise ValueError('DeepSpeed is not compatible with MP. '
                                 f'n_gpu: {torch.cuda.device_count()}, '
                                 f'local_world_size: {self.local_world_size}.')
            require_version('deepspeed')
            if self.deepspeed.endswith('.json') or os.path.isfile(self.deepspeed):
                with open(self.deepspeed, 'r', encoding='utf-8') as f:
                    self.deepspeed = json.load(f)
            logger.info(f'Using deepspeed: {self.deepspeed}')

        if self.gradient_accumulation_steps is None:
            self.gradient_accumulation_steps = math.ceil(16 / self.batch_size / self.world_size)
        template_info = TEMPLATE_MAPPING[self.template_type]
        self._handle_streaming_args()
        if self.lazy_tokenize is None and not self.streaming:
            self.lazy_tokenize = template_info.get('lazy_tokenize', False)
            logger.info(f'Setting args.lazy_tokenize: {self.lazy_tokenize}')
        if self.dataloader_num_workers is None:
            if 'dataloader_num_workers' in template_info:
                self.dataloader_num_workers = template_info['dataloader_num_workers']
            elif platform.system() == 'Windows':
                self.dataloader_num_workers = 0
            else:
                self.dataloader_num_workers = 1
            logger.info(f'Setting args.dataloader_num_workers: {self.dataloader_num_workers}')
        if 'dataloader_pin_memory' in template_info:
            self.dataloader_pin_memory = template_info['dataloader_pin_memory']
            logger.info(f'Setting args.dataloader_pin_memory: {self.dataloader_pin_memory}')
        if 'qwen-audio' in self.model_type:
            assert self.preprocess_num_proc == 1 or self.lazy_tokenize, 'not support'
        support_gradient_checkpointing = model_info.get('support_gradient_checkpointing', True)
        if self.gradient_checkpointing is None:
            self.gradient_checkpointing = support_gradient_checkpointing
        elif not support_gradient_checkpointing and self.gradient_checkpointing:
            logger.warning(f'{self.model_type} not support gradient_checkpointing.')

        if use_torchacc():
            self.dataloader_drop_last = True

        if self.train_backend == 'transformers':
            self._init_training_args()
        else:
            assert is_dist(), 'Please start in distributed mode.'
            dist.init_process_group(backend=self.ddp_backend)
            if self.min_lr is None:
                self.min_lr = self.learning_rate * 0.1
        if self.add_output_dir_suffix is None:
            self.add_output_dir_suffix = True
        if self.add_output_dir_suffix:
            if self.train_backend == 'megatron':
                self.output_dir = os.path.join(self.output_dir, f'{self.model_type}-tp{self.tp}-pp{self.pp}')
            else:
                self.output_dir = os.path.join(self.output_dir, self.model_type)
            self.output_dir = add_version_to_work_dir(self.output_dir)
            logger.info(f'output_dir: {self.output_dir}')
            if self.train_backend == 'transformers':
                self.training_args.output_dir = self.output_dir
                self.training_args.run_name = self.output_dir
        if is_local_master():
            os.makedirs(self.output_dir, exist_ok=True)
        if self.logging_dir is None:
            self.logging_dir = f'{self.output_dir}/runs'
            if self.train_backend == 'transformers':
                self.training_args.logging_dir = self.logging_dir
        self.handle_generation_config()

    def _init_training_args(self) -> None:
        self.train_type = self.rlhf_type if hasattr(self, 'rlhf_type') else 'sft'
        training_args_cls, kwargs = TrainerFactory.get_training_args_info(self)
        additional_saved_files = []
        if self.sft_type == 'full':
            additional_saved_files = get_additional_saved_files(self.model_type)

        if self.neftune_backend != 'swift':
            kwargs['neftune_noise_alpha'] = self.neftune_noise_alpha

        parameters = inspect.signature(training_args_cls.__init__).parameters
        for k in ['lr_scheduler_kwargs', 'include_num_input_tokens_seen', 'auto_find_batch_size']:
            if k in parameters:
                kwargs[k] = getattr(self, k)
        if 'eval_strategy' in parameters:
            kwargs['eval_strategy'] = self.evaluation_strategy
        else:
            kwargs['evaluation_strategy'] = self.evaluation_strategy

        if 'accelerator_config' in parameters:
            kwargs['accelerator_config'] = {'dispatch_batches': False}

        training_args = training_args_cls(
            output_dir=self.output_dir,
            logging_dir=self.logging_dir,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.eval_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            max_grad_norm=self.max_grad_norm,
            num_train_epochs=self.num_train_epochs,
            max_steps=self.max_steps,
            lr_scheduler_type=self.lr_scheduler_type,
            warmup_ratio=self.warmup_ratio,
            warmup_steps=self.warmup_steps,
            logging_steps=self.logging_steps,
            save_strategy=self.save_strategy,
            save_steps=self.save_steps,
            save_total_limit=self.save_total_limit,
            remove_unused_columns=False,
            bf16=self.bf16,
            fp16=self.fp16,
            eval_steps=self.eval_steps,
            dataloader_num_workers=self.dataloader_num_workers,
            dataloader_pin_memory=self.dataloader_pin_memory,
            metric_for_best_model='rouge-l' if self.predict_with_generate else 'loss',
            greater_is_better=self.predict_with_generate,
            full_determinism=self.full_determinism,
            optim=self.optim,
            adam_beta1=self.adam_beta1,
            adam_beta2=self.adam_beta2,
            adam_epsilon=self.adam_epsilon,
            hub_model_id=self.hub_model_id,
            hub_private_repo=self.hub_private_repo,
            hub_strategy=self.hub_strategy,
            hub_token=self.hub_token,
            push_to_hub=self.push_to_hub,
            resume_from_checkpoint=self.resume_from_checkpoint,
            ignore_data_skip=self.ignore_data_skip,
            ddp_backend=self.ddp_backend,
            gradient_checkpointing=self.gradient_checkpointing,
            local_rank=self.local_rank,
            save_only_model=self.save_only_model,
            train_sampler_random=self.train_sampler_random,
            report_to=self.report_to,
            deepspeed=self.deepspeed,
            additional_saved_files=additional_saved_files,
            disable_tqdm=self.disable_tqdm,
            save_on_each_node=self.save_on_each_node,
            acc_strategy=self.acc_strategy,
            save_safetensors=self.save_safetensors,
            logging_first_step=True,
            metric_warmup_step=self.metric_warmup_step,
            fsdp=self.fsdp,
            fsdp_config=self.fsdp_config,
            dataloader_drop_last=self.dataloader_drop_last,
            seed=self.seed,
            data_seed=self.dataset_seed,
            loss_name=self.loss_name,
            **kwargs)

        training_args.ddp_find_unused_parameters = self.ddp_find_unused_parameters
        training_args.ddp_broadcast_buffers = self.ddp_broadcast_buffers
        training_args.ddp_timeout = self.ddp_timeout
        if is_dist() and training_args.ddp_find_unused_parameters is None:
            if self.gradient_checkpointing:
                training_args.ddp_find_unused_parameters = False
            else:
                training_args.ddp_find_unused_parameters = True

        if is_dist() and training_args.ddp_broadcast_buffers is None:
            if self.gradient_checkpointing:
                training_args.ddp_broadcast_buffers = False
            else:
                training_args.ddp_broadcast_buffers = True

        self.training_args = training_args

    def _handle_pai_compat(self) -> None:
        assert is_pai_training_job()
        logger.info('Handle pai compat...')
        pai_tensorboard_dir = get_pai_tensorboard_dir()
        if self.logging_dir is None and pai_tensorboard_dir is not None:
            self.logging_dir = pai_tensorboard_dir
            logger.info(f'Setting args.logging_dir: {self.logging_dir}')
        if self.add_output_dir_suffix is None:
            self.add_output_dir_suffix = False
            logger.info(f'Setting args.add_output_dir_suffix: {self.add_output_dir_suffix}')

    def _handle_streaming_args(self) -> None:
        if not self.streaming:
            return
        if self.max_steps == -1:
            raise ValueError('Please specify `max_steps` in streaming mode.')

        if self.packing:
            self.packing = False
            logger.warning('Packing is not supported for streaming dataset, set to False')

        if self.test_oom_error:
            self.test_oom_error = False
            logger.warning('test_oom_error is not supported for streaming dataset, set to False')

        if self.lazy_tokenize:
            self.lazy_tokenize = False
            logger.info('lazy_tokenize set to False in streaming dataset')

        if self.train_dataset_mix_ratio > 0:
            logger.warning('train_dataset_mix_ratio is not supported for streaming dataset, set to 0')
            self.train_dataset_mix_ratio = 0

        if self.dataset_test_ratio > 0:
            logger.info('Set dataset_test_ratio to 0 in streaming mode.'
                        'You can manually set val_dataset and val_dataset_sample.'
                        'or set streaming_val_size instead to split from train dataset')
            self.dataset_test_ratio = 0

        if self.train_dataset_sample > 0:
            logger.warning('train_dataset_sample is not supported for streaming dataset, set to -1')
            self.train_dataset_sample = -1

        if self.dataloader_num_workers is None or self.dataloader_num_workers > 0:
            logger.info('Set dataloader_num_workers to 0 in streaming mode')
            self.dataloader_num_workers = 0


@dataclass
class PtArguments(SftArguments):
    sft_type: Literal['lora', 'full', 'longlora', 'adalora', 'ia3', 'llamapro', 'vera', 'boft'] = 'full'
    target_modules: List[str] = field(default_factory=lambda: ['ALL'])
    lazy_tokenize: Optional[bool] = True
    eval_steps: int = 500


@dataclass
class RLHFArguments(SftArguments):
    rlhf_type: Literal['dpo', 'orpo', 'simpo', 'kto', 'cpo'] = 'dpo'
    ref_model_type: Optional[str] = field(
        default=None, metadata={'help': f'model_type choices: {list(MODEL_MAPPING.keys())}'})
    ref_model_id_or_path: Optional[str] = None
    ref_model_revision: Optional[str] = None

    beta: Optional[float] = None
    label_smoothing: float = 0
    # dpo: 'sigmoid', 'hinge', 'ipo', 'exo_pair', 'nca_pair', 'robust', 'bco_pair',
    #      'sppo_hard', 'aot', 'aot_pair', 'apo_zero', 'apo_down'
    # cpo: 'sigmoid', 'hinge', 'ipo', 'simpo'
    loss_type: Optional[str] = None
    # DPO
    # The alpha parameter from the [RPO](https://huggingface.co/papers/2404.19733) paper V3.
    # The paper recommends `rpo_alpha=1.0`.
    rpo_alpha: float = 1.
    # CPO
    cpo_alpha: float = 1.
    # SimPO
    simpo_gamma: float = 1
    # KTO
    desirable_weight: float = 1.0
    undesirable_weight: float = 1.0

    def __post_init__(self):
        self._check_simpo()
        self._set_default()
        self.ref_model_free = self.rlhf_type in ['cpo', 'orpo']
        super().__post_init__()

    def _check_simpo(self):
        if self.rlhf_type != 'simpo':
            return

        self.rlhf_type = 'cpo'
        if self.loss_type is None:
            self.loss_type = 'simpo'
        if self.beta is None:
            self.beta = 2.

    def _set_default(self):
        if self.beta is None:
            self.beta = 0.1
        if self.loss_type is None:
            if self.rlhf_type in ['dpo', 'cpo']:
                self.loss_type = 'sigmoid'  # else None