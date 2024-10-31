import inspect
import itertools
import os
from dataclasses import dataclass, field
from functools import partial, update_wrapper
from types import MethodType
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
import transformers
from packaging import version
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig, PretrainedConfig,
                          PreTrainedModel, PreTrainedTokenizerBase)
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.utils import is_torch_bf16_gpu_available, is_torch_cuda_available, is_torch_npu_available
from transformers.utils.versions import require_version

from swift.utils import get_dist_setting, get_logger, is_ddp_plus_mp, is_dist, is_unsloth_available, use_torchacc
from .utils import AttnImpl, HfConfigFactory, safe_snapshot_download

MODEL_MAPPING: Dict[str, Dict[str, Any]] = {}

ARCH_MAPPING: Optional[Dict[str, Dict[str, List[str]]]] = None

GetModelTokenizerFunction = Callable[..., Tuple[Optional[PreTrainedModel], PreTrainedTokenizerBase]]
logger = get_logger()


# [TODO:eos_token -> template]
def register_model(model_meta: ModelMeta, *, exist_ok: bool = False, **kwargs) -> None:
    """
    model_type: The unique ID for the model type. Models with the same model_type share
        the same architectures, template, get_function, etc.
    """
    model_type = model_meta.model_type
    if not exist_ok and model_type in MODEL_MAPPING:
        raise ValueError(f'The `{model_type}` has already been registered in the MODEL_MAPPING.')
    from .constant import MLLMModelType
    if not model_meta.is_multimodal:
        assert model_type not in MLLMModelType.__dict__

    model_info = {'model_meta': model_meta, **kwargs}
    MODEL_MAPPING[model_type] = model_info


def load_by_unsloth(model_dir: str,
                    torch_dtype: torch.dtype,
                    max_seq_length: Optional[int] = None,
                    load_in_4bit: bool = True):
    """Load model by unsloth"""
    # TODO:check
    assert is_unsloth_available(), 'please install unsloth if using `use_unsloth=True`'
    from unsloth import FastLanguageModel
    return FastLanguageModel.from_pretrained(
        model_name=model_dir,
        dtype=torch_dtype,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        trust_remote_code=True,
    )


def get_model_tokenizer_from_local(model_dir: str,
                                   model_config: PretrainedConfig,
                                   model_kwargs: Dict[str, Any],
                                   load_model: bool = True,
                                   tokenizer=None,
                                   automodel_class=AutoModelForCausalLM,
                                   quant_method: Optional[str] = None,
                                   quant_bits: Optional[int] = 0,
                                   is_training: bool = False,
                                   **kwargs):
    """Load the model and tokenizer from the local model_dir."""

    torch_dtype = model_config.torch_dtype
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    model = None
    if load_model:
        if kwargs.get('use_unsloth', False):
            unsloth_kwargs = kwargs.get('unsloth_kwargs') or {}
            logger.info(f'unsloth_kwargs: {unsloth_kwargs}')
            model, tokenizer = load_by_unsloth(model_dir, torch_dtype, **unsloth_kwargs)
        else:
            logger.info(f'model_kwargs: {model_kwargs}')
            model = automodel_class.from_pretrained(
                model_dir, config=model_config, torch_dtype=torch_dtype, trust_remote_code=True, **model_kwargs)
        model.quant_method = quant_method  # TODO: check bnb
        model.quant_bits = quant_bits
        model.is_training = is_training

    return model, tokenizer


def get_model_tokenizer_with_flash_attn(model_dir: str,
                                        model_config: PretrainedConfig,
                                        model_kwargs: Dict[str, Any],
                                        load_model: bool = True,
                                        **kwargs):
    AttnImpl.update_attn_impl(model_config, kwargs.get('attn_impl'))
    return get_model_tokenizer_from_local(model_dir, model_config, model_kwargs, load_model, **kwargs)


def get_model_tokenizer_multimodal(model_dir: str, *args, **kwargs):
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model_dir)
    model, tokenizer = get_model_tokenizer_with_flash_attn(model_dir, *args, **kwargs)
    tokenizer.processor = processor
    return model, tokenizer


def fix_transformers_upgrade(module: PreTrainedModel) -> None:
    # from 4.35, transformers changes its arguments of _set_gradient_checkpointing
    if version.parse(transformers.__version__) >= version.parse('4.35'):
        if isinstance(module, PreTrainedModel) and hasattr(module, '_set_gradient_checkpointing') \
                and 'value' in inspect.signature(module._set_gradient_checkpointing).parameters.keys():
            module._set_gradient_checkpointing = MethodType(PreTrainedModel._set_gradient_checkpointing, module)


def fix_gradient_checkpointing_warning(is_moe: bool = False) -> None:
    torch_version = version.parse(torch.__version__)
    if torch_version < version.parse('2'):
        return
    elif torch_version < version.parse('2.1'):
        # fix https://github.com/Dao-AILab/flash-attention/issues/341
        _use_reentrant = True
    else:
        _use_reentrant = is_moe
    if hasattr(torch.utils.checkpoint, '_checkpoint_origin'):
        return
    # fix torch
    _checkpoint_origin = torch.utils.checkpoint.checkpoint
    torch.utils.checkpoint._checkpoint_origin = _checkpoint_origin
    checkpoint = update_wrapper(
        lambda *args, use_reentrant=_use_reentrant, **kwargs: _checkpoint_origin(
            *args, use_reentrant=use_reentrant, **kwargs),
        _checkpoint_origin)
    torch.utils.checkpoint.checkpoint = checkpoint

    try:
        # fix gradient_checkpointing_enable
        import transformers.modeling_utils
        if hasattr(transformers.modeling_utils, 'checkpoint'):
            transformers.modeling_utils.checkpoint = checkpoint
    except ImportError:
        pass


def fix_do_sample_warning(generation_config) -> None:
    # Use the default values of temperature/top_p/top_k in generation_config.
    if generation_config.do_sample is False:
        generation_config.temperature = 1.
        generation_config.top_p = 1.
        generation_config.top_k = 50


def get_default_device_map():
    if is_deepspeed_zero3_enabled() or os.environ.get('ACCELERATE_USE_FSDP', 'False') == 'true':
        return None
    local_rank = get_dist_setting()[1]
    if is_torch_npu_available():
        if local_rank >= 0:
            return f'npu:{local_rank}'
        else:
            return 'npu:0'
    if torch.cuda.device_count() == 0:
        return 'cpu'
    elif torch.cuda.device_count() == 1:
        return 'cuda:0'
    elif is_dist() and not is_ddp_plus_mp():
        return f'cuda:{local_rank}'
    else:
        return 'auto'


def _check_torch_dtype(torch_dtype: torch.dtype):
    if is_torch_cuda_available() or is_torch_npu_available():

        if torch_dtype == torch.bfloat16:
            support_bf16 = is_torch_bf16_gpu_available()
            if not support_bf16:
                logger.warning(f'torch_dtype: {torch_dtype}, but support_bf16: {support_bf16}.')
    else:
        # cpu
        if torch_dtype == torch.float16:
            logger.warning(f'torch_dtype: {torch_dtype}. The CPU does not support matrix multiplication with FP16.')


def get_default_torch_dtype(torch_dtype: torch.dtype):
    # torch_dtype: torch_dtype in config.json
    if is_torch_cuda_available() or is_torch_npu_available():
        if is_torch_bf16_gpu_available():
            if torch_dtype in {torch.float16, torch.bfloat16}:
                res = torch_dtype
            else:
                res = torch.bfloat16
        else:
            res = torch.float16
    else:
        # cpu
        res = torch.float32
    return res


def get_model_tokenizer(
        model_id_or_path: str,
        torch_dtype: Optional[torch.dtype] = None,
        device_map: Union[str, Dict[str, Any], None] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        load_model: bool = True,
        *,
        model_type: Optional[str] = None,
        is_training: bool = False,
        attn_impl: Literal['flash_attn', 'sdpa', 'eager', None] = None,
        use_hf: Optional[bool] = None,
        revision: Optional[str] = None,
        download_model: Optional[bool] = None,
        **kwargs  #
) -> Tuple[Optional[PreTrainedModel], PreTrainedTokenizerBase]:
    """
    model_id_or_path: The path to the model or the model_id from modelscope/huggingface (controlled by `use_hf`).
    torch_dtype: If you pass `None`, it will retrieve the torch_dtype from the config.json file.
    model_kwargs: Passed to `automodel_class.from_pretrained`.
    load_model: Whether to load the model. If set to False, the model will return `None`.
    use_hf: Indicates whether the model download hub is modelscope or huggingface.
    model_type: If it is not possible to uniquely determine the model_type from the architecture in config.json,
        it needs to be provided.
    attn_impl: If set to 'flash_attn': It will automatically convert names based on the model.
        If set to None : It will be automatically selected between sdpa and eager.
    download_model: Whether to download the model weights. If `None`, it will be selected based on load_model.
    """

    if model_kwargs is None:
        model_kwargs = {}
    if download_model is None:
        download_model = load_model
    # download config.json
    model_dir = safe_snapshot_download(
                model_id_or_path, revision=revision, download_model=False, use_hf=use_hf)
    model_info = HfConfigFactory.get_model_info(model_dir)

    if download_model:
        safe_snapshot_download(
            model_id_or_path, revision=revision, download_model=download_model, use_hf=use_hf)

    if not use_torchacc() and device_map is None:
        device_map = get_default_device_map()
    model_kwargs['device_map'] = device_map

    model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    if model_type is None:
        model_type = model_info.model_type
        logger.info(f'Setting model_type: {model_type}')
    if torch_dtype is None:
        torch_dtype = get_default_torch_dtype(model_info.torch_dtype)
        logger.info(f'Setting torch_dtype: {torch_dtype}')
    _check_torch_dtype(torch_dtype)
    model_config.torch_dtype = torch_dtype
    HfConfigFactory.compat_zero3(model_config)
    rope_scaling = kwargs.get('rope_scaling')
    if rope_scaling is not None:  # TODO: dict?
        HfConfigFactory.set_config_attr(model_config, 'rope_scaling', rope_scaling)

    if model_info.quant_method is not None:
        kwargs['quant_method'] = model_info.quant_method
        kwargs['quant_bits'] = model_info.quant_bits
    kwargs.update({'is_training': is_training, 'model_type': model_type, 'attn_impl': attn_impl})

    model_meta = get_model_meta(model_type)
    model_meta.check_requires()
    model_meta.check_flash_attn(attn_impl)

    get_function = model_meta.get_function
    model, tokenizer = get_function(model_dir, model_config, model_kwargs, load_model, **kwargs)

    tokenizer.model_info = model_info
    tokenizer.model_meta = model_meta

    if model is not None:
        model.model_info = model_info
        model.model_meta = model_meta
        fix_gradient_checkpointing_warning(model_meta.is_moe)
        fix_transformers_upgrade(model)

        # generation_config
        generation_config_path = os.path.join(model_dir, 'generation_config.json')
        # TODO:model.llm.generation_config: deepseek-vl
        if not hasattr(model, 'generation_config') and os.path.isfile(generation_config_path):
            model.generation_config = GenerationConfig.from_pretrained(model_dir)
        # fix llama2 warning
        fix_do_sample_warning(model.generation_config)
    return model, tokenizer


def get_arch_mapping() -> Dict[str, Dict[str, List[str]]]:
    global ARCH_MAPPING
    if ARCH_MAPPING is None:
        # arch(str) -> Dict[model_type(str), List[model_name(str)]]
        ARCH_MAPPING = {}
        for model_type, model_info in MODEL_MAPPING.items():
            model_meta = model_info['model_meta']
            archs = model_meta.architectures
            model_names = model_meta.get_model_names()
            for arch in archs:
                if arch not in ARCH_MAPPING:
                    ARCH_MAPPING[arch] = {}
                ARCH_MAPPING[arch][model_type] = model_names
    return ARCH_MAPPING


def get_model_meta(model_type: str) -> ModelMeta:
    return MODEL_MAPPING[model_type]['model_meta']
