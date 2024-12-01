# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import sys
from typing import Any, Dict

from transformers import AutoModel

from swift.llm import TemplateType
from ..constant import MLLMModelType
from ..model_arch import ModelArch
from ..register import Model, ModelGroup, ModelMeta, get_model_tokenizer_with_flash_attn, register_model
from ..utils import ModelInfo, git_clone_github, safe_snapshot_download


def get_model_tokenizer_emu3_gen(model_dir: str,
                                 model_info: ModelInfo,
                                 model_kwargs: Dict[str, Any],
                                 load_model: bool = True,
                                 **kwargs):
    import sys
    sys.path.append(model_dir)
    from processing_emu3 import Emu3Processor
    vq_hub = safe_snapshot_download('BAAI/Emu3-VisionTokenizer')
    from transformers import AutoModel, AutoImageProcessor
    image_processor = AutoImageProcessor.from_pretrained(vq_hub, trust_remote_code=True)
    image_tokenizer = AutoModel.from_pretrained(vq_hub, trust_remote_code=True).eval()
    model, tokenizer = get_model_tokenizer_with_flash_attn(model_dir, model_info, model_kwargs, load_model, **kwargs)
    processor = Emu3Processor(image_processor, image_tokenizer, tokenizer)
    processor.vision_tokenizer.to('cuda:0')
    return model, processor


register_model(
    ModelMeta(
        MLLMModelType.emu3_gen,
        [
            ModelGroup([
                Model('BAAI/Emu3-Gen', 'BAAI/Emu3-Gen'),
            ], tags=['multi-modal', 't2i']),
        ],
        TemplateType.emu3_gen,
        get_model_tokenizer_emu3_gen,
        architectures=['LlavaForConditionalGeneration'],
        model_arch=ModelArch.emu3_chat,
    ))


def get_model_tokenizer_emu3_chat(model_dir: str,
                                  model_info: ModelInfo,
                                  model_kwargs: Dict[str, Any],
                                  load_model: bool = True,
                                  **kwargs):
    model, tokenizer = get_model_tokenizer_with_flash_attn(model_dir, model_info, model_kwargs, load_model, **kwargs)

    # download and load vision tokenizer
    from transformers import AutoImageProcessor
    vq_model = safe_snapshot_download('BAAI/Emu3-VisionTokenizer')
    image_processor = AutoImageProcessor.from_pretrained(vq_model, trust_remote_code=True)
    image_tokenizer = AutoModel.from_pretrained(vq_model, device_map=model_kwargs['device_map'], trust_remote_code=True)
    image_tokenizer.requires_grad_(False)
    image_tokenizer.to('cuda:0')  # TODO: check npu

    # load processor
    if 'local_repo_path' in kwargs:
        local_repo_path = kwargs['local_repo_path']
    else:
        local_repo_path = git_clone_github('https://github.com/baaivision/Emu3.git')
    sys.path.append(os.path.join(local_repo_path))
    from emu3.mllm.processing_emu3 import Emu3Processor
    processor = Emu3Processor(image_processor, image_tokenizer, tokenizer)

    return model, processor


register_model(
    ModelMeta(
        MLLMModelType.emu3_chat,
        [
            ModelGroup(
                [
                    Model('BAAI/Emu3-Chat', 'BAAI/Emu3-Chat'),
                ], tags=['vision'], requires=['transformers>=4.44.0']),
        ],
        TemplateType.emu3_chat,
        get_model_tokenizer_emu3_chat,
        architectures=['LlavaForConditionalGeneration'],
        model_arch=ModelArch.emu3_chat,
    ))
