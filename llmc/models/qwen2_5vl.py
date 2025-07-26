
from typing import Optional, Union

import torch
import torch.nn as nn
from accelerate import Accelerator, DistributedType
from loguru import logger
from transformers import AutoConfig, AutoProcessor, AutoTokenizer

try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except Exception:
    logger.warning(
        'Can not import Qwen2_5_VLForConditionalGeneration. '
        'If you need it, please upgrade transformers.'
    )

try:
    from qwen_vl_utils import process_vision_info
except Exception:
    logger.warning(
        'Can not import qwen_vl_utils. '
        'If you need it, please pip install qwen-vl-utils'
    )

from llmc.utils.registry_factory import MODEL_REGISTRY

from .qwen2vl import Qwen2VL


@MODEL_REGISTRY
class Qwen2_5VL(Qwen2VL):
    def __init__(self, config, device_map=None, use_cache=False):
        super().__init__(config, device_map, use_cache)

    def build_model(self):
        self.eval_name = 'Qwen2_5VLEval'
        self.vlm_model_config = AutoConfig.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        if not self.use_cache:
            if hasattr(self.vlm_model_config, 'use_cache'):
                self.vlm_model_config.use_cache = False
        logger.info(f'self.vlm_model_config : {self.vlm_model_config}')

        self.vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            config=self.vlm_model_config,
            trust_remote_code=True,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
        )
        self.mm_model = self.vlm_model
        logger.info(f'self.vlm_model : {self.vlm_model}')

        self.vision_model = self.vlm_model.visual
        self.language_model = self.vlm_model.model
        self.vision_projector = self.vision_model.merger
        self.model = self.vlm_model

        self.model_config = self.vlm_model_config

        self.min_pixels = 256 * 28 * 28
        self.max_pixels = 1280 * 28 * 28
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels
        )
        self.pruning_config = {
            'is_video_model': False,
            'image_token_index': self.vlm_model_config.image_token_id,
            'vision_end_token_id': self.vlm_model_config.vision_end_token_id,
            'vision_start_token_id': self.vlm_model_config.vision_start_token_id,
            'vision_token_start_index': 15
        }
        self.first_turn_question = True

    # todo: check
    def get_subsets_in_block(self, block):
        if self.get_modality() == 'language':
            return super().get_subsets_in_block(block)
        elif self.get_modality() == 'vision':
            return [
                {
                    'layers': {
                        'attn.qkv': block.attn.qkv,
                    },
                    'prev_op': [block.norm1],
                    'input': ['attn.qkv'],
                    'inspect': block.attn,
                    'has_kwargs': True,
                },
                {
                    'layers': {'attn.proj': block.attn.proj},
                    'prev_op': [block.attn.qkv],
                    'input': ['attn.proj'],
                    'inspect': block.attn.proj,
                    'has_kwargs': False,
                },
                {
                    'layers': {
                        'mlp.gate_proj': block.mlp.gate_proj,
                        'mlp.up_proj': block.mlp.up_proj,
                    },
                    'prev_op': [block.norm2],
                    'input': ['mlp.gate_proj'],
                    'inspect': block.mlp,
                    'has_kwargs': False,
                    'is_mlp': True,
                },
                {
                    'layers': {'mlp.down_proj': block.mlp.down_proj},
                    'prev_op': [block.mlp.up_proj],
                    'input': ['mlp.down_proj'],
                    'inspect': block.mlp.down_proj,
                    'has_kwargs': False,
                    'is_mlp': True,
                },
            ]
        else:
            raise Exception(f'Qwen2_5VL do not support {self.get_modality()} modality.')


try:
    from lmms_eval.api.model import lmms
    from lmms_eval.models.qwen2_5_vl import Qwen2_5_VL

    @MODEL_REGISTRY
    class Qwen2_5VLEval(Qwen2_5_VL):
        def __init__(
            self,
            llmc_model,
            pretrained: str = 'Qwen/Qwen2.5-VL-3B-Instruct',
            device: Optional[str] = 'cuda',
            device_map: Optional[str] = 'auto',
            batch_size: Optional[Union[int, str]] = 1,
            use_cache=True,
            attn_implementation: Optional[str] = None,
            min_pixels: int = 256 * 28 * 28,
            max_pixels: int = 1605632,
            max_num_frames: int = 32,
            use_custom_video_loader: Optional[bool] = False,
            fps: Optional[float] = None,
            max_image_size: Optional[int] = None,
            system_prompt: Optional[str] = 'You are a helpful assistant.',
            interleave_visuals: Optional[bool] = False,
            reasoning_prompt: Optional[str] = None,
            **kwargs,
        ) -> None:
            lmms.__init__(self)
            # Do not use kwargs for now
            assert kwargs == {}, f'Unexpected kwargs: {kwargs}'

            # Validate attention implementation
            valid_attn_implementations = [None, 'flash_attention_2', 'sdpa', 'eager']
            if attn_implementation not in valid_attn_implementations:
                raise ValueError(
                    f'attn_implementation must be one of {valid_attn_implementations}, \
                    got {attn_implementation}'
                )

            self.use_custom_video_loader = use_custom_video_loader
            self.fps = fps
            # if self.fps and not self.use_custom_video_loader:
            #     raise ValueError("FPS is only applicable if use_custom_video_loader is True")
            self.max_image_size = max_image_size
            if self.max_image_size and not self.use_custom_video_loader:
                raise ValueError(
                    'max_image_size is only applicable if use_custom_video_loader is True'
                )

            accelerator = Accelerator()
            if accelerator.num_processes > 1:
                self._device = torch.device(f'cuda:{accelerator.local_process_index}')
                self.device_map = f'cuda:{accelerator.local_process_index}'
            else:
                self._device = torch.device(device)
                self.device_map = device_map if device_map else device

            # Prepare model loading arguments
            model_kwargs = {
                'torch_dtype': 'auto',
                'device_map': self.device_map,
            }

            # Add attention implementation if specified
            if attn_implementation is not None:
                model_kwargs['attn_implementation'] = attn_implementation

            self._model = llmc_model.eval().cuda()
            self.max_pixels = max_pixels
            self.min_pixels = min_pixels
            self.max_num_frames = max_num_frames

            if reasoning_prompt:
                self.reasoning_prompt = reasoning_prompt.replace('\\n', '\n')
            else:
                self.reasoning_prompt = None
            self.processor = AutoProcessor.from_pretrained(
                pretrained,
                ax_pixels=max_pixels,
                min_pixels=min_pixels
            )
            self._tokenizer = AutoTokenizer.from_pretrained(pretrained)
            self.system_prompt = system_prompt
            self.interleave_visuals = interleave_visuals

            self._config = self.model.config
            self._max_length = kwargs.get('max_length', 2048)
            self.batch_size_per_gpu = int(batch_size)
            self.use_cache = use_cache

            if accelerator.num_processes > 1:
                assert accelerator.distributed_type in [
                    DistributedType.FSDP,
                    DistributedType.MULTI_GPU,
                ], 'Unsupported distributed type provided. Only DDP and FSDP are supported.'
                if accelerator.distributed_type == DistributedType.FSDP:
                    self._model = accelerator.prepare(self.model)
                else:
                    self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
                self.accelerator = accelerator
                if self.accelerator.is_local_main_process:
                    logger.info(f'Using {accelerator.num_processes} devices with data parallelism')
                self._rank = self.accelerator.local_process_index
                self._world_size = self.accelerator.num_processes
            else:
                self._rank = 0
                self._world_size = 1
except Exception:
    logger.warning(
        'Can not import lmms_eval. '
        'If you need it, please upgrade transformers.'
    )
