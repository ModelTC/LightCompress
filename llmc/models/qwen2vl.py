import inspect
from typing import Optional, Union

import torch
import torch.nn as nn
from accelerate import Accelerator, DistributedType
from loguru import logger
from transformers import AutoConfig, AutoProcessor, AutoTokenizer

try:
    from transformers import Qwen2VLForConditionalGeneration
except Exception:
    logger.warning(
        'Can not import Qwen2VLForConditionalGeneration. '
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

from .qwen2 import Qwen2


@MODEL_REGISTRY
class Qwen2VL(Qwen2):
    def __init__(self, config, device_map=None, use_cache=False):
        super().__init__(config, device_map, use_cache)

    def build_model(self):
        self.eval_name = 'Qwen2VLEval'
        self.vlm_model_config = AutoConfig.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        if not self.use_cache:
            if hasattr(self.vlm_model_config, 'use_cache'):
                self.vlm_model_config.use_cache = False
        logger.info(f'self.vlm_model_config : {self.vlm_model_config}')
        self.vlm_model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_path,
            config=self.vlm_model_config,
            trust_remote_code=True,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
        )
        self.mm_model = self.vlm_model
        logger.info(f'self.vlm_model : {self.vlm_model}')

        self.vision_model = self.vlm_model.visual
        self.vision_projector = self.vision_model.merger
        self.model = self.vlm_model
        self.model_config = self.vlm_model_config

        self.min_pixels = 256 * 28 * 28
        self.max_pixels = 1280 * 28 * 28
        logger.warning(f'min_pixels is set to: {self.min_pixels}')
        logger.warning(f'max_pixels is set to: {self.max_pixels}')
        logger.warning('You can refer to the link https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct '
                       'to get more info of image resolution for performance boost.')
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels
        )

    def get_extra_rot_module_besides_embed_layers(self):
        return [self.vision_projector.mlp[-1]]

    def batch_process(self, img_qas, calib_or_eval='eval', apply_chat_template=True, return_inputs=True): # noqa
        assert calib_or_eval == 'calib' or calib_or_eval == 'eval'
        assert apply_chat_template
        messages = []
        answers = []
        for idx in range(len(img_qas)):
            img_path = img_qas[idx]['image']
            if img_path is not None:
                content = []
                if not isinstance(img_path, list):
                    img_path = [img_path]
                for img_idx in range(len(img_path)):
                    content.append({'type': 'image', 'image': img_path[img_idx]})
                content.append({'type': 'text', 'text': img_qas[idx]['question']})
                message = [
                    {
                        'role': 'user',
                        'content': content
                    }
                ]
            else:
                message = [
                    {
                        'role': 'user',
                        'content': [
                            {'type': 'text', 'text': img_qas[idx]['question']}
                        ]
                    }
                ]
            messages.append(message)
            answers.append(img_qas[idx]['answer'] + '<|im_end|>')
        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages
        ]
        if calib_or_eval == 'calib' and self.config['calib'].get('add_answer', False):
            texts = [
                texts[n] + answers[n]
                for n in range(len(texts))
            ]
        if calib_or_eval == 'calib':
            logger.info(f'Calib data is:\n{texts}')
        if not return_inputs:
            return texts
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors='pt',
        ).to(next(self.vlm_model.parameters()).dtype)
        return inputs

    def find_blocks(self):
        if self.get_modality() == 'language':
            super().find_blocks()
        elif self.get_modality() == 'vision':
            self.blocks = self.vision_model.blocks
        else:
            raise Exception(f'Qwen2VL do not support {self.get_modality()} modality.')

    def get_layernorms_in_block(self, block):
        if self.get_modality() == 'language':
            return super().get_layernorms_in_block(block)
        elif self.get_modality() == 'vision':
            return {
                'norm1': block.norm1,
                'norm2': block.norm2,
            }
        else:
            raise Exception(f'Qwen2VL do not support {self.get_modality()} modality.')

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
                    'layers': {'mlp.fc1': block.mlp.fc1},
                    'prev_op': [block.norm2],
                    'input': ['mlp.fc1'],
                    'inspect': block.mlp.fc1,
                    'has_kwargs': False,
                    'is_mlp': True,
                },
                {
                    'layers': {'mlp.fc2': block.mlp.fc2},
                    'prev_op': [block.mlp.fc1],
                    'input': ['mlp.fc2'],
                    'inspect': block.mlp.fc2,
                    'has_kwargs': False,
                    'is_mlp': True,
                    'do_trans': False
                },
            ]
        else:
            raise Exception(f'Qwen2VL do not support {self.get_modality()} modality.')


try:
    from lmms_eval.api.model import lmms
    from lmms_eval.models.qwen2_vl import Qwen2_VL

    @MODEL_REGISTRY
    class Qwen2VLEval(Qwen2_VL):
        def __init__(
            self,
            llmc_model,
            pretrained: str = 'Qwen/Qwen2-VL-7B-Instruct',
            device: Optional[str] = 'cuda',
            device_map: Optional[str] = 'cuda',
            batch_size: Optional[Union[int, str]] = 1,
            use_cache=True,
            use_flash_attention_2: Optional[bool] = False,
            max_pixels: int = 12845056,
            min_pixels: int = 3136,
            max_num_frames: int = 32,
            **kwargs,
        ) -> None:
            lmms.__init__(self)
            # Do not use kwargs for now
            assert kwargs == {}, f'Unexpected kwargs: {kwargs}'

            accelerator = Accelerator()
            if accelerator.num_processes > 1:
                self._device = torch.device(f'cuda:{accelerator.local_process_index}')
                self.device_map = f'cuda:{accelerator.local_process_index}'
            elif accelerator.num_processes == 1 and device_map == 'auto':
                self._device = torch.device(device)
                self.device_map = device_map
            else:
                self._device = torch.device(f'cuda:{accelerator.local_process_index}')
                self.device_map = f'cuda:{accelerator.local_process_index}'

            self._model = llmc_model.eval().cuda()
            self.processor = AutoProcessor.from_pretrained(
                pretrained,
                max_pixels=max_pixels,
                min_pixels=min_pixels
            )
            self.max_pixels = max_pixels
            self.min_pixels = min_pixels
            self.max_num_frames = max_num_frames
            self._tokenizer = AutoTokenizer.from_pretrained(pretrained)

            self._config = self.model.config
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
