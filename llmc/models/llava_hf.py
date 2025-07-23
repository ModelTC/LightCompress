from typing import List, Optional, Tuple, Union

import torch
from accelerate import Accelerator, DistributedType
from accelerate.state import AcceleratorState
from lmms_eval.api.model import lmms
from lmms_eval.models.llava_hf import LlavaHf as LlavaHF
from loguru import logger
from PIL import Image
from transformers import (AutoConfig, AutoProcessor,
                          LlavaForConditionalGeneration)

from llmc.utils.registry_factory import MODEL_REGISTRY

from .llama import Llama


@MODEL_REGISTRY
class LlavaHf(Llama):
    def __init__(self, config, device_map=None, use_cache=False):
        super().__init__(config, device_map, use_cache)

    def build_model(self):
        self.vlm_model_config = AutoConfig.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        if not self.use_cache:
            self.vlm_model_config.text_config.use_cache = False
        logger.info(f'self.vlm_model_config : {self.vlm_model_config}')
        self.vlm_model = LlavaForConditionalGeneration.from_pretrained(
            self.model_path,
            config=self.vlm_model_config,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
        )
        self.eval_name = 'LlavaHfEval'
        self.mm_model = self.vlm_model
        logger.info(f'self.vlm_model : {self.vlm_model}')
        self.vision_model = self.vlm_model.vision_tower
        self.vision_projector = self.vlm_model.multi_modal_projector
        self.model = self.vlm_model.language_model
        self.model_config = self.vlm_model_config.text_config
        self.pruning_config = {
            'is_video_model': False,
            'image_token_start_index': 5,
            'image_token_length': self.vlm_model_config.image_seq_length,
            'select_layer': self.vlm_model_config.vision_feature_layer,
            'select_feature': self.vlm_model_config.vision_feature_select_strategy,
            'image_token_index': self.vlm_model_config.image_token_index,
            'vision_token_start_index': 35,
        }

        self.processor = AutoProcessor.from_pretrained(self.model_path)

    def get_extra_rot_module_besides_embed_layers(self):
        return [self.vision_projector.linear_2]

    def batch_process(
        self,
        img_qas,
        calib_or_eval='eval',
        apply_chat_template=True,
        return_inputs=True,
    ):  # noqa
        assert calib_or_eval == 'calib' or calib_or_eval == 'eval'
        assert apply_chat_template
        messages = []
        images = []
        answers = []
        for idx in range(len(img_qas)):
            img_path = img_qas[idx]['image']
            if img_path is not None:
                image = Image.open(img_path)
                message = [
                    {
                        'role': 'user',
                        'content': [
                            {'type': 'image'},
                            {'type': 'text', 'text': img_qas[idx]['question']},
                        ],
                    }
                ]
                images.append(image)
            else:
                message = [
                    {
                        'role': 'user',
                        'content': [{'type': 'text', 'text': img_qas[idx]['question']}],
                    }
                ]
            messages.append(message)
            answers.append(img_qas[idx]['answer'])
        texts = [
            self.processor.apply_chat_template(messages[n], add_generation_prompt=True)
            for n in range(len(messages))
        ]
        if calib_or_eval == 'calib' and self.config['calib'].get('add_answer', False):
            texts = [texts[n] + ' ' + answers[n] for n in range(len(texts))]
        if calib_or_eval == 'calib':
            logger.info(f'Calib data is:\n{texts}')
        if not return_inputs:
            return texts
        inputs = self.processor(
            text=texts,
            images=images if len(images) else None,
            padding=True,
            return_tensors='pt',
        ).to(
            next(self.vlm_model.parameters()).dtype
        )  # noqa
        return inputs

    def find_blocks(self):
        if self.get_modality() == 'language':
            super().find_blocks()
        elif self.get_modality() == 'vision':
            self.blocks = self.vision_model.vision_model.encoder.layers
        else:
            raise Exception(f'Llava do not support {self.get_modality()} modality.')

    def get_layernorms_in_block(self, block):
        if self.get_modality() == 'language':
            return super().get_layernorms_in_block(block)
        elif self.get_modality() == 'vision':
            return {
                'layer_norm1': block.layer_norm1,
                'layer_norm2': block.layer_norm2,
            }
        else:
            raise Exception(f'Llava do not support {self.get_modality()} modality.')

    def get_subsets_in_block(self, block):
        if self.get_modality() == 'language':
            return super().get_subsets_in_block(block)
        elif self.get_modality() == 'vision':
            return [
                {
                    'layers': {
                        'self_attn.q_proj': block.self_attn.q_proj,
                        'self_attn.k_proj': block.self_attn.k_proj,
                        'self_attn.v_proj': block.self_attn.v_proj,
                    },
                    'prev_op': [block.layer_norm1],
                    'input': ['self_attn.q_proj'],
                    'inspect': block.self_attn,
                    'has_kwargs': True,
                },
                {
                    'layers': {'self_attn.out_proj': block.self_attn.out_proj},
                    'prev_op': [block.self_attn.v_proj],
                    'input': ['self_attn.out_proj'],
                    'inspect': block.self_attn.out_proj,
                    'has_kwargs': False,
                },
                {
                    'layers': {'mlp.fc1': block.mlp.fc1},
                    'prev_op': [block.layer_norm2],
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
                    'do_trans': False,
                },
            ]
        else:
            raise Exception(f'Llava do not support {self.get_modality()} modality.')


@MODEL_REGISTRY
class LlavaHfEval(LlavaHF):
    def __init__(
        self,
        llmc_model,
        pretrained: str = 'llava-hf/llava-1.5-7b-hf',
        revision: str = 'main',
        device: str = 'cuda',
        dtype: Optional[Union[str, torch.dtype]] = 'auto',
        batch_size: int = 1,
        trust_remote_code: Optional[bool] = False,
        attn_implementation: Optional[str] = None,
        device_map: str = '',
        chat_template: Optional[str] = None,
        use_cache: bool = False,
        max_frames_num: Optional[int] = 32,
        **kwargs,
    ) -> None:

        lmms.__init__(self)
        # Do not use kwargs for now
        assert kwargs == {}, f'Unexpected kwargs: {kwargs}'

        accelerator = Accelerator()
        if accelerator.num_processes > 1 and device_map == '':
            self._device = torch.device(f'cuda:{accelerator.local_process_index}')
            self.device_map = f'cuda:{accelerator.local_process_index}'
        else:
            self._device = torch.device(device)
            self.device_map = device_map
        if isinstance(dtype, str) and dtype != 'auto':
            dtype = getattr(torch, dtype)

        self._model = llmc_model.cuda()
        self.pretrained = pretrained
        self._image_processor = AutoProcessor.from_pretrained(
            pretrained, revision=revision, trust_remote_code=trust_remote_code
        )
        # Pad from left for batched generation:
        # https://huggingface.co/docs/transformers/v4.39.3/en/model_doc/llava#usage-tips
        self._image_processor.tokenizer.padding_side = 'left'
        self._tokenizer = self._image_processor.tokenizer
        self._config = self._model.config
        self.batch_size_per_gpu = int(batch_size)
        self.chat_template = chat_template
        self.use_cache = use_cache
        if accelerator.num_processes > 1 and device_map == '':
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    'train_micro_batch_size_per_gpu': self.batch_size_per_gpu,
                    'train_batch_size': self.batch_size_per_gpu
                    * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(
                    must_match=True, **kwargs
                )
                logger.info(
                    'Detected that you are using DistributedType.DEEPSPEED. \
                            Make sure you run `accelerate config` and set zero stage to 0'
                )
            if (
                accelerator.distributed_type == DistributedType.FSDP
                or accelerator.distributed_type == DistributedType.DEEPSPEED
            ):
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(
                    self.model, evaluation_mode=True
                )
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                logger.info(
                    f'Using {accelerator.num_processes} devices with data parallelism'
                )
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        elif accelerator.num_processes == 1 and device_map == 'auto':
            logger.info(
                f'Using {accelerator.num_processes} devices with pipeline parallelism'
            )
            self._rank = 0
            self._world_size = 1
        else:
            logger.info(f'Using single device: {self._device}')
            self.model.to(self._device)
            self._rank = 0
            self._world_size = 1
        self.accelerator = accelerator
