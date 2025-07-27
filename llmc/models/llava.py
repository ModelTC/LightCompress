import os
import types
from datetime import timedelta
from typing import Optional, Union

import torch
from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from lmms_eval.api.model import lmms
from lmms_eval.models.llava import Llava as LLaVA
from loguru import logger
from packaging import version
from PIL import Image
from transformers import AutoConfig, AutoTokenizer

from llmc.utils.registry_factory import MODEL_REGISTRY

from .llama import Llama

try:
    from llava.constants import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                                 DEFAULT_IMAGE_PATCH_TOKEN,
                                 DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)
    from llava.conversation import SeparatorStyle, conv_templates
    from llava.mm_utils import (get_model_name_from_path, process_images,
                                tokenizer_image_token)
    from llava.model.builder import load_pretrained_model
    from llava.model.language_model.llava_llama import LlavaConfig
except Exception as e:
    logger.debug('LLaVA is not installed. Please install LLaVA to use this model.\nError: %s' % e)


@MODEL_REGISTRY
class Llava(Llama):
    def __init__(self, config, device_map=None, use_cache=False):
        super().__init__(config, device_map, use_cache)

    def build_tokenizer(self):
        pass

    def build_model(self):

        self.vlm_model_config = AutoConfig.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        logger.info(f'self.vlm_model_config : {self.vlm_model_config}')
        model_name = get_model_name_from_path(self.model_path)
        self.tokenizer, self.vlm_model, self.image_processor, context_len = load_pretrained_model(
            self.model_path,
            None,
            model_name,
            device_map='cpu',
            attn_implementation='sdpa'
        )

        self.eval_name = 'LlavaEval'
        self.mm_model = self.vlm_model
        logger.info(f'self.vlm_model : {self.vlm_model}')
        self.vision_model = self.vlm_model.get_vision_tower()
        self.language_model = self.vlm_model.model
        self.vision_projector = self.vlm_model.model.mm_projector
        # Llava merges the language model with the vision projector and vision model
        self.model = self.vlm_model
        self.model_config = self.vlm_model_config.text_config
        self.pruning_config = {
            'is_video_model': False,
            'image_token_length': self.vlm_model_config.image_seq_length,
            'select_layer': self.vision_model.select_layer,
            'select_feature': self.vision_model.select_feature,
            'image_token_index': IMAGE_TOKEN_INDEX,
            'IMAGE_TOKEN_INDEX': IMAGE_TOKEN_INDEX,  # for llava
            'vision_token_start_index': 35,
        }
        if 'v1.6' in model_name.lower():
            self.pruning_config['image_token_length'] = None
        self.processor = None
        self.first_turn_question = True

    def get_extra_rot_module_besides_embed_layers(self):
        return [self.vision_projector[2]]

    def find_blocks(self):
        if self.get_modality() == 'language':
            super().find_blocks()
        elif self.get_modality() == 'vision':
            self.blocks = self.vision_model.vision_tower.vision_model.encoder.layers
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
                    'do_trans': False
                },
            ]
        else:
            raise Exception(f'Llava do not support {self.get_modality()} modality.')

    def eval_custom_samples_just_infer(
        self,
        img_qas,
        eval_cfg
    ):  # noqa

        custom_samples_ans = img_qas.copy()

        self.vlm_model.cuda()

        def load_image(image_file):
            image = Image.open(image_file).convert('RGB')
            return image

        def load_images(image_files):
            out = []
            for image_file in image_files:
                image = load_image(image_file)
                out.append(image)
            return out

        for data_idx, questions in enumerate(img_qas):
            self.first_turn_question = True

            custom_samples_ans[data_idx]['answer'] = []

            image_files = questions['image']
            image_files = [os.path.join(eval_cfg.path, 'images', image_file) for image_file in image_files] # noqa
            images = load_images(image_files)
            image_sizes = [x.size for x in images]
            images_tensor = process_images(
                images,
                self.image_processor,
                self.vlm_model.config
            ).to(self.vlm_model.device, dtype=torch.float16)

            input_ids_old = None

            for question_idx, question in enumerate(questions['question']):

                conv_mode = 'llava_v1'
                conv = conv_templates[conv_mode].copy()
                if question_idx > 0:
                    conv.system = ''
                    qs = question
                    self.first_turn_question = False
                else:
                    qs = DEFAULT_IMAGE_TOKEN + '\n' + question
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()

                input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda() # noqa
                # print(f"input_ids 1: {input_ids}, {input_ids.shape}")
                if input_ids_old is not None:
                    input_ids = torch.cat((input_ids_old, input_ids), dim=1)
                # print(f"input_ids 2: {input_ids}, {input_ids.shape}")

                with torch.inference_mode():
                    output_ids = self.vlm_model.generate(
                        input_ids,
                        attention_mask=input_ids.new_ones(input_ids.shape, dtype=torch.bool),
                        images=images_tensor,
                        image_sizes=image_sizes,
                        do_sample=False,
                        top_p=None,
                        num_beams=1,
                        max_new_tokens=eval_cfg.max_new_tokens,
                        use_cache=True,
                    )

                    # print(f"output_ids: {output_ids}, {output_ids.shape}")

                outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

                print('--------------------------------')
                print(f'data_idx: {data_idx}')
                print(f'question_idx: {question_idx}')
                print(f'question: {question}')
                print(f'outputs: {outputs}')
                print('--------------------------------')

                custom_samples_ans[data_idx]['answer'].append(outputs[0])

                input_ids_old = torch.cat((input_ids, output_ids), dim=1)

        return custom_samples_ans


if version.parse(torch.__version__) >= version.parse('2.1.2'):
    best_fit_attn_implementation = 'sdpa'
else:
    best_fit_attn_implementation = 'eager'


@MODEL_REGISTRY
class LlavaEval(LLaVA):
    def __init__(
        self,
        llmc_model,
        pretrained: str = 'liuhaotian/llava-v1.5-7b',
        truncation: Optional[bool] = True,
        device: Optional[str] = 'cuda',
        batch_size: Optional[Union[int, str]] = 1,
        model_name=None,
        attn_implementation=best_fit_attn_implementation,
        device_map: str = '',
        conv_template='vicuna_v1',
        use_cache: bool = False,
        tie_weights: bool = True,
        truncate_context=False,  # set it False for LLaVA-1.6 no matter truncate
        customized_config=None,  # ends in json
        **kwargs,
    ) -> None:
        lmms.__init__(self)
        # Do not use kwargs for now
        assert kwargs == {}, f'Unexpected kwargs: {kwargs}'

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f'cuda:{accelerator.local_process_index}')
            self.device_map = f'cuda:{accelerator.local_process_index}'
        elif accelerator.num_processes == 1 and device_map == 'auto':
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f'cuda:{accelerator.local_process_index}')
            self.device_map = f'cuda:{accelerator.local_process_index}'

        llava_model_args = {
            'multimodal': True,
        }
        if customized_config is not None:
            llava_model_args['customized_config'] = customized_config
        if attn_implementation is not None:
            llava_model_args['attn_implementation'] = attn_implementation
        if 'use_flash_attention_2' in kwargs:
            llava_model_args['use_flash_attention_2'] = kwargs['use_flash_attention_2']
        model_name = model_name if model_name is not None else get_model_name_from_path(pretrained)

        self._model = llmc_model.cuda()
        self._config = self._model.config
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained, use_fast=False)
        self._image_processor = None
        if 'llava' in model_name.lower():
            mm_use_im_start_end = getattr(self._config, 'mm_use_im_start_end', False)
            mm_use_im_patch_token = getattr(self._config, 'mm_use_im_patch_token', True)
            if mm_use_im_patch_token:
                self._tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            if mm_use_im_start_end:
                self._tokenizer.add_tokens(
                    [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN],
                    special_tokens=True
                )
            self._image_processor = self._model.get_vision_tower().image_processor
        if hasattr(self._config, 'max_sequence_length'):
            self._max_length = self._config.max_sequence_length
        else:
            self._max_length = 2048

        self.model.eval()
        if tie_weights:
            self.model.tie_weights()

        self.truncation = truncation
        self.batch_size_per_gpu = int(batch_size)
        self.conv_template = conv_template
        self.use_cache = use_cache
        self.truncate_context = truncate_context
        # assert self.batch_size_per_gpu == 1, (
        #     "Llava currently does not support batched generation. "
        #     "See: https://github.com/haotian-liu/LLaVA/issues/754. "
        #     "HF Llava also has this issue."
        # )
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
                DistributedType.DEEPSPEED], (
                    'Unsupported distributed type provided. '
                    'Only DDP and FSDP are supported.')
            # To use DistributedType.DEEPSPEED, run `accelerate config` first.
            # You must select zero stage 0 (equivalent to DDP) for model preparation to work.
            # Attempts to support zero stage 2 via kwargs failed.
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    'train_micro_batch_size_per_gpu': self.batch_size_per_gpu,
                    'train_batch_size': self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(
                    must_match=True, **kwargs
                )
                logger.info(
                    'Detected that you are using DistributedType.DEEPSPEED. '
                    'Make sure you run `accelerate config` and set zero stage to 0'
                )

            if (
                accelerator.distributed_type == DistributedType.FSDP
                or accelerator.distributed_type == DistributedType.DEEPSPEED
            ):
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                logger.info(f'Using {accelerator.num_processes} devices with data parallelism')
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        elif accelerator.num_processes == 1 and device_map == 'auto':
            logger.info(f'Using {accelerator.num_processes} devices with tensor parallelism')
            self._rank = 0
            self._world_size = 1
        else:
            logger.info(f'Using single device: {self._device}')
            self.model.to(self._device)
            self._rank = 0
            self._world_size = 1
