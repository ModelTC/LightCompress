import gc
import inspect
import json
import os
from abc import ABCMeta, abstractmethod
from collections import defaultdict

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from loguru import logger
from safetensors import safe_open
from torch.nn import functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from llmc.compression.quantization.module_utils import (
    _LLMC_LINEAR_TYPES_, _LLMC_LN_TYPES_, _TRANSFORMERS_LINEAR_TYPES_,
    _TRANSFORMERS_LN_TYPES_, LlmcFp8Linear, VllmQuantLinearFp8,
    VllmQuantLinearInt8)


class BaseModel(metaclass=ABCMeta):
    def __init__(self, config, device_map=None, use_cache=False):
        self.config = config
        self.model_type = self.config.model.type
        self.model_path = self.config.model.path
        self.tokenizer_mode = self.config.model.get('tokenizer_mode', 'fast')
        self.use_cpu_to_save_cuda_mem_for_catcher = self.config.model.get('use_cpu_to_save_cuda_mem_for_catcher', False) # noqa
        torch_dtype = self.config.model.torch_dtype
        self.torch_dtype = torch_dtype if torch_dtype in ['auto'] else eval(torch_dtype)
        self.block_wise_quant = self.config.model.get('block_wise_quant', False)
        if self.block_wise_quant:
            assert self.torch_dtype == torch.float8_e4m3fn
        self.device_map = device_map
        self.use_cache = use_cache
        self.mm_model = None
        self.vision_model = None
        self.vision_projector = None
        self.audio_model = None
        self.audio_projector = None
        self.modality = 'language'
        self.kvcache_buffer = []
        self.build_tokenizer()
        self.build_model()
        try:
            self.model.eval()
        except: # noqa
            pass
        self.update_key_info()
        if self.mm_model:
            self.mm_model.eval()

    def set_modality(self, modality='language'):
        assert modality in ['audio', 'vision', 'language', 'video_gen']
        self.modality = modality
        self.update_key_info()

    def get_modality(self):
        assert self.modality in ['audio', 'vision', 'language']
        return self.modality

    def update_key_info(self):
        self.find_blocks()
        self.find_embed_layers()
        self.find_block_name()
        self.add_layernorms_class()

    def reset_kv(self):
        for kvcache in self.kvcache_buffer:
            kvcache._reset_states()

    @abstractmethod
    def find_blocks(self):
        pass

    def find_block_name(self):
        pass

    def get_model(self):
        return self.model

    def get_extra_rot_module_besides_embed_layers(self):
        return []

    def get_blocks(self):
        return self.blocks

    @abstractmethod
    def find_embed_layers(self):
        pass

    @abstractmethod
    def get_embed_layers(self):
        pass

    @abstractmethod
    def get_layers_except_blocks(self):
        pass

    def get_matmul_in_block(self):
        return {}

    def get_act_fn_in_block(self):
        return {}

    def get_softmax_in_block(self):
        return {}

    @abstractmethod
    def get_subsets_in_block(self, block):
        pass

    @abstractmethod
    def skip_layer_name(self):
        pass

    @abstractmethod
    def has_bias(self):
        pass

    def build_tokenizer(self):
        if self.model_type not in ['Vit', 'WanT2V', 'WanI2V']:
            assert self.tokenizer_mode in ['fast', 'slow']
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, use_fast=self.tokenizer_mode, trust_remote_code=True
            )
            if 'Intern' in self.model_type:
                self.tokenizer.padding_side = 'left'
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer = None

    def get_tokenizer(self):
        return self.tokenizer

    def get_attention_rotary_layers(self):
        return []

    def get_num_attention_heads(self):
        return self.model_config.num_attention_heads

    def apply_chat_template(self, prompt):
        messages = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return text

    def batch_process(self, samples, calib_or_eval='eval', apply_chat_template=False, return_inputs=True): # noqa
        assert calib_or_eval == 'calib' or calib_or_eval == 'eval'
        texts = []
        for idx in range(len(samples)):
            question = samples[idx]['question']
            if apply_chat_template:
                question = self.apply_chat_template(question)
            texts.append(question)
        if not return_inputs:
            return texts
        model_inputs = self.tokenizer(texts, return_tensors='pt', padding=True)
        input_ids = model_inputs['input_ids']
        attention_mask = model_inputs['attention_mask']
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }
        return inputs

    def get_catcher(self, first_block_input):
        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module
                self.signature = inspect.signature(module.forward)

            def forward(self, *args, **kwargs):
                params = list(self.signature.parameters.keys())
                for i, arg in enumerate(args):
                    if i > 0:
                        kwargs[params[i]] = arg
                first_block_input['data'].append(args[0])
                if 'output_router_logits' in kwargs:
                    assert kwargs['output_router_logits'] is False
                    kwargs.pop('output_router_logits')
                first_block_input['kwargs'].append(kwargs)
                raise ValueError
        return Catcher

    def __str__(self):
        return f'\nConfig: \n{str(self.model_config)} \nModel: \n{str(self.model)}'

    def build_model(self):
        self.model_config = AutoConfig.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        if not self.use_cache:
            if hasattr(self.model_config, 'use_cache'):
                self.model_config.use_cache = False
        logger.info(f'self.model_config : {self.model_config}')
        if self.torch_dtype in [torch.float8_e4m3fn, torch.int8]:
            with init_empty_weights():
                self.model = AutoModelForCausalLM.from_config(config=self.model_config,
                                                              torch_dtype=torch.float16,
                                                              trust_remote_code=True)
            self.find_blocks()
            if self.torch_dtype == torch.float8_e4m3fn:
                if self.block_wise_quant:
                    self.fp8_block_size \
                        = self.model_config.quantization_config['weight_block_size'][0]
                    params_dict = {'block_size': self.fp8_block_size}
                    quant_linear_cls = LlmcFp8Linear
                else:
                    params_dict = {}
                    quant_linear_cls = VllmQuantLinearFp8
            elif self.torch_dtype == torch.int8:
                params_dict = {}
                quant_linear_cls = VllmQuantLinearInt8

            for block_idx, block in enumerate(self.blocks):
                self.replace_module_block(quant_linear_cls,
                                          block,
                                          block_idx,
                                          params_dict)

            self.load_quant_weight()

            logger.info(f'fp8 block size: {self.fp8_block_size}')
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                config=self.model_config,
                device_map=self.device_map,
                trust_remote_code=True,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True,
            )
        logger.info(f'self.model : {self.model}')

    def load_quant_weight(self):
        state_dict = self.model.state_dict()
        model_index_file = os.path.join(self.model_path, 'model.safetensors.index.json')

        with open(model_index_file, 'r') as f:
            model_index = json.load(f)
        weight_map = model_index['weight_map']

        shard_to_tensors = defaultdict(list)
        for weight_name in state_dict:
            shard_path = weight_map[weight_name]
            shard_to_tensors[shard_path].append(weight_name)

        for shard_path, tensor_names in shard_to_tensors.items():
            full_shard_path = os.path.join(self.model_path, shard_path)
            logger.info(f'Loading Quant shard: {full_shard_path}')
            with safe_open(full_shard_path, framework='pt', device='cpu') as f:
                for weight_name in tensor_names:
                    tensor = f.get_tensor(weight_name)
                    state_dict[weight_name] = tensor
        self.model.load_state_dict(state_dict, assign=True)

    def add_layernorms_class(self):
        ln_class_list = []
        single_block = self.blocks[0]
        ln_dict = self.get_layernorms_in_block(single_block)
        for ln_name in ln_dict:
            ln_class = ln_dict[ln_name].__class__
            if ln_class not in ln_class_list:
                ln_class_list.append(ln_class)
        for ln_class in ln_class_list:
            if ln_class not in _TRANSFORMERS_LN_TYPES_:
                _TRANSFORMERS_LN_TYPES_.append(ln_class)
        logger.info(f'_TRANSFORMERS_LN_TYPES_ : {_TRANSFORMERS_LN_TYPES_}')

    @torch.no_grad()
    def collect_first_block_input(self, calib_data, padding_mask=None):
        first_block_input = defaultdict(list)

        Catcher = self.get_catcher(first_block_input)

        if not self.use_cpu_to_save_cuda_mem_for_catcher:
            self.move_embed_to_device('cuda')
            if self.vision_model:
                self.vision_model.cuda()
            if self.vision_projector:
                self.vision_projector.cuda()
            if self.audio_model:
                self.audio_model.cuda()
            if self.audio_projector:
                self.audio_projector.cuda()
            self.blocks[0] = self.blocks[0].cuda()
        self.blocks[0] = Catcher(self.blocks[0])

        for data in calib_data:
            data = {
                k: (v.cuda() if torch.is_tensor(v) else v)
                for k, v in data.items()
            }
            try:
                if not self.mm_model:
                    self.model(**data)
                else:
                    self.mm_model.generate(**data, max_new_tokens=128, do_sample=False)
            except ValueError:
                pass
        self.first_block_input = first_block_input
        assert len(self.first_block_input) > 0, 'Catch input data failed.'
        if padding_mask:
            for idx in range(len(self.first_block_input['data'])):
                token_num = self.first_block_input['data'][idx].shape[1]
                if token_num != padding_mask[idx].shape[1]:
                    padding_mask[idx] = F.pad(
                        padding_mask[idx],
                        self.get_one_pad_setting(
                            self.tokenizer.padding_side,
                            token_num - padding_mask[idx].shape[1]
                        ),
                        value=1
                    )
        self.padding_mask = padding_mask
        if not self.use_cpu_to_save_cuda_mem_for_catcher:
            if self.vision_model:
                self.vision_model.cpu()
            if self.vision_projector:
                self.vision_projector.cpu()
            if self.audio_model:
                self.audio_model.cpu()
            if self.audio_projector:
                self.audio_projector.cpu()
            self.blocks[0] = self.blocks[0].cpu()
            self.move_embed_to_device('cpu')
        self.blocks[0] = self.blocks[0].module

    def get_one_pad_setting(self, padding_side, length):
        if padding_side == 'left':
            return [0, length]
        elif padding_side == 'right':
            return [length, 0]
        else:
            raise Exception(f'Not support padding_side: {padding_side}.')

    def get_first_block_input(self):
        return self.first_block_input

    def get_padding_mask(self):
        return self.padding_mask

    def get_model_config(self):
        return self.model_config

    def move_embed_to_device(self, device):
        for embed_layer in self.get_embed_layers():
            embed_layer.to(device)
        for attention_rotary_layer in self.get_attention_rotary_layers():
            attention_rotary_layer.to(device)

    def get_block_linears(self, block):
        return {
            name: m
            for name, m in block.named_modules()
            if isinstance(m, tuple(_LLMC_LINEAR_TYPES_ + _TRANSFORMERS_LINEAR_TYPES_))
        }

    def get_all_linears(self, module):
        return {
            name: m
            for name, m in module.named_modules()
            if isinstance(m, tuple(_LLMC_LINEAR_TYPES_ + _TRANSFORMERS_LINEAR_TYPES_))
        }

    def get_extra_modules(self, block):
        return {}

    def get_moe_gate(self, block):
        return None

    def replace_module_all(self, module, params_dict, keep_device=False):
        for block_idx in range(len(self.blocks)):
            logger.info(f'Replace block index: {block_idx}/{len(self.blocks)}')
            if keep_device:
                self.replace_module_block(module, self.blocks[block_idx], block_idx, params_dict)
            else:
                self.blocks[block_idx].cuda()
                self.replace_module_block(module, self.blocks[block_idx], block_idx, params_dict)
                self.blocks[block_idx].cpu()
            gc.collect()
            torch.cuda.empty_cache()

    def replace_module_block(self, module, block, block_idx, params_dict):
        if module in _LLMC_LN_TYPES_ + _TRANSFORMERS_LN_TYPES_:
            self.replace_module_layernorm(
                module, block, self.get_layernorms_in_block(block), block_idx, params_dict
            )
        else:
            self.replace_module_subset(module,
                                       block,
                                       {'layers': self.get_block_linears(block)},
                                       block_idx,
                                       params_dict)

    def replace_module_subset(self, module, block, subset, block_idx, params_dict):
        if module in _LLMC_LINEAR_TYPES_ + _TRANSFORMERS_LINEAR_TYPES_:
            layers_dict = {
                name: layer for name, layer in subset['layers'].items()
                if isinstance(layer, tuple(_LLMC_LINEAR_TYPES_ + _TRANSFORMERS_LINEAR_TYPES_))
            }
        else:
            layers_dict = subset['layers']

        for name, m in layers_dict.items():
            if hasattr(m, 'no_quant') and m.no_quant:
                continue

            M = module.new(m, **params_dict)

            name_tmp = name.rsplit('.', 1)
            if len(name_tmp) == 2:
                parent_name = name_tmp[0]
                parent = block.get_submodule(parent_name)
                child_name = name_tmp[1]
            elif len(name_tmp) == 1:
                parent = block
                child_name = name_tmp[0]

            setattr(parent, child_name, M)
            del M

            logger.info(f'replace >>> {name} in {block_idx}-th block')

        del layers_dict
        gc.collect()
        torch.cuda.empty_cache()

    def replace_module_layernorm(self, module, block, lns, i, params_dict):
        for name, m in lns.items():
            if isinstance(m, module):
                continue
            M = module.new(m, **params_dict)

            name_tmp = name.rsplit('.', 1)
            if len(name_tmp) == 2:
                parent_name = name_tmp[0]
                parent = block.get_submodule(parent_name)
                child_name = name_tmp[1]
            elif len(name_tmp) == 1:
                parent = block
                child_name = name_tmp[0]

            setattr(parent, child_name, M)
            del M

        del lns
        gc.collect()
        torch.cuda.empty_cache()

    def convert_dtype(self, dtype='torch.float16'):
        for i in range(len(self.blocks)):
            self.blocks[i] = self.blocks[i].to(dtype)
