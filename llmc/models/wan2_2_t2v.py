import gc
import inspect
from collections import defaultdict

import torch
import torch.nn as nn
from diffusers import AutoencoderKLWan, WanPipeline
from loguru import logger

from llmc.compression.quantization.module_utils import LlmcWanTransformerBlock
from llmc.utils.registry_factory import MODEL_REGISTRY

from .base_model import BaseModel


@MODEL_REGISTRY
class Wan2T2V(BaseModel):
    """Wan2.2-T2V with MoE: two experts (high-noise + low-noise), same block structure as Wan2.1."""

    def __init__(self, config, device_map=None, use_cache=False):
        super().__init__(config, device_map, use_cache)
        if 'calib' in config:
            self.calib_bs = config.calib.bs
            self.sample_steps = config.calib.sample_steps
            self.target_height = config.calib.get('target_height', 480)
            self.target_width = config.calib.get('target_width', 832)
            self.num_frames = config.calib.get('num_frames', 81)
            self.guidance_scale = config.calib.get('guidance_scale', 5.0)
            self.guidance_scale_2 = config.calib.get('guidance_scale_2', 3.0)
        else:
            self.sample_steps = None

    def build_model(self):
        vae = AutoencoderKLWan.from_pretrained(
            self.model_path,
            subfolder='vae',
            torch_dtype=torch.float32,
            use_safetensors=True,
        )
        # Wan2.2: one pipeline, two transformer experts (transformer + transformer_2).
        # Pipeline switches by SNR; both use WanTransformer3DModel with same block layout as Wan2.1.
        self.Pipeline = WanPipeline.from_pretrained(
            self.model_path,
            vae=vae,
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
        )
        self.find_llmc_model()
        # Wrap both experts with LlmcWanTransformerBlock (same as Wan2.1 per-block layout).
        for block_idx, block in enumerate(self.Pipeline.transformer.blocks):
            new_block = LlmcWanTransformerBlock.new(block)
            self.Pipeline.transformer.blocks[block_idx] = new_block
        if hasattr(self.Pipeline, 'transformer_2') and self.Pipeline.transformer_2 is not None:
            for block_idx, block in enumerate(self.Pipeline.transformer_2.blocks):
                new_block = LlmcWanTransformerBlock.new(block)
                self.Pipeline.transformer_2.blocks[block_idx] = new_block
            self.num_transformer_blocks = len(self.Pipeline.transformer.blocks)
            self.blocks = list(self.Pipeline.transformer.blocks) + list(self.Pipeline.transformer_2.blocks)
            logger.info(
                'Wan2.2 MoE: both experts wrapped (high-noise + low-noise, 80 blocks total).'
            )
        else:
            self.blocks = list(self.Pipeline.transformer.blocks)
            self.num_transformer_blocks = len(self.blocks)
            logger.info('Wan2.2: single transformer wrapped (40 blocks).')
        logger.info('Model: %s', self.model)

    def find_llmc_model(self):
        self.model = self.Pipeline.transformer

    def find_blocks(self):
        self.blocks = list(self.Pipeline.transformer.blocks)
        self.num_transformer_blocks = len(self.blocks)
        if hasattr(self.Pipeline, 'transformer_2') and self.Pipeline.transformer_2 is not None:
            self.blocks += list(self.Pipeline.transformer_2.blocks)

    def _expert_name_from_block_idx(self, block_idx):
        if block_idx < self.num_transformer_blocks:
            return 'transformer'
        return 'transformer_2'

    def get_blockwise_input(self, block_idx, fallback_input):
        if not hasattr(self, 'blockwise_inputs'):
            return fallback_input
        return self.blockwise_inputs[self._expert_name_from_block_idx(block_idx)]

    def set_blockwise_input(self, block_idx, block_input):
        if not hasattr(self, 'blockwise_inputs'):
            return
        self.blockwise_inputs[self._expert_name_from_block_idx(block_idx)] = block_input

    def get_catcher(self, first_block_input):
        sample_steps = self.sample_steps

        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module
                self.signature = inspect.signature(module.forward)
                self.step = 0

            def forward(self, *args, **kwargs):
                params = list(self.signature.parameters.keys())
                for i, arg in enumerate(args):
                    if i > 0:
                        kwargs[params[i]] = arg
                first_block_input['data'].append(args[0])
                first_block_input['kwargs'].append(kwargs)
                self.step += 1
                if self.step == sample_steps:
                    raise ValueError
                else:
                    return self.module(*args)

        return Catcher

    @torch.no_grad()
    def collect_first_block_input(self, calib_data, padding_mask=None):
        first_block_input = {
            'transformer': defaultdict(list),
            'transformer_2': defaultdict(list),
        }
        sample_steps = self.sample_steps

        class Catcher(nn.Module):
            def __init__(self, module, expert_name):
                super().__init__()
                self.module = module
                self.signature = inspect.signature(module.forward)
                self.expert_name = expert_name

            def _to_cpu(self, x):
                if torch.is_tensor(x):
                    return x.detach().cpu()
                if isinstance(x, tuple):
                    return tuple(self._to_cpu(t) for t in x)
                return x

            def forward(self, *args, **kwargs):
                params = list(self.signature.parameters.keys())
                for i, arg in enumerate(args):
                    if i > 0:
                        kwargs[params[i]] = arg
                cur_num = len(first_block_input[self.expert_name]['data'])
                if cur_num < sample_steps:
                    first_block_input[self.expert_name]['data'].append(
                        args[0].detach().cpu() if torch.is_tensor(args[0]) else args[0]
                    )
                    first_block_input[self.expert_name]['kwargs'].append(
                        {k: self._to_cpu(v) for k, v in kwargs.items()}
                    )
                if all(len(first_block_input[name]['data']) >= sample_steps for name in first_block_input):
                    raise ValueError
                return self.module(*args)

        first_block = self.Pipeline.transformer.blocks[0]
        self.Pipeline.transformer.blocks[0] = Catcher(first_block, 'transformer')
        first_block_2 = None
        if hasattr(self.Pipeline, 'transformer_2') and self.Pipeline.transformer_2 is not None:
            first_block_2 = self.Pipeline.transformer_2.blocks[0]
            self.Pipeline.transformer_2.blocks[0] = Catcher(first_block_2, 'transformer_2')

        self.Pipeline.to('cuda')
        for data in calib_data:
            try:
                pipe_kw = {
                    'prompt': data['prompt'],
                    'negative_prompt': data['negative_prompt'],
                    'height': self.target_height,
                    'width': self.target_width,
                    'num_frames': self.num_frames,
                    'guidance_scale': self.guidance_scale,
                }
                if hasattr(self, 'guidance_scale_2'):
                    pipe_kw['guidance_scale_2'] = self.guidance_scale_2
                self.Pipeline(**pipe_kw)
            except ValueError:
                pass
            gc.collect()
            torch.cuda.empty_cache()

        self.Pipeline.transformer.blocks[0] = self.Pipeline.transformer.blocks[0].module
        if first_block_2 is not None:
            self.Pipeline.transformer_2.blocks[0] = self.Pipeline.transformer_2.blocks[0].module
        self.Pipeline.to('cpu')

        assert len(first_block_input['transformer']['data']) > 0, 'Catch transformer input data failed.'
        if hasattr(self.Pipeline, 'transformer_2') and self.Pipeline.transformer_2 is not None:
            assert len(first_block_input['transformer_2']['data']) > 0, \
                'Catch transformer_2 input data failed.'

        self.blockwise_inputs = first_block_input
        self.first_block_input = self.blockwise_inputs['transformer']
        self.n_samples = sum(len(v['data']) for v in self.blockwise_inputs.values())
        logger.info(
            'Retrieved Wan2.2 calibration samples: transformer=%s, transformer_2=%s.',
            len(self.blockwise_inputs['transformer']['data']),
            len(self.blockwise_inputs['transformer_2']['data']),
        )

    def get_padding_mask(self):
        return None

    def has_bias(self):
        return True

    def __str__(self):
        return '\nWan2.2 MoE Model:\n%s\nTotal params: ~27B (14B active per step)' % (
            str(self.model),
        )

    def get_layernorms_in_block(self, block):
        return {
            'affine_norm1': block.affine_norm1,
            'norm2': block.norm2,
            'affine_norm3': block.affine_norm3,
        }

    def get_subsets_in_block(self, block):
        return [
            {
                'layers': {
                    'attn1.to_q': block.attn1.to_q,
                    'attn1.to_k': block.attn1.to_k,
                    'attn1.to_v': block.attn1.to_v,
                },
                'prev_op': [block.affine_norm1],
                'input': ['attn1.to_q'],
                'inspect': block.attn1,
                'has_kwargs': True,
                'sub_keys': {'rotary_emb': 'rotary_emb'},
            },
            {
                'layers': {
                    'attn2.to_q': block.attn2.to_q,
                },
                'prev_op': [block.norm2],
                'input': ['attn2.to_q'],
                'inspect': block.attn2,
                'has_kwargs': True,
                'sub_keys': {'encoder_hidden_states': 'encoder_hidden_states'},
            },
            {
                'layers': {
                    'ffn.net.0.proj': block.ffn.net[0].proj,
                },
                'prev_op': [block.affine_norm3],
                'input': ['ffn.net.0.proj'],
                'inspect': block.ffn,
                'has_kwargs': True,
            },
        ]

    def find_embed_layers(self):
        pass

    def get_embed_layers(self):
        pass

    def get_layers_except_blocks(self):
        pass

    def skip_layer_name(self):
        pass
