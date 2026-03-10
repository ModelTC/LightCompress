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
            self.blocks = list(self.Pipeline.transformer.blocks) + list(
                self.Pipeline.transformer_2.blocks
            )
            logger.info(
                'Wan2.2 MoE: both experts wrapped (high-noise + low-noise, 80 blocks total).'
            )
        else:
            self.blocks = list(self.Pipeline.transformer.blocks)
            logger.info('Wan2.2: single transformer wrapped (40 blocks).')
        logger.info('Model: %s', self.model)

    def find_llmc_model(self):
        self.model = self.Pipeline.transformer

    def find_blocks(self):
        self.blocks = self.model.blocks

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
        first_block_input = defaultdict(list)
        Catcher = self.get_catcher(first_block_input)
        # Install Catcher on the pipeline's first block so forward passes go through it.
        first_block = self.Pipeline.transformer.blocks[0]
        self.Pipeline.transformer.blocks[0] = Catcher(first_block)
        self.Pipeline.to('cuda')
        for data in calib_data:
            self.Pipeline.transformer.blocks[0].step = 0
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

        self.first_block_input = first_block_input
        assert len(self.first_block_input['data']) > 0, 'Catch input data failed.'
        self.n_samples = len(self.first_block_input['data'])
        logger.info('Retrieved %s calibration samples for Wan2.2 T2V.', self.n_samples)
        self.Pipeline.transformer.blocks[0] = self.Pipeline.transformer.blocks[0].module
        self.Pipeline.to('cpu')

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
