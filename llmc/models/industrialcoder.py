"""
IndustrialCoder (IQuestCoder) model adapter for LLMC quantization.

Model structure follows IQuestCoderForCausalLM / IQuestCoderModel:
  - model.model.embed_tokens, model.model.layers, model.model.norm, model.model.rotary_emb
  - model.lm_head
  - Each layer: input_layernorm, self_attn (q_proj, k_proj, v_proj, o_proj),
    post_attention_layernorm, mlp (gate_proj, up_proj, down_proj)

Layout is the same as Qwen2-style decoders; this module provides a dedicated
adapter so IndustrialCoder is supported as its own model type, not as Qwen2.
"""

from importlib.metadata import version

import packaging

from llmc.utils.registry_factory import MODEL_REGISTRY

from .base_model import BaseModel


@MODEL_REGISTRY
class IndustrialCoder(BaseModel):
    """IndustrialCoder (IQuestCoder) – standalone adapter for blockwise quantization."""

    def __init__(self, config, device_map=None, use_cache=False):
        super().__init__(config, device_map, use_cache)

    def find_blocks(self):
        # IQuestCoderForCausalLM.model -> IQuestCoderModel with .layers
        self.blocks = self.model.model.layers

    def find_embed_layers(self):
        base = self.model.model
        self.embed_tokens = base.embed_tokens
        if hasattr(base, 'rotary_emb') and (
            packaging.version.parse(version('transformers')) >= packaging.version.parse('4.45.0')
        ):
            self.rotary_emb = base.rotary_emb

    def find_block_name(self):
        self.block_name_prefix = 'model.layers'

    def get_embed_layers(self):
        return [self.embed_tokens]

    def get_attn_in_block(self, block):
        return {'self_attn': block.self_attn}

    def get_attention_rotary_layers(self):
        if packaging.version.parse(version('transformers')) >= packaging.version.parse('4.45.0'):
            return [self.rotary_emb] if hasattr(self, 'rotary_emb') and self.rotary_emb is not None else []
        return []

    def get_head_layers(self):
        return [self.model.lm_head]

    def get_pre_head_layernorm_layers(self):
        return [self.model.model.norm]

    def get_layers_except_blocks(self):
        if packaging.version.parse(version('transformers')) >= packaging.version.parse('4.45.0'):
            rotary = [self.rotary_emb] if hasattr(self, 'rotary_emb') and self.rotary_emb is not None else []
            return [self.embed_tokens] + rotary + [self.model.model.norm, self.model.lm_head]
        return [self.embed_tokens, self.model.model.norm, self.model.lm_head]

    def skip_layer_name(self):
        return ['lm_head']

    def has_bias(self):
        # IQuestCoder config: attention_bias, mlp_bias (often False)
        cfg = self.model_config
        return getattr(cfg, 'attention_bias', False) or getattr(cfg, 'mlp_bias', False)

    def get_layernorms_in_block(self, block):
        return {
            'input_layernorm': block.input_layernorm,
            'post_attention_layernorm': block.post_attention_layernorm,
        }

    def get_subsets_in_block(self, block):
        # Same layout as Qwen2 / IQuestCoderDecoderLayer
        return [
            {
                'layers': {
                    'self_attn.q_proj': block.self_attn.q_proj,
                    'self_attn.k_proj': block.self_attn.k_proj,
                    'self_attn.v_proj': block.self_attn.v_proj,
                },
                'prev_op': [block.input_layernorm],
                'input': ['self_attn.q_proj'],
                'inspect': block.self_attn,
                'has_kwargs': True,
            },
            {
                'layers': {'self_attn.o_proj': block.self_attn.o_proj},
                'prev_op': [block.self_attn.v_proj],
                'input': ['self_attn.o_proj'],
                'inspect': block.self_attn.o_proj,
                'has_kwargs': False,
            },
            {
                'layers': {
                    'mlp.gate_proj': block.mlp.gate_proj,
                    'mlp.up_proj': block.mlp.up_proj,
                },
                'prev_op': [block.post_attention_layernorm],
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
