from llmc.utils.registry_factory import MODEL_REGISTRY

from .base_model import BaseModel


def _has_legacy_moe(block):
    return hasattr(block, 'block_sparse_moe')


@MODEL_REGISTRY
class Mixtral(BaseModel):
    def __init__(self, config, device_map=None, use_cache=False):
        super().__init__(config, device_map, use_cache)

    def find_blocks(self):
        self.blocks = self.model.model.layers

    def find_embed_layers(self):
        self.embed_tokens = self.model.model.embed_tokens

    def find_block_name(self):
        self.block_name_prefix = 'model.layers'

    def get_embed_layers(self):
        return [self.embed_tokens]

    def get_layers_except_blocks(self):
        return [self.embed_tokens, self.model.model.norm, self.model.lm_head]

    def skip_layer_name(self):
        return ['lm_head']

    def has_bias(self):
        return False

    def get_layernorms_in_block(self, block):
        return {
            'input_layernorm': block.input_layernorm,
            'post_attention_layernorm': block.post_attention_layernorm,
        }

    def get_extra_modules(self, block):
        if _has_legacy_moe(block):
            return {
                'block_sparse_moe': block.block_sparse_moe
            }
        return {
            'mlp': block.mlp
        }

    def get_moe_gate(self, block):
        if _has_legacy_moe(block):
            return block.block_sparse_moe.gate
        return block.mlp.gate

    def get_subsets_in_block(self, block):
        if _has_legacy_moe(block):
            return self._get_subsets_legacy(block)
        return self._get_subsets_fused(block)

    def _get_subsets_legacy(self, block):
        """Transformers <5.0: block.block_sparse_moe with ModuleList
        experts."""
        moe = block.block_sparse_moe
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
                    **{f'block_sparse_moe.experts.{i}.w1': moe.experts[i].w1 for i in range(len(moe.experts))},  # noqa
                    **{f'block_sparse_moe.experts.{i}.w3': moe.experts[i].w3 for i in range(len(moe.experts))},  # noqa
                    'block_sparse_moe.gate': moe.gate,
                },
                'prev_op': [block.post_attention_layernorm],
                'input': ['block_sparse_moe'],
                'inspect': moe,
                'has_kwargs': False,
                'is_mlp': True,
            },
            *[
                {
                    'layers': {f'block_sparse_moe.experts.{i}.w2': moe.experts[i].w2},
                    'prev_op': [moe.experts[i].w3],
                    'input': [f'block_sparse_moe.experts.{i}.w2'],
                    'inspect': moe.experts[i].w2,
                    'has_kwargs': False,
                    'is_mlp': True,
                }
                for i in range(len(moe.experts))
            ],
        ]

    def _get_subsets_fused(self, block):
        """Transformers >=5.0: block.mlp with fused MixtralExperts."""
        moe = block.mlp
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
                    'mlp.gate': moe.gate,
                },
                'prev_op': [block.post_attention_layernorm],
                'input': ['mlp'],
                'inspect': moe,
                'has_kwargs': False,
                'is_mlp': True,
            },
        ]
