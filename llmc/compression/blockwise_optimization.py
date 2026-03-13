import os
from abc import ABCMeta, abstractmethod

import torch
from loguru import logger


class BlockwiseOpt(metaclass=ABCMeta):
    def __init__(self, model, compress_config, input, padding_mask, config):
        self.model = model
        self.blocks = model.get_blocks()
        self.quant_config = compress_config
        self.sparsity_config = compress_config
        self.input = input
        self.padding_mask = padding_mask
        self.data_free = False if self.input else True
        self.config = config
        self.block_idx = None
        self.num_blocks = len(self.blocks)
        if self.input:
            for i in range(len(input['kwargs'])):
                if 'use_cache' in input['kwargs'][i]:
                    input['kwargs'][i].pop('use_cache')
            for i in range(len(input['kwargs'])):
                if 'past_key_value' in input['kwargs'][i]:
                    input['kwargs'][i]['past_key_value'] = None
            self.n_samples = 0
            for i in range(len(input['data'])):
                self.n_samples += input['data'][i].shape[0]

    def run_block_loop(self):
        for i in range(len(self.blocks)):
            self.block_idx = i
            if self.input and hasattr(self.model, 'get_blockwise_input'):
                self.input = self.model.get_blockwise_input(self.block_idx, self.input)
            logger.info(
                f'\nblock index: {self.block_idx}/{len(self.blocks)} '
                f'\nblock: {self.blocks[self.block_idx]}'
            )
            self.block_opt(self.blocks[self.block_idx])
            if self.input and hasattr(self.model, 'set_blockwise_input'):
                self.model.set_blockwise_input(self.block_idx, self.input)

        if hasattr(self, 'save_scale') and self.save_scale:
            os.makedirs(self.scale_path, exist_ok=True)
            torch.save(self.act_scales, os.path.join(self.scale_path, 'scales.pth'))
            if hasattr(self, 'act_shifts') and self.act_shifts:
                torch.save(self.act_shifts, os.path.join(self.scale_path, 'shifts.pth'))

        if hasattr(self, 'save_clip') and self.save_clip:
            os.makedirs(self.clip_path, exist_ok=True)
            torch.save(
                self.auto_clipper.weight_clips,
                os.path.join(self.clip_path, 'clips.pth'),
            )

    def cache_input_hook(self, m, x, y, name, feat_dict):
        inputs = [i.detach().cpu() for i in x]
        if len(inputs) == 1:
            inp = inputs[0]
            if len(inp.shape) == 2:
                inp = inp.unsqueeze(0)
            feat_dict[name].append(inp)
        else:
            feat_dict[name].append(tuple(inputs))

    def kv_cache_input_hook(self, attn_layer):
        def hook_fn(module, args, kwargs):
            kvcache = getattr(module, 'kvcache')
            kwargs['past_key_value'] = kvcache
            if self.config.eval.get('type', None) == 'decode_ppl':
                # For eval decoding PPL (Perplexity).
                past_seen_tokens = kvcache.get_seq_length()
                cache_position = torch.arange(
                    past_seen_tokens,
                    past_seen_tokens + kwargs['hidden_states'].shape[1],
                    device=kwargs['hidden_states'].device,
                )
                kwargs['cache_position'] = cache_position
                position_ids = cache_position.unsqueeze(0)
                kwargs['position_ids'] = position_ids
                if 'position_embeddings' in kwargs:
                    kwargs['position_embeddings'] = self.model.rotary_emb(
                        kwargs['hidden_states'], position_ids
                    )
            if kwargs['hidden_states'].shape[1] == 1:
                from .sparsification.kvsparse import ShadowKVCache
                if isinstance(kvcache, ShadowKVCache):
                    hidden_states = kwargs['hidden_states'][:, -1, :].unsqueeze(0)
                    kwargs['hidden_states'] = hidden_states
                    bsz, q_len, _ = hidden_states.size()
                    tmp_query_states = \
                        attn_layer.q_proj(hidden_states).view(bsz,
                                                              q_len,
                                                              -1,
                                                              attn_layer.head_dim).transpose(1, 2)
                    retrieval_position_ids = \
                        kvcache.get_retrieval_position_ids(layer_idx=attn_layer.layer_idx,
                                                           query_states=tmp_query_states)
                    kwargs['retrieval_position_ids'] = retrieval_position_ids
                    kwargs['cos_sin_cache'] = self.cos_sin_cache

            return args, kwargs

        return hook_fn

    @abstractmethod
    def block_opt(self, block):
        pass

    def layer_init(self, layer):
        pass

    def subset_init(self, subset):
        pass

    def block_init(self, block):
        pass
