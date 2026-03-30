import functools
import gc
from collections import defaultdict

import torch
from loguru import logger

from llmc.utils import copy_files
from llmc.utils.registry_factory import KV_REGISTRY

from ..blockwise_optimization import BlockwiseOpt
from .attn_utils import _LLMC_ATTN_MAP_


class BaseBlockwiseSparsification(BlockwiseOpt):
    def __init__(self, model, sparsity_config, input, padding_mask, config):
        super().__init__(model, sparsity_config, input, padding_mask, config)
        self.set_sparsity_config()

    def block_init(self, block):
        pass

    def set_sparsity_config(self):
        if 'sparsity_out' in self.sparsity_config and self.sparsity_config[
            'sparsity_out'
        ]:
            self.sparsity_out = True
        else:
            self.sparsity_out = False
        logger.info(f'use sparsity_out {self.sparsity_out}')

        # set kv cache sparse config
        if 'kvcache' in self.sparsity_config:
            self.sparse_kvcache = True
            self.set_kv_sparse_config()
        else:
            self.sparse_kvcache = False

        if 'weight' in self.sparsity_config:
            if 'sparsity' in self.sparsity_config['weight']:
                self.sparsity = self.sparsity_config['weight']['sparsity']
                self.W_mask = None
            elif 'n_prune_layers' in self.sparsity_config['weight']:
                self.n_prune_layers = self.sparsity_config['weight']['n_prune_layers']

    def set_kv_sparse_config(self):
        kv_sparse_config = {}
        if self.sparsity_config['kvcache']['method'] == 'ShadowKV':
            assert self.config['model']['type'] in ['Llama']
            assert self.config['eval'].get('type', None) != 'decode_ppl'
            inv_freq = \
                self.model.model.model.layers[0].self_attn.rotary_emb.inv_freq.cuda()
            cos_cache, sin_cache = self.set_cos_sin_cache(inv_freq)
            self.cos_sin_cache = (cos_cache, sin_cache)
            kv_sparse_config['config'] = self.model.model_config
        elif self.sparsity_config['kvcache']['method'] == 'SinkKV':
            kv_sparse_config['num_hidden_layers'] = self.model.model_config.num_hidden_layers
            kv_sparse_config['window_length'] = self.sparsity_config['kvcache']['window_length']
            kv_sparse_config['num_sink_tokens'] = self.sparsity_config['kvcache']['num_sink_tokens']
        self.kv_module = KV_REGISTRY[self.sparsity_config['kvcache']['method']](**kv_sparse_config)
        self.replace_attn = self.sparsity_config['kvcache'].get('replace_attn', False)
        self.model.kvcache_buffer.append(self.kv_module)

    def set_cos_sin_cache(self, inv_freq):
        max_length = 64 * 1024
        t = torch.arange(max_length + 1024, device=torch.device('cuda'), dtype=inv_freq.dtype)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(torch.bfloat16), emb.sin().to(torch.bfloat16)

    @torch.no_grad()
    def register_kv_cache(self, block):
        attn_layers_dict = self.model.get_attn_in_block(block)
        attn_layer = attn_layers_dict[list(attn_layers_dict.keys())[0]]
        setattr(attn_layer, 'kvcache', self.kv_module)
        attn_layer.register_forward_pre_hook(
            self.kv_cache_input_hook(attn_layer), with_kwargs=True
        )

    def replace_attention(self, block):
        attn_layers_dict = self.model.get_attn_in_block(block)
        layers_dict = {'layers': attn_layers_dict}
        kv_method = self.sparsity_config['kvcache']['method']
        model_type = self.config['model']['type']
        attn_module = _LLMC_ATTN_MAP_[kv_method][model_type]
        self.model.replace_module_subset(
            attn_module,
            block,
            layers_dict,
            self.block_idx,
            {}
        )

    def block_forward(self, block, input_data=None):
        output = []
        if input_data is None:
            input_data = self.input['data']

        for i in range(len(input_data)):
            input_data[i] = input_data[i].to(device=next(block.parameters()).device)
            if 'attention_mask' in self.input[
                'kwargs'
            ][i] and self.input['kwargs'][i]['attention_mask'] is not None:
                self.input['kwargs'][i]['attention_mask'] = self.input['kwargs'][i][
                    'attention_mask'
                ].cuda()
            with torch.no_grad():
                out = block(input_data[i], **self.input['kwargs'][i])[0]
                output.append(out)
        return output

    def block_opt(self, block):
        if self.sparse_kvcache:
            if self.replace_attn:
                self.replace_attention(block)
            self.register_kv_cache(block)
        block = block.cuda()

        if not self.data_free:
            named_linears = self.model.get_block_linears(block)
            logger.info(f'named_linears: {named_linears}')
            input_feat = defaultdict(list)
            handles = []
            self.block_init(block)

            for name in named_linears:
                handles.append(
                    named_linears[name].register_forward_hook(
                        functools.partial(
                            self.cache_input_hook, name=name, feat_dict=input_feat
                        )
                    )
                )

            if not self.sparsity_out:
                self.input['data'] = self.block_forward(block)
            else:
                self.block_forward(block)
            for h in handles:
                h.remove()
            torch.cuda.empty_cache()

            self.block_transform(block, input_feat, self.input['kwargs'])

            if self.sparsity_out:
                self.input['data'] = self.block_forward(block)

            block = block.cpu()
            del input_feat
            gc.collect()
            torch.cuda.empty_cache()

        else:
            self.block_transform(block)

    def block_transform(self, block, input_feat, block_kwargs):
        logger.info(f'Start transform the {self.block_idx + 1}-th block')
        subsets = self.model.get_subsets_in_block(block)
        for index, subset in enumerate(subsets):
            if not self.filter_subset(subset):
                continue
            # logger.info(f"subset: {subset}")
            prev_op = subset['prev_op']
            layers_dict = subset['layers']
            input_name = subset['input'][0]
            inspect_module = subset['inspect']
            inspect_has_kwargs = subset['has_kwargs']
            subset_kwargs = block_kwargs if inspect_has_kwargs else {}
            self.subset_transform(
                layers_dict,
                input_feat,
                prev_op,
                input_name,
                inspect_module,
                subset_kwargs
            )
        logger.info(f'End transform the {self.block_idx + 1}-th block')

    def filter_subset(self, subset):
        return True

    @torch.no_grad()
    def deploy(self, deploy_format):
        logger.info('-- deploy_sparsity_model start --')
        logger.info(f'sparsity_config : {self.sparsity_config}')
        logger.info('-- deploy_sparsity_model done --')

    @torch.no_grad()
    def copy_tokenizer(self, path):
        self.model.tokenizer.save_pretrained(path)
        logger.info('copy tokenizer done --')

    @torch.no_grad()
    def save_model(self, path):
        if self.config.model.type == 'Llava':
            self.model.llava_model.language_model = self.model.get_model()
            self.model.llava_model.save_pretrained(path)
            logger.info('save model done --')
            self.copy_tokenizer(path)
            copy_files(self.config.model.path, path, 'preprocessor_config')
        else:
            self.model.get_model().save_pretrained(path)
            logger.info('save model done --')
            self.copy_tokenizer(path)
