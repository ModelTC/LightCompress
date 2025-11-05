import copy
import functools
import gc
import json
import os
import re
from collections import defaultdict
from functools import partial

import torch
import torch.distributed as dist
import torch.nn as nn
from loguru import logger

from llmc.utils.registry_factory import KV_REGISTRY, TOKEN_REDUCTION_REGISTRY

from ..blockwise_optimization import BlockwiseOpt
from .attn_utils import _LLMC_ATTN_MAP_
from .auto_clip import AutoClipper
from .utils import is_fp8_supported_gpu

if is_fp8_supported_gpu():
    from .kernel import weight_cast_to_bf16, weight_cast_to_fp8
    logger.info('Successfully imported Triton kernel.')
else:
    from .quant import weight_cast_to_bf16, weight_cast_to_fp8
    logger.info(
        'Triton kernel not available: non-Hopper GPU detected.\n'
        'Using LLMC Quantizer implementation instead.'
    )

from .hadamard_utils import apply_exact_had_to_linear, get_hadK
from .module_utils import (_LLMC_LINEAR_TYPES_, _LLMC_LN_TYPES_,
                           _REALQUANT_LINEAR_MAP_, _TRANSFORMERS_LINEAR_TYPES_,
                           _TRANSFORMERS_LN_TYPES_, EffcientFakeQuantLinear,
                           FakeQuantLinear, LlmcActFn, OriginFloatLinear,
                           RotateLinear)
from .quant import FloatQuantizer, IntegerQuantizer, Weight48IntegerQuantizer


class BaseBlockwiseQuantization(BlockwiseOpt):
    def __init__(self, model, quant_config, input, padding_mask, config):
        super().__init__(model, quant_config, input, padding_mask, config)
        self.set_quant_config()

    def w_qdq(self, module, wquantizer):
        args = {'lowbound_factor': None, 'upbound_factor': None}
        if hasattr(module, 'buf_lowbound_factor'):
            args['lowbound_factor'] = module.buf_lowbound_factor
        if hasattr(module, 'buf_upbound_factor'):
            args['upbound_factor'] = module.buf_upbound_factor

        if module.weight.data.dtype == torch.float8_e4m3fn:
            tmp_weight \
                = weight_cast_to_bf16(module.weight,
                                      module.weight_scale_inv,
                                      self.fp8_block_size).to(torch.bfloat16)
        else:
            tmp_weight = module.weight

        tmp_weight = wquantizer.fake_quant_weight_dynamic(tmp_weight, args)

        if module.weight.data.dtype == torch.float8_e4m3fn:
            tmp_weight, module.weight_scale_inv.data \
                = weight_cast_to_fp8(tmp_weight, self.fp8_block_size)

        return tmp_weight

    def w_q(self, module, wquantizer):
        return wquantizer.real_quant_weight_dynamic(module.weight.data)

    def a_qdq(self, act, module, aquantizer, input_index=0):
        if self.act_static:
            args = {
                'scales': (getattr(module, f'buf_act_scales_{input_index}', None)),
                'zeros': (getattr(module, f'buf_act_zeros_{input_index}', None)),
                'qmax': (getattr(module, f'buf_act_qmax_{input_index}', None)),
                'qmin': (getattr(module, f'buf_act_qmin_{input_index}', None)),
            }
            return aquantizer.fake_quant_act_static(act, args)
        else:
            return aquantizer.fake_quant_act_dynamic(act)

    def get_replacement_params(self, mode='fake_quant', w_only=False, name=None):
        params_dict = {}
        if mode in ['fake_quant', 'fake_quant_wo_kv']:
            params_dict['a_qdq'] = (
                partial(self.a_qdq, aquantizer=self.aquantizer)
                if not w_only
                else None
            )
            params_dict['w_qdq'] = partial(self.w_qdq, wquantizer=self.wquantizer)

        elif mode in _REALQUANT_LINEAR_MAP_.keys():
            params_dict['w_q'] = partial(self.w_q, wquantizer=self.wquantizer)
            params_dict['quant_config'] = self.quant_config

        elif mode == 'online_rotate':
            had_K, K = get_hadK(
                self.intermediate_size if 'down_proj' in name else self.num_heads
            )
            params_dict = {
                'had_K': had_K,
                'K': K,
                'online_full_had': 'down_proj' in name,
                'online_partial_had': 'o_proj' in name,
                'had_dim': (
                    None if 'down_proj' in name else self.hidden_size // self.num_heads
                ),
                'fp32_had': self.fp32_had,
            }

        elif mode == 'quant_attn':
            params_dict = {
                'matmul_a1_qdq': partial(
                    self.a_qdq, aquantizer=self.aquantizer, input_index=0
                ),
                'matmul_a2_qdq': partial(
                    self.a_qdq, aquantizer=self.aquantizer, input_index=1
                ),
                'softmax_a_qdq': (
                    partial(self.a_qdq, aquantizer=self.aquantizer)
                    if self.quant_softmax
                    else None
                ),
            }

        elif mode == 'quant_act_fn':
            params_dict = {'a_qdq': partial(self.a_qdq, aquantizer=self.aquantizer)}

        return params_dict

    def set_quant_config(self):
        if self.model.torch_dtype == torch.float8_e4m3fn:
            self.fp8_block_size = self.model.fp8_block_size

        if 'ignored_layers' in self.config:
            self.mixed_precision = True
            self.ignored_block_ids = self.config.ignored_layers.get('block_ids', [])
            self.ignored_layer_names = self.config.ignored_layers.get('layer_names', [])
            self.ignored_speical_names = self.config.ignored_layers.get('speical_names', [])
        else:
            self.mixed_precision = False
        logger.info(f'mixed_precision = {self.mixed_precision}')

        self.quant_out = self.quant_config.get('quant_out', False)
        self.tp = self.quant_config.get('tp', 1)
        self.quant_config['weight']['tp'] = self.tp

        # select quantizer
        # weight
        quant_type = self.quant_config['weight'].get('quant_type', 'int-quant')
        if quant_type == 'int-quant':
            if self.quant_config['weight']['bit'] == 48:
                self.weight_quant_module = Weight48IntegerQuantizer
            else:
                self.weight_quant_module = IntegerQuantizer
        elif quant_type == 'float-quant':
            self.weight_quant_module = FloatQuantizer
        logger.info(f'The used Weight Quant Module is {self.weight_quant_module}')
        self.wquantizer = self.weight_quant_module(**self.quant_config['weight'])

        # act
        if 'act' in self.quant_config:
            if self.quant_config['weight']['granularity'] == 'per_block':
                assert self.quant_config['act']['granularity'] == 'per_group'
                assert self.quant_config['act']['group_size'] \
                    == self.quant_config['weight']['block_size']
            self.w_only = False
            quant_type = self.quant_config['act'].get('quant_type', 'int-quant')
            if quant_type == 'int-quant':
                if self.quant_config['act']['bit'] == 48:
                    self.act_quant_module = Weight48IntegerQuantizer
                else:
                    self.act_quant_module = IntegerQuantizer
            elif quant_type == 'float-quant':
                self.act_quant_module = FloatQuantizer
            self.quant_config['act']['tp'] = self.tp
            self.aquantizer = self.act_quant_module(**self.quant_config['act'])
            self.act_static = self.quant_config['act'].get('static', False)
            if self.act_static:
                assert (
                    self.quant_config['act']['granularity'] == 'per_tensor'
                ), 'Only support per_tensor static quant'
            self.quant_attn = self.quant_config['act'].get('quant_attn', False)
            if self.quant_attn:
                assert self.config['model']['type'] in ['Vit', 'DeepseekV2']
                self.quant_softmax = self.quant_config['act'].get(
                    'quant_softmax', False
                )
            self.quant_act_fn = self.quant_config['act'].get('quant_act_fn', False)
        else:
            self.w_only = True
            self.aquantizer = None
            self.act_static = False
            self.quant_attn = False
            self.quant_softmax = False
            self.quant_act_fn = False

        # set kv cache quant config
        if 'kvcache' in self.quant_config:
            self.quant_config['kvcache']['static'] = self.act_static
            kv_special_cfg = self.quant_config['kvcache'].get('special', {})
            act_static_cfg = {}
            if self.act_static:
                act_static_cfg.update(self.config.calib.n_sample)
                act_static_cfg.update(self.config.calib.bs)
            kv_quant_type = self.quant_config['kvcache'].get('quant_type', 'int-quant')
            self.kv_module = KV_REGISTRY[self.quant_config['kvcache']['method']](
                kv_quant_type, self.quant_config['kvcache'],
                self.model.model_config.num_hidden_layers, **kv_special_cfg, **act_static_cfg
            )
            self.quant_kvcache = True
            self.model.kvcache_buffer.append(self.kv_module)
        else:
            self.quant_kvcache = False

        # set special quant config
        special_config = self.quant_config.get('special', {})
        self.true_sequential = special_config.get('true_sequential', False)

        # set weight clip config
        self.weight_clip = special_config.get('weight_clip', False)
        if self.weight_clip or special_config.get('search_clip_init', False):
            self.save_clip = special_config.get('save_clip', False)
            if self.save_clip:
                self.clip_path = special_config['clip_path']
            self.clip_version = special_config.get('clip_version', 'v1')
            if self.clip_version == 'v2':
                assert self.wquantizer.calib_algo == 'learnable'
            clip_sym = special_config.get('clip_sym', self.wquantizer.sym)
            self.auto_clipper = AutoClipper(
                w_only=self.w_only,
                wquantizer=self.wquantizer,
                aquantizer=self.aquantizer,
                clip_version=self.clip_version,
                clip_sym=clip_sym,
                save_clip=self.save_clip,
                padding_mask=self.padding_mask,
            )

        # set transformation config
        self.save_scale = special_config.get('save_scale', False)
        if self.save_scale:
            self.scale_path = special_config['scale_path']
            self.act_scales = {}

        # set online-rotation config
        self.online_rotate = special_config.get('online_rotate', False)
        if self.online_rotate:
            assert (
                self.config['model']['type'] in ['Opt', 'Llama']
            ), 'Please set online_rotate=False'
            self.fp32_had = special_config.get('fp32_had', False)
        if self.quant_config.modality != 'video_gen':
            self.set_model_config()
        self.modality = self.quant_config.modality
        logger.info(f'self.quant_objects : {self.quant_config.modality}')

        # set token reduction config
        if 'token_reduction' in self.quant_config:
            token_reduction_cfg = self.quant_config['token_reduction']
            TOKEN_REDUCTION_REGISTRY[self.quant_config['token_reduction']['method']](
                token_reduction_cfg, self.model, self.blocks
            )

        self.do_gqa_trans = special_config.get('do_gqa_trans', False)
        logger.info(f'self.do_gqa_trans : {self.do_gqa_trans}')

    def set_model_config(self):
        self.hidden_size = self.model.model_config.hidden_size
        self.num_heads = self.model.model_config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        if hasattr(self.model.model_config, 'intermediate_size'):
            self.intermediate_size = self.model.model_config.intermediate_size
        if hasattr(self.model.model_config, 'num_key_value_heads'):
            self.num_key_value_heads = self.model.model_config.num_key_value_heads
            self.num_key_value_groups = self.num_heads // self.num_key_value_heads
            if self.num_key_value_groups > 1:
                self.has_gqa = True
            else:
                self.has_gqa = False
        else:
            self.has_gqa = False

    def replace_rotate_linears(self, block):
        for n, m in block.named_modules():
            if isinstance(m, nn.Linear) and (
                'down_proj' in n or 'o_proj' in n or 'fc2' in n or 'out_proj' in n
            ):
                subset = {'layers': {n: m}}
                self.model.replace_module_subset(
                    RotateLinear,
                    block,
                    subset,
                    None,
                    self.get_replacement_params(
                        mode='online_rotate', w_only=self.w_only, name=n
                    ),
                )

    def replace_act_fn(self, block, extra_modules):
        act_fn_dict = self.model.get_act_fn_in_block(block)
        layers_dict = {'layers': act_fn_dict}
        self.model.replace_module_subset(
            LlmcActFn,
            block,
            layers_dict,
            self.block_idx,
            self.get_replacement_params(
                mode='quant_act_fn', w_only=self.w_only, name=None
            ),
        )
        extra_modules.update(act_fn_dict)

    def replace_attention(self, block, extra_modules):
        attn_layers_dict = self.model.get_attn_in_block(block)
        layers_dict = {'layers': attn_layers_dict}
        attn_module = _LLMC_ATTN_MAP_[self.config['model']['type']]
        self.model.replace_module_subset(
            attn_module,
            block,
            layers_dict,
            self.block_idx,
            self.get_replacement_params(
                mode='quant_attn', w_only=self.w_only, name=None
            ),
        )

        matmul_modules = self.model.get_matmul_in_block(block)
        softmax_modules = (
            self.model.get_softmax_in_block(block) if self.quant_softmax else {}
        )
        extra_modules.update(matmul_modules)
        extra_modules.update(softmax_modules)

    @torch.no_grad()
    def collect_block_qparams(self, block):
        named_linears = self.model.get_block_linears(block)
        for n, m in named_linears.items():
            args = {}
            if hasattr(m, 'buf_lowbound_factor'):
                args['lowbound_factor'] = m.buf_lowbound_factor
            if hasattr(m, 'buf_upbound_factor'):
                args['upbound_factor'] = m.buf_upbound_factor

            if m.weight.data.dtype == torch.float8_e4m3fn:
                tmp_weight_data = weight_cast_to_bf16(m.weight.data,
                                                      m.weight_scale_inv.data,
                                                      self.fp8_block_size).to(torch.bfloat16)
            else:
                tmp_weight_data = m.weight.data

            (
                tensor,
                scales,
                zeros,
                max_int,
                min_int,
            ) = self.wquantizer.get_tensor_qparams(tmp_weight_data, args=args)

            m.register_buffer('buf_scales', scales.detach())
            m.register_buffer('buf_zeros', zeros.detach())
            m.register_buffer('buf_qmax', torch.tensor(max_int).to(self.dev))
            m.register_buffer('buf_qmin', torch.tensor(min_int).to(self.dev))

    def block_forward(self, block, input_data=None):
        output = []

        if input_data is None:
            input_data = self.input['data']

        for i in range(len(input_data)):
            input_data[i] = input_data[i].to(device=next(block.parameters()).device)
            for k in self.input['kwargs'][i]:
                if torch.is_tensor(self.input['kwargs'][i][k]):
                    self.input['kwargs'][i][k] = self.input['kwargs'][i][k].to(
                        device=next(block.parameters()).device
                    )
                if isinstance(self.input['kwargs'][i][k], tuple):
                    self.input['kwargs'][i][k] = tuple(
                        tmp.to(device=next(block.parameters()).device)
                        for tmp in self.input['kwargs'][i][k]
                    )
            with torch.no_grad():
                out = block(input_data[i], **self.input['kwargs'][i])
                if isinstance(out, tuple):
                    out = out[0]
                output.append(out)
        return output

    def block_opt(self, block):

        if self.quant_kvcache:
            self.register_kv_cache(block)

        block = block.cuda()
        named_linears = self.model.get_block_linears(block)
        extra_modules = self.model.get_extra_modules(block)

        if self.quant_attn:
            self.replace_attention(block, extra_modules)
        if self.quant_act_fn:
            self.replace_act_fn(block, extra_modules)

        input_feat_modules = {
            k: v for d in [named_linears, extra_modules] for k, v in d.items()
        }
        logger.info(f'input_feat_modules: {input_feat_modules}')
        input_feat = defaultdict(list)

        handles = self.register_hooks(input_feat_modules, input_feat)

        self.block_init(block)

        self.run(block, input_feat, handles)

        block = block.cpu()
        del input_feat, block
        gc.collect()
        torch.cuda.empty_cache()

    def register_hooks(self, input_feat_modules, input_feat):
        handles = []
        if not self.data_free:
            for name in input_feat_modules:
                handles.append(
                    input_feat_modules[name].register_forward_hook(
                        functools.partial(
                            self.cache_input_hook, name=name, feat_dict=input_feat
                        )
                    )
                )
        return handles

    def run(self, block, input_feat, handles):
        if not self.data_free:
            if self.quant_out:
                self.block_forward(block)
            else:
                self.input['data'] = self.block_forward(block)

            for h in handles:
                h.remove()
            torch.cuda.empty_cache()

            self.block_transform(block, input_feat, self.input['kwargs'])
        else:
            self.block_transform(block)

        if not self.data_free and self.quant_out:
            self.model.replace_module_block(
                FakeQuantLinear,
                block,
                self.block_idx,
                self.get_replacement_params(
                    mode='fake_quant', w_only=self.w_only, name=None
                ),
            )
            self.set_non_linear_mode('fake_quant', block, False)
            self.input['data'] = self.block_forward(block)
        torch.cuda.empty_cache()

    def block_transform(self, block, input_feat, block_kwargs):
        logger.info(f'Start transform the {self.block_idx}-th block')
        subsets = self.model.get_subsets_in_block(block)

        if self.act_static:
            self.register_non_linear_qparams(block, input_feat)

        self.set_non_linear_mode('fake_quant', block, False)

        for index, subset in enumerate(subsets):
            logger.info(f'subset: {subset}')
            layers_dict = subset['layers']
            input_name = subset['input'][0]
            inspect_has_kwargs = subset['has_kwargs']
            if inspect_has_kwargs:
                if 'sub_keys' in subset:
                    subset_kwargs = []
                    for i in range(len(block_kwargs)):
                        for k, v in subset['sub_keys'].items():
                            subset_kwargs.append({k: block_kwargs[i][v]})
                else:
                    subset_kwargs = block_kwargs
            else:
                subset_kwargs = {}
            self.subset_transform(
                subset,
                input_feat,
                subset_kwargs,
            )
            if self.act_static:
                input_tensors = copy.deepcopy(input_feat[input_name])
                self.register_act_qparams(layers_dict, input_tensors)
                del input_tensors

            if self.true_sequential and index != len(subsets) - 1:
                next_subset = subsets[index + 1]
                input_feat_subset = self.rehook_next_subset(block, subset, next_subset)
                input_feat.update(input_feat_subset)

        self.set_non_linear_mode('fake_quant', block, True)
        logger.info(f'End transform the {self.block_idx}-th block')

    def rehook_next_subset(self, block, subset, next_subset):
        self.subset_init(next_subset)
        self.model.replace_module_subset(
            FakeQuantLinear,
            block,
            subset,
            self.block_idx,
            self.get_replacement_params(
                mode='fake_quant', w_only=self.w_only, name=None
            ),
        )

        input_feat_subset = defaultdict(list)
        input_feat_modules = next_subset['layers']
        handles = self.register_hooks(input_feat_modules, input_feat_subset)

        self.block_forward(block)
        for h in handles:
            h.remove()

        return input_feat_subset

    def collect_layers_weights(self, layers, tensor_parallelize_style=None):
        weights = []
        for _m in layers:
            if _m.weight.data.dtype == torch.float8_e4m3fn:
                fp8_scale = _m.weight_scale_inv
                tmp_weight = weight_cast_to_bf16(_m.weight, fp8_scale).to(torch.bfloat16)
                weights.append(tmp_weight)
            else:
                weights.append(_m.weight)
        return weights

    @torch.no_grad()
    def register_kv_cache(self, block):
        attn_layers_dict = self.model.get_attn_in_block(block)
        attn_layer = attn_layers_dict[list(attn_layers_dict.keys())[0]]
        setattr(attn_layer, 'kvcache', self.kv_module)
        attn_layer.register_forward_pre_hook(
            self.kv_cache_input_hook(attn_layer), with_kwargs=True
        )

    @torch.no_grad()
    def register_non_linear_qparams(self, block, input_feat):
        layer_types = [
            ('quant_attn', self.model.get_matmul_in_block),
            ('quant_softmax', self.model.get_softmax_in_block, 'quant_attn'),
            ('quant_act_fn', self.model.get_act_fn_in_block),
        ]

        for mode, layer_func, *dependency in layer_types:
            if getattr(self, mode, True) and all(
                getattr(self, dep, True) for dep in dependency
            ):
                layers_dict = layer_func(block)
                for name, layer in layers_dict.items():
                    input_tensors = copy.deepcopy(input_feat[name])
                    self.register_act_qparams({name: layer}, input_tensors)
                    del input_tensors

    @torch.no_grad()
    def register_act_qparams(self, layers_dict, act_tensors):
        scales_list, zeros_list, qmin_list, qmax_list = (
            self.aquantizer.get_batch_tensors_qparams(act_tensors)
        )
        world_size = int(os.environ['WORLD_SIZE'])

        for i, (scales, zeros, qmin, qmax) in enumerate(
            zip(scales_list, zeros_list, qmin_list, qmax_list)
        ):
            scales = scales.cuda()
            dist.all_reduce(scales, op=dist.ReduceOp.SUM)
            scales = scales / world_size

            for name, layer in layers_dict.items():
                if not isinstance(
                    layer, tuple(_LLMC_LINEAR_TYPES_ + _TRANSFORMERS_LINEAR_TYPES_)
                ):
                    continue
                layer.register_buffer(f'buf_act_scales_{i}', scales)
                layer.register_buffer(f'buf_act_zeros_{i}', zeros.cuda())
                layer.register_buffer(f'buf_act_qmin_{i}', qmin.cuda())
                layer.register_buffer(f'buf_act_qmax_{i}', qmax.cuda())

    @torch.no_grad()
    def repeat_gqa_scales(self, scales):
        scales = scales.view(1, self.num_key_value_heads, self.head_dim)
        scales = torch.repeat_interleave(scales, dim=1, repeats=self.num_key_value_groups)
        return scales

    @torch.no_grad()
    def apply_scale(self, scales, prev_op, layers):
        assert (
            len(prev_op) == 1
        ), 'Only support single prev_op. If multi prev_ops, code need to be updated.'
        if isinstance(
            prev_op[0], tuple(_LLMC_LINEAR_TYPES_ + _TRANSFORMERS_LINEAR_TYPES_)
        ):
            assert len(layers) == 1
            logger.info('apply scale between fc and fc')
            self.scale_fc_fc(prev_op[0], layers[0], scales)
        elif isinstance(prev_op[0], tuple(_LLMC_LN_TYPES_ + _TRANSFORMERS_LN_TYPES_)):
            logger.info('apply scale between ln and fc')
            self.scale_ln_fcs(prev_op[0], layers, scales)
        else:
            raise NotImplementedError(f'prev_op {type(prev_op[0])} not supported yet!')

    @torch.no_grad()
    def apply_shift(self, shifts, prev_op, layers):
        if shifts is None:
            return

        assert (
            len(prev_op) == 1
        ), 'Only support single prev_op. If multi prev_ops, code need to be updated.'
        if isinstance(
            prev_op[0], tuple(_LLMC_LINEAR_TYPES_ + _TRANSFORMERS_LINEAR_TYPES_)
        ):
            assert len(layers) == 1
            self.shift_fc_fc(prev_op[0], layers[0], shifts)
        elif isinstance(prev_op[0], tuple(_LLMC_LN_TYPES_ + _TRANSFORMERS_LN_TYPES_)):
            self.shift_ln_fcs(prev_op[0], layers, shifts)
        else:
            raise NotImplementedError(f'prev_op {type(prev_op[0])} not supported yet!')

    @torch.no_grad()
    def scale_fc_fc(self, fc1, fc2, scales):
        scales = scales.to(fc1.weight.device)
        if fc1.out_features == fc2.in_features * 3:
            logger.info('fc1.out_features == fc2.in_features * 3')
            num_heads = self.model.get_num_attention_heads()
            fc1.weight.t_()
            org_shape = fc1.weight.shape
            fc1.weight.data = fc1.weight.data.reshape(org_shape[0] * num_heads, 3, -1)
            value = fc1.weight.data[:, 2, :].reshape(org_shape[0], -1)
            fc1.weight.data[:, 2, :] = value.div(scales.view(-1)).reshape(
                fc1.weight[:, 2, :].shape
            )
            fc1.weight.data = fc1.weight.data.reshape(org_shape).t_()
            if hasattr(fc1, 'bias') and fc1.bias is not None:
                fc1.bias.data = fc1.bias.data.reshape(num_heads, 3, -1)

                value = fc1.bias.data[:, 2, :].reshape(-1)

                fc1.bias.data[:, 2, :] = value.div(scales.view(-1)).reshape(
                    fc1.bias[:, 2, :].shape
                )
                fc1.bias.data = fc1.bias.data.reshape(-1)
        elif fc1.out_features == fc2.in_features * 2:
            logger.info('fc1.out_features == fc2.in_features * 2')
            fc1.weight.data[fc1.weight.data.shape[0] // 2:].div_(scales.view(-1, 1))
            if hasattr(fc1, 'bias') and fc1.bias is not None:
                fc1.bias.data[fc1.bias.data.shape[0] // 2:].div_(scales.view(-1))
        elif fc1.out_features == fc2.in_features:
            logger.info('fc1.out_features == fc2.in_features')
            assert fc1.out_features == fc2.in_features

            if hasattr(fc1, 'bias') and fc1.bias is not None:
                fc1.bias.div_(scales.view(-1))

            if fc1.weight.data.dtype == torch.float8_e4m3fn:
                fp8_scale = fc1.weight_scale_inv
                tmp_weight_data = weight_cast_to_bf16(fc1.weight.data,
                                                      fp8_scale,
                                                      self.fp8_block_size).to(torch.bfloat16)
                tmp_weight_data.div_(scales.view(-1, 1))

                fc1.weight.data, fc1.weight_scale_inv.data \
                    = weight_cast_to_fp8(tmp_weight_data, self.fp8_block_size)
            else:
                fc1.weight.div_(scales.view(-1, 1))

        elif self.has_gqa and self.do_gqa_trans:
            if hasattr(fc1, 'bias') and fc1.bias is not None:
                fc1.bias.div_(scales.view(-1))
            fc1.weight.div_(scales.view(-1, 1))

            if fc1.out_features != fc2.in_features:
                logger.info('GQA scale this fc-fc.')
                scales = self.repeat_gqa_scales(scales)
        else:
            logger.error(f'fc1.out_features: {fc1.out_features}')
            logger.error(f'fc2.in_features: {fc2.in_features}')
            raise Exception('Can not scale this fc-fc.')

        if fc2.weight.data.dtype == torch.float8_e4m3fn:
            fp8_scale = fc2.weight_scale_inv
            tmp_weight_data = weight_cast_to_bf16(fc2.weight.data,
                                                  fp8_scale,
                                                  self.fp8_block_size).to(torch.bfloat16)
            tmp_weight_data.mul_(scales.view(1, -1))
            fc2.weight.data, fc2.weight_scale_inv.data \
                = weight_cast_to_fp8(tmp_weight_data, self.fp8_block_size)
        else:
            fc2.weight.mul_(scales.view(1, -1))

    @torch.no_grad()
    def shift_fc_fc(self, fc1, fc2, shifts):
        if fc1.out_features == fc2.in_features * 3:
            num_heads = self.model.get_model_config().to_dict().get('n_head', None)
            if hasattr(fc1, 'bias') and fc1.bias is not None:
                fc1.bias.data = fc1.bias.data.reshape(num_heads, 3, -1)

                value = fc1.bias.data[:, 2, :].reshape(-1)
                fc1.bias.data[:, 2, :] = (value - shifts).reshape(
                    fc1.bias[:, 2, :].shape
                )
                fc1.bias.data = fc1.bias.data.reshape(-1)
        else:
            assert fc1.out_features == fc2.in_features

            if hasattr(fc1, 'bias') and fc1.bias is not None:
                fc1.bias.sub_(shifts)

        if hasattr(fc2, 'bias') and fc2.bias is not None:
            fc2.bias.add_(fc2.weight @ shifts)
        else:
            if hasattr(self, 'use_shift') and self.use_shift:
                del fc2.bias
                fc2.register_buffer('bias', fc2.weight @ shifts)

    @torch.no_grad()
    def shift_ln_fcs(self, ln, fcs, shifts):
        if not isinstance(fcs, list):
            fcs = [fcs]

        if self.model.has_bias():
            ln.bias.sub_(shifts)

        for fc in fcs:
            if self.model.has_bias():
                fc.bias.add_(fc.weight @ shifts)
            else:
                if hasattr(self, 'use_shift') and self.use_shift:
                    del fc.bias
                    fc.register_buffer('bias', fc.weight @ shifts)

        for p in ln.parameters():
            assert torch.isnan(p).sum() == 0
        for fc in fcs:
            for p in fc.parameters():
                assert torch.isnan(p).sum() == 0

    @torch.no_grad()
    def scale_ln_fcs(self, ln, fcs, scales):
        if not isinstance(fcs, list):
            fcs = [fcs]

        scales = scales.to(ln.weight.device)
        scales = scales.to(ln.weight.dtype)

        ln.weight.div_(scales)

        if hasattr(ln, 'bias') and ln.bias is not None:
            ln.bias.div_(scales)

        for fc in fcs:
            if fc.weight.data.dtype == torch.float8_e4m3fn:
                fp8_scale = fc.weight_scale_inv.data
                tmp_weight_data = weight_cast_to_bf16(fc.weight.data,
                                                      fp8_scale,
                                                      self.fp8_block_size).to(torch.bfloat16)
                tmp_weight_data.mul_(scales.view(1, -1))
                fc.weight.data, fc.weight_scale_inv.data \
                    = weight_cast_to_fp8(tmp_weight_data, self.fp8_block_size)
            else:
                fc.weight.mul_(scales.view(1, -1))

        for p in ln.parameters():
            assert torch.isnan(p).sum() == 0
        for fc in fcs:
            for p in fc.parameters():
                assert torch.isnan(p).sum() == 0

    def rotate_pre_layers(self, pre_layers, Q):
        for layer in pre_layers:
            if layer.weight.data.dtype == torch.float8_e4m3fn:
                layer.weight.data \
                    = weight_cast_to_bf16(layer.weight.data,
                                          layer.weight_scale_inv.data,
                                          self.fp8_block_size).to(torch.bfloat16)
            dtype = layer.weight.dtype
            layer.weight.data = torch.matmul(layer.weight.data.double(), Q).to(dtype)

            if hasattr(layer, 'weight_scale_inv'):
                layer.weight.data, layer.weight_scale_inv.data \
                    = weight_cast_to_fp8(layer.weight.data, self.fp8_block_size)
            torch.cuda.empty_cache()

    def rotate_post_layers(self, post_layers, Q, exact_had=False):
        for layer in post_layers:
            if layer.weight.data.dtype == torch.float8_e4m3fn:
                layer.weight.data \
                    = weight_cast_to_bf16(layer.weight.data,
                                          layer.weight_scale_inv.data,
                                          self.fp8_block_size).to(torch.bfloat16)
            dtype = layer.weight.dtype
            layer.weight.data = torch.matmul(Q.T, layer.weight.data.double()).to(dtype)

            if exact_had and self.online_rotate:
                apply_exact_had_to_linear(layer, had_dim=-1, output=False)

            if hasattr(layer, 'bias') and layer.bias is not None:
                b = layer.bias.data.to(torch.float64)
                layer.bias.data = torch.matmul(Q.T, b).to(dtype)

            if hasattr(layer, 'weight_scale_inv'):
                layer.weight.data, layer.weight_scale_inv.data \
                    = weight_cast_to_fp8(layer.weight.data, self.fp8_block_size)
            torch.cuda.empty_cache()

    def rotate_embeddings(self, Q):
        embeddings = self.model.get_embed_layers()
        assert len(embeddings) == 1
        for layer in embeddings:
            dtype = layer.weight.data.dtype
            W = layer.weight.data.to(device=self.dev, dtype=torch.float64)
            layer.weight.data = torch.matmul(W, Q).to(device='cpu', dtype=dtype)

    def rotate_head(self, Q):
        heads = self.model.get_head_layers()
        for layer in heads:
            dtype = layer.weight.data.dtype
            W = layer.weight.data.to(device=self.dev, dtype=torch.float64)
            layer.weight.data = torch.matmul(W, Q).to(device='cpu', dtype=dtype)

    def fuse_ln_fcs(self, ln, fcs):
        for fc in fcs:
            if fc.weight.data.dtype == torch.float8_e4m3fn:
                fc.weight.data \
                    = weight_cast_to_bf16(fc.weight.data,
                                          fc.weight_scale_inv.data,
                                          self.fp8_block_size).to(torch.bfloat16)
            fc_dtype = fc.weight.dtype
            if hasattr(ln, 'bias') and ln.bias is not None:
                W = fc.weight.data.double().clone()
            fc.weight.data = (fc.weight.data.double() * ln.weight.double()).to(fc_dtype)
            if hasattr(ln, 'bias') and ln.bias is not None:
                if fc.bias is None:
                    fc.bias = torch.nn.Parameter(
                        torch.zeros(fc.out_features, dtype=torch.float64)
                    )
                fc.bias.data = fc.bias.data.double().to(device=W.device) + torch.matmul(
                    W, ln.bias.double()
                )
                fc.bias.data = fc.bias.data.to(fc_dtype)

            if hasattr(fc, 'weight_scale_inv'):
                fc.weight.data, fc.weight_scale_inv.data \
                    = weight_cast_to_fp8(fc.weight.data, self.fp8_block_size)
            torch.cuda.empty_cache()

    def remove_mean_from_embed(self):
        embeddings = self.model.get_embed_layers()
        for layer in embeddings:
            W = layer.weight.data.double()
            layer.weight.data = (W - W.mean(dim=-1, keepdim=True)).to(
                layer.weight.data.dtype
            )

    def bake_mean_into_fc(self, fc):
        fc_dtype = fc.weight.dtype
        W_ = fc.weight.data.double()
        fc.weight.data = W_ - W_.mean(dim=-2, keepdim=True)
        fc.weight.data = fc.weight.data.to(fc_dtype)
        if hasattr(fc, 'bias') and fc.bias is not None:
            b_ = fc.bias.data.double()
            fc.bias.data = b_ - b_.mean()
            fc.bias.data = fc.bias.data.to(fc_dtype)

    @torch.no_grad()
    def scaling_input(self, x, scales, is_gqa):
        if is_gqa:
            scales_tmp = self.repeat_gqa_scales(scales)
        else:
            scales_tmp = scales
        if hasattr(self, '_bs') and self._bs < x.shape[0]:
            x_tmp = torch.empty_like(x)
            for i, batch in enumerate(x):
                batch_scale = scales_tmp.view(1, -1)
                x_tmp[i] = batch / batch_scale
        else:
            x_tmp = x / scales_tmp.view(1, -1)
        return x_tmp

    @torch.no_grad()
    def update_input_feat(self, scale, input_feat, layers_dict, is_gqa):
        for layer_name in layers_dict:
            for i in range(len(input_feat[layer_name])):
                inp = input_feat[layer_name][i]
                scale = scale.to(inp.device)
                input_feat[layer_name][i] = self.scaling_input(inp, scale, is_gqa)

    @torch.no_grad()
    def set_non_linear_mode(self, quant_format, module, mode):
        assert mode in [True, False]
        if quant_format != 'fake_quant':
            return
        for name, m in module.named_modules():
            if 'kvcache' in name:
                continue
            if getattr(m, 'calib', None) is not None:
                m.calib = mode

    def set_no_quant_layer(self):
        if self.ignored_speical_names:
            assert hasattr(self.model, 'block_name_prefix'), \
                'block_name_prefix missing in model'
        ignored_block_ids = []
        for item in self.ignored_block_ids:
            match = re.match(r'(\d+)-(\d+)', str(item))
            if match:
                start, end = int(match.group(1)), int(match.group(2))
                ignored_block_ids.extend(range(start, end + 1))
            else:
                ignored_block_ids.append(int(item))

        for idx, block in enumerate(self.blocks):
            for n, m in block.named_modules():
                if idx in ignored_block_ids and n in self.ignored_layer_names:
                    m.register_buffer('no_quant', torch.tensor(True))
                else:
                    layer_name = f'{self.model.block_name_prefix}.{idx}.{n}'
                    if layer_name in self.ignored_speical_names:
                        m.register_buffer('no_quant', torch.tensor(True))

    @torch.no_grad()
    def deploy(self, quant_format, keep_device=False):
        logger.info(f'-- deploy_{quant_format}_model start --')
        logger.info(f'quant_config : {self.quant_config}')

        module_mapping = {
            'origin_float': OriginFloatLinear,
            'fake_quant': EffcientFakeQuantLinear,
            'fake_quant_wo_kv': EffcientFakeQuantLinear,
        }
        module_mapping.update(_REALQUANT_LINEAR_MAP_)

        if quant_format not in module_mapping:
            raise NotImplementedError(
                f"Quant format '{quant_format}' is not implemented."
            )
        if self.mixed_precision and 'quant' in quant_format:
            self.set_no_quant_layer()

        module = module_mapping[quant_format]

        self.model.set_modality(self.modality)
        logger.info(f'set modality: {self.modality}')
        if self.modality in ('vision', 'language', 'video_gen'):
            self.model.replace_module_all(
                module,
                self.get_replacement_params(mode=quant_format, w_only=self.w_only),
                keep_device=keep_device,
            )

        self.set_non_linear_mode(quant_format, self.model.model, False)

        if self.quant_kvcache:
            if quant_format == 'origin_float':
                self.kv_module.use_org_kv = True
            elif quant_format == 'fake_quant_wo_kv':
                self.kv_module.use_org_kv = True
            elif quant_format == 'fake_quant':
                self.kv_module.use_org_kv = False
                if self.act_static:
                    self.kv_module.calib = False

        if self.model.mm_model is not None:
            logger.info(f'Now, the mm_model is: {self.model.mm_model}')

        logger.info(f'-- deploy_{quant_format}_model done --')

    @torch.no_grad()
    def copy_tokenizer(self, path):
        if self.model.tokenizer is not None:
            self.model.tokenizer.save_pretrained(path)
            logger.info('copy tokenizer done --')
        else:
            logger.info('no tokenizer, skip --')

    @torch.no_grad()
    def contiguous_params(self):
        if self.model.mm_model is not None:
            for name, param in self.model.mm_model.named_parameters():
                if not param.is_contiguous():
                    param.data = param.data.contiguous()

            for name, param in self.model.mm_model.named_buffers():
                if not param.is_contiguous():
                    param.data = param.data.contiguous()
        else:
            for name, param in self.model.model.named_parameters():
                if not param.is_contiguous():
                    param.data = param.data.contiguous()

            for name, param in self.model.model.named_buffers():
                if not param.is_contiguous():
                    param.data = param.data.contiguous()

    @torch.no_grad()
    def save_model(self, path):
        if int(os.environ['RANK']) != 0:
            return
        self.contiguous_params()
        if self.config.model.type in ['Llava', 'InternVL2', 'Mllama', 'Qwen2vl']:
            self.model.vlm_model.language_model = self.model.get_model()
            self.model.vlm_model.save_pretrained(path)
            logger.info('save model done --')
            self.copy_tokenizer(path)
        elif self.config.model.type in ['Qwen2Audio']:
            self.model.alm_model.language_model = self.model.get_model()
            self.model.alm_model.save_pretrained(path)
            logger.info('save model done --')
            self.copy_tokenizer(path)
        elif self.config.model.type in ['InternOmni']:
            self.model.avlm_model.language_model = self.model.get_model()
            self.model.avlm_model.save_pretrained(path)
            logger.info('save model done --')
            self.copy_tokenizer(path)
        else:
            self.model.get_model().save_pretrained(path)
            logger.info('save model done --')
            self.copy_tokenizer(path)
