import copy
import functools
import gc
import math
import os
import random
from contextlib import nullcontext
from math import inf
from random import sample

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from tqdm import tqdm

from llmc.utils.registry_factory import ALGO_REGISTRY

from .base_blockwise_quantization import BaseBlockwiseQuantization
from .module_utils import FakeQuantLinear, RectifiedSigmoid
from .train_utils import AvgMeter, LossFunction, NativeScalerWithGradNormCount


@ALGO_REGISTRY
class TesseraQ(BaseBlockwiseQuantization):
    def __init__(self, model, quant_config, input, padding_mask, config):
        super().__init__(model, quant_config, input, padding_mask, config)
        self.add_quant_config()

        self.attention_mask = self.input['kwargs'][0].get('attention_mask')
        model_type = self.config['model']['type']
        self.position_ids = (
            self.input['kwargs'][0].get('position_ids')
            if model_type in ['Llama', 'Mistral', 'Qwen2']
            else None
        )

        if self.deactive_amp:
            self.batch_mask = self._repeat_attention_mask()
        else:
            self.batch_mask = (
                self._repeat_attention_mask().float()
                if self.attention_mask is not None
                else None
            )

        self.dev = torch.device('cuda')
        self.model_dtype = next(self.model.model.parameters()).dtype
        logger.info('self model dtype: {}'.format(self.model_dtype))

        self.sigmoid = RectifiedSigmoid(-0.1, 1.1)

    def _repeat_attention_mask(self):
        if self.attention_mask is not None:
            return self.attention_mask.repeat(
                self.input['data'][0].shape[0], 1, 1, 1
            ).cuda()
        return None

    def w_q(self, module, wquantizer):
        args = {}
        if self.optimize_scale:
            args['output_scale_factor'] = 2 * self.sigmoid(module.buf_output_scale_factor)
        if hasattr(module, 'buf_upbound_factor'):
            args['upbound_factor'] = module.buf_upbound_factor
            args['lowbound_factor'] = None
        if hasattr(module, 'buf_lowbound_factor'):
            args['lowbound_factor'] = module.buf_lowbound_factor

        return wquantizer.real_quant_weight_dynamic(module.weight.data, args)

    def add_quant_config(self):
        self.prefix = self.model.block_name_prefix
        self.loss_func = LossFunction(method='l2')
        special_config = self.quant_config.get('special', {})

        self.deactive_amp = special_config.get('deactive_amp', False)
        self.wd = special_config.get('wd', None)
        self.lr = special_config.get('lr', None)
        self.iterations = special_config.get('iterations', 0)
        self.batch_size = special_config.get('batch_size', 1)
        self.optimize_scale = special_config.get('optimize_scale', False)
        self.thresholds = special_config.get('thresholds', [])
        self.load_transform = special_config.get('load_transform', False)
        self.reduce_memory = special_config.get('reduce_memory', False)

        if self.load_transform:
            assert 'scale_path' in special_config, \
                'scale_path must be specified when load_transform is True'
            self.scale_path = special_config['scale_path']
            self.act_scales = torch.load(os.path.join(self.scale_path, 'scales.pth'),
                                         map_location='cpu')
            for k in self.act_scales:
                self.act_scales[k] = self.act_scales[k].to(torch.float32)

        self.scale_lr = special_config.get('scale_lr', None)

        if self.deactive_amp:
            self.dtype = torch.float
            self.traincast = nullcontext
        else:
            self.dtype = torch.bfloat16
            self.traincast = torch.cuda.amp.autocast

        self.aug_loss = special_config.get('aug_loss', None)

        if self.weight_clip and self.clip_version == 'v2':
            self.wquantizer.calib_algo = 'learnable'
            self.clip_path = special_config.get('clip_path', None)
            if self.clip_path:
                self.weight_clips = torch.load(os.path.join(self.clip_path, 'clips.pth'),
                                               map_location='cpu')

        self.change_ratio = {}

    def block_forward(self, block, input_data=None):
        output = []

        if input_data is None:
            input_data = self.input['data']

        for i in range(len(input_data)):
            input_data[i] = input_data[i].to(device=next(block.parameters()).device)
            if (
                'attention_mask' in self.input['kwargs'][i]
                and self.input['kwargs'][i]['attention_mask'] is not None
            ):
                self.input['kwargs'][i]['attention_mask'] = self.input['kwargs'][i][
                    'attention_mask'
                ].cuda()
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    out = block(input_data[i], **self.input['kwargs'][i])[0]
                    output.append(out)
        return output

    def get_original_out(self, block):
        if self.block_idx == 0:
            self.ori_out = self.block_forward(block)
            if self.aug_loss:
                self.ori_out2 = self.ori_out
        else:
            self.ori_out = self.block_forward(block, self.ori_out)
            if self.aug_loss:
                self.ori_out2 = self.block_forward(block)

    @torch.no_grad()
    def collect_block_qparams(self, block, input_feat):
        named_linears = self.model.get_block_linears(block)
        for n, m in named_linears.items():
            args = {}
            if hasattr(m, 'buf_lowbound_factor'):
                args['lowbound_factor'] = m.buf_lowbound_factor
            if hasattr(m, 'buf_upbound_factor'):
                args['upbound_factor'] = m.buf_upbound_factor
            (
                tensor,
                scales,
                zeros,
                max_int,
                min_int,
            ) = self.wquantizer.get_tensor_qparams(m.weight.data, args=args)
            m.register_buffer('buf_scales', scales)
            m.register_buffer('buf_zeros', zeros)
            m.register_buffer('buf_qmax', torch.tensor(max_int).to(self.dev))
            m.register_buffer('buf_qmin', torch.tensor(min_int).to(self.dev))

            if self.act_static:
                subsets = self.model.get_subsets_in_block(block)
                for index, subset in enumerate(subsets):
                    layers_dict = subset['layers']
                    input_name = subset['input'][0]
                    input_tensors = copy.deepcopy(input_feat[input_name])
                    self.register_act_qparams(layers_dict, input_tensors)
                    del input_tensors

    @torch.no_grad()
    def block_transform(self, block, input_feat, block_kwargs):
        logger.info(f'Start transform the {self.block_idx+1}-th block')

        with torch.no_grad():
            block.float()

        if self.online_rotate:
            self.replace_rotate_linears(block)

        for i in range(len(self.input['data'])):
            self.input['data'][i] = self.input['data'][i].to(self.dtype)
        self.get_original_out(block)  # collect block output

        if self.load_transform:
            self.tesseraq_load_transform(block, input_feat)
        if self.weight_clip:
            self.tesseraq_weight_clip(block, input_feat)

        self.collect_block_qparams(block, input_feat)  # collect quant range after transformation
        self.register_tesseraq_parameters(block)

        self.tesseraq_train(block)
        self.merge_tesseraq_parameters_and_clear_tmp(block)
        self.set_rounding_opt_mode(block, on=False)

        # convert it back to original dtype
        if self.reduce_memory:
            block.to(self.model_dtype)

        logger.info(f'End transform the {self.block_idx+1}-th block')

    def tesseraq_train(self, block):
        self.set_dynamic_tmp_quant(block, on=True)
        for n, p in block.named_parameters():
            p.requires_grad = False

        thresholds = self.thresholds
        self.input['data'] = torch.cat(self.input['data'], dim=0)
        self.ori_out = torch.cat(self.ori_out, dim=0)

        # evaluate loss before reconstruction
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                loss_prev = self.get_tesseraq_loss(
                    block, self.input['data'][:4], self.ori_out[:4]
                )
                logger.info(
                    'Before TesseraQ, the reconstruction loss: {}'.format(loss_prev.item())
                )

        for i in range(len(thresholds)):
            self.set_rounding_opt_mode(block, on=True)
            self.update_mask(block, quantile_threshold=thresholds[i])

            params_r, params_s = self.get_rounding_parameters(block)
            if self.optimize_scale:
                optimizer = torch.optim.Adam(
                    [
                        {'params': params_r, 'lr': self.lr},
                        {
                            'params': params_s,
                            'lr': self.scale_lr or self.lr,
                            'weight_decay': 1e-4,
                        },
                    ],
                    lr=self.lr,
                )
            else:
                optimizer = torch.optim.Adam(params_r, self.lr)

            loss_scaler = NativeScalerWithGradNormCount()

            with torch.enable_grad():
                for p in params_r + params_s:
                    p.requires_grad = True

                for iters in range(self.iterations):
                    indices = torch.randperm(self.config['calib']['n_samples'])[
                        : self.batch_size
                    ]

                    with self.traincast():
                        target2 = self.ori_out2[indices] if self.aug_loss else None
                        loss = self.get_tesseraq_loss(
                            block,
                            self.input['data'][indices],
                            self.ori_out[indices],
                            target2,
                        )

                    if not math.isfinite(loss.item()):
                        logger.info('Loss is NAN, stopping training')

                    optimizer.zero_grad()

                    norm = loss_scaler(loss, optimizer, parameters=params_r + params_s)

                logger.info(
                    f'block {self.block_idx} iter {i+1} loss:{loss.item():5f} \
                    norm:{norm.item():4f} HR progress:{(1-thresholds[i])*100:1f}% '
                )
                for p in params_r + params_s:
                    p.requires_grad = False

            del optimizer

        for n, m in block.named_modules():
            if isinstance(m, FakeQuantLinear):
                # set to hard masking
                m.buf_rounding = 100 * m.buf_rounding.sign()

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                loss_now = self.get_tesseraq_loss(
                    block, self.input['data'][:4], self.ori_out[:4]
                )
                self.low_now = loss_now.item()
                logger.info(
                    'After TesseraQ, the reconstruction loss: {}'.format(loss_now.item())
                )

        self.input['data'] = list(
            torch.split(self.input['data'], split_size_or_sections=1, dim=0)
        )
        self.ori_out = list(torch.split(self.ori_out, split_size_or_sections=1, dim=0))

    @torch.no_grad()
    def tesseraq_load_transform(self, block, input_feat):
        logger.info('loading scales...')
        subsets = self.model.get_subsets_in_block(block)
        for index, subset in enumerate(subsets):
            prev_op = subset['prev_op']
            layers_dict = subset['layers']
            layers = list(layers_dict.values())

            if (
                isinstance(prev_op[0], (nn.Linear, FakeQuantLinear))
                and prev_op[0].out_features != layers[0].in_features * 3
                and prev_op[0].out_features != layers[0].in_features
            ):
                logger.info('Cannot apply scale. Do not transform this subset.')
                continue

            for n in layers_dict:
                layer_name = f'{self.model.block_name_prefix}.{self.block_idx}.{n}'
            scale = self.act_scales[layer_name].cuda()
            self.apply_scale(scale, prev_op, layers)
            self.update_input_feat(scale, input_feat, layers_dict)

    @torch.no_grad()
    def update_input_feat(self, scale, input_feat, layers_dict):
        for layer_name in layers_dict:
            for i in range(len(input_feat[layer_name])):
                inp = input_feat[layer_name][i]
                inp.div_(scale.view(1, -1).to(inp.device))

    def tesseraq_weight_clip(self, block, input_feat):
        if self.clip_version == 'v1':
            self.auto_clipper.run(block, self.block_idx, input_feat, n_sample_token=512)
        elif self.clip_version == 'v2':
            logger.info('loading clips...')
            for n, m in block.named_modules():
                if isinstance(m, nn.Linear):
                    if any([_ in n for _ in ['q_', 'k_', 'query', 'key', 'Wqkv']]):
                        m.register_buffer('buf_upbound_factor', None)
                        m.register_buffer('buf_lowbound_factor', None)
                        continue
                    layer_name = f'{n}.weight_quantizer.'
                    upbound_factor = self.weight_clips[self.block_idx][
                        layer_name + 'upbound_factor'
                    ]
                    lowbound_factor = self.weight_clips[self.block_idx][
                        layer_name + 'lowbound_factor'
                    ]
                    m.register_buffer(
                        'buf_upbound_factor',
                        upbound_factor.cuda().float(),
                    )
                    m.register_buffer(
                        'buf_lowbound_factor',
                        lowbound_factor.cuda().float()
                        if lowbound_factor is not None
                        else None,
                    )

    def get_tesseraq_loss(self, block, x, target, target2=None):
        if self.position_ids is not None:
            quant_out = block(
                x, attention_mask=self.batch_mask, position_ids=self.position_ids
            )[0]
        else:
            quant_out = block(x, attention_mask=self.batch_mask)[0]

        loss = self.loss_func(target, quant_out)
        if target2 is not None:
            loss = (loss + self.loss_func(target2, quant_out)) / 2
        return loss

    def register_tesseraq_parameters(self, block):
        module = FakeQuantLinear
        self.model.replace_module_block(
            module,
            block,
            self.block_idx,
            self.get_replacement_params(
                mode='fake_quant', w_only=self.w_only, name=None
            ),
        )
        self.register_rounding_parameters(block)

    def register_rounding_parameters(self, block):
        for n, m in block.named_modules():
            if isinstance(m, FakeQuantLinear):
                rounding = m.weight.data.clone()
                scales = m.buf_scales
                rounding = self.wquantizer.reshape_tensor(rounding).div(scales)
                rounding = rounding - torch.floor(rounding)
                rounding = self.sigmoid.inverse(rounding)

                m.register_buffer('buf_rounding', rounding)

                if self.optimize_scale:
                    m.register_buffer('buf_output_scale_factor', torch.zeros_like(scales))

    @torch.no_grad()
    def update_mask(self, block, quantile_threshold):
        for n, m in block.named_modules():
            if isinstance(m, FakeQuantLinear):
                score = (self.sigmoid(m.buf_rounding) - 0.5).abs().cpu()
                value = np.quantile(score.numpy(), q=quantile_threshold)
                m.buf_rounding[self.sigmoid(m.buf_rounding) > (value + 0.5)] = float('inf')
                m.buf_rounding[self.sigmoid(m.buf_rounding) < (0.5 - value)] = -float('inf')
                del score

    def set_rounding_opt_mode(self, block, on=True):
        for n, m in block.named_modules():
            if isinstance(m, FakeQuantLinear):
                if not hasattr(m, 'buf_rounding_opt'):
                    m.register_buffer('buf_rounding_opt', torch.tensor(on))
                else:
                    m.buf_rounding_opt = torch.tensor(on)

    def set_dynamic_tmp_quant(self, block, on=True):
        for n, m in block.named_modules():
            if isinstance(m, FakeQuantLinear):
                m.dynamic_quant_tmp_weight = on

    def get_rounding_parameters(self, block):
        params_r = []
        params_s = []
        for n, m in block.named_modules():
            if isinstance(m, FakeQuantLinear):
                params_r += [m.buf_rounding]
                if self.optimize_scale:
                    params_s += [m.buf_output_scale_factor]
        return params_r, params_s

    def merge_tesseraq_parameters_and_clear_tmp(self, block):
        for n, m in block.named_modules():
            if isinstance(m, FakeQuantLinear):
                m.buf_rounding = (m.buf_rounding > 0).float()
                w_shape = m.weight.shape
                W = self.wquantizer.reshape_tensor(m.weight.data) / m.buf_scales
                m.buf_rounding = m.buf_rounding - (W - torch.floor(W) > 0.5).float()

                cr = torch.count_nonzero(m.buf_rounding) / m.buf_rounding.numel()
                if n not in self.change_ratio:
                    self.change_ratio[n] = 0
                self.change_ratio[n] = self.change_ratio[n] + cr
                logger.info('layer {}, change ratio: {}%'
                            .format(n, self.change_ratio[n] / (self.block_idx + 1)))
                m.buf_rounding *= 0.5 * m.buf_scales
                m.buf_rounding = self.wquantizer.restore_tensor(m.buf_rounding, w_shape)
                m.weight.data.add_(m.buf_rounding.to(self.model_dtype))

                delattr(m, 'buf_rounding')
                delattr(m, 'tmp_weight')
                delattr(m, 'tmp_bias')
                m.dynamic_quant_weight = False
                m.dynamic_quant_tmp_weight = False

                gc.collect()
                torch.cuda.empty_cache()

    def cache_input_hook(self, m, x, y, name, feat_dict):
        super(TesseraQ, self).cache_input_hook(m, x, y, name, feat_dict)
        if len(feat_dict[name]) > 128:
            del feat_dict[name][-1]

    def w_qdq(self, module, wquantizer):
        weight = module.weight

        args = {}
        args['scales'] = module.buf_scales
        if hasattr(module, 'buf_zeros'):
            args['zeros'] = module.buf_zeros
        else:
            args['zeros'] = None
        args['qmax'] = module.buf_qmax
        args['qmin'] = module.buf_qmin

        if hasattr(module, 'buf_rounding_opt') and module.buf_rounding_opt:
            args['rounding'] = self.sigmoid(module.buf_rounding)

        if self.optimize_scale:
            args['output_scale_factor'] = 2 * self.sigmoid(module.buf_output_scale_factor)

        weight = wquantizer.fake_quant_weight_static(weight, args)

        return weight

    def deploy(self, quant_format):
        super().deploy(quant_format)
        self.model.convert_dtype(self.model_dtype)

    def save_model(self, path):
        self.model.convert_dtype(self.model_dtype)
        super().save_model(path)
