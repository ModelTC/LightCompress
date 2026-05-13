import copy
import functools
import math
import time
from collections import defaultdict

import torch
import torch.nn as nn
import transformers
from loguru import logger

from llmc.utils.registry_factory import ALGO_REGISTRY

from .base_blockwise_quantization import BaseBlockwiseQuantization
from .module_utils import FakeQuantLinear
from .quant import IntegerQuantizer


@ALGO_REGISTRY
class SpQR(BaseBlockwiseQuantization):
    def __init__(self, model, quant_config, input, padding_mask, config):
        super().__init__(model, quant_config, input, padding_mask, config)
        assert (
            self.wquantizer.granularity == 'per_group'
        ), 'SpQR only supports per_group quantization'
        self.dev = torch.device('cuda')
        self.model_dtype = next(self.model.model.parameters()).dtype
        self.add_quant_config()
        self.layers_cache = {}
        self.model_qparams = defaultdict()

    @torch.no_grad()
    def add_quant_config(self):
        special_config = self.quant_config['special']

        self.prefix = self.model.block_name_prefix
        self.true_sequential = special_config['true_sequential']
        self.actorder = special_config['actorder']
        self.percdamp = special_config['percdamp']
        self.blocksize = special_config['blocksize']
        self.relative_threshold = special_config['relative_threshold']
        self.simplified_outliers = special_config['simplified_outliers']

        if self.wquantizer.granularity == 'per_group' and self.actorder:
            self.need_perm = True

        if self.relative_threshold == 'inf':
            self.relative_threshold = math.inf

        scale_config = special_config['scale']
        zero_config = special_config['zero']

        self.quant_type = self.quant_config.get('quant_type', 'int-quant')
        assert self.quant_type != 'float-quant', 'SPQR do not support Float quant now.'
        self.scale_quantizer = IntegerQuantizer(**scale_config)
        self.zero_quantizer = IntegerQuantizer(**zero_config)
        self.Q = IntegerQuantizer(
            self.wquantizer.bit, self.wquantizer.sym, 'per_channel', round_zp=False
        )

    @torch.no_grad()
    def block_transform_true_sequential(self, block, input_feat):

        subsets = self.model.get_subsets_in_block(block)
        for subset in subsets:
            handles = []
            self.subset_init(subset)

            for name in subset['layers']:
                handles.append(
                    subset['layers'][name].register_forward_hook(
                        functools.partial(
                            self.cache_input_hook, name=name, feat_dict=input_feat
                        )
                    )
                )
            self.block_forward(block)
            for h in handles:
                h.remove()
            torch.cuda.empty_cache()

            self.subset_transform(subset['layers'])
            self.model.replace_module_subset(
                FakeQuantLinear,
                block,
                subset,
                self.block_idx,
                self.get_replacement_params(mode='fake_quant', w_only=True),
            )

    @torch.no_grad()
    def block_transform(self, block, input_feat, *block_kwargs):
        logger.info(f'Start transform the {self.block_idx + 1}-th block')

        if self.true_sequential:
            self.block_transform_true_sequential(block, input_feat)
        else:
            layers_dict = self.model.get_block_linears(block)
            self.subset_transform(layers_dict)
            self.model.replace_module_block(
                FakeQuantLinear,
                block,
                self.get_replacement_params(mode='fake_quant', w_only=True),
            )

        logger.info(f'End transform the {self.block_idx + 1}-th block')

    @torch.no_grad()
    def subset_transform(self, layers_dict):
        for name in layers_dict:
            layer = layers_dict[name]
            self.layer_transform(layer, name)
            self.free(name)

    @torch.no_grad()
    def layer_transform(self, layer, name):
        self.qparams = {}
        self.columns = self.layers_cache[name]['columns']
        W = layer.weight.data.clone()
        if isinstance(layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(layer, transformers.Conv1D):
            W = W.t()

        W = W.float()

        tick = time.time()

        self.groups = [None] * (self.columns // self.wquantizer.group_size)

        H = self.layers_cache[name]['H']
        del self.layers_cache[name]['H']

        if self.actorder:
            self.perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, self.perm]
            H = H[self.perm][:, self.perm]
            self.invperm = torch.argsort(self.perm)
            layer.register_buffer('buf_perm', self.perm)
            layer.register_buffer('buf_invperm', self.invperm)

        dead = torch.diag(H) == 0
        if self.percdamp > 0:
            damp = self.percdamp * abs(torch.diag(H)).mean()
            diag = torch.arange(self.columns, device=self.dev)
            H[diag, diag] += damp
            del diag
        H[dead, dead] = 1
        W[:, dead] = 0

        Losses = torch.zeros_like(W)
        tmp = torch.zeros_like(W)

        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H
        mask = torch.zeros_like(W, dtype=torch.bool)
        self.weight_transform(W, Hinv, Losses, tmp, mask)

        torch.cuda.synchronize()
        logger.info(f'time {time.time() - tick}')
        logger.info(f'error {torch.sum(Losses).item()}')

        if self.actorder:
            tmp = tmp[:, self.invperm]
            mask = mask[:, self.invperm]

        if isinstance(layer, transformers.Conv1D):
            tmp = tmp.t()
            mask = mask.t()

        assert layer.weight.shape == tmp.shape
        layer.weight.data = tmp

        logger.info(f'tmp {tmp}')
        logger.info(f'outliers {torch.sum(mask)} / {mask.numel()}')

        if self.wquantizer.granularity == 'per_group':
            self.set_model_qparams(layer)
            layer.register_buffer('buf_mask', mask.float().to_sparse())

    @torch.no_grad()
    def weight_transform(self, W, Hinv, Losses, tmp, mask):
        def outliers(G, HinvGD):
            indices = torch.arange(G.shape[1], device=G.device)
            indices = indices[1:] - (indices[:, None] >= indices[1:]).to(indices.dtype)
            LooG = G[:, indices]

            _, s, z, N, P = self.Q.get_tensor_qparams(LooG.flatten(0, 1))
            LooRG = self.Q.quant_dequant(LooG.flatten(0, 1), s, z, N, P).reshape(
                LooG.shape
            )
            LooHinvGD = HinvGD[indices]
            LooError = ((LooRG - LooG) / LooHinvGD).square().sum(-1)

            _, s, z, N, P = self.Q.get_tensor_qparams(G)
            BaseRG = self.Q.quant_dequant(G, s, z, N, P)
            BaseError = ((BaseRG - G) / HinvGD).square().sum(dim=1, keepdim=True)

            return BaseError - LooError

        outlier_scale = (W.var(dim=0) / torch.diag(Hinv).square()).mean().item()
        threshold = self.relative_threshold * outlier_scale
        logger.info(f'threshold {threshold}')

        for i1 in range(0, self.columns, self.blocksize):
            i2 = min(i1 + self.blocksize, self.columns)
            Err1 = torch.zeros((W.shape[0], i2 - i1), device=W.device)
            Losses1 = torch.zeros((W.shape[0], i2 - i1), device=W.device)

            for i in range(i1, i2):
                if i % self.wquantizer.group_size == 0:
                    G = W[:, i: i + self.wquantizer.group_size]

                    if self.simplified_outliers or threshold == math.inf:
                        self.get_group_qparams(G, i)
                    else:
                        HinvGD = torch.diag(Hinv)[i: i + self.wquantizer.group_size]
                        E = outliers(G, HinvGD)
                        M = (E > threshold).float()
                        mean = torch.sum(G * (1 - M), dim=1, keepdim=True) / torch.sum(
                            1 - M, dim=1, keepdim=True
                        ).clamp_min(1)
                        newG = G * (1 - M) + mean * M
                        self.get_group_qparams(newG, i)
                        del HinvGD, E, M, mean, newG

                    del G

                q = self.wquantizer.quant_dequant(
                    W[:, i].unsqueeze(1),
                    self.qparams['scales'],
                    self.qparams['zeros'],
                    self.qparams['qmax'],
                    self.qparams['qmin'],
                ).squeeze(1)

                err = (W[:, i] - q) / Hinv[i, i]
                if threshold != math.inf:
                    mask[:, i] = err.square() > threshold
                    M = mask[:, i].float()
                    newq = q * (1 - M) + W[:, i] * M
                    err = (W[:, i] - newq) / Hinv[i, i]
                tmp[:, i] = W[:, i]
                Losses1[:, i - i1] = err.square()
                W[:, i + 1: i2] -= err.unsqueeze(1).matmul(
                    Hinv[i, i + 1: i2].unsqueeze(0)
                )
                Err1[:, i - i1] = err

            Losses[:, i1:i2] = Losses1
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

    @torch.no_grad()
    def cache_input_hook(self, m, inp, out, name, feat_dict):
        self.add_batch(self.named_layers[name], name, inp[0].data, out.data)

    @torch.no_grad()
    def add_batch(self, layer, name, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(layer, (FakeQuantLinear, nn.Linear, transformers.Conv1D)):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        if isinstance(layer, nn.Conv2d):
            unfold = nn.Unfold(
                layer.kernel_size,
                dilation=layer.dilation,
                padding=layer.padding,
                stride=layer.stride,
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)

        self.layers_cache[name]['H'] *= self.layers_cache[name]['nsamples'] / (
            self.layers_cache[name]['nsamples'] + tmp
        )
        self.layers_cache[name]['nsamples'] += tmp
        inp = math.sqrt(2 / self.layers_cache[name]['nsamples']) * inp.float()
        self.layers_cache[name]['H'] += inp.matmul(inp.t())

    @torch.no_grad()
    def layer_init(self, layer, name):
        W = layer.weight.data.clone()
        if isinstance(layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(layer, transformers.Conv1D):
            W = W.t()
        self.layers_cache[name]['H'] = torch.zeros(
            (W.shape[1], W.shape[1]), device=self.dev
        )
        self.layers_cache[name]['nsamples'] = 0
        self.layers_cache[name]['columns'] = W.shape[1]

    @torch.no_grad()
    def subset_init(self, subset):
        self.named_layers = subset['layers']
        for name in self.named_layers:
            self.layers_cache[name] = {}
            self.layer_init(self.named_layers[name], name)

    @torch.no_grad()
    def block_init(self, block):
        self.named_layers = self.model.get_block_linears(block)
        for name in self.named_layers:
            self.layers_cache[name] = {}
            self.layer_init(self.named_layers[name], name)

    @torch.no_grad()
    def merge_qparams(self, qparams):
        if isinstance(qparams, int):
            return qparams
        elif self.wquantizer.granularity == 'per_group':
            qparams = torch.stack(qparams, dim=1)
            qparams = qparams.reshape(-1, 1)
        return qparams

    @torch.no_grad()
    def get_group_qparams(self, c_tensor, idx):
        """Get qparams for a group, idx is the index of a column within a
        group, c_tensor is a group."""
        _, s, z, qmax, qmin = self.wquantizer.get_tensor_qparams(c_tensor)
        _, ss, zs, Ps, Ns = self.scale_quantizer.get_tensor_qparams(s)
        args = {}
        args['scales'] = ss
        args['zeros'] = zs
        args['qmin'] = Ns
        args['qmax'] = Ps
        scales = self.scale_quantizer.fake_quant_weight_static(s.data, args)
        _, sz, zz, Pz, Nz = self.zero_quantizer.get_tensor_qparams(z)
        args['scales'] = sz
        args['zeros'] = zz
        args['qmin'] = Nz
        args['qmax'] = Pz
        zeros = self.zero_quantizer.fake_quant_weight_static(z.data, args)
        self.qparams['scales'] = scales
        self.qparams['zeros'] = zeros
        self.qparams['qmax'] = qmax
        self.qparams['qmin'] = qmin
        qparams = copy.deepcopy(self.qparams)
        self.groups[idx // self.wquantizer.group_size] = qparams

    @torch.no_grad()
    def set_model_qparams(self, layer):
        d = defaultdict(list)
        d['scales'] = self.merge_qparams([g['scales'] for g in self.groups])
        d['zeros'] = self.merge_qparams([g['zeros'] for g in self.groups])
        for k, v in d.items():
            layer.register_buffer('buf_' + k, copy.deepcopy(v))
        layer.register_buffer('buf_qmax', torch.tensor(self.groups[0]['qmax']))
        layer.register_buffer('buf_qmin', torch.tensor(self.groups[0]['qmin']))

    @torch.no_grad()
    def free(self, name):
        del self.layers_cache[name]
        torch.cuda.empty_cache()

    @torch.no_grad()
    def w_q(self, weight, qargs):
        pass

    @torch.no_grad()
    def w_qdq(self, module, wquantizer):
        mask = module.buf_mask.to_dense()
        weight = module.weight
        out = (mask * weight).to(self.model_dtype)
        if hasattr(self, 'need_perm'):
            perm = module.buf_perm
            weight = weight[:, perm]

        args = {}
        args['scales'] = module.buf_scales
        args['zeros'] = module.buf_zeros
        args['qmax'] = module.buf_qmax
        args['qmin'] = module.buf_qmin

        weight = wquantizer.fake_quant_weight_static(weight, args).to(self.model_dtype)

        if hasattr(self, 'need_perm'):
            invperm = module.buf_invperm
            weight = weight[:, invperm]
        weight = (weight * (1 - mask) + out).to(self.model_dtype)
        return weight

    @torch.no_grad()
    def deploy(self, quant_format):
        if quant_format == 'real_quant':
            assert False, 'SpQR does not support real quantization'
        super().deploy(quant_format)

    @torch.no_grad()
    def save_model(self, path):
        self.model.convert_dtype(self.model_dtype)
        super().save_model(path)
