import gc
import os
import sys

import torch
from loguru import logger

from .utils import ceil_div

try:
    from qtorch.quant import float_quantize
except Exception:
    logger.warning(
        'qtorch not found, please install qtorch.'
        'Please install qtorch (pip install qtorch).'
    )
    float_quantize = None


def weight_cast_to_bf16(weight, scale, block_size):
    quantizer = FloatQuantizer(
        bit='e4m3',
        symmetric=True,
        granularity='per_block',
        block_size=block_size,
        use_qtorch=True,
    )
    scale = scale.view(scale.shape[0], 1, scale.shape[1], 1)
    org_shape = weight.shape
    weight = quantizer.reshape_tensor(weight)
    weight = quantizer.dequant(weight.float(), scale, 0)
    weight = quantizer.restore_tensor(weight, org_shape)
    return weight.to(torch.bfloat16)


def weight_cast_to_fp8(weight, block_size):
    quantizer = FloatQuantizer(
        bit='e4m3',
        symmetric=True,
        granularity='per_block',
        block_size=block_size,
        use_qtorch=True,
    )
    fp8_weight, fp8_scale, _ = quantizer.real_quant_weight_dynamic(weight)
    return fp8_weight, fp8_scale


class BaseQuantizer(object):
    def __init__(self, bit, symmetric, granularity, **kwargs):
        self.bit = bit
        self.sym = symmetric
        self.granularity = granularity
        self.kwargs = kwargs

        self.calib_algo = self.kwargs.get('calib_algo', 'minmax')

        if self.granularity == 'per_group':
            self.group_size = self.kwargs['group_size']
        elif self.granularity == 'per_head':
            self.head_num = self.kwargs['head_num']
        elif self.granularity == 'per_block':
            assert self.calib_algo == 'minmax' and self.sym
            self.block_size = self.kwargs['block_size']

        if self.kwargs.get('ste', False):
            self.round_func = lambda x: (x.round() - x).detach() + x
        else:
            self.round_func = torch.round
        if 'ste_all' in self.kwargs and self.kwargs['ste_all']:
            self.round_func = torch.round
            self.ste_all = True
        else:
            self.ste_all = False

        self.round_zp = self.kwargs.get('round_zp', True)
        self.sigmoid = torch.nn.Sigmoid()

        # mse config
        self.mse_b_num = self.kwargs.get('mse_b_num', 1)
        self.maxshrink = self.kwargs.get('maxshrink', 0.8)
        self.mse_grid = self.kwargs.get('mse_grid', 100)

        # hist config
        self.bins = self.kwargs.get('bins', 2048)
        self.upsample_rate = (
            16  # used to reduce quantization errors when upscaling histogram
        )

        # hqq config
        self.lp_norm = self.kwargs.get('lp_norm', 0.7)
        self.beta = self.kwargs.get('beta', 10)
        self.kappa = self.kwargs.get('kappa', 1.01)
        self.iters = self.kwargs.get('iters', 20)
        if self.lp_norm == 1:
            self.shrink_op = lambda x, beta: torch.sign(x) * torch.nn.functional.relu(
                torch.abs(x) - 1.0 / self.beta
            )
        else:
            self.shrink_op = lambda x, beta, p=self.lp_norm: torch.sign(
                x
            ) * torch.nn.functional.relu(
                torch.abs(x) - (1.0 / self.beta) * torch.pow(torch.abs(x), p - 1)
            )

    def reshape_batch_tensors(self, act_tensors):
        assert len(act_tensors) > 0, (
            'Calibration data is insufficient. Please provide more data to ensure '
            'all experts in the MOE receive an adequate number of tokens.'
        )

        if isinstance(act_tensors[0], tuple):
            # Handle multiple inputs by stacking tensors.
            unzipped_inputs = zip(*act_tensors)
            act_tensors = [torch.stack(tensor_list) for tensor_list in unzipped_inputs]
        else:
            if len(act_tensors) == 1:
                # Handle batch-size=-1 case.
                tensor_list = [act_tensors[0][i] for i in range(act_tensors[0].size(0))]
                act_tensors[0] = tensor_list
            else:
                act_tensors = [act_tensors]
        return act_tensors

    def get_tensor_range(self, tensor, args={}):
        if self.calib_algo == 'minmax':
            return self.get_minmax_range(tensor)
        elif self.calib_algo == 'mse':
            return self.get_mse_range(tensor)
        elif self.calib_algo == 'learnable':
            return self.get_learnable_range(tensor, **args)
        else:
            return self.get_minmax_range(tensor)

    def get_minmax_range(self, tensor):
        if self.granularity == 'per_tensor':
            max_val = torch.max(tensor)
            min_val = torch.min(tensor)
        elif self.granularity == 'per_block':
            min_val = tensor.abs().float().amin(dim=(1, 3), keepdim=True)
            max_val = tensor.abs().float().amax(dim=(1, 3), keepdim=True)
        else:
            max_val = tensor.amax(dim=-1, keepdim=True)
            min_val = tensor.amin(dim=-1, keepdim=True)

        return (min_val, max_val)

    def get_mse_range(self, tensor, norm=2.4, bs=256):

        assert (
            self.mse_b_num >= 1 and tensor.shape[0] % self.mse_b_num == 0
        ), 'Batch number must be divisible by tensor.shape[0],'
        bs = tensor.shape[0] // self.mse_b_num
        tensor = tensor.float()
        min_val, max_val = self.get_minmax_range(tensor)

        dev = tensor.device

        for b_num in range(self.mse_b_num):
            _tensor = tensor[b_num * bs : (b_num + 1) * bs, :]
            _min_val, _max_val = (
                min_val[b_num * bs : (b_num + 1) * bs, :],
                max_val[b_num * bs : (b_num + 1) * bs, :],
            )

            best = torch.full([_tensor.shape[0]], float('inf'), device=dev)

            best_min_val, best_max_val = _min_val, _max_val

            for i in range(int(self.maxshrink * self.mse_grid)):
                p = 1 - i / self.mse_grid

                xmin = p * _min_val
                xmax = p * _max_val

                if self.quant_type == 'float-quant' and not self.use_qtorch:
                    clip_tensor, scales = self.get_float_qparams(
                        _tensor, (xmin, xmax), dev
                    )
                    zeros, qmin, qmax = 0, None, None
                    q_tensor = self.quant_dequant(
                        clip_tensor, scales, zeros, qmax, qmin
                    )

                else:
                    scales, zeros, qmax, qmin = self.get_qparams((xmin, xmax), dev)
                    q_tensor = self.quant_dequant(_tensor, scales, zeros, qmax, qmin)

                q_tensor -= _tensor
                q_tensor.abs_()
                q_tensor.pow_(norm)
                err = torch.sum(q_tensor, 1)

                tmp = err < best

                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    best_min_val[tmp] = xmin[tmp]
                    best_max_val[tmp] = xmax[tmp]

            (
                min_val[b_num * bs : (b_num + 1) * bs, :],
                max_val[b_num * bs : (b_num + 1) * bs, :],
            ) = (best_min_val, best_max_val)

        return (min_val, max_val)

    def get_learnable_range(self, tensor, lowbound_factor=None, upbound_factor=None):
        min_val, max_val = self.get_minmax_range(tensor)
        if self.sym:
            if upbound_factor is not None:
                abs_max = torch.max(max_val.abs(), min_val.abs())
                abs_max = abs_max.clamp(min=1e-5)
                abs_max = self.sigmoid(upbound_factor) * abs_max
                min_val = -abs_max
                max_val = abs_max
        else:
            if upbound_factor is not None and lowbound_factor is not None:
                min_val = self.sigmoid(lowbound_factor) * min_val
                max_val = self.sigmoid(upbound_factor) * max_val

        return (min_val, max_val)

    def get_minmax_stats(self, act_tensors):
        stats_min_max = {}
        for input_idx, tensors in enumerate(act_tensors):
            for tensor in tensors:
                tensor = self.reshape_tensor(tensor)
                tensor_range = self.get_minmax_range(tensor)
                min_val, max_val = tensor_range[0], tensor_range[1]

                if input_idx not in stats_min_max:
                    stats_min_max[input_idx] = {}
                    stats_min_max[input_idx]['min'] = torch.tensor(
                        [min_val], dtype=torch.float32
                    )
                    stats_min_max[input_idx]['max'] = torch.tensor(
                        [max_val], dtype=torch.float32
                    )
                else:
                    stats_min_max[input_idx]['min'] = torch.cat(
                        [
                            stats_min_max[input_idx]['min'],
                            torch.tensor([min_val], dtype=torch.float32),
                        ]
                    )
                    stats_min_max[input_idx]['max'] = torch.cat(
                        [
                            stats_min_max[input_idx]['max'],
                            torch.tensor([max_val], dtype=torch.float32),
                        ]
                    )

        return stats_min_max

    def get_static_minmax_range(self, act_tensors):
        act_tensors = self.reshape_batch_tensors(act_tensors)
        stats_min_max = self.get_minmax_stats(act_tensors)
        min_vals, max_vals = [], []
        for input_idx, tensor_range in stats_min_max.items():
            min_val = tensor_range['min'].mean()
            max_val = tensor_range['max'].mean()
            min_vals.append(min_val)
            max_vals.append(max_val)

        return min_vals, max_vals

    def get_norm(
        self, delta_begin: torch.Tensor, delta_end: torch.Tensor, density: torch.Tensor
    ) -> torch.Tensor:
        r"""Compute the norm of the values uniformaly distributed between
        delta_begin and delta_end. Currently only L2 norm is supported.

        norm = density * (integral_{begin, end} x^2)
             = density * (end^3 - begin^3) / 3
        """
        norm = (
            delta_end * delta_end * delta_end - delta_begin * delta_begin * delta_begin
        ) / 3
        return density * norm

    def get_quantization_error(
        self, histogram, min_val, max_val, next_start_bin, next_end_bin
    ):
        r"""Compute the quantization error if we use start_bin to end_bin as
        the min and max to do the quantization."""
        bin_width = (max_val.item() - min_val.item()) / self.bins

        dst_bin_width = bin_width * (next_end_bin - next_start_bin + 1) / self.dst_nbins
        if dst_bin_width == 0.0:
            return 0.0

        src_bin = torch.arange(self.bins, device=histogram.device)
        # distances from the beginning of first dst_bin to the beginning and
        # end of src_bin
        src_bin_begin = (src_bin - next_start_bin) * bin_width
        src_bin_end = src_bin_begin + bin_width

        # which dst_bins the beginning and end of src_bin belong to?
        dst_bin_of_begin = torch.clamp(
            torch.div(src_bin_begin, dst_bin_width, rounding_mode='floor'),
            0,
            self.dst_nbins - 1,
        )
        dst_bin_of_begin_center = (dst_bin_of_begin + 0.5) * dst_bin_width

        dst_bin_of_end = torch.clamp(
            torch.div(src_bin_end, dst_bin_width, rounding_mode='floor'),
            0,
            self.dst_nbins - 1,
        )
        density = histogram / bin_width

        norm = torch.zeros(self.bins, device=histogram.device)

        delta_begin = src_bin_begin - dst_bin_of_begin_center
        delta_end = dst_bin_width / 2
        norm += self.get_norm(
            delta_begin,
            torch.ones(self.bins, device=histogram.device) * delta_end,
            density,
        )

        norm += (dst_bin_of_end - dst_bin_of_begin - 1) * self.get_norm(
            torch.tensor(-dst_bin_width / 2), torch.tensor(dst_bin_width / 2), density
        )

        dst_bin_of_end_center = dst_bin_of_end * dst_bin_width + dst_bin_width / 2

        delta_begin = -dst_bin_width / 2
        delta_end = src_bin_end - dst_bin_of_end_center
        norm += self.get_norm(torch.tensor(delta_begin), delta_end, density)

        return norm.sum().item()

    def _upscale_histogram(self, histogram, orig_min, orig_max, update_min, update_max):
        # this turns the histogram into a more fine-coarsed histogram to reduce
        # bin quantization errors
        histogram = histogram.repeat_interleave(self.upsample_rate) / self.upsample_rate
        bin_size = (orig_max - orig_min) / (self.bins * self.upsample_rate)
        mid_points_histogram = (
            torch.linspace(
                orig_min,
                orig_max,
                self.bins * self.upsample_rate + 1,
                device=orig_min.device,
            )[:-1].to(histogram.device)
            + 0.5 * bin_size
        )
        boundaries_new_histogram = torch.linspace(
            update_min, update_max, self.bins + 1, device=update_min.device
        ).to(histogram.device)
        # this maps the mid-poits of the histogram to the new histogram's space
        bucket_assignments = (
            torch.bucketize(mid_points_histogram, boundaries_new_histogram, right=True)
            - 1
        )
        # this then maps the histogram mid-points in the new space,
        # weighted by the original histogram's values
        # this is just the old histogram in the new histogram's space

        # In case due to numerical issues the values land higher/lower than the maximum/minimum
        bucket_assignments[bucket_assignments >= self.bins] = self.bins - 1
        bucket_assignments[bucket_assignments < 0] = 0

        update_histogram = torch.bincount(
            bucket_assignments, weights=histogram, minlength=self.bins
        )
        return update_histogram

    def _combine_histograms(
        self, orig_hist, orig_min, orig_max, update_hist, update_min, update_max
    ):
        # If the new min and max are the same as the current min and max,
        # we can just add the new histogram to the original histogram
        if update_min == orig_min and update_max == orig_max:
            return orig_hist + update_hist

        # If the orig hist only has one value (i.e., the min and max are the same)
        # we can just add it into new histogram
        if orig_min == orig_max:
            bin_value = torch.sum(update_hist)
            transformed_orig_hist = (
                torch.histc(
                    orig_min, bins=self.bins, min=update_min, max=update_max
                )  # type: ignore[arg-type]
                * bin_value
            )
            return transformed_orig_hist + update_hist

        # We assume the update_hist is already in the target range, we will map the orig_max to it
        assert update_min <= orig_min
        assert update_max >= orig_max

        # Now we need to turn the old_histogram, into the range of the new histogram
        transformed_orig_hist = self._upscale_histogram(
            orig_hist,
            orig_min,
            orig_max,
            update_min,
            update_max,
        )

        return update_hist + transformed_orig_hist

    def get_hist_threshold(self, histogram, min_val, max_val):

        assert histogram.size()[0] == self.bins, 'bins mismatch'
        bin_width = (max_val - min_val) / self.bins

        # cumulative sum
        total = torch.sum(histogram).item()
        cSum = torch.cumsum(histogram, dim=0)

        stepsize = 1e-8
        alpha = 0.0  # lower bound
        beta = 1.0  # upper bound
        start_bin = 0
        end_bin = self.bins - 1
        norm_min = float('inf')

        while alpha < beta:
            # Find the next step
            next_alpha = alpha + stepsize
            next_beta = beta - stepsize

            # find the left and right bins between the quantile bounds
            left = start_bin
            right = end_bin
            while left < end_bin and cSum[left] < next_alpha * total:
                left = left + 1
            while right > start_bin and cSum[right] > next_beta * total:
                right = right - 1

            # decide the next move
            next_start_bin = start_bin
            next_end_bin = end_bin
            if (left - start_bin) > (end_bin - right):
                # move the start bin
                next_start_bin = left
                alpha = next_alpha
            else:
                # move the end bin
                next_end_bin = right
                beta = next_beta

            if next_start_bin == start_bin and next_end_bin == end_bin:
                continue

            # calculate the quantization error using next_start_bin and next_end_bin
            norm = self.get_quantization_error(
                histogram, min_val, max_val, next_start_bin, next_end_bin
            )

            if norm > norm_min:
                break
            norm_min = norm
            start_bin = next_start_bin
            end_bin = next_end_bin

        new_min = min_val + bin_width * start_bin
        new_max = min_val + bin_width * (end_bin + 1)
        return new_min, new_max

    def get_static_hist_range(self, act_tensors):
        act_tensors = self.reshape_batch_tensors(act_tensors)
        stats_min_max = self.get_minmax_stats(act_tensors)
        min_vals, max_vals = [], []
        histograms = []
        for input_idx, tensors in enumerate(act_tensors):
            min_val, max_val = None, None
            histogram = torch.zeros(self.bins)
            tensor_range = stats_min_max[input_idx]
            for idx, tensor in enumerate(tensors):
                tensor = tensor.float()
                x_min, x_max = tensor_range['min'][idx], tensor_range['max'][idx]
                if min_val is None or max_val is None:
                    new_histogram = torch.histc(
                        tensor, self.bins, min=x_min.item(), max=x_max.item()
                    )
                    histogram.detach_().resize_(new_histogram.shape)
                    histogram.copy_(new_histogram)

                    min_val, max_val = x_min, x_max
                else:
                    current_min, current_max = min_val, max_val
                    update_min, update_max = x_min, x_max
                    new_min = torch.min(current_min, update_min)
                    new_max = torch.max(current_max, update_max)

                    update_histogram = torch.histc(
                        tensor, self.bins, min=new_min.item(), max=new_max.item()
                    ).to(histogram.device)

                    if new_min == current_min and new_max == current_max:
                        combined_histogram = histogram + update_histogram
                        histogram.detach_().resize_(combined_histogram.shape)
                        histogram.copy_(combined_histogram)
                    else:
                        combined_histogram = self._combine_histograms(
                            histogram,
                            current_min,
                            current_max,
                            update_histogram,
                            new_min,
                            new_max,
                        )
                        histogram.detach_().resize_(combined_histogram.shape)
                        histogram.copy_(combined_histogram)

                    min_val, max_val = new_min, new_max

            min_vals.append(min_val)
            max_vals.append(max_val)
            histograms.append(histogram)

        new_min_vals, new_max_vals = [], []
        for i in range(len(histograms)):
            histogram = histograms[i]
            min_val, max_val = min_vals[i], max_vals[i]
            new_min, new_max = self.get_hist_threshold(histogram, min_val, max_val)
            new_min_vals.append(new_min)
            new_max_vals.append(new_max)

        return new_min_vals, new_max_vals

    def get_static_moving_minmax_range(self, act_tensors, alpha):
        act_tensors = self.reshape_batch_tensors(act_tensors)
        moving_min_vals, moving_max_vals = [], []
        for tensors in act_tensors:
            moving_min_val, moving_max_val = None, None
            for tensor in tensors:
                tensor = self.reshape_tensor(tensor)
                tensor_range = self.get_minmax_range(tensor)
                min_val, max_val = tensor_range[0], tensor_range[1]

                if moving_min_val is None or moving_max_val is None:
                    moving_min_val = min_val
                    moving_max_val = max_val
                else:
                    moving_min_val = moving_min_val + alpha * (min_val - moving_min_val)
                    moving_max_val = moving_max_val + alpha * (max_val - moving_max_val)
            moving_min_vals.append(moving_min_val)
            moving_max_vals.append(moving_max_val)

        return moving_min_vals, moving_max_vals

    def get_qparams(self, tensor_range, device):
        min_val, max_val = tensor_range[0], tensor_range[1]
        qmin = self.qmin.to(device)
        qmax = self.qmax.to(device)
        if self.sym:
            abs_max = torch.max(max_val.abs(), min_val.abs())
            abs_max = abs_max.clamp(min=1e-5)
            scales = abs_max / qmax
            zeros = torch.tensor(0.0)
        else:
            scales = (max_val - min_val).clamp(min=1e-5) / (qmax - qmin)
            zeros = (qmin - torch.round(min_val / scales)).clamp(qmin, qmax)
            if not self.round_zp:
                zeros = qmin - (min_val / scales)
        return scales, zeros, qmax, qmin

    def get_batch_tensors_qparams(self, act_tensors, alpha=0.01, args={}):
        scales_list, zeros_list, qmin_list, qmax_list = [], [], [], []

        if self.calib_algo == 'static_hist':
            assert (
                self.sym is True and self.granularity == 'per_tensor'
            ), 'Only support per tensor static symmetric int quantize.'
            min_vals, max_vals = self.get_static_hist_range(act_tensors)
        elif self.calib_algo == 'static_minmax':
            min_vals, max_vals = self.get_static_minmax_range(act_tensors)
        elif self.calib_algo == 'static_moving_minmax':
            min_vals, max_vals = self.get_static_moving_minmax_range(act_tensors, alpha)
        else:
            raise ValueError(f'Unsupported calibration algorithm: {self.calib_algo}')

        for i in range(len(min_vals)):
            min_val, max_val = min_vals[i], max_vals[i]
            scales, zeros, qmax, qmin = self.get_qparams(
                (min_val, max_val), min_val.device
            )
            scales_list.append(scales)
            zeros_list.append(zeros)
            qmin_list.append(qmin)
            qmax_list.append(qmax)

        return scales_list, zeros_list, qmin_list, qmax_list

    def optimize_weights_proximal(self, tensor, scales, zeros, qmax, qmin):
        best_error = 1e4
        current_beta = self.beta
        current_kappa = self.kappa
        scales = 1 / scales
        for i in range(self.iters):
            W_q = torch.round(tensor * scales + zeros).clamp(qmin, qmax)
            W_r = (W_q - zeros) / scales
            W_e = self.shrink_op(tensor - W_r, current_beta)

            zeros = torch.mean(W_q - (tensor - W_e) * scales, axis=-1, keepdim=True)
            current_beta *= current_kappa
            current_error = float(torch.abs(tensor - W_r).mean())

            if current_error < best_error:
                best_error = current_error
            else:
                break

        torch.cuda.empty_cache()
        scales = 1 / scales

        return scales, zeros

    def reshape_tensor(self, tensor, allow_padding=False):
        if self.granularity == 'per_group':
            if tensor.shape[-1] >= self.group_size:
                if tensor.shape[-1] % self.group_size == 0:
                    t = tensor.reshape(-1, self.group_size)
                elif allow_padding:
                    deficiency = self.group_size - tensor.shape[1] % self.group_size
                    prefix = tensor.shape[:-1]
                    pad_zeros = torch.zeros(
                        (*prefix, deficiency), device=tensor.device, dtype=tensor.dtype
                    )
                    t = torch.cat((tensor, pad_zeros), dim=-1).reshape(
                        -1, self.group_size
                    )
                else:
                    raise ValueError(
                        f'Dimension {tensor.shape[-1]} '
                        f'not divisible by group size {self.group_size}'
                    )
            else:
                t = tensor
        elif self.granularity == 'per_head':
            t = tensor.reshape(self.head_num, -1)
        elif self.granularity == 'per_block':
            m, n = tensor.shape
            t_padded = torch.zeros((ceil_div(m, self.block_size) * self.block_size, ceil_div(n, self.block_size) * self.block_size), dtype=tensor.dtype, device=tensor.device)
            t_padded[:m, :n] = tensor
            t = t_padded.view(-1, self.block_size, t_padded.size(1) // self.block_size, self.block_size)
        else:
            t = tensor
        return t

    def restore_tensor(self, tensor, shape):
        if tensor.shape == shape:
            t = tensor
        elif self.granularity == 'per_block':
            try:
                t = tensor.reshape(-1, shape[-1])[:shape[0], :]
            except RuntimeError:
                t = tensor.reshape(shape[0], -1)[:, :shape[1]]
        else:
            try:
                t = tensor.reshape(shape)
            except RuntimeError:
                deficiency = self.group_size - shape[1] % self.group_size
                t = tensor.reshape(*shape[:-1], -1)[..., :-deficiency]
        return t


class IntegerQuantizer(BaseQuantizer):
    def __init__(self, bit, symmetric, granularity, **kwargs):
        super().__init__(bit, symmetric, granularity, **kwargs)
        self.quant_type = 'int-quant'
        if 'int_range' in self.kwargs:
            self.qmin = self.kwargs['int_range'][0]
            self.qmax = self.kwargs['int_range'][1]
        else:
            if self.sym:
                self.qmin = -(2 ** (self.bit - 1))
                self.qmax = 2 ** (self.bit - 1) - 1
            else:
                self.qmin = 0.0
                self.qmax = 2**self.bit - 1

        self.qmin = torch.tensor(self.qmin)
        self.qmax = torch.tensor(self.qmax)
        self.dst_nbins = 2**bit

    def get_hqq_qparams(self, tensor, args):
        tensor = tensor.float()
        tensor = self.reshape_tensor(tensor)
        tensor_range = self.get_minmax_range(tensor)
        scales, zeros, qmax, qmin = self.get_qparams(tensor_range, tensor.device)
        best_scales, best_zeros = self.optimize_weights_proximal(
            tensor, scales, zeros, qmax, qmin
        )
        return tensor, best_scales, best_zeros, qmax, qmin

    def get_tensor_qparams(self, tensor, args={}):
        if self.calib_algo == 'hqq':
            return self.get_hqq_qparams(tensor, args)
        else:
            tensor = self.reshape_tensor(tensor)
            tensor_range = self.get_tensor_range(tensor, args)
            scales, zeros, qmax, qmin = self.get_qparams(tensor_range, tensor.device)
            return tensor, scales, zeros, qmax, qmin

    def quant(self, tensor, scales, zeros, qmax, qmin):
        if self.round_zp:
            tensor = torch.clamp(self.round_func(tensor / scales) + zeros, qmin, qmax)
        else:
            tensor = torch.clamp(
                self.round_func(tensor / scales.clamp_min(1e-9) + zeros),
                qmin,
                qmax,
            )
        return tensor

    def dequant(self, tensor, scales, zeros):
        tensor = (tensor - zeros) * scales
        return tensor

    def quant_dequant(self, tensor, scales, zeros, qmax, qmin, output_scale_factor=1):
        tensor = self.quant(tensor, scales, zeros, qmax, qmin)
        tensor = self.dequant(tensor, scales * output_scale_factor, zeros)
        return tensor

    def fake_quant_act_static(self, act, args={}):
        if 'int_indices' in args:
            q_act = act[:, :, args['int_indices']]
            fp_act = act[:, :, args['fp_indices']]
        else:
            q_act = act

        if 'current_bit' in args:
            org_bit = self.bit
            self.bit = args['current_bit']

        org_act_shape = q_act.shape
        org_act_dtype = q_act.dtype

        scales, zeros, qmax, qmin = (
            args['scales'],
            args['zeros'],
            args['qmax'],
            args['qmin'],
        )
        q_act = self.reshape_tensor(q_act)
        q_act = self.quant_dequant(q_act, scales, zeros, qmax, qmin)
        q_act = self.restore_tensor(q_act, org_act_shape).to(org_act_dtype)

        if 'current_bit' in args:
            self.bit = org_bit

        if 'int_indices' in args:
            mix_act = torch.zeros_like(act)
            mix_act[:, :, args['int_indices']] = q_act
            mix_act[:, :, args['fp_indices']] = fp_act
            return mix_act

        return q_act

    def fake_quant_act_dynamic(self, act, args={}):
        if 'int_indices' in args:
            q_act = act[:, :, args['int_indices']]
            fp_act = act[:, :, args['fp_indices']]
        else:
            q_act = act

        if 'current_bit' in args:
            org_bit = self.bit
            self.bit = args['current_bit']

        org_act_shape = q_act.shape
        org_act_dtype = q_act.dtype

        q_act, scales, zeros, qmax, qmin = self.get_tensor_qparams(q_act, args)
        q_act = self.quant_dequant(q_act, scales, zeros, qmax, qmin)

        q_act = self.restore_tensor(q_act, org_act_shape).to(org_act_dtype)

        if 'current_bit' in args:
            self.bit = org_bit

        if 'int_indices' in args:
            mix_act = torch.zeros_like(act)
            mix_act[:, :, args['int_indices']] = q_act
            mix_act[:, :, args['fp_indices']] = fp_act
            return mix_act
        if self.ste_all:
            return (q_act - act).detach() + act
        return q_act

    def fake_quant_weight_static(self, weight, args):
        if 'int_indices' in args:
            if self.granularity == 'per_group':
                assert len(args['int_indices']) % self.group_size == 0
            q_weight = weight[:, args['int_indices']]
            fp_weight = weight[:, args['fp_indices']]

        elif 'dim' in args and 'ic' in args['dim']:
            q_weight = weight.T
        else:
            q_weight = weight

        if 'rounding' in args:
            org_round_func = self.round_func
            self.round_func = lambda x: torch.floor(x) + args['rounding']

        org_w_shape = q_weight.shape
        org_w_dtype = q_weight.dtype
        scales, zeros, qmax, qmin = (
            args['scales'],
            args['zeros'],
            args['qmax'],
            args['qmin'],
        )
        output_scale_factor = (
            args['output_scale_factor'] if 'output_scale_factor' in args else 1
        )

        q_weight = self.reshape_tensor(q_weight)
        q_weight = self.quant_dequant(
            q_weight, scales, zeros, qmax, qmin, output_scale_factor
        )
        q_weight = self.restore_tensor(q_weight, org_w_shape).to(org_w_dtype)

        if 'int_indices' in args:
            mix_weight = torch.zeros_like(weight)
            mix_weight[:, args['int_indices']] = q_weight
            mix_weight[:, args['fp_indices']] = fp_weight
            return mix_weight

        elif 'dim' in args and 'ic' in args['dim']:
            q_weight = q_weight.T

        if 'rounding' in args:
            self.round_func = org_round_func

        return q_weight

    def fake_quant_weight_dynamic(self, weight, args={}):
        if 'int_indices' in args:
            if self.granularity == 'per_group':
                assert len(args['int_indices']) % self.group_size == 0
            q_weight = weight[:, args['int_indices']]
            fp_weight = weight[:, args['fp_indices']]

        elif 'dim' in args and 'ic' in args['dim']:
            q_weight = weight.T
        else:
            q_weight = weight

        if 'current_bit' in args:
            org_bit = self.bit
            self.bit = args['current_bit']

        org_w_shape = q_weight.shape
        org_w_dtype = q_weight.dtype

        q_weight, scales, zeros, qmax, qmin = self.get_tensor_qparams(q_weight, args)
        q_weight = self.quant_dequant(q_weight, scales, zeros, qmax, qmin)

        q_weight = self.restore_tensor(q_weight, org_w_shape).to(org_w_dtype)

        if 'current_bit' in args:
            self.bit = org_bit

        if 'int_indices' in args:
            mix_weight = torch.zeros_like(weight)
            mix_weight[:, args['int_indices']] = q_weight
            mix_weight[:, args['fp_indices']] = fp_weight
            return mix_weight

        elif 'dim' in args and 'ic' in args['dim']:
            q_weight = q_weight.T

        return q_weight

    def real_quant_weight_static(self, weight, args):
        org_w_shape = weight.shape
        if 'output_scale_factor' in args:
            output_scale_factor = args['output_scale_factor']
            del args['output_scale_factor']
        else:
            output_scale_factor = 1
        scales, zeros, qmax, qmin = (
            args['scales'],
            args['zeros'],
            args['qmax'],
            args['qmin'],
        )
        weight = self.reshape_tensor(weight)
        weight = self.quant(weight, scales, zeros, qmax, qmin)
        weight = self.restore_tensor(weight, org_w_shape)

        scales = scales * output_scale_factor

        if self.bit == 8:
            if self.qmin != 0:
                dtype = torch.int8
            else:
                dtype = torch.uint8
        else:
            dtype = torch.int32
        weight = weight.to(dtype)
        if not self.sym and self.round_zp:
            zeros = zeros.to(dtype)
        elif self.sym:
            zeros = None

        if self.granularity == 'per_tensor':
            qparams_shape = 1
        elif self.granularity == 'per_block':
            qparams_shape = (scales.shape[0], scales.shape[2])
        else:
            qparams_shape = (weight.shape[0], -1)

        if zeros is not None:
            zeros = zeros.view(qparams_shape)
        scales = scales.view(qparams_shape)

        return weight, scales, zeros

    def real_quant_weight_dynamic(self, weight, args={}):
        org_w_shape = weight.shape
        if 'output_scale_factor' in args:
            output_scale_factor = args['output_scale_factor']
            del args['output_scale_factor']
        else:
            output_scale_factor = 1
        weight, scales, zeros, qmax, qmin = self.get_tensor_qparams(weight, args)
        weight = self.quant(weight, scales, zeros, qmax, qmin)
        weight = self.restore_tensor(weight, org_w_shape)

        scales = scales * output_scale_factor

        if self.bit == 8:
            if self.qmin != 0:
                dtype = torch.int8
            else:
                dtype = torch.uint8
        else:
            dtype = torch.int32
        weight = weight.to(dtype)
        if not self.sym and self.round_zp:
            zeros = zeros.to(dtype)
        elif self.sym:
            zeros = None

        if self.granularity == 'per_tensor':
            qparams_shape = 1
        elif self.granularity == 'per_block':
            qparams_shape = (scales.shape[0], scales.shape[2])
        else:
            qparams_shape = (weight.shape[0], -1)

        if zeros is not None:
            zeros = zeros.view(qparams_shape)
        scales = scales.view(qparams_shape)

        return weight, scales, zeros

    def __repr__(self):
        return (
            f'IntegerQuantizer(bit={self.bit}, sym={self.sym},'
            f'granularity={self.granularity},'
            f'kwargs={self.kwargs}, qmin={self.qmin}, qmax={self.qmax})'
        )


class FloatQuantizer(BaseQuantizer):
    def __init__(self, bit, symmetric, granularity, **kwargs):
        super().__init__(bit, symmetric, granularity, **kwargs)
        self.sym = True
        self.quant_type = 'float-quant'
        self.e_bits = int(self.bit[1])
        self.m_bits = int(self.bit[-1])
        self.sign_bits = 1
        self.num_bits = self.e_bits + self.m_bits + self.sign_bits
        self.default_bias = 2 ** (self.e_bits - 1)
        self.dst_nbins = 2**self.num_bits
        self.use_qtorch = self.kwargs.get('use_qtorch')
        if self.use_qtorch:
            assert (
                float_quantize is not None
            ), 'Please install qtorch (pip install qtorch). Or set use_qtorch=False'
            if 'float_range' in self.kwargs:
                self.qmin, self.qmax = self.kwargs['float_range']
            else:
                bit_ranges = {
                    ('e4m3', 8): torch.float8_e4m3fn,
                    ('e5m2', 8): torch.float8_e5m2,
                    ('e3m2', 6): (-28, 28),
                    ('e4m7', 12): (-510, 510),
                    ('e2m1', 4): (-6, 6),
                }

                key = (self.bit, self.num_bits)
                if key in bit_ranges:
                    if isinstance(bit_ranges[key], tuple):
                        self.qmin, self.qmax = bit_ranges[key]
                    else:
                        finfo = torch.finfo(bit_ranges[key])
                        self.qmin, self.qmax = finfo.min, finfo.max
                else:
                    raise NotImplementedError(
                        'Only 4, 6, 8, and \
                                                12-bit quantization is supported.'
                    )
            self.qmax = torch.tensor(self.qmax)
            self.qmin = torch.tensor(self.qmin)

    def get_float_qparams(self, tensor, tensor_range, device):
        min_val, max_val = tensor_range[0], tensor_range[1]
        maxval = torch.max(max_val, -min_val)

        e_bits = torch.tensor(self.e_bits, dtype=torch.float32).cuda()
        m_bits = torch.tensor(self.m_bits, dtype=torch.float32).cuda()

        if maxval.shape[0] != 1 and len(maxval.shape) != len(tensor.shape):
            maxval = maxval.view([-1] + [1] * (len(tensor.shape) - 1))

        if e_bits >= 5:
            maxval = maxval.to(dtype=torch.float32)

        bias = 2**e_bits - torch.log2(maxval) + torch.log2(2 - 2 ** (-m_bits)) - 1

        xc = torch.min(torch.max(tensor, -maxval), maxval)

        log_scales = torch.clamp(
            (torch.floor(torch.log2(torch.abs(xc)) + bias)).detach(), 1.0
        )
        scales = 2.0 ** (log_scales - m_bits - bias)

        return xc, scales

    def get_hqq_qparams(self, tensor, args):
        tensor = tensor.float()
        tensor = self.reshape_tensor(tensor)
        tensor_range = self.get_minmax_range(tensor)
        if self.use_qtorch:
            scales, zeros, qmax, qmin = self.get_qparams(tensor_range, tensor.device)
        else:
            tensor, scales = self.get_float_qparams(tensor, tensor_range, tensor.device)
            zeros, qmin, qmax = torch.tensor(0), None, None
        best_scales, best_zeros = self.optimize_weights_proximal(
            tensor, scales, zeros, qmax, qmin
        )
        return tensor, best_scales, best_zeros, qmax, qmin

    def get_tensor_qparams(self, tensor, args={}):
        if self.calib_algo == 'hqq':
            return self.get_hqq_qparams(tensor, args)
        else:
            tensor = self.reshape_tensor(tensor)
            tensor_range = self.get_tensor_range(tensor, args)
            if self.use_qtorch:
                scales, zeros, qmax, qmin = self.get_qparams(
                    tensor_range, tensor.device
                )
            else:
                tensor, scales = self.get_float_qparams(
                    tensor, tensor_range, tensor.device
                )
                zeros, qmin, qmax = torch.tensor(0), None, None

            return tensor, scales, zeros, qmax, qmin

    def quant(self, tensor, scales, zeros, qmax, qmin):
        scales[scales == 0] = 1
        scaled_tensor = tensor / scales + zeros
        if self.use_qtorch:
            org_dtype = scaled_tensor.dtype
            q_tensor = float_quantize(
                scaled_tensor.float(), self.e_bits, self.m_bits, rounding='nearest'
            )
            q_tensor.to(org_dtype)
        else:
            q_tensor = self.round_func(scaled_tensor)
        return q_tensor

    def dequant(self, tensor, scales, zeros):
        tensor = (tensor - zeros) * scales
        return tensor

    def quant_dequant(self, tensor, scales, zeros, qmax, qmin):
        tensor = self.quant(tensor, scales, zeros, qmax, qmin)
        tensor = self.dequant(tensor, scales, zeros)
        return tensor

    def fake_quant_act_static(self, act, args={}):
        q_act = act
        org_act_shape = q_act.shape
        org_act_dtype = q_act.dtype

        scales, zeros, qmax, qmin = (
            args['scales'],
            args['zeros'],
            args['qmax'],
            args['qmin'],
        )
        q_act = self.reshape_tensor(q_act)
        q_act = self.quant_dequant(q_act, scales, zeros, qmax, qmin)
        q_act = self.restore_tensor(q_act, org_act_shape).to(org_act_dtype)

        return q_act

    def fake_quant_act_dynamic(self, act, args={}):
        q_act = act
        org_act_shape = q_act.shape
        org_act_dtype = q_act.dtype

        q_act, scales, zeros, qmax, qmin = self.get_tensor_qparams(q_act, args)
        q_act = self.quant_dequant(q_act, scales, zeros, qmax, qmin)

        q_act = self.restore_tensor(q_act, org_act_shape).to(org_act_dtype)
        return q_act

    def fake_quant_weight_static(self, weight, args):

        if 'dim' in args and 'ic' in args['dim']:
            q_weight = weight.T
        else:
            q_weight = weight

        if 'rounding' in args:
            org_round_func = self.round_func
            self.round_func = lambda x: torch.floor(x) + args['rounding']

        org_w_shape = q_weight.shape
        org_w_dtype = q_weight.dtype
        scales, zeros, qmax, qmin = (
            args['scales'],
            args['zeros'],
            args['qmax'],
            args['qmin'],
        )
        q_weight = self.reshape_tensor(q_weight)
        q_weight = self.quant_dequant(q_weight, scales, zeros, qmax, qmin)
        q_weight = self.restore_tensor(q_weight, org_w_shape).to(org_w_dtype)

        if 'dim' in args and 'ic' in args['dim']:
            q_weight = q_weight.T

        if 'rounding' in args:
            self.round_func = org_round_func

        return q_weight

    def fake_quant_weight_dynamic(self, weight, args={}):

        if 'dim' in args and 'ic' in args['dim']:
            q_weight = weight.T
        else:
            q_weight = weight

        org_w_shape = q_weight.shape
        org_w_dtype = q_weight.dtype

        q_weight, scales, zeros, qmax, qmin = self.get_tensor_qparams(q_weight, args)
        q_weight = self.quant_dequant(q_weight, scales, zeros, qmax, qmin)
        q_weight = self.restore_tensor(q_weight, org_w_shape).to(org_w_dtype)

        if 'dim' in args and 'ic' in args['dim']:
            q_weight = q_weight.T

        return q_weight

    def real_quant_weight_static(self, weight, args):
        assert self.bit in ['e4m3', 'e5m2'], 'Only FP8 E4M3 and E5M2 support real quant'
        dtype = torch.float8_e4m3fn if self.e_bits == 4 else torch.float8_e5m2

        org_w_shape = weight.shape
        if 'output_scale_factor' in args:
            output_scale_factor = args['output_scale_factor']
            del args['output_scale_factor']
        else:
            output_scale_factor = 1
        scales, zeros, qmax, qmin = (
            args['scales'],
            args['zeros'],
            args['qmax'],
            args['qmin'],
        )
        weight = self.reshape_tensor(weight)
        weight = self.quant(weight, scales, zeros, qmax, qmin)
        weight = self.restore_tensor(weight, org_w_shape)

        scales = scales * output_scale_factor

        weight = weight.to(dtype)
        zeros = None
        if self.granularity == 'per_tensor':
            qparams_shape = 1
        elif self.granularity == 'per_block':
            qparams_shape = (scales.shape[0], scales.shape[2])
        else:
            qparams_shape = (weight.shape[0], -1)

        scales = scales.view(qparams_shape)
        return weight, scales, zeros

    def real_quant_weight_dynamic(self, weight, args={}):
        assert self.bit in ['e4m3', 'e5m2'], 'Only FP8 E4M3 and E5M2 support real quant'
        dtype = torch.float8_e4m3fn if self.e_bits == 4 else torch.float8_e5m2

        org_w_shape = weight.shape
        if 'output_scale_factor' in args:
            output_scale_factor = args['output_scale_factor']
            del args['output_scale_factor']
        else:
            output_scale_factor = 1
        weight, scales, zeros, qmax, qmin = self.get_tensor_qparams(weight, args)
        weight = self.quant(weight, scales, zeros, qmax, qmin)
        weight = self.restore_tensor(weight, org_w_shape)

        scales = scales * output_scale_factor

        weight = weight.to(dtype)
        zeros = None
        if self.granularity == 'per_tensor':
            qparams_shape = 1
        elif self.granularity == 'per_block':
            qparams_shape = (scales.shape[0], scales.shape[2])
        else:
            qparams_shape = (weight.shape[0], -1)

        scales = scales.view(qparams_shape)
        return weight, scales, zeros

    def __repr__(self):
        return (
            f'FloatQuantizer(bit={self.bit},'
            f'e_bits={self.e_bits}, m_bits={self.m_bits},'
            f'granularity={self.granularity},'
            f'kwargs={self.kwargs}, qmin={self.qmin}, qmax={self.qmax})'
        )

class Weight48IntegerQuantizer(BaseQuantizer):
    # flake8: noqa
    def __init__(self, bit, bit4, bit8, **kwargs):
        super().__init__(bit, None, None, **kwargs)
        self.quant_type = 'int-quant-w48'
        assert self.bit == 48, 'Only support 48-bit quantization'
        self.bit_settings = {}
        self.bit_settings[4] = bit4
        self.bit_settings[8] = bit8
        for bit in [4, 8]:
            if 'int_range' in self.bit_settings[bit]:
                self.bit_settings[bit]['qmin'] = self.bit_settings[bit]['int_range'][0]
                self.bit_settings[bit]['qmax'] = self.bit_settings[bit]['int_range'][1]
            else:
                if self.bit_settings[bit]['symmetric']:
                    self.bit_settings[bit]['qmin'] = -(2 ** (bit - 1))
                    self.bit_settings[bit]['qmax'] = 2 ** (bit - 1) - 1
                else:
                    self.bit_settings[bit]['qmin'] = 0
                    self.bit_settings[bit]['qmax'] = 2**bit - 1
            self.bit_settings[bit]['qmin'] = torch.tensor(
                self.bit_settings[bit]['qmin']
            )
            self.bit_settings[bit]['qmax'] = torch.tensor(
                self.bit_settings[bit]['qmax']
            )
            if 'scales_bit' in self.bit_settings[bit]:
                if self.bit_settings[bit]['scales_symmetric']:
                    self.bit_settings[bit]['scales_qmin'] = -(
                        2 ** (self.bit_settings[bit]['scales_bit'] - 1)
                    )
                    self.bit_settings[bit]['scales_qmax'] = (
                        2 ** (self.bit_settings[bit]['scales_bit'] - 1) - 1
                    )
                else:
                    self.bit_settings[bit]['scales_qmin'] = 0
                    self.bit_settings[bit]['scales_qmax'] = (
                        2 ** self.bit_settings[bit]['scales_bit'] - 1
                    )
            else:
                self.bit_settings[bit]['scales_qmin'] = -torch.inf
                self.bit_settings[bit]['scales_qmax'] = torch.inf
            if 'zeros_bit' in self.bit_settings[bit]:
                if self.bit_settings[bit]['zeros_symmetric']:
                    self.bit_settings[bit]['zeros_qmin'] = -(
                        2 ** (self.bit_settings[bit]['scales_bit'] - 1)
                    )
                    self.bit_settings[bit]['zeros_qmax'] = (
                        2 ** (self.bit_settings[bit]['scales_bit'] - 1) - 1
                    )
                else:
                    self.bit_settings[bit]['zeros_qmin'] = 0
                    self.bit_settings[bit]['zeros_qmax'] = (
                        2 ** self.bit_settings[bit]['scales_bit'] - 1
                    )
            else:
                self.bit_settings[bit]['zeros_qmin'] = self.bit_settings[bit]['qmin']
                self.bit_settings[bit]['zeros_qmax'] = self.bit_settings[bit]['qmax']

    def reshape_tensor(self, tensor, bit=4):
        granularity = self.bit_settings[bit].get('granularity')
        if granularity == 'per_group':
            group_size = self.bit_settings[bit].get('group_size')
            if tensor.shape[-1] % group_size == 0:
                t = tensor.reshape(-1, group_size)
            else:
                raise ValueError(
                    f'Dimension {tensor.shape[-1]} '
                    f'not divisible by group size {group_size}'
                )
        else:
            t = tensor
        return t

    def get_qparams(self, tensor_range, device, bit):
        min_val, max_val = tensor_range[0], tensor_range[1]
        qmin = self.bit_settings[bit]['qmin'].to(device)
        qmax = self.bit_settings[bit]['qmax'].to(device)
        sym = self.bit_settings[bit]['symmetric']
        if sym:
            abs_max = torch.max(max_val.abs(), min_val.abs())
            abs_max = abs_max.clamp(min=1e-5)
            scales = abs_max / qmax
            zeros = torch.tensor(0.0)
        else:
            scales = (max_val - min_val).clamp(min=1e-5) / (qmax - qmin)
            zeros = qmin - torch.round(min_val / scales)
        scales = scales.clamp(
            self.bit_settings[bit]['scales_qmin'], self.bit_settings[bit]['scales_qmax']
        )
        zeros = zeros.clamp(
            self.bit_settings[bit]['zeros_qmin'], self.bit_settings[bit]['zeros_qmax']
        )
        return scales, zeros, qmax, qmin

    def quant(self, tensor, scales, zeros, qmax, qmin):
        tensor = torch.clamp(self.round_func(tensor / scales) + zeros, qmin, qmax)
        return tensor

    def dequant(self, tensor, scales, zeros):
        tensor = (tensor - zeros) * scales
        return tensor

    def quant_dequant(self, tensor, scales, zeros, qmax, qmin):
        tensor = self.quant(tensor, scales, zeros, qmax, qmin)
        tensor = self.dequant(tensor, scales, zeros)
        return tensor

    def fake_quant_weight_dynamic(self, weight, args={}):
        # step 1: quantize to 8-bit
        org_shape16 = weight.shape
        org_dtype16 = weight.dtype
        weight = self.reshape_tensor(weight, bit=8)
        weight_range = self.get_tensor_range(weight)
        scales816, zeros816, qmax816, qmin816 = self.get_qparams(
            weight_range, weight.device, bit=8
        )
        weight = self.quant(weight, scales816, zeros816, qmax816, qmin816)

        # step 2: quantize to 4-bit
        org_shape8 = weight.shape
        org_dtype8 = weight.dtype
        weight = self.reshape_tensor(weight, bit=4)
        weight_range = self.get_tensor_range(weight)
        scales48, zeros48, qmax48, qmin48 = self.get_qparams(
            weight_range, weight.device, bit=4
        )
        weight = self.quant(weight, scales48, zeros48, qmax48, qmin48)

        # step 3: dequantize to 8-bit
        weight = self.dequant(weight, scales48, zeros48)
        weight = self.restore_tensor(weight, org_shape8).to(org_dtype8)

        # step 4: dequantize to 16-bit
        weight = self.dequant(weight, scales816, zeros816)
        weight = self.restore_tensor(weight, org_shape16).to(org_dtype16)

        return weight
