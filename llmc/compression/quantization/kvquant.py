import copy

import torch
from loguru import logger
from transformers import DynamicCache

from llmc.utils.registry_factory import KV_REGISTRY

from .quant import FloatQuantizer, IntegerQuantizer


@KV_REGISTRY.register('Naive')
class NaiveQuantKVCache(DynamicCache):
    def __init__(self, quant_type, kvquant_cfg, num_hidden_layers, num_samples=128, bsz=1):
        super().__init__()

        # Copy the config to avoid mutating the original quantization config in static KV calibration.
        kvquant_cfg = copy.deepcopy(kvquant_cfg)
        assert kvquant_cfg.granularity in ['per_token', 'per_tensor', 'per_group', 'per_head']
        self.num_hidden_layers, self.num_samples, self.bsz = (
            num_hidden_layers,
            num_samples,
            bsz,
        )
        if kvquant_cfg.get('static', False) and kvquant_cfg.get(
            'calib_algo', 'minmax'
        ) == 'minmax':
            # Static KV calibration uses the batched tensor statistics path, so convert the default
            # minmax setting to static_minmax here to avoid a later calibration algo name mismatch.
            kvquant_cfg['calib_algo'] = 'static_minmax'
        if quant_type == 'int-quant':
            self.kvquantizer = IntegerQuantizer(**kvquant_cfg)
        elif quant_type == 'float-quant':
            self.kvquantizer = FloatQuantizer(**kvquant_cfg)

        self.kvquant_cfg = kvquant_cfg
        self.static = kvquant_cfg.get('static', False)
        self._quantized_key_cache = []
        self._quantized_value_cache = []
        self.use_org_kv = False

        if self.static:
            self._reset_buffers()
            self.calib_key_cache = [
                [] for i in range(self.num_hidden_layers)
            ]
            self.calib_value_cache = [
                [] for i in range(self.num_hidden_layers)
            ]
            self.calib = True
        else:
            self.calib = False

    def update(
        self,
        key_states,
        value_states,
        layer_idx,
        cache_kwargs,
    ):
        if self.use_org_kv:
            return super().update(key_states, value_states, layer_idx, cache_kwargs)
        elif self.static and self.calib:
            self._calibration(layer_idx, key_states, value_states)
            keys_to_return, values_to_return = key_states, value_states
        else:
            if layer_idx == 0:
                self._seen_tokens += key_states.shape[-2]

            if len(self._quantized_key_cache) <= layer_idx:
                # Prefill
                q_keys = self._quantize(key_states.contiguous(), layer_idx, is_key=True)
                q_values = self._quantize(
                    value_states.contiguous(), layer_idx, is_key=False
                )
                self._quantized_key_cache.append(q_keys)
                self._quantized_value_cache.append(q_values)
                keys_to_return = self._dequantize(q_keys)
                values_to_return = self._dequantize(q_values)
            else:
                # Decode
                dequant_key = self._dequantize(self._quantized_key_cache[layer_idx])
                dequant_value = self._dequantize(self._quantized_value_cache[layer_idx])
                keys_to_return = [dequant_key, key_states]
                values_to_return = [dequant_value, value_states]

                keys_to_return = torch.cat(keys_to_return, dim=-2)
                values_to_return = torch.cat(values_to_return, dim=-2)

                self._quantized_key_cache[layer_idx] = self._quantize(
                    keys_to_return.contiguous(), layer_idx, is_key=True
                )
                self._quantized_value_cache[layer_idx] = self._quantize(
                    values_to_return.contiguous(), layer_idx, is_key=False
                )

        return keys_to_return, values_to_return

    def _check_pass_all_calib_data(self, layer_idx):
        return (
            self.bsz == 1 and len(self.calib_value_cache[layer_idx]) == self.num_samples
        ) or (
            self.bsz == -1
            and self.calib_value_cache[layer_idx][0].shape[0] == self.num_samples
        )

    def _calibration(self, layer_idx, key_states, value_states):
        # Calibration data can be provided through the prompt or
        # the preprocessed decode data.
        # Therefore, calibration occurs only during the prefill stage.
        self.calib_key_cache[layer_idx].append(key_states)
        self.calib_value_cache[layer_idx].append(value_states)

        if self._check_pass_all_calib_data(layer_idx):
            # Get and store calibration parameters for keys and values
            for data, buffer, scale_buffer, zero_buffer, qmin_buffer, qmax_buffer in [
                (
                    self.calib_key_cache[layer_idx],
                    self.k_scales_buffer,
                    self.k_scales_buffer,
                    self.k_zeros_buffer,
                    self.k_qmin_buffer,
                    self.k_qmax_buffer,
                ),
                (
                    self.calib_value_cache[layer_idx],
                    self.v_scales_buffer,
                    self.v_scales_buffer,
                    self.v_zeros_buffer,
                    self.v_qmin_buffer,
                    self.v_qmax_buffer,
                ),
            ]:
                scales, zeros, qmin, qmax = self.get_qparams(data)
                (
                    scale_buffer[layer_idx],
                    zero_buffer[layer_idx],
                    qmin_buffer[layer_idx],
                    qmax_buffer[layer_idx],
                ) = (scales, zeros, qmin, qmax)

            # Clear the calibration caches
            self.calib_key_cache[layer_idx].clear()
            self.calib_value_cache[layer_idx].clear()

    def _quantize(self, tensor, layer_idx, is_key):
        org_shape = tensor.shape
        tensor = self.kvquantizer.reshape_tensor(tensor)

        if self.static:
            scales = (
                self.k_scales_buffer[layer_idx]
                if is_key
                else self.v_scales_buffer[layer_idx]
            )
            zeros = (
                self.k_zeros_buffer[layer_idx]
                if is_key
                else self.v_zeros_buffer[layer_idx]
            )
            qmax = (
                self.k_qmax_buffer[layer_idx]
                if is_key
                else self.v_qmax_buffer[layer_idx]
            )
            qmin = (
                self.k_qmin_buffer[layer_idx]
                if is_key
                else self.v_qmin_buffer[layer_idx]
            )
        else:
            tensor_range = self.kvquantizer.get_tensor_range(tensor, {})
            scales, zeros, qmax, qmin = self.kvquantizer.get_qparams(
                tensor_range, tensor.device
            )

        q_tensor = self.kvquantizer.quant(tensor, scales, zeros, qmax, qmin)
        q_tensor = self.kvquantizer.restore_tensor(q_tensor, org_shape)

        q_tensors = {
            'q_tensor': q_tensor,
            'scales': scales,
            'zeros': zeros,
        }

        return q_tensors

    def _dequantize(self, q_tensors):
        q_tensor = q_tensors['q_tensor']
        scales = q_tensors['scales']
        zeros = q_tensors['zeros']
        org_shape = q_tensor.shape
        q_tensor = self.kvquantizer.reshape_tensor(q_tensor)
        qdq_tensor = self.kvquantizer.dequant(q_tensor, scales, zeros)
        qdq_tensor = self.kvquantizer.restore_tensor(qdq_tensor, org_shape)
        return qdq_tensor

    def _reset_buffers(self):
        self.k_scales_buffer = [torch.zeros(0)] * self.num_hidden_layers
        self.k_zeros_buffer = [torch.zeros(0)] * self.num_hidden_layers
        self.k_qmin_buffer = [0] * self.num_hidden_layers
        self.k_qmax_buffer = [0] * self.num_hidden_layers

        self.v_scales_buffer = [torch.zeros(0)] * self.num_hidden_layers
        self.v_zeros_buffer = [torch.zeros(0)] * self.num_hidden_layers
        self.v_qmin_buffer = [0] * self.num_hidden_layers
        self.v_qmax_buffer = [0] * self.num_hidden_layers

    def _reset_states(self):
        self._quantized_key_cache = []
        self._quantized_value_cache = []
        self.key_cache = []
        self.value_cache = []
        self._seen_tokens = 0

    def get_qparams(self, tensor):
        scales_list, zeros_list, qmin_list, qmax_list = (
            self.kvquantizer.get_batch_tensors_qparams(tensor)
        )
        scales, zeros, qmin, qmax = (
            scales_list[0],
            zeros_list[0],
            qmin_list[0],
            qmax_list[0],
        )
        return scales, zeros, qmin, qmax

    def get_seq_length(self, layer_idx=0):
        if self.use_org_kv:
            return super().get_seq_length()
        if len(self._quantized_key_cache) <= layer_idx:
            return 0
        return self._seen_tokens if layer_idx == 0 else self._seen_tokens - 1


@KV_REGISTRY.register('Kivi')
class KiviQuantKVCache(NaiveQuantKVCache):
    def __init__(self, quant_type, kvquant_cfg, num_hidden_layers, num_samples=128, bsz=1):
        super().__init__(quant_type, kvquant_cfg, num_hidden_layers, num_samples, bsz)
        assert not self.static, 'Only support dynamic quantization for KIVI'
        self.residual_length = kvquant_cfg.get('residual_length', 128)

    def update(
        self,
        key_states,
        value_states,
        layer_idx,
        cache_kwargs,
    ):
        if self.use_org_kv:
            return super().update(key_states, value_states, layer_idx, cache_kwargs)
        else:
            if layer_idx == 0:
                self._seen_tokens += key_states.shape[-2]

            if len(self.key_cache) <= layer_idx:
                self._quantized_key_cache.append(self._quantize(key_states.contiguous(),
                                                                layer_idx,
                                                                is_key=True))
                self._quantized_value_cache.append(self._quantize(value_states.contiguous(),
                                                                  layer_idx,
                                                                  is_key=False))
                self.key_cache.append(torch.zeros(0,
                                                  dtype=key_states.dtype,
                                                  device=key_states.device))
                self.value_cache.append(torch.zeros(0,
                                                    dtype=key_states.dtype,
                                                    device=key_states.device))
                keys_to_return, values_to_return = key_states, value_states
            else:
                dequant_key = self._dequantize(self._quantized_key_cache[layer_idx])
                dequant_value = self._dequantize(self._quantized_value_cache[layer_idx])
                keys_to_return = [dequant_key, self.key_cache[layer_idx], key_states]
                values_to_return = [dequant_value, self.value_cache[layer_idx], value_states]

                keys_to_return = torch.cat(keys_to_return, dim=-2)
                values_to_return = torch.cat(values_to_return, dim=-2)
                if (
                    self.key_cache[layer_idx].dim() == 4
                    and self.key_cache[layer_idx].shape[-2] + 1 >= self.residual_length
                ):
                    self._quantized_key_cache[layer_idx] = \
                        self._quantize(keys_to_return.contiguous(), layer_idx, is_key=True)
                    self._quantized_value_cache[layer_idx] = self._quantize(
                        values_to_return.contiguous(), layer_idx, is_key=False
                    )
                    self.key_cache[layer_idx] = torch.zeros(0,
                                                            dtype=key_states.dtype,
                                                            device=key_states.device)
                    self.value_cache[layer_idx] = torch.zeros(0,
                                                              dtype=key_states.dtype,
                                                              device=key_states.device)
                else:
                    self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states],
                                                          dim=-2)
                    self.value_cache[layer_idx] = \
                        torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

            return keys_to_return, values_to_return
