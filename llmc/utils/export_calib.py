import torch


def _to_jsonable(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def _to_tensor(value, dtype=torch.float32):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().to(dtype)
    return torch.as_tensor(value, dtype=dtype)


def _collect_lightllm_kv_scale(scales, zeros, qmin, qmax):
    if isinstance(scales, torch.Tensor) and scales.numel() == 0:
        return None

    scales_tensor = _to_tensor(scales)
    zeros_tensor = _to_tensor(zeros, dtype=scales_tensor.dtype)
    qmin_tensor = _to_tensor(qmin, dtype=scales_tensor.dtype)
    qmax_tensor = _to_tensor(qmax, dtype=scales_tensor.dtype)
    min_tensor = (qmin_tensor - zeros_tensor) * scales_tensor
    max_tensor = (qmax_tensor - zeros_tensor) * scales_tensor
    absmax_tensor = torch.maximum(min_tensor.abs(), max_tensor.abs())
    fp8_qmax = torch.tensor(
        torch.finfo(torch.float8_e4m3fn).max, dtype=absmax_tensor.dtype
    )
    return absmax_tensor / fp8_qmax


def collect_lightllm_kv_calib_json(blockwise_opt):
    if not getattr(blockwise_opt, 'quant_kvcache', False):
        raise ValueError(
            'save_lightllm_kv_cache_calib requires kvcache quantization.'
        )

    kv_cfg = blockwise_opt.quant_config['kvcache']
    granularity = kv_cfg.get('granularity')
    if granularity not in ['per_tensor', 'per_head']:
        raise ValueError(
            f'LightLLM calib export only supports per_tensor/per_head, got {granularity}'
        )

    num_layers = blockwise_opt.model.model_config.num_hidden_layers
    num_head = int(
        getattr(
            blockwise_opt.model.model_config,
            'num_key_value_heads',
            blockwise_opt.model.get_num_attention_heads(),
        )
    )
    scales = []
    for layer_idx in range(num_layers):
        key_scale = _collect_lightllm_kv_scale(
            blockwise_opt.kv_module.k_scales_buffer[layer_idx],
            blockwise_opt.kv_module.k_zeros_buffer[layer_idx],
            blockwise_opt.kv_module.k_qmin_buffer[layer_idx],
            blockwise_opt.kv_module.k_qmax_buffer[layer_idx],
        )
        value_scale = _collect_lightllm_kv_scale(
            blockwise_opt.kv_module.v_scales_buffer[layer_idx],
            blockwise_opt.kv_module.v_zeros_buffer[layer_idx],
            blockwise_opt.kv_module.v_qmin_buffer[layer_idx],
            blockwise_opt.kv_module.v_qmax_buffer[layer_idx],
        )
        if key_scale is None or value_scale is None:
            raise ValueError(f'Calibration scale for layer {layer_idx} is empty.')

        scale_row = torch.cat([key_scale.reshape(-1), value_scale.reshape(-1)]).tolist()
        scales.append(scale_row)

    scale_width = len(scales[0]) if scales else 0
    if granularity == 'per_tensor' and scale_width != 2:
        raise ValueError(f'per_tensor export expects 2 scales per layer, got {scale_width}')
    if granularity == 'per_head' and scale_width != num_head * 2:
        raise ValueError(
            f'per_head export expects {num_head * 2} scales per layer, got {scale_width}'
        )

    architectures = getattr(blockwise_opt.model.model_config, 'architectures', None)
    if isinstance(architectures, list) and len(architectures) > 0:
        architectures = architectures[0]
    elif architectures is None:
        architectures = blockwise_opt.config.model.type

    return {
        'version': '1.0',
        'architectures': architectures,
        'quant_type': granularity,
        'qmin': float(torch.finfo(torch.float8_e4m3fn).min),
        'qmax': float(torch.finfo(torch.float8_e4m3fn).max),
        'num_layers': num_layers,
        'num_head': num_head,
        'scales_shape': [num_layers, scale_width],
        'scales': _to_jsonable(scales),
    }
