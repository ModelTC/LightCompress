import json


def update_vllm_quant_config(
    model,
    config,
    save_quant_path,
    vllm_quant_method='compressed-tensors',

):
    need_pack = config.quant.weight.get('need_pack', False)
    weight_quant_type = config.quant.weight.get('quant_type', 'int-quant')
    if 'act' in config.quant:
        act_quant_type = config.quant.act.get('quant_type', 'int-quant')
        assert act_quant_type == weight_quant_type
    else:
        act_quant_type = None
    if act_quant_type is not None and act_quant_type == 'float-quant':
        if config.quant.act.get('static', False):
            quant_config = {
                'activation_scheme': 'static',
                'ignored_layers': [
                    model.skip_layer_name()
                ],
                'quant_method': 'fp8'
            }
            config_file = save_quant_path + '/config.json'
            with open(config_file, 'r') as file:
                config_vllm = json.load(file)
            config_vllm['quantization_config'] = quant_config
            with open(config_file, 'w') as file:
                json.dump(config_vllm, file, indent=4)
            return
        # elif config.quant.weight.get('granularity', 'per_block'):
        elif config.quant.weight.get('granularity') == 'per_block':
            quant_config = {
                'activation_scheme': 'dynamic',
                'fmt': 'e4m3',
                'quant_method': 'fp8',
                'weight_block_size': [
                    config.quant.weight.block_size,
                    config.quant.weight.block_size
                ]
            }
            config_file = save_quant_path + '/config.json'
            with open(config_file, 'r') as file:
                config_vllm = json.load(file)
            config_vllm['quantization_config'] = quant_config
            with open(config_file, 'w') as file:
                json.dump(config_vllm, file, indent=4)
            return
        else:
            vllm_quant_format = 'float-quantized'
            quant_type = 'float'
            w_num_bits = 8
            a_num_bits = 8
    elif need_pack:
        vllm_quant_format = 'pack-quantized'
        quant_type = 'int'
        w_num_bits = config.quant.weight.bit
    elif weight_quant_type == 'float-quant':
        vllm_quant_format = 'float-quantized'
        quant_type = 'float'
        w_num_bits = 8
    else:
        vllm_quant_format = 'int-quantized'
        quant_type = 'int'
        w_num_bits = config.quant.weight.bit
        if 'act' in config.quant:
            a_num_bits = config.quant.act.bit

    if config.quant.weight.granularity == 'per_group':
        group_size = config.quant.weight.group_size
    else:
        group_size = None

    if 'act' in config.quant and 'static' in config.quant.act:
        dynamic = not config.quant.act.static
    else:
        dynamic = True

    quant_config = {
        'config_groups': {
            'group_0': {
                'targets': ['Linear'],  # Now only support "Linear".
                'input_activations': {
                    'dynamic': dynamic,
                    'group_size': None,   # Don't support activations per-group quant.
                    'num_bits': a_num_bits,
                    'observer': 'minmax',
                    'observer_kwargs': {},
                    'strategy': 'token'
                                if config.quant.act.granularity == 'per_token'
                                else 'tensor',
                    'symmetric': config.quant.act.symmetric,
                    'type': quant_type
                } if 'act' in config.quant else None,
                'weights': {
                    'dynamic': False,
                    'group_size': group_size,
                    'num_bits': w_num_bits,
                    'observer': 'minmax',  # Now only support "minmax".
                    'observer_kwargs': {},
                    'strategy': (
                        'group'
                        if config.quant.weight.granularity == 'per_group'
                        else 'channel'
                    ),
                    'symmetric': config.quant.weight.symmetric,
                    'type': quant_type,
                },
            }
        },
        'format': vllm_quant_format,
        'ignore': model.skip_layer_name(),
        'quant_method': vllm_quant_method,
    }

    config_file = save_quant_path + '/config.json'
    with open(config_file, 'r') as file:
        config_vllm = json.load(file)
    if weight_quant_type == 'int-quant' and 'quantization_config' in config_vllm:
        del config_vllm['quantization_config']
    config_vllm['compression_config'] = quant_config
    with open(config_file, 'w') as file:
        json.dump(config_vllm, file, indent=4)
