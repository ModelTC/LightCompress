import argparse
import gc
import json
import os
import sys
import time

import torch
import torch.distributed as dist
import yaml
from easydict import EasyDict
from loguru import logger
from torch.distributed import destroy_process_group, init_process_group

from llmc.compression.quantization import *
from llmc.compression.sparsification import *
from llmc.compression.token_reduction import *
from llmc.data import BaseDataset
from llmc.eval.utils import eval_model, get_eval_list
from llmc.models import *
from llmc.utils import (check_config, deploy_all_modality, get_modality,
                        mkdirs, print_important_package_version, seed_all,
                        update_autoawq_quant_config,
                        update_lightx2v_quant_config, update_vllm_quant_config)
from llmc.utils.registry_factory import ALGO_REGISTRY, MODEL_REGISTRY


def main(config):
    model = MODEL_REGISTRY[config.model.type](config)

    logger.info(f'model: {model}')
    logger.info(f'tokenizer: {model.get_tokenizer()}')

    eval_list = get_eval_list(model, config)
    eval_model(model, None, eval_list, eval_pos='pretrain')

    blockwise_opts = []
    modalities, modality_configs = get_modality(config)

    for modality, modality_config in zip(modalities, modality_configs):
        model.set_modality(modality)
        if not config.get('calib', False):
            blockwise_opt = ALGO_REGISTRY[modality_config.method](
                model,
                modality_config,
                input=None,
                padding_mask=None,
                config=config,
            )
            blockwise_opt.run_block_loop()
            blockwise_opts.append(blockwise_opt)
            dist.barrier()
        else:
            dataset = BaseDataset(
                model.get_tokenizer(), config.calib, model.batch_process
            )
            calib_data, padding_mask = dataset.get_calib_dataset()
            model.collect_first_block_input(calib_data, padding_mask)
            del calib_data
            gc.collect()
            torch.cuda.empty_cache()
            blockwise_opt = ALGO_REGISTRY[modality_config.method](
                model,
                modality_config,
                model.get_first_block_input(),
                model.get_padding_mask(),
                config,
            )
            blockwise_opt.run_block_loop()
            blockwise_opts.append(blockwise_opt)
            dist.barrier()

    eval_model(model, blockwise_opts, eval_list, eval_pos='transformed')
    if int(os.environ['RANK']) == 0:
        if 'save' in config and config.save.get('save_trans', False):
            blockwise_opt.save_model(save_trans_path)

        if 'save' in config and config.save.get('save_trtllm', False):
            blockwise_opt.save_model(save_trtllm_trans_path)
            from llmc.utils.export_trtllm import cvt_trtllm_engine

            cvt_trtllm_engine(
                save_trtllm_trans_path,
                save_trtllm_engine_path,
                config.save.get('trtllm_cfg'),
            )

        eval_model(model, blockwise_opts, eval_list, eval_pos='fake_quant')
        eval_model(model, blockwise_opts, eval_list, eval_pos='fake_quant_wo_kv')

        if 'save' in config and config.save.get('save_fake', False):
            deploy_all_modality(blockwise_opts, 'fake_quant')
            blockwise_opt.save_model(save_fake_path)

        if 'save' in config:
            if (
                config.save.get('save_vllm', False)
                or config.save.get('save_sgl', False)
                or config.save.get('save_lightllm', False)
            ):
                for modality_config in modality_configs:
                    w, a = modality_config.weight, modality_config.get('act')

                    if isinstance(w.bit, str):
                        assert w.symmetric, 'Only symmetric quant is supported.'
                        assert w.bit in ['e4m3', 'e3m4'], 'Supported quant: w8a16.'
                        if a:
                            assert (
                                w.symmetric and a.symmetric
                            ), 'Only symmetric quant is supported.'
                            assert (
                                w.bit == a.bit
                                and w.bit in ['e4m3', 'e5m2']
                                and a.bit in ['e4m3', 'e5m2']
                            ), 'Only WA FP8 quant is supported'
                    else:
                        assert w.symmetric, 'Only symmetric quant is supported.'
                        assert w.bit in [4, 8], 'Supported quant: w4a16, w8a16, w8a8.'
                        if a:
                            assert a.symmetric, 'Only symmetric quant is supported.'
                            assert a.bit == 8, 'Supported quant: w4a16, w8a16, w8a8.'

                if config.save.get('save_vllm', False):
                    deploy_all_modality(blockwise_opts, 'vllm_quant')
                elif config.save.get('save_lightllm', False):
                    deploy_all_modality(blockwise_opts, 'lightllm_quant')
                elif config.save.get('save_sgl', False):
                    deploy_all_modality(blockwise_opts, 'sgl_quant')

                blockwise_opt.save_model(save_quant_path)
                update_vllm_quant_config(blockwise_opt.model, config, save_quant_path)

            elif config.save.get('save_autoawq', False):
                for modality_config in modality_configs:
                    assert (
                        modality_config.weight.bit in [4] and 'act' not in modality_config
                    ), 'AutoAWQ supports only 4-bit weight-only quantization.'
                    assert (
                        not modality_config.weight.symmetric
                    ), 'Only asymmetric quant is supported.'

                deploy_all_modality(blockwise_opts, 'autoawq_quant')
                blockwise_opt.save_model(save_quant_path)
                update_autoawq_quant_config(config, save_quant_path)

            elif config.save.get('save_mlcllm', False):
                for modality_config in modality_configs:
                    assert (
                        modality_config.weight.bit in [4] and 'act' not in modality_config
                    ), 'MlcLLM supports only 4-bit weight-only quantization.'
                    assert (
                        not modality_config.weight.symmetric
                    ), 'Only asymmetric quant is supported.'

                deploy_all_modality(blockwise_opts, 'mlcllm_quant')
                blockwise_opt.save_model(save_quant_path)
                update_autoawq_quant_config(config, save_quant_path)

            elif config.save.get('save_lightx2v', False):
                deploy_all_modality(blockwise_opts, 'lightx2v_quant')
                blockwise_opt.save_model(save_quant_path)
                update_lightx2v_quant_config(save_quant_path)

        if 'opencompass' in config:
            assert config.save.get('save_trans', False)
            cfg_path = config['opencompass']['cfg_path']
            output_path = config['opencompass']['output_path']
            eval_model_path = os.path.abspath(save_trans_path)
            opencompass_cmd = (
                f'opencompass {cfg_path} -w {output_path} '
                f'--llmc_cfg {args.config} '
                f'--llmc_eval_mode quant '
                f'--llmc_model_path {eval_model_path}'
            )
            logger.info(f'opencompass_cmd : {opencompass_cmd}')
            os.system(opencompass_cmd)
    dist.barrier()


if __name__ == '__main__':
    logger.add(sys.stdout, level='INFO')
    llmc_start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--task_id', type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    config = EasyDict(config)

    init_process_group(backend='nccl')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))

    if int(os.environ['RANK']) != 0:
        logger.remove()

    check_config(config)

    logger.info(f'args: {args}')
    logger.info(f'config:\n{json.dumps(config, ensure_ascii=False, indent=4)}')

    print_important_package_version()

    logger.info(f'WORLD_SIZE : {int(os.environ["WORLD_SIZE"])}')

    seed_all(config.base.seed + int(os.environ['RANK']))

    # Ensure only the main process creates directories
    if int(os.environ['RANK']) == 0:
        if 'save' in config:
            if config.save.get('save_trans', False):
                save_trans_path = os.path.join(
                    config.save.save_path, 'transformed_model'
                )
                mkdirs(save_trans_path)
            if config.save.get('save_trtllm', False):
                save_trtllm_trans_path = os.path.join(
                    config.save.save_path, 'trtllm_transformed_model'
                )
                mkdirs(save_trtllm_trans_path)
                save_trtllm_engine_path = os.path.join(
                    config.save.save_path, 'trtllm_engine'
                )
                mkdirs(save_trtllm_engine_path)
            if config.save.get('save_vllm', False):
                save_quant_path = os.path.join(
                    config.save.save_path, 'vllm_quant_model'
                )
                mkdirs(save_quant_path)
            if config.save.get('save_lightllm', False):
                save_quant_path = os.path.join(
                    config.save.save_path, 'lightllm_quant_model'
                )
                mkdirs(save_quant_path)
            if config.save.get('save_sgl', False):
                save_quant_path = os.path.join(config.save.save_path, 'sgl_quant_model')
                mkdirs(save_quant_path)
            if config.save.get('save_autoawq', False):
                save_quant_path = os.path.join(
                    config.save.save_path, 'autoawq_quant_model'
                )
                mkdirs(save_quant_path)
            if config.save.get('save_mlcllm', False):
                save_quant_path = os.path.join(
                    config.save.save_path, 'mlcllm_quant_model'
                )
                mkdirs(save_quant_path)
            if config.save.get('save_lightx2v', False):
                save_quant_path = os.path.join(
                    config.save.save_path, 'lightx2v_quant_model'
                )
                mkdirs(save_quant_path)
            if config.save.get('save_fake', False):
                save_fake_path = os.path.join(config.save.save_path, 'fake_quant_model')
                mkdirs(save_fake_path)

    # Synchronize all processes after directory creation
    dist.barrier()

    main(config)

    destroy_process_group()

    llmc_end_time = time.time()
    llmc_duration_time = llmc_end_time - llmc_start_time
    logger.info(f'llmc_duration_time: {llmc_duration_time} s')
    logger.info('--- llmc finished ---')
