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
    # 从注册表拿模型并实例化
    # 动态分配模型
    model = MODEL_REGISTRY[config.model.type](config)

    # 打印模型和tokenizer
    logger.info(f'model: {model}')
    logger.info(f'tokenizer: {model.get_tokenizer()}')

    # 获得需要的评测种类
    eval_list = get_eval_list(model, config)
    # 真正执行评测
    eval_model(model, None, eval_list, eval_pos='pretrain')

    blockwise_opts = []
    # 取出处理模态
    modalities, modality_configs = get_modality(config)

    for modality, modality_config in zip(modalities, modality_configs):
        model.set_modality(modality)
        if not config.get('calib', False):
            # 不需要校准数据 直接构造算法对象
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
            # 需要校准数据
            dataset = BaseDataset(
                model.get_tokenizer(), config.calib, model.batch_process
            )
            calib_data, padding_mask = dataset.get_calib_dataset()
            # 收集第一层block输入 为后续blockwise算法需要的输入缓存下来
            model.collect_first_block_input(calib_data, padding_mask)
            del calib_data
            gc.collect()
            torch.cuda.empty_cache()
            # 构造算法对象
            blockwise_opt = ALGO_REGISTRY[modality_config.method](
                model,
                modality_config,
                model.get_first_block_input(),
                model.get_padding_mask(),
                config,
            )
            # 项目逐层block做优化
            blockwise_opt.run_block_loop()
            blockwise_opts.append(blockwise_opt)
            dist.barrier()

    # 对变化后的浮点模型做评测
    eval_model(model, blockwise_opts, eval_list, eval_pos='transformed')
    # 只有rank 0继续做保存和导出
    if int(os.environ['RANK']) == 0:
        if 'save' in config and config.save.get('save_calib_json', False):
            # 收集各个模态/量化器导出的校准结果。
            calib_json_list = [
                blockwise_opt.collect_calib_json()
                for blockwise_opt in blockwise_opts
                if hasattr(blockwise_opt, 'collect_calib_json')
            ]
            # 单模态时保持扁平结构，兼容 LightLLM 的校准文件格式。
            calib_json_payload = (
                calib_json_list[0] if len(calib_json_list) == 1 else calib_json_list
            )
            # 将最终的校准 JSON 写入配置指定的输出路径。
            with open(save_calib_json_path, 'w') as file:
                json.dump(calib_json_payload, file, ensure_ascii=False, indent=4)
            logger.info(f'save calib json done -- {save_calib_json_path}')

        # 保存变换后的浮点模型
        if 'save' in config and config.save.get('save_trans', False):
            blockwise_opt.save_model(save_trans_path)

        # 保存TensorRT-LLM格式并构建engine
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

        # 切换到fake quant部署模式再保存
        if 'save' in config and config.save.get('save_fake', False):
            deploy_all_modality(blockwise_opts, 'fake_quant')
            blockwise_opt.save_model(save_fake_path)

        if 'save' in config:
            # 导出真实量化模型给推理后端
            if (
                # 导出前进行遍历检查
                config.save.get('save_vllm', False)
                or config.save.get('save_sgl', False)
                or config.save.get('save_lightllm', False)
            ):
                for modality_config in modality_configs:
                    w, a = modality_config.weight, modality_config.get('act')

                    # 只允许特定bit类型
                    if isinstance(w.bit, str):
                        # 必须对称量化
                        assert w.symmetric, 'Only symmetric quant is supported.'
                        assert w.bit in ['e4m3', 'e3m4'], 'Supported quant: w8a16.'
                        # 有激活量化的话，那激活也要满足对称、bit合法的要求
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
                        # 是整数则必须是4 or 8
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

            # 给特定后端（AutoAWQ导出
            elif config.save.get('save_autoawq', False):
                for modality_config in modality_configs:
                    # 只能4 bit 仅含有weight 不支持act
                    assert (
                        modality_config.weight.bit in [4] and 'act' not in modality_config
                    ), 'AutoAWQ supports only 4-bit weight-only quantization.'
                    assert (
                    # 不能对称量化
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

        # 判断是否有opencompass
        if 'opencompass' in config:
            assert config.save.get('save_trans', False)
            # 从配置里读取cfg_path, output_path
            cfg_path = config['opencompass']['cfg_path']
            output_path = config['opencompass']['output_path']
            # 取路径
            eval_model_path = os.path.abspath(save_trans_path)
            # 拼指令
            opencompass_cmd = (
                f'opencompass {cfg_path} -w {output_path} '
                f'--llmc_cfg {args.config} '
                f'--llmc_eval_mode quant '
                f'--llmc_model_path {eval_model_path}'
            )
            logger.info(f'opencompass_cmd : {opencompass_cmd}')
            # 执行
            os.system(opencompass_cmd)
    dist.barrier()


if __name__ == '__main__':
    logger.add(sys.stdout, level='INFO')
    llmc_start_time = time.time()
    parser = argparse.ArgumentParser()
    # 解析命令行参数
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--task_id', type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        # 读取配置文件
        config = yaml.safe_load(file)
    config = EasyDict(config)

    init_process_group(backend='nccl')
    # 初始化分布式环境 设置GPU
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))

    # 检查配置 打印依赖版本
    if int(os.environ['RANK']) != 0:
        logger.remove()

    # 检查配置是否合法
    check_config(config)

    logger.info(f'args: {args}')
    logger.info(f'config:\n{json.dumps(config, ensure_ascii=False, indent=4)}')

    print_important_package_version()

    logger.info(f'WORLD_SIZE : {int(os.environ["WORLD_SIZE"])}')

    seed_all(config.base.seed + int(os.environ['RANK']))

    # Ensure only the main process creates directories
    if int(os.environ['RANK']) == 0:
        if 'save' in config:
            if config.save.get('save_calib_json', False):
                mkdirs(config.save.save_path)
                save_calib_json_path = os.path.join(
                    config.save.save_path,
                    config.save.get('calib_json_name', 'calib_scales.json'),
                )
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
    
