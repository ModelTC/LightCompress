import copy
import os

from loguru import logger

from llmc.eval import (AccuracyEval, CustomGenerate, CustomGenerateJustInfer,
                       DecodePerplexityEval, HumanEval, PerplexityEval,
                       TokenConsistencyEval, VideoGenerateEval, VQAEval)
from llmc.utils import deploy_all_modality


def get_eval_list(model, config):
    eval_list = []
    if int(os.environ['RANK']) == 0:
        if 'eval' in config:
            if 'type' in config.eval and config.eval.type == 'decode_ppl':
                if 'pretrain' in config.eval.eval_pos:
                    raise ValueError(
                        'Unsupported: Evaluating decode_ppl with a pretrained model. '
                    )
                    # Pretrained models do not use key-value caching.
                    # Please use a transformed model to evaluate decode_ppl
                    # for the original model.

            if not isinstance(config.eval, list):
                eval_config_list = [config.eval]
            else:
                eval_config_list = config.eval
            for eval_config in eval_config_list:
                config_tmp = copy.deepcopy(config)
                config_tmp.eval = eval_config
                if 'type' not in config_tmp.eval:
                    config_tmp.eval['type'] = 'ppl'
                if 'eval' in config_tmp and len(config_tmp.eval.eval_pos):
                    name_list = (
                        config_tmp.eval.name
                        if not isinstance(config_tmp.eval.name, str)
                        else [config_tmp.eval.name]
                    )
                    for name in name_list:
                        config_for_eval = copy.deepcopy(config_tmp)
                        config_for_eval.eval.name = name
                        if len(name_list) != 1:  # eval multi datasets
                            config_for_eval.eval.path = os.path.join(
                                config_tmp.eval.path, name
                            )
                        if 'type' not in config_tmp.eval:
                            config_tmp.eval.type == 'ppl'
                        if config_tmp.eval.type == 'acc':
                            eval_class = AccuracyEval(config_for_eval)
                        elif config_tmp.eval.type == 'vqa':
                            eval_class = VQAEval(config_for_eval)
                        elif (
                            config_tmp.eval.type == 'code'
                            and config_tmp.eval.name == 'human_eval'
                        ):
                            eval_class = HumanEval(model, config_for_eval)
                        elif config_tmp.eval.type == 'generate_only':
                            eval_class = CustomGenerate(model, config_for_eval)
                        elif config_tmp.eval.type == 'just_infer':
                            eval_class = CustomGenerateJustInfer(model, config_for_eval)
                        elif config_tmp.eval.type == 'token_acc':
                            eval_class = TokenConsistencyEval(model, config_for_eval)
                        elif config_tmp.eval.type == 'ppl':
                            eval_class = PerplexityEval(model, config_for_eval)
                        elif config_tmp.eval.type == 'decode_ppl':
                            eval_class = DecodePerplexityEval(model, config_for_eval)
                        elif config_tmp.eval.type == 'video_gen':
                            eval_class = VideoGenerateEval(model, config_for_eval)
                        else:
                            raise ValueError(
                                f'Unsupported eval type: {config_tmp.eval.type}'
                            )
                        eval_list.append((eval_class, config_for_eval))
    return eval_list


def eval_model(model, blockwise_opts, eval_list, eval_pos):
    if int(os.environ['RANK']) == 0:
        do_eval = False
        for _, config_for_eval in eval_list:
            if eval_pos in config_for_eval.eval.eval_pos:
                do_eval = True
        if do_eval:
            if eval_pos == 'transformed':
                deploy_all_modality(blockwise_opts, 'origin_float')
            elif eval_pos in ['fake_quant', 'fake_quant_wo_kv']:
                deploy_all_modality(blockwise_opts, 'fake_quant')
            for eval_class, config_for_eval in eval_list:
                if eval_pos in config_for_eval.eval.eval_pos:
                    res = eval_class.eval(model, eval_pos)
                    eval_name = config_for_eval.eval.type
                    dataset_name = config_for_eval.eval.name
                    logger.info(f'EVAL: {eval_name} on {dataset_name} is {res}')
