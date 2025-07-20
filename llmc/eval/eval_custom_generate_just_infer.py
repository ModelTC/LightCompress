import glob
import json
import os

import torch
from human_eval.data import stream_jsonl, write_jsonl
from human_eval.evaluation import evaluate_functional_correctness
from loguru import logger
from tqdm import tqdm

from .eval_base import BaseEval


class CustomGenerateJustInfer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.eval_cfg = config.eval

    @torch.no_grad()
    def eval(self, model, eval_pos=None):
        logger.info('start inference')

        with open(os.path.join(self.eval_cfg.path, 'samples.json'), 'r') as f:
            questions_list = json.load(f)

        custom_samples_ans = self.model.eval_custom_samples_just_infer(
            questions_list,
            self.eval_cfg
        )

        with open(os.path.join('custom_samples_ans.json'), 'w') as f:
            json.dump(custom_samples_ans, f, indent=4)

        torch.cuda.empty_cache()
        return 'custom gen done.'
