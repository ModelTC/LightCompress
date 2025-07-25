import json
import os

import torch
from loguru import logger


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

        self.eval_answer(custom_samples_ans)

        with open(os.path.join(self.config.save.save_path), 'w') as f:
            json.dump(custom_samples_ans, f, indent=4)

        torch.cuda.empty_cache()
        return 'custom gen done.'

    def eval_answer(self, data):
        T1V = 0
        T1V_T2V = 0

        def create_pairs(lst):
            return [(lst[i], lst[i + 1]) for i in range(0, len(lst), 2)]

        def check_acc(gt, answer, turn):
            if gt[turn].lower() in answer[turn].lower():
                return True
            return False

        pair_data = create_pairs(data)

        for idx, item in enumerate(pair_data):
            assert item[0]['image'] == item[1]['image']

            pair1 = item[0]
            pair2 = item[1]

            if check_acc(pair1['gt'], pair1['answer'], 0):
                T1V += 1
                if check_acc(pair2['gt'], pair2['answer'], 1):
                    T1V_T2V += 1
                assert pair1['question'][0] == pair2['question'][1]

            if check_acc(pair2['gt'], pair2['answer'], 0):
                T1V += 1
                if check_acc(pair1['gt'], pair1['answer'], 1):
                    T1V_T2V += 1
                assert pair2['question'][0] == pair1['question'][1]

        logger.info(f'CustomGenerateJustInfer T1V: {T1V}, T1V_T2V: {T1V_T2V}')
        logger.info(f'CustomGenerateJustInfer Possibility: {T1V_T2V / T1V}')
