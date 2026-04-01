import gc
import json
import os
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
from datasets import load_dataset, load_from_disk
from human_eval.data import read_problems
from loguru import logger


class BaseEval:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.tokenizer = self.model.get_tokenizer()
        # eval_cfg
        self.eval_cfg = config.eval
        self.model_type = config.model.type
        logger.info(f'eval_cfg : {self.eval_cfg}')
        self.eval_dataset_name = self.eval_cfg['name']
        self.dataset_type = self.eval_cfg.get('type', 'ppl')
        assert self.eval_dataset_name in [
            'wikitext2',
            'c4',
            'ptb',
            'pile',
            'custom',
            'human_eval',
            'mme',
            'custom_ppl',
            'custom_gen',
            't2v',
            'i2v',
        ], f'Not support {self.eval_dataset_name} dataset now.'
        self.seq_len = self.eval_cfg.get('seq_len', None)
        self.num_samples = self.eval_cfg.get('num_samples', None)
        self.num_eval_tokens = self.eval_cfg.get('num_eval_tokens', None)
        self.eval_dataset_bs = self.eval_cfg['bs']
        self.eval_dataset_path = self.eval_cfg.get('path', None)
        self.apply_chat_template = self.eval_cfg.get('apply_chat_template', False)
        self.download = self.eval_cfg.get('download', False)
        self.load_from_txt = self.eval_cfg.get('load_from_txt', False)
        self.inference_per_block = self.eval_cfg.get('inference_per_block', False)
        self.testenc = self.build_data()

    @torch.no_grad()
    def build_data(self):
        # load data
        if self.eval_dataset_name == 'human_eval':
            testenc = read_problems()
        else:
            if self.download:
                if self.eval_dataset_name == 'wikitext2':
                    testdata = load_dataset(
                        'wikitext', 'wikitext-2-raw-v1', split='test'
                    )
                elif self.eval_dataset_name == 'c4':
                    testdata = load_dataset(
                        'allenai/c4',
                        data_files={
                            'validation': 'en/c4-validation.00000-of-00008.json.gz'
                        },
                        split='validation',
                    )
                elif self.eval_dataset_name == 'ptb':
                    testdata = load_dataset(
                        'ptb_text_only', 'penn_treebank', split='test'
                    )
                elif self.eval_dataset_name == 'pile':
                    testdata = load_dataset(
                        'mit-han-lab/pile-val-backup', split='validation'
                    )
            else:
                if self.eval_dataset_name in ['custom_gen', 'custom_ppl', 't2v', 'i2v']:
                    testdata = self.get_cutomdata(self.eval_dataset_path)
                else:
                    assert self.eval_dataset_path, 'Please set path in eval_cfg.'
                    testdata = load_from_disk(self.eval_dataset_path)
            self.testdata = testdata
            # encode data
            if self.dataset_type == 'decode_ppl':
                assert self.eval_dataset_name == 'wikitext2'
                testenc = testdata['text']
            elif self.eval_dataset_name == 'wikitext2':
                testenc = self.tokenizer(
                    '\n\n'.join(testdata['text']), return_tensors='pt'
                )
            elif self.eval_dataset_name == 'c4':
                testenc = self.tokenizer(
                    ' '.join(testdata[:1100]['text']), return_tensors='pt'
                )
                testenc.input_ids = testenc.input_ids[:, : (256 * self.seq_len)]
            elif self.eval_dataset_name == 'ptb':
                testenc = self.tokenizer(
                    ' '.join(testdata['sentence']), return_tensors='pt'
                )
            elif self.eval_dataset_name == 'pile':
                testenc = self.tokenizer(
                    '\n\n'.join(testdata['text'][:1000]), return_tensors='pt'
                )
            elif self.eval_dataset_name == 'custom_ppl':
                testenc = self.tokenizer(
                    '\n'.join([data['question'] + data['answer'] if 'answer' in data else data['question'] for data in testdata]), # noqa
                    return_tensors='pt',
                )
            elif self.eval_dataset_name == 'custom_gen':
                testenc = []
                if self.eval_dataset_bs < 0:
                    testenc.append(
                        self.model.batch_process(
                            testdata,
                            calib_or_eval='eval',
                            apply_chat_template=self.apply_chat_template
                        )
                    )
                elif self.eval_dataset_bs == 1:
                    testenc = [
                        self.model.batch_process(
                            [sample],
                            calib_or_eval='eval',
                            apply_chat_template=self.apply_chat_template
                        )
                        for sample in testdata
                    ]  # noqa
                elif self.eval_dataset_bs > 1:
                    for i in range(0, len(testdata), self.eval_dataset_bs):
                        start = i
                        end = min(i + self.eval_dataset_bs, len(testdata))
                        batch = testdata[start:end]
                        testenc.append(
                            self.model.batch_process(
                                batch,
                                calib_or_eval='eval',
                                apply_chat_template=self.apply_chat_template
                            )
                        )
            elif self.eval_dataset_name in ['t2v', 'i2v']:
                testenc = self.testdata
        return testenc

    def get_cutomdata(self, custom_dataset):
        audio_img_qa_json = os.path.join(custom_dataset, 'samples.json')
        fp = open(audio_img_qa_json)
        custom_data_samples = json.load(fp)
        for idx in range(len(custom_data_samples)):
            if 'audio' in custom_data_samples[idx]:
                if isinstance(custom_data_samples[idx]['audio'], list):
                    for audio_idx in range(len(custom_data_samples[idx]['audio'])):
                        custom_data_samples[idx]['audio'][audio_idx] = os.path.join(
                            custom_dataset, custom_data_samples[idx]['audio'][audio_idx]
                        )
                else:
                    custom_data_samples[idx]['audio'] = os.path.join(
                        custom_dataset, custom_data_samples[idx]['audio']
                    )
            else:
                custom_data_samples[idx]['audio'] = None
            if 'image' in custom_data_samples[idx]:
                if isinstance(custom_data_samples[idx]['image'], list):
                    for img_idx in range(len(custom_data_samples[idx]['image'])):
                        custom_data_samples[idx]['image'][img_idx] = os.path.join(
                            custom_dataset, custom_data_samples[idx]['image'][img_idx]
                        )
                else:
                    custom_data_samples[idx]['image'] = os.path.join(
                        custom_dataset, custom_data_samples[idx]['image']
                    )
            else:
                custom_data_samples[idx]['image'] = None
            if 'question' not in custom_data_samples[idx]:
                custom_data_samples[idx]['question'] = ''
            if 'answer' not in custom_data_samples[idx]:
                custom_data_samples[idx]['answer'] = ''
            if 'prompt' not in custom_data_samples[idx]:
                custom_data_samples[idx]['prompt'] = ''
            if 'negative_prompt' not in custom_data_samples[idx]:
                custom_data_samples[idx]['negative_prompt'] = ''
        return custom_data_samples

    @torch.no_grad()
    def forward_pre_hook(self, m, x):
        m.cuda()

    @torch.no_grad()
    def forward_hook(self, m, x, y):
        with ThreadPoolExecutor() as executor:
            executor.submit(self.load_layer_to_cpu, m)

    @torch.no_grad()
    def load_layer_to_cpu(self, m):
        m.cpu()

    def register_hooks(self, model):
        handles = []
        for layer in model.get_blocks():
            handles.append(layer.register_forward_pre_hook(self.forward_pre_hook))
        for layer in model.get_blocks():
            handles.append(layer.register_forward_hook(self.forward_hook))
        for layer in model.get_layers_except_blocks():
            layer.cuda()
        return handles

    @torch.no_grad()
    def eval(self, model_llmc, eval_pos=None):
        handles = []
        if self.inference_per_block:
            handles = self.register_hooks(model_llmc)
        else:
            if model_llmc.mm_model:
                model_llmc.mm_model.cuda()
            else:
                model_llmc.model.cuda()

        if model_llmc.mm_model:
            model_llmc.mm_model.eval()
        else:
            model_llmc.model.eval()

        eval_res = self.eval_func(
            model_llmc,
            self.testenc,
            self.seq_len,
            self.eval_dataset_bs,
            eval_pos,
        )
        if self.inference_per_block:
            for h in handles:
                h.remove()

        if model_llmc.mm_model:
            model_llmc.mm_model.cpu()
        else:
            model_llmc.model.cpu()

        gc.collect()
        torch.cuda.empty_cache()
        return eval_res

    def post_process(self, testenc):
        pass
