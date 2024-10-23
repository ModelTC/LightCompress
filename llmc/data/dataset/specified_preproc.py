import json
import os
import random

import torch

from llmc.utils.registry_factory import PREPROC_REGISTRY


@PREPROC_REGISTRY
def wikitext2_gptq(calib_dataset, tokenizer, n_samples, seq_len):
    trainenc = tokenizer('\n\n'.join(calib_dataset['text']), return_tensors='pt')
    samples = []
    for _ in range(n_samples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seq_len - 1)
        j = i + seq_len
        inp = trainenc.input_ids[:, i:j]
        samples.append(inp)
    return samples


@PREPROC_REGISTRY
def ptb_gptq(calib_dataset, tokenizer, n_samples, seq_len):
    trainenc = tokenizer(' '.join(calib_dataset['sentence']), return_tensors='pt')
    samples = []
    for _ in range(n_samples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seq_len - 1)
        j = i + seq_len
        inp = trainenc.input_ids[:, i:j]
        samples.append(inp)
    return samples


@PREPROC_REGISTRY
def c4_gptq(calib_dataset, tokenizer, n_samples, seq_len):
    samples = []
    for _ in range(n_samples):
        while True:
            i = random.randint(0, len(calib_dataset) - 1)
            trainenc = tokenizer(calib_dataset[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seq_len:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seq_len - 1)
        j = i + seq_len
        inp = trainenc.input_ids[:, i:j]
        samples.append(inp)
    return samples


@PREPROC_REGISTRY
def pileval_awq(calib_dataset, tokenizer, n_samples, seq_len):
    dataset = calib_dataset.shuffle(seed=42)
    samples = []
    n_run = 0
    for data in dataset:
        line = data['text']
        line = line.strip()
        line_encoded = tokenizer.encode(line)
        if len(line_encoded) > seq_len:
            continue
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        n_run += 1
        if n_run == n_samples:
            break
    samples = torch.cat(samples, dim=1)
    n_split = samples.shape[1] // seq_len
    samples = [samples[:, i * seq_len: (i + 1) * seq_len] for i in range(n_split)]
    return samples


@PREPROC_REGISTRY
def pileval_smooth(calib_dataset, tokenizer, n_samples, seq_len):
    dataset = calib_dataset.shuffle(seed=42)
    samples = []
    n_run = 0
    for data in dataset:
        line = data['text']
        trainenc = tokenizer(
            line, return_tensors='pt', max_length=seq_len, truncation=True
        )
        line_encoded = trainenc.input_ids
        samples.append(line_encoded)
        n_run += 1
        if n_run == n_samples:
            break
    return samples


@PREPROC_REGISTRY
def pileval_omni(calib_dataset, tokenizer, n_samples, seq_len):
    trainenc = tokenizer('\n\n'.join(calib_dataset['text'][:1000]), return_tensors='pt')
    samples = []
    for _ in range(n_samples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seq_len - 1)
        j = i + seq_len
        inp = trainenc.input_ids[:, i:j]
        samples.append(inp)
    return samples


@PREPROC_REGISTRY
def vlm_general(calib_dataset, tokenizer, preprocess, n_samples):
    img_qa_json = os.path.join(calib_dataset, 'img_qa.json')
    fp = open(img_qa_json)
    img_qas = json.load(fp)
    for idx in range(len(img_qas)):
        img_qas[idx]['img'] = os.path.join(calib_dataset, img_qas[idx]['img'])
    random.shuffle(img_qas)
    if len(img_qas) > n_samples:
        img_qas = img_qas[:n_samples]
    vlm_data = preprocess(img_qas)
    samples = []
    for data in vlm_data:
        if 'input_ids' in data:
            samples.append(data)
        elif 'text' in data:
            trainenc = tokenizer(data['text'], return_tensors='pt')
            inp = trainenc.input_ids
            samples.append(
                {
                    'pixel_values': data['pixel_values'],
                    'input_ids': inp
                }
            )
        else:
            raise Exception(f'Both input_ids and text are not in data. data is: {data}')
    return samples


@PREPROC_REGISTRY
def img_sampler(calib_dataset, processor, n_samples):
    random.shuffle(calib_dataset)
    samples = []
    n_run = 0
    for image in calib_dataset:
        inp = processor(images=image, return_tensors='pt')
        samples.append(inp)
        n_run += 1
        if n_run == n_samples:
            break
    return samples


@PREPROC_REGISTRY
def random_truncate_txt(calib_dataset, tokenizer, n_samples, seq_len):
    random.shuffle(calib_dataset)
    trainenc = tokenizer('\n\n'.join(calib_dataset), return_tensors='pt')
    samples = []
    for _ in range(n_samples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seq_len - 1)
        j = i + seq_len
        inp = trainenc.input_ids[:, i:j]
        samples.append(inp)
    return samples


@PREPROC_REGISTRY
def original_txt(calib_dataset, tokenizer, n_samples, seq_len=None):
    random.shuffle(calib_dataset)
    n_samples = min(n_samples, len(calib_dataset))
    samples = []
    for i in range(n_samples):
        trainenc = tokenizer(calib_dataset[i], return_tensors='pt')
        inp = trainenc.input_ids
        samples.append(inp)
    return samples
