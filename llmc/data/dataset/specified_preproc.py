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
def pile_gptq(calib_dataset, tokenizer, n_samples, seq_len):
    trainenc = tokenizer('\n\n'.join(calib_dataset['text'][:1000]), return_tensors='pt')
    samples = []
    for _ in range(n_samples):
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
def img_general(calib_dataset, tokenizer, batch_process, n_samples):
    random.shuffle(calib_dataset)
    if len(calib_dataset) > n_samples:
        calib_dataset = calib_dataset[:n_samples]
    samples = batch_process(calib_dataset)
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
def ultrachat_general(calib_dataset, tokenizer, n_samples, seq_len):
    calib_dataset = calib_dataset.shuffle(seed=42).select(range(n_samples))
    texts = []
    samples = []
    for example in calib_dataset:
        text = tokenizer.apply_chat_template(
            example['messages'],
            tokenize=False,
        )
        texts.append(text)

    for i in range(n_samples):
        trainenc = tokenizer(
            texts[i],
            padding=False,
            max_length=seq_len,
            truncation=True,
            add_special_tokens=False,
            return_tensors='pt'
        )
        inp = trainenc.input_ids
        samples.append(inp)
    return samples


@PREPROC_REGISTRY
def txt_general_preproc(calib_dataset, tokenizer, n_samples, seq_len, key):
    dataset = calib_dataset.shuffle(seed=42)
    samples = []
    n_run = 0
    for data in dataset:
        line = data[key]
        trainenc = tokenizer(
            line, return_tensors='pt', max_length=seq_len, truncation=True
        )
        line_encoded = trainenc.input_ids
        if line_encoded.shape[1] < seq_len:
            continue
        samples.append(line_encoded)
        n_run += 1
        if n_run == n_samples:
            break
    return samples
