# Downloads the specified datasets for ppl evaluation.
# This is particularly useful when running in environments
# where the GPU nodes do not have internet access.
# You can pre-download them and set the local path in config yml file.

import argparse
import os

from datasets import load_dataset
from loguru import logger


def download(calib_dataset_name, path):
    if 'c4' in calib_dataset_name:
        calib_dataset = load_dataset(
            'allenai/c4',
            data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'},
            split='validation',
        )
        save_path = os.path.join(path, 'c4')
        calib_dataset.save_to_disk(save_path)
        logger.info('download c4 for eval finished.')
    if 'wikitext2' in calib_dataset_name:
        calib_dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        save_path = os.path.join(path, 'wikitext2')
        calib_dataset.save_to_disk(save_path)
        logger.info('download wikitext2 for eval finished.')
    if 'ptb' in calib_dataset_name:
        calib_dataset = load_dataset(
            'ptb_text_only', 'penn_treebank', split='test', trust_remote_code=True
        )
        save_path = os.path.join(path, 'ptb')
        calib_dataset.save_to_disk(save_path)
        logger.info('download ptb for eval finished.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_name', type=str, default=['c4', 'wikitext2', 'ptb'], nargs='*'
    )
    parser.add_argument('--save_path', type=str, required=True)
    args = parser.parse_args()
    logger.info(f'args : {args}')
    download(args.dataset_name, args.save_path)
