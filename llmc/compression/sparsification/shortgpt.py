import gc
import json
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from transformers.models.mistral.modeling_mistral import MistralRMSNorm

from llmc.utils import copy_files
from llmc.utils.registry_factory import ALGO_REGISTRY

from .base_blockwise_sparsification import BaseBlockwiseSparsification


@ALGO_REGISTRY
class ShortGPT(BaseBlockwiseSparsification):
    def __init__(self, model, sparsity_config, input, padding_mask, config):
        super().__init__(model, sparsity_config, input, padding_mask, config)
        self.importances = np.zeros(len(self.blocks))

    def block_opt(self, block):
        block = block.cuda()

        output_feat = self.block_forward(block)
        torch.cuda.empty_cache()
        self.block_transform(self.input['data'], output_feat)
        self.input['data'] = output_feat

    def block_transform(self, input_feat, output_feat):
        logger.info(f'Start transform the {self.block_idx + 1}-th block')
        self.subset_transform(
            input_feat,
            output_feat
        )

    @torch.no_grad()
    def compute_bi(
            self,
            input_feat: torch.Tensor,
            output_feat: torch.Tensor
    ):
        _, _, d = input_feat.shape
        input_feat = input_feat.reshape(-1, d)
        output_feat = output_feat.reshape(-1, d)

        norm_input = input_feat.norm(dim=-1, keepdim=True)
        norm_output = output_feat.norm(dim=-1, keepdim=True)

        sim = (input_feat @ output_feat.T) / (norm_input * norm_output)
        sim = sim.diagonal().nan_to_num(nan=0.5)

        return 1 - sim

    @torch.no_grad()
    def subset_transform(
        self,
        input_feat,
        output_feat
    ):
        # calculate BI score
        self.importances[self.block_idx] = self.compute_bi(
            input_feat[0], output_feat[0]
        ).sum().cpu().item()

    @torch.no_grad()
    def remove_layers(
        self,
        layers_to_remove: Optional[List[int]] = []
    ):
        if not layers_to_remove and self.n_prune_layers:
            layers_to_remove = np.argsort(
                np.array(self.importances)
            )[:self.n_prune_layers].tolist()

        for idx in sorted(layers_to_remove, reverse=True):
            try:
                del self.blocks[idx]
            except IndexError:
                logger.info(f'layer {idx} does not exist')
        return layers_to_remove

    @torch.no_grad()
    def deploy(self, deploy_format):
        logger.info(f'After compute, BI scores are {self.importances}')
        logger.info('-- deploy_sparsity_model start --')
        logger.info(f'sparsity_config : {self.sparsity_config}')
        logger.info('-- begin remove layers --')
        layers_to_remove = self.remove_layers()
        logger.info(f'remove layers: {layers_to_remove}')
        logger.info('-- deploy_sparsity_model done --')

    @torch.no_grad()
    def save_model(self, path):
        if self.config.model.type == 'Llava':
            self.model.llava_model.language_model = self.model.get_model()
            self.model.llava_model.save_pretrained(path)
            logger.info('save model done --')
            self.copy_tokenizer(path)
            copy_files(self.config.model.path, path, 'preprocessor_config')
        else:
            self.model.get_model().save_pretrained(path)
            config_file = path + '/config.json'

            logger.info('save model done --')
            self.copy_tokenizer(path)
            with open(config_file, 'r') as file:
                config_new = json.load(file)
            config_new['num_hidden_layers'] = len(self.blocks)
            with open(config_file, 'w') as file:
                json.dump(config_new, file, indent=4)
