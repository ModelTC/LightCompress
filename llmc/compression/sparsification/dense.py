from loguru import logger

from llmc.utils.registry_factory import ALGO_REGISTRY

from .base_blockwise_sparsification import BaseBlockwiseSparsification


@ALGO_REGISTRY
class Dense(BaseBlockwiseSparsification):
    def __init__(self, model, sparsity_config, input, padding_mask, config):
        super().__init__(model, sparsity_config, input, padding_mask, config)

    def block_transform(self, block):
        logger.info(f'Start transform the {self.block_idx + 1}-th block')
        logger.info(block)
        logger.info(f'End transform the {self.block_idx + 1}-th block')
