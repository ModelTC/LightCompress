import torch
from loguru import logger

from llmc.utils.registry_factory import ALGO_REGISTRY

from .base_blockwise_quantization import BaseBlockwiseQuantization
from .module_utils import FakeQuantLinear


@ALGO_REGISTRY
class LlmInt8(BaseBlockwiseQuantization):
    def __init__(self, model, quant_config, input, padding_mask, config):
        super().__init__(model, quant_config, input, padding_mask, config)
        self.add_quant_config()

    @torch.no_grad()
    def add_quant_config(self):
        self.threshold = self.quant_config['special']['threshold']

    @torch.no_grad()
    def block_opt(self, *opt_kwargs):
        pass

    @torch.no_grad()
    def get_outlier_indices(self, act):
        tmp = act.abs().amax(dim=1)

        fp_indices = torch.where(tmp >= self.threshold)[1]
        all_idx = torch.arange(act.shape[2]).to(act.device)

        tensor_is_not_in = torch.isin(all_idx, fp_indices, invert=True)
        int_indices = all_idx[tensor_is_not_in]

        return int_indices, fp_indices

    @torch.no_grad()
    def w_qdq(self, module, wquantizer):
        weight = module.weight
        args = {}
        args['int_indices'] = module.buf_int_ids
        args['fp_indices'] = module.buf_fp_ids

        weight = self.wquantizer.fake_quant_weight_dynamic(weight, args)

        return weight

    @torch.no_grad()
    def a_qdq(self, act, module, aquantizer):
        args = {}

        int_indices, fp_indices = self.get_outlier_indices(act)

        args['int_indices'] = int_indices
        args['fp_indices'] = fp_indices

        module.register_buffer('buf_int_ids', int_indices)
        module.register_buffer('buf_fp_ids', fp_indices)

        act = self.aquantizer.fake_quant_act_dynamic(act, args)

        return act

    @torch.no_grad()
    def deploy(self, quant_format):
        assert not quant_format != 'fake_quant'
        logger.info(f'-- deploy_{quant_format}_model start --')
        logger.info(f'quant_config : {self.quant_config}')

        self.model.replace_module_all(
            FakeQuantLinear,
            self.get_replacement_params(
                mode='fake_quant', w_only=self.w_only, name=None
            ),
        )
        logger.info(f'-- deploy_{quant_format}_model done --')
