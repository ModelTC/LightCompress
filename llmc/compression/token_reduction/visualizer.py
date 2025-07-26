import functools

from llmc.utils.registry_factory import TOKEN_REDUCTION_REGISTRY
from llmc.utils.visualizer import (visualize_grid_to_grid, visualize_heads,
                                   visualize_kept_patches)

from .token_reduction_module import TokenReductionModule
from .utils import prefill_wrapper


@TOKEN_REDUCTION_REGISTRY.register('Visualizer')
class Visualizer(TokenReductionModule):
    def __init__(self, config, model, blocks):
        super().__init__(config, model, blocks)
        self.add_sparse_config()
        self.register_reduction_modules()

    def add_sparse_config(self):
        self.pruning_paras = self.special_config
        self.pruning_paras['attentions'] = []

    def register_reduction_modules(self):

        @prefill_wrapper
        def update_attentions_hook(module, args, kwargs):
            kwargs['output_attentions'] = True
            return args, kwargs

        @prefill_wrapper
        def get_images_hook(module, input_args, pruning_paras):
            pruning_paras['images'] = input_args[0]
            return input_args

        @prefill_wrapper
        def get_attentions_hook(module, inps, layer_outs, pruning_paras):
            pruning_paras['attentions'].append(layer_outs[1])
            return layer_outs

        @prefill_wrapper
        def visualizer_hook(module, inps, layer_outs, pruning_paras):
            attention_maps = pruning_paras['attentions'][0]
            visual_attention_maps = attention_maps[:, :, 35: 35 + 576, 35: 35 + 576]
            image = pruning_paras['images'][0]

            visualize_heads(
                visual_attention_maps[:, :6],
                cols=4,
                save_path=''
            )
            visualize_grid_to_grid(
                visual_attention_maps[0, 4, :, :],
                300,
                image,
                grid_size=24,
                save_path=''
            )
            visualize_kept_patches(
                pruning_paras['images'][0],
                pruning_paras['visual_keep_indexs'],
                save_path='',
            )
            return layer_outs

        self.model.vision_model.register_forward_pre_hook(
            functools.partial(get_images_hook, pruning_paras=self.pruning_paras),
        )

        for idx, blk in enumerate(self.blocks):
            if idx == 5:
                blk.register_forward_pre_hook(update_attentions_hook, with_kwargs=True)
                blk.register_forward_hook(
                    functools.partial(get_attentions_hook, pruning_paras=self.pruning_paras),
                )
            if idx == (len(self.blocks) - 1):
                blk.register_forward_hook(
                    functools.partial(visualizer_hook, pruning_paras=self.pruning_paras),
                )
