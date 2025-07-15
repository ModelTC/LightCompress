import functools
from functools import wraps
from types import MethodType

import torch

from llmc.utils.registry_factory import TOKEN_REDUCTION_REGISTRY

from .token_reduction_module import TokenReductionModule
from .utils import prefill_wrapper


def pairwise_cosine_similarity(matrix):
    norm_matrix = matrix / matrix.norm(dim=1, keepdim=True)
    cosine_similarity = torch.mm(norm_matrix, norm_matrix.t())
    return cosine_similarity


def divprune(
    visual_feature_vectors,
    image_feature_length,
    cosine_matrix=None,
    threshold_ratio=0.1,
):
    threshold_terms = int(round(threshold_ratio * image_feature_length))
    if cosine_matrix is None:
        cosine_matrix = 1.0 - (pairwise_cosine_similarity(visual_feature_vectors))

    s = torch.empty(
        threshold_terms, dtype=torch.long, device=visual_feature_vectors.device
    )
    for i in range(threshold_terms):
        if i == 0:
            m2 = cosine_matrix
        else:
            m2 = torch.index_select(
                cosine_matrix,
                0,
                torch.index_select(
                    s, 0, torch.arange(0, i, device=cosine_matrix.device)
                ),
            )

        if i == 0:
            scores = torch.topk(m2, 2, dim=0, largest=False).values[
                1, :
            ]  # for distance
        else:
            scores = torch.min(m2, dim=0).values  # for distance

        phrase_to_add_idx = torch.argmax(scores)
        s[i] = phrase_to_add_idx
    return s, cosine_matrix


def divprune_post_hook(
    input_ids,
    position_ids,
    attention_mask,
    past_key_values,
    inputs_embeds,
    labels,
    pruning_paras=None,
):
    rate = pruning_paras['rate']
    SYS_TOKEN_LEN = pruning_paras['image_token_start_index']
    img_feature_len = pruning_paras['image_token_length']
    device = inputs_embeds.device
    visual_tokens = inputs_embeds[0][SYS_TOKEN_LEN: SYS_TOKEN_LEN + img_feature_len]
    selected_visual_tokens, cosine_matrix = divprune(
        visual_tokens, img_feature_len, None, threshold_ratio=rate
    )

    selected_visual_tokens += SYS_TOKEN_LEN
    keep_indexs = torch.cat(
        (
            torch.arange(SYS_TOKEN_LEN, device=device),
            selected_visual_tokens,
            torch.arange(
                SYS_TOKEN_LEN + img_feature_len, inputs_embeds.shape[1], device=device
            ),
        )
    )
    keep_indexs = keep_indexs.sort().values

    inputs_embeds = inputs_embeds[:, keep_indexs]
    if position_ids is not None:
        position_ids = position_ids[:, keep_indexs, :]
    if attention_mask is not None:
        attention_mask = attention_mask[:, keep_indexs]

    return (
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        inputs_embeds,
        labels,
    )


@TOKEN_REDUCTION_REGISTRY.register('DivPrune')
class DivPrune(TokenReductionModule):
    def __init__(self, config, model, blocks):
        super().__init__(config, model, blocks)
        self.add_sparse_config()
        self.register_reduction_modules()

    def add_sparse_config(self):
        self.special_config['image_token_length'] = self.model.pruning_config[
            'image_token_length'
        ]

        self.pruning_paras = self.special_config

    def register_reduction_modules(self):

        def input_hook_llava(fn, pruning_paras):
            @wraps(fn)
            def wrapper(self, *args, **kwargs):
                if len(args) == 0:
                    return fn(*args, **kwargs)
                input_args = args[0]
                if hasattr(input_args[0], 'shape') and input_args[0].shape[0] == 1:
                    return fn(*args, **kwargs)

                input_ids = args[0]
                attention_mask = args[2]
                token_indices = input_ids[0][attention_mask[0]] == IMAGE_TOKEN_INDEX
                pruning_paras['image_token_start_index'] = torch.where(token_indices)[
                    0
                ][0].item()

                outputs = fn(*args, **kwargs)

                return divprune_post_hook(*outputs, pruning_paras=pruning_paras)

            return wrapper

        if self.model.__class__.__name__ == 'Llava':
            from llava.constants import IMAGE_TOKEN_INDEX

            hook_fn = input_hook_llava(
                self.model.vlm_model.prepare_inputs_labels_for_multimodal,
                self.pruning_paras,
            )
            self.model.vlm_model.prepare_inputs_labels_for_multimodal = MethodType(
                hook_fn, self.model.vlm_model
            )
