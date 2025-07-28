from functools import wraps
from types import MethodType

import torch

from llmc.utils.registry_factory import TOKEN_REDUCTION_REGISTRY

from .token_reduction_module import TokenReductionModule


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
    threshold_terms = round(threshold_ratio * image_feature_length)
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


def divprune_post_hook(*args, pruning_paras=None):
    args = list(args)
    position_ids, attention_mask, inputs_embeds = args[1], args[2], args[4]
    rate = pruning_paras['reduction_ratio']
    SYS_TOKEN_LEN = pruning_paras['vision_token_start_index']
    img_feature_len = pruning_paras['vision_token_length']
    device = inputs_embeds.device
    visual_tokens = inputs_embeds[0][SYS_TOKEN_LEN: SYS_TOKEN_LEN + img_feature_len]
    selected_visual_tokens, cosine_matrix = divprune(
        visual_tokens, img_feature_len, None, threshold_ratio=1 - rate
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

    if position_ids is not None:
        args[1] = position_ids[:, keep_indexs, :]
    if attention_mask is not None:
        args[2] = attention_mask[:, keep_indexs]
    args[4] = inputs_embeds[:, keep_indexs]

    return tuple(args)


@TOKEN_REDUCTION_REGISTRY.register('DivPrune')
class DivPrune(TokenReductionModule):
    def __init__(self, config, model, blocks):
        super().__init__(config, model, blocks)
        self.add_sparse_config()
        self.register_reduction_modules()

    def add_sparse_config(self):
        self.pruning_paras = self.special_config

    def register_reduction_modules(self):

        def input_hook_llava(fn, pruning_paras, llava_next):
            @wraps(fn)
            def wrapper(self, *args, **kwargs):
                if args[0].shape[1] == 1:
                    return fn(*args, **kwargs)
                outs = fn(*args, **kwargs)

                if llava_next:
                    message = (
                        'To obtain the vision_token_length for LLaVA-1.6, you should append '
                        '`image_features[0].shape[0]` to the return value of the function '
                        '`prepare_inputs_labels_for_multimodal`, and modify the related code.'
                    )
                    assert len(outs) == 7, message
                    pruning_paras['vision_token_length'] = outs[-1]
                return divprune_post_hook(*outs, pruning_paras=pruning_paras)
            return wrapper

        if self.model.__class__.__name__ == 'Llava':

            self.model.vlm_model.prepare_inputs_labels_for_multimodal = MethodType(
                input_hook_llava(
                    self.model.vlm_model.prepare_inputs_labels_for_multimodal,
                    self.pruning_paras,
                    llava_next=self.special_config['vision_token_length'] is None
                ), self.model.vlm_model
            )
