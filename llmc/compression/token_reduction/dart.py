import functools
from types import MethodType

import torch

from llmc.utils.registry_factory import TOKEN_REDUCTION_REGISTRY

from .token_reduction_module import TokenReductionModule
from .utils import prefill_wrapper


@TOKEN_REDUCTION_REGISTRY.register('DART')
class DART(TokenReductionModule):
    def __init__(self, config, model, blocks):
        super().__init__(config, model, blocks)
        self.add_sparse_config()
        self.register_reduction_modules()

    def add_sparse_config(self):
        self.pruning_loc = self.special_config['pruning_loc']

        self.pruning_paras = self.special_config

    def register_reduction_modules(self):

        @prefill_wrapper
        def vtoken_length_hook(module, args, pruning_paras):
            input_ids = args[0]
            token_indices = torch.where(
                input_ids[0] == pruning_paras['vision_token_index']
            )[0]
            pruning_paras['vision_token_length'] = token_indices.shape[0]

        @prefill_wrapper
        def get_any_states_hook(module, args, kwargs, layer_outs, pruning_paras, layer_idx):
            past_key_value = kwargs['past_key_value']
            if past_key_value is None:
                raise ValueError('DART needs past_key_value but got None.')
            pruning_paras['any_states'] = past_key_value.key_cache[layer_idx]

        @prefill_wrapper
        def pruning_hook(module, args, kwargs, pruning_paras, normlayer):

            image_token_start_index = pruning_paras['vision_token_start_index']
            image_token_length = pruning_paras['vision_token_length']
            any_states = pruning_paras['any_states']

            hidden_states = args[0]
            attention_mask = kwargs['attention_mask']
            seq_length = hidden_states.shape[1]
            device = hidden_states.device
            last_layer_state = normlayer(hidden_states)

            # keep index
            retained_image_tokens_index = get_retained_image_token(
                pruning_paras, last_layer_state, any_states)

            keep_indexs = torch.cat(
                (
                    torch.arange(image_token_start_index, device=device),
                    retained_image_tokens_index,
                    torch.arange(
                        image_token_start_index + image_token_length,
                        seq_length,
                        device=device
                    )
                )
            )
            # sort index
            keep_indexs = keep_indexs.sort().values
            hidden_states = hidden_states[:, keep_indexs, :]
            position_ids = keep_indexs.unsqueeze(0)
            if attention_mask is not None:
                attention_mask = attention_mask[
                    :, :, :hidden_states.shape[1], :hidden_states.shape[1]
                ]
                kwargs['attention_mask'].resize_as_(attention_mask).copy_(attention_mask.clone())
            kwargs['cache_position'].resize_as_(position_ids.squeeze(0)).copy_(
                position_ids.squeeze(0).clone())
            kwargs['position_ids'].resize_as_(position_ids).copy_(position_ids.clone())

            position_embeddings = kwargs['position_embeddings']
            index_dim = 1 if position_embeddings[0].dim() == 3 else 2
            new_pe0 = position_embeddings[0].index_select(index_dim, keep_indexs).clone()
            new_pe1 = position_embeddings[1].index_select(index_dim, keep_indexs).clone()
            position_embeddings[0].resize_as_(new_pe0).copy_(new_pe0)
            position_embeddings[1].resize_as_(new_pe0).copy_(new_pe1)

            return (hidden_states,), kwargs

        if self.special_config['vision_token_length'] is None:
            if self.model.__class__.__name__ == 'Llava':
                self.model.vlm_model.prepare_inputs_labels_for_multimodal = MethodType(
                    self.vtoken_length_for_llava_hook(
                        self.model.vlm_model.prepare_inputs_labels_for_multimodal,
                        self.pruning_paras
                    ), self.model.vlm_model
                )
            else:
                self.model.embed_tokens.register_forward_pre_hook(
                    functools.partial(vtoken_length_hook, pruning_paras=self.pruning_paras)
                )

        self.blocks[self.pruning_loc - 1].register_forward_hook(
            functools.partial(
                get_any_states_hook,
                pruning_paras=self.pruning_paras,
                layer_idx=self.pruning_loc - 1
            ),
            with_kwargs=True
        )

        self.blocks[self.pruning_loc].register_forward_pre_hook(
            functools.partial(
                pruning_hook,
                pruning_paras=self.pruning_paras,
                normlayer=self.model.language_model.norm
            ),
            with_kwargs=True
        )


def get_retained_image_token(pruning_paras, last_layer_state, any_states):
    image_token_start_index = pruning_paras['vision_token_start_index']
    image_token_length = pruning_paras['vision_token_length']
    pivot_image_token = pruning_paras['pivot_image_token']
    pivot_text_token = pruning_paras['pivot_text_token']
    reduction_ratio = pruning_paras['reduction_ratio']
    TOKEN_TOPK = int(
        image_token_length * (1 - reduction_ratio) / (pivot_image_token + pivot_text_token)
    )
    device = last_layer_state.device

    any_states = any_states.permute(0, 2, 1, 3)
    any_states = any_states.reshape(any_states.shape[0], any_states.shape[1], -1)

    k_states_image_token = any_states[0][
        image_token_start_index:image_token_start_index + image_token_length, :
    ]
    k_states_query_token = any_states[0][image_token_start_index + image_token_length:, :]

    k_states_image_token_L1_norm = torch.norm(k_states_image_token, p=1, dim=-1)
    k_states_query_token_L1_norm = torch.norm(k_states_query_token, p=1, dim=-1)

    image_indices = (
        k_states_image_token_L1_norm.topk(pivot_image_token).indices
        + image_token_start_index
    ).tolist()
    query_indices = (
        k_states_query_token_L1_norm.topk(pivot_text_token).indices
        + image_token_start_index + image_token_length
    ).tolist()
    indices_set = set(image_indices + query_indices)

    valid_indices = set(
        range(image_token_start_index, image_token_start_index + image_token_length)
    ) - set(image_indices)

    valid_indices_list = list(valid_indices)
    for item in list(indices_set):
        valid_vectors = last_layer_state[0][valid_indices_list, :]
        cos_sim = -torch.nn.functional.cosine_similarity(
            last_layer_state[0][item, :],
            valid_vectors,
            dim=-1
        )
        top_k_indices = cos_sim.topk(TOKEN_TOPK).indices

        top_k_real_indices = [valid_indices_list[i] for i in top_k_indices]
        indices_set.update(top_k_real_indices)

        valid_indices.difference_update(top_k_real_indices)
        valid_indices_list = list(valid_indices)

    indices_set.difference_update(query_indices)

    retained_image_tokens_index = torch.tensor(list(indices_set), device=device)

    return retained_image_tokens_index
