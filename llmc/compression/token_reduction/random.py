import functools
from types import MethodType

import torch

from llmc.utils.registry_factory import TOKEN_REDUCTION_REGISTRY

from .token_reduction_module import TokenReductionModule
from .utils import prefill_wrapper


@TOKEN_REDUCTION_REGISTRY.register('RandomPrune')
class RandomPrune(TokenReductionModule):
    def __init__(self, config, model, blocks):
        super().__init__(config, model, blocks)
        self.add_sparse_config()
        self.register_reduction_modules()

    def add_sparse_config(self):

        self.pruning_loc = self.special_config['pruning_loc']
        self.pruning_paras = self.special_config

    def register_reduction_modules(self):

        @prefill_wrapper
        def random_pruning_hook(module, args, kwargs, pruning_paras):

            rate = pruning_paras['prune_ratio']
            image_token_start_index = pruning_paras['vision_token_start_index']
            image_token_length = pruning_paras['vision_token_length']

            hidden_states = args[0]
            causal_mask = kwargs['attention_mask']

            device = hidden_states.device
            vision_indexes = torch.arange(
                image_token_start_index,
                image_token_start_index + image_token_length,
                device=device,
            )
            if self.model.first_turn_question:
                num_keep = round(image_token_length * (1 - rate))
                rand_idx = torch.randperm(image_token_length, device=device)[:num_keep]
                vision_indexes = vision_indexes[rand_idx]

                # save rand_idx to module
                module.register_buffer('rand_idx', rand_idx)
            else:
                # load vision_indexes from module (prompt cache)
                rand_idx = module.rand_idx
                vision_indexes = vision_indexes[rand_idx]

            # keep index
            keep_indexs = torch.cat(
                (
                    torch.arange(image_token_start_index, device=device),
                    vision_indexes,
                    torch.arange(
                        image_token_start_index + image_token_length,
                        hidden_states.shape[1],
                        device=device,
                    ),
                )
            )

            keep_indexs = keep_indexs.sort().values
            # filter hidden states &
            hidden_states = hidden_states[:, keep_indexs, :]
            # update position ids
            position_ids = keep_indexs.unsqueeze(0)
            # update attention mask
            if causal_mask is not None:
                causal_mask = causal_mask[
                    :, :, : hidden_states.shape[1], : hidden_states.shape[1]
                ]
                kwargs['attention_mask'].resize_as_(causal_mask).copy_(
                    causal_mask.clone()
                )
            kwargs['cache_position'].resize_as_(position_ids.squeeze(0)).copy_(
                position_ids.squeeze(0).clone()
            )
            kwargs['position_ids'].resize_as_(position_ids).copy_(position_ids.clone())

            position_embeddings = kwargs['position_embeddings']
            new_pe0 = position_embeddings[0][:, keep_indexs, :].clone()
            new_pe1 = position_embeddings[1][:, keep_indexs, :].clone()
            position_embeddings[0].resize_as_(new_pe0).copy_(new_pe0)
            position_embeddings[1].resize_as_(new_pe0).copy_(new_pe1)

            return (hidden_states,), kwargs

        @prefill_wrapper
        def holitom_merge_hook(module, args, kwargs, pruning_paras):

            rate = pruning_paras['prune_ratio']
            image_token_start_index = pruning_paras['vision_token_start_index']
            image_token_length = pruning_paras['vision_token_length']

            hidden_states = args[0]
            causal_mask = kwargs['attention_mask']

            device = hidden_states.device
            last_layer_attention = pruning_paras['attn_scores']
            # compute average attention over different head
            last_layer_attention_avg = torch.mean(
                last_layer_attention, dim=1
            )[0]
            # generate new attention mask based on the average attention,
            # sample the top ATTENTION_RANK tokens with highest attention
            last_layer_attention_avg_last_tok = (
                last_layer_attention_avg[-1]
            )
            # get the attention in image token
            last_layer_attention_avg_last_tok_image = \
                last_layer_attention_avg_last_tok[
                    image_token_start_index:
                    image_token_start_index + image_token_length
                ]
            # get the indexes of the top ATTENTION_RANK tokens
            top_attention_rank_index = (
                last_layer_attention_avg_last_tok_image.topk(
                    round(
                        image_token_length * (1 - rate)
                    )
                ).indices
                + image_token_start_index
            )

            all_indices = torch.arange(
                image_token_length, device=device
            )
            non_topk_mask = ~torch.isin(
                all_indices,
                top_attention_rank_index
                - image_token_start_index,
            )
            non_topk_indices = (
                all_indices[non_topk_mask]
                + image_token_start_index
            )
            non_topk_states = hidden_states[
                :, non_topk_indices, :
            ]  # [batch_size, len(non_topk), hidden_size]
            topk_states = hidden_states[
                :, top_attention_rank_index, :
            ]  # [batch_size, len(topk), hidden_size]
            non_topk_norm = torch.norm(
                non_topk_states, dim=-1, keepdim=True
            )  # [batch_size, len(non_topk), 1]
            topk_norm = torch.norm(
                topk_states, dim=-1, keepdim=True
            )  # [batch_size, len(topk), 1]
            dot_product = torch.bmm(
                non_topk_states, topk_states.transpose(1, 2)
            )  # [batch_size, len(non_topk), len(topk)]
            sim_matrix = dot_product / (
                non_topk_norm * topk_norm.transpose(1, 2)
            )
            sim_max, sim_max_index = torch.max(sim_matrix, dim=-1)

            batch_size = hidden_states.size(0)
            num_topk = len(top_attention_rank_index)
            num_non_topk = len(non_topk_indices)
            topk_counter = torch.ones((batch_size, num_topk, 1), device=hidden_states.device)

            for b in range(batch_size):
                for i in range(num_non_topk):
                    topk_rel_idx = sim_max_index[b, i].item()  # 这是 topk 中的相对索引
                    topk_abs_idx = top_attention_rank_index[topk_rel_idx]  # 得到绝对索引
                    non_topk_abs_idx = non_topk_indices[i]

                    # 累加non-topk到topk token上（就地）
                    hidden_states[b, topk_abs_idx, :] += hidden_states[b, non_topk_abs_idx, :]
                    # 增加计数
                    topk_counter[b, topk_rel_idx] += 1

            # 平均化所有topk token（包含自己和所有被合并的）
            for b in range(batch_size):
                for i in range(num_topk):
                    topk_abs_idx = top_attention_rank_index[i]
                    hidden_states[b, topk_abs_idx, :] /= topk_counter[b, i]

            keep_indexs = torch.cat(
                (
                    torch.arange(
                        image_token_start_index,
                        device=device,
                    ),
                    top_attention_rank_index,
                    torch.arange(
                        image_token_start_index
                        + image_token_length,
                        hidden_states.shape[1],
                        device=device,
                    ),
                )
            )

            # sort index
            keep_indexs = keep_indexs.sort().values
            # filter hidden states &
            hidden_states = hidden_states[:, keep_indexs, :]
            # update position ids
            position_ids = keep_indexs.unsqueeze(0)
            # update attention mask
            if causal_mask is not None:
                causal_mask = causal_mask[:, :, :hidden_states.shape[1], :hidden_states.shape[1]]
                kwargs['attention_mask'].resize_as_(causal_mask).copy_(causal_mask.clone())
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

        def update_output_attentions_hook(module, args, kwargs):
            kwargs['output_attentions'] = True
            return args, kwargs

        def store_attention_hook(m, x, layer_outputs, pruning_paras):
            layer_attention = layer_outputs[1]
            pruning_paras['attn_scores'] = layer_attention

        if self.special_config['vision_token_length'] is None:
            if self.model.__class__.__name__ == 'Llava':
                self.model.vlm_model.prepare_inputs_labels_for_multimodal = MethodType(
                    self.vtoken_length_for_llava_hook(
                        self.model.vlm_model.prepare_inputs_labels_for_multimodal,
                        self.pruning_paras
                    ), self.model.vlm_model
                )

        if self.special_config['metric'] == 'random':
            self.blocks[self.pruning_loc].register_forward_pre_hook(
                functools.partial(random_pruning_hook, pruning_paras=self.pruning_paras),
                with_kwargs=True
            )
        elif self.special_config['metric'] == 'holitom_merge':
            self.blocks[self.pruning_loc - 1].register_forward_pre_hook(
                update_output_attentions_hook,
                with_kwargs=True
            )
            self.blocks[self.pruning_loc - 1].register_forward_hook(
                functools.partial(store_attention_hook, pruning_paras=self.pruning_paras),
            )
            self.blocks[self.pruning_loc].register_forward_pre_hook(
                functools.partial(holitom_merge_hook, pruning_paras=self.pruning_paras),
                with_kwargs=True
            )
