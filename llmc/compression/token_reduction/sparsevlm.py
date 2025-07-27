import functools
import math
from functools import wraps
from types import MethodType

import einops as ein
import torch

from llmc.utils.registry_factory import TOKEN_REDUCTION_REGISTRY

from .token_reduction_module import TokenReductionModule
from .utils import prefill_wrapper, prefill_wrapper_model

layer_dict = {}
prune_flag = True
merge_flag = True
sparse_token_list_192 = []
sparse_token_list_128 = []
sparse_token_list_64 = []
sparse_token_list_640 = []
sparse_token_list_320 = []
sparse_token_list_160 = []
sparse_token_dict = {}


@TOKEN_REDUCTION_REGISTRY.register('SparseVLM')
class SparseVLM(TokenReductionModule):
    def __init__(self, config, model, blocks):
        super().__init__(config, model, blocks)
        self.add_sparse_config()
        self.register_reduction_modules()

    def add_sparse_config(self):

        self.pruning_loc = self.special_config.get('pruning_loc', [2, 6, 15])
        global layer_dict, prune_flag, merge_flag
        layer_dict = {layer: idx for idx, layer in enumerate(self.pruning_loc)}
        prune_flag = self.special_config.get('prune_flag', True)
        merge_flag = self.special_config.get('merge_flag', True)
        update_list()
        self.pruning_paras = self.special_config
        self.pruning_paras['pre_prompt_length_list'] = []

    def register_reduction_modules(self):
        @prefill_wrapper
        def input_hook(module, args, pruning_paras):
            input_ids = args[0]
            pre_prompt_length_list = []

            # find the position of the first image token
            for seq in input_ids:
                image_token_index = (
                    seq == pruning_paras['vision_token_index']
                ).nonzero(as_tuple=True)[0]
                if len(image_token_index) > 0:
                    pre_prompt_length_list.append(image_token_index[0].item())
                else:
                    pre_prompt_length_list.append(0)
            pruning_paras['pre_prompt_length_list'] = pre_prompt_length_list

        def input_hook_llava(fn, pruning_paras, llava_next=False):
            @wraps(fn)
            def wrapper(self, *args, **kwargs):
                if args[0].shape[1] == 1:
                    return fn(*args, **kwargs)

                input_ids = args[0]
                attention_mask = args[2]

                if attention_mask is None:
                    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
                else:
                    attention_mask = attention_mask.bool()

                pre_prompt_length_list = []
                for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask):
                    seq = cur_input_ids[cur_attention_mask]
                    image_token_index = (
                        [-1]
                        + torch.where(seq == pruning_paras['vision_token_index'])[0].tolist()
                        + [seq.shape[0]]
                    )
                    pre_prompt_length_list.append(image_token_index[1])

                pruning_paras['pre_prompt_length_list'] = pre_prompt_length_list

                outs = fn(*args, **kwargs)
                if llava_next:
                    pruning_paras['vision_token_length'] = outs[-1]
                return outs
            return wrapper

        @prefill_wrapper_model
        def register_module_paras(module, args, kwargs, pruning_paras):
            pre_prompt_length_list = pruning_paras['pre_prompt_length_list']
            hidden_states = kwargs['inputs_embeds']
            if hidden_states is None:
                hidden_states = module.embed_tokens(kwargs['input_ids'])

            B, L, _ = hidden_states.shape
            pruning_paras['B'] = B

            v_token_start = pre_prompt_length_list[0] if len(
                pre_prompt_length_list) != 0 else 0
            text_token_start = v_token_start + pruning_paras['vision_token_length']
            pruning_paras['v_token_start'] = v_token_start  # 35
            pruning_paras['text_token_start'] = text_token_start  # 611
            pruning_paras['v_token_num'] = pruning_paras['vision_token_length']  # 576
            pruning_paras['retained_tokens'] = round(
                pruning_paras['vision_token_length'] * (1 - pruning_paras['reduction_ratio'])
            )

            if (len(pre_prompt_length_list) != 0 and hidden_states.shape[1] != 1):
                v_t = hidden_states[:, v_token_start: text_token_start, :]
                t_t = hidden_states[:, text_token_start:, :]
                m_v_t = v_t @ t_t.transpose(1, 2)
                m_v_t = m_v_t.softmax(2).mean(1)
                pruning_paras['t_token_idx'] = torch.where(m_v_t > m_v_t.mean())

            return args, kwargs

        def update_output_attentions_hook(module, args, kwargs, pruning_paras, layer_idx):
            kwargs['output_attentions'] = True
            if layer_idx != self.pruning_loc[0]:
                kwargs['position_ids'] = pruning_paras['position_ids']
                kwargs['attention_mask'] = pruning_paras['attention_mask']
                kwargs['cache_position'] = pruning_paras['cache_position']
                kwargs['position_embeddings'] = pruning_paras['position_embeddings']
            return args, kwargs

        def update_kwargs_hook(module, args, kwargs, pruning_paras, layer_idx):

            if len(kwargs['position_ids'][0]) == 1:
                return args, kwargs
            if layer_idx != self.pruning_loc[0]:
                kwargs['position_ids'] = pruning_paras['position_ids']
                kwargs['attention_mask'] = pruning_paras['attention_mask']
                kwargs['cache_position'] = pruning_paras['cache_position']
                kwargs['position_embeddings'] = pruning_paras['position_embeddings']
            else:
                pruning_paras['position_ids'] = kwargs['position_ids']
                pruning_paras['attention_mask'] = kwargs['attention_mask']
                pruning_paras['cache_position'] = kwargs['cache_position']
                pruning_paras['position_embeddings'] = kwargs['position_embeddings']
            return args, kwargs

        def get_attn_logits_hook(module, args, kwargs, layer_outs, pruning_paras, layer_idx):

            if len(kwargs['position_ids'][0]) == 1:
                return layer_outs

            from transformers.models.llama.modeling_llama import \
                apply_rotary_pos_emb

            hidden_states = kwargs['hidden_states']
            position_embeddings = kwargs['position_embeddings']
            position_ids = kwargs['position_ids']
            past_key_value = layer_outs[2]
            attention_mask = kwargs['attention_mask']

            t_token_idx = pruning_paras['t_token_idx']
            v_token_start = pruning_paras['v_token_start']
            v_token_num = pruning_paras['v_token_num']

            bsz, q_len, _ = hidden_states.size()
            query_states = module.q_proj(hidden_states)
            key_states = module.k_proj(hidden_states)
            value_states = module.v_proj(hidden_states)
            query_states = query_states.view(
                bsz, q_len, module.num_heads, module.head_dim
            ).transpose(1, 2)
            key_states = key_states.view(
                bsz, q_len, module.num_key_value_heads, module.head_dim
            ).transpose(1, 2)
            value_states = value_states.view(
                bsz, q_len, module.num_key_value_heads, module.head_dim
            ).transpose(1, 2)

            if position_embeddings is None:
                cos, sin = module.rotary_emb(value_states, position_ids)
            else:
                cos, sin = position_embeddings

            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
            if past_key_value is not None:
                key_states = past_key_value.key_cache[layer_idx]
                value_states = past_key_value.value_cache[layer_idx]
            t_token_idx = t_token_idx[1] + v_token_start + v_token_num
            L, S = query_states.size(-2), key_states.size(-2)
            scale_factor = 1 / math.sqrt(query_states.size(-1))
            attn_bias = torch.zeros(L, S, dtype=query_states.dtype)
            if module.is_causal:
                assert attention_mask is None
                temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
                attn_bias.masked_fill_(temp_mask.logical_not(), float('-inf'))
                attn_bias.to(query_states.dtype)

            attn_logits = query_states @ key_states.transpose(2, 3) * scale_factor
            attn_logits += attn_bias.to(query_states.device)
            attn_logits = torch.softmax(attn_logits, dim=-1)

            pruning_paras['attn_logits'] = attn_logits

            return layer_outs

        @prefill_wrapper
        def decoder_attn_hook(module, inputs, kwargs, layer_outputs, pruning_paras, layer_idx):

            if 'attn_logits' not in pruning_paras:
                attn_logits = layer_outputs[1]  # for LlavaHf, but error
            else:
                attn_logits = pruning_paras['attn_logits']
            prune_flag = pruning_paras.get('prune_flag', True)
            merge_flag = pruning_paras['merge_flag']
            v_token_start = pruning_paras['v_token_start']
            v_token_num = pruning_paras['v_token_num']
            text_token_start = pruning_paras['text_token_start']
            t_token_idx = pruning_paras['t_token_idx']
            retained_tokens = pruning_paras['retained_tokens']

            B = pruning_paras['B']
            pre_prompt_length_list = pruning_paras['pre_prompt_length_list']
            vision_token_length = pruning_paras['vision_token_length']

            attention_mask = kwargs['attention_mask']
            position_embeddings = kwargs['position_embeddings']

            hidden_states = inputs[0]  # [B, L, D]
            pred_score_vis, s_flag, relation_vis_text = attn_postprocess_topk(
                attn_logits,
                v_token_start,
                v_token_num,
                text_token_start,
                t_token_idx,
                layer_idx,
                retained_tokens,
                pruning_paras['reduction_ratio']
            )
            if not prune_flag:
                pred_score_vis = torch.zeros_like(relation_vis_text, dtype=bool)
            policy = torch.ones(B, hidden_states.shape[1], dtype=hidden_states.dtype,
                                device=hidden_states.device)
            policy[:, v_token_start:text_token_start] = \
                pred_score_vis.type(dtype=hidden_states.dtype)

            for batch in range(len(pre_prompt_length_list)):
                # keep pre prompt
                prompt_length = pre_prompt_length_list[batch]
                policy[batch, :prompt_length] = 1
                # keep question
                text_token_start = prompt_length + vision_token_length
                policy[batch, text_token_start:] = 1

            if self.model.first_turn_question:
                vision_mask = policy[:, v_token_start:v_token_start + v_token_num]
                module.register_buffer('vision_mask', vision_mask)
            else:
                vision_mask = module.vision_mask
                policy[:, v_token_start:v_token_start + v_token_num] = vision_mask

            total_sparse_token_idx = torch.where(policy == 0)[1].unsqueeze(0)

            # merge and cluster
            if s_flag and merge_flag and total_sparse_token_idx.shape[1] > 0:
                total_sparse_token = batch_index_select(
                    layer_outputs[0], total_sparse_token_idx
                )

                merge_token_idx_stage1 = torch.where(pred_score_vis == 0)[1]
                merge_token_stage1 = relation_vis_text[0][merge_token_idx_stage1]
                if prune_flag:
                    merge_token_num_stage1 = int(merge_token_idx_stage1.shape[0] * 0.3) + 1
                else:
                    merge_token_num_stage1 = (
                        merge_token_idx_stage1.shape[0]
                        - sparse_token_dict[retained_tokens][layer_dict[layer_idx]]
                    )
                merge_token_stage2_idx = merge_token_stage1.topk(merge_token_num_stage1)[1]
                if not prune_flag:
                    all_idx = torch.arange(
                        merge_token_stage1.size(0),
                        device=merge_token_stage1.device
                    )
                    non_topk_idx = all_idx[~torch.isin(all_idx, merge_token_stage2_idx)]
                    pred_score_vis[0][non_topk_idx] = 1
                    policy[:, v_token_start:text_token_start] = \
                        pred_score_vis.type(dtype=hidden_states.dtype)

                merge_token_stage2 = total_sparse_token[:, merge_token_stage2_idx, :]
                cluster_num = int(merge_token_stage2.shape[1] / 10) + 1
                if cluster_num == 0:
                    cluster_num = merge_token_stage2.shape[1]
                merge_sparse_token, index_down = cluster_and_merge(merge_token_stage2, cluster_num)

                cluster_idx = total_sparse_token_idx.squeeze(0)[merge_token_stage2_idx[index_down]]
                cluster_idx = cluster_idx.squeeze(0)
                select_token_idx = torch.where(policy == 1)[1].unsqueeze(0)
                select_token = batch_index_select(layer_outputs[0], select_token_idx)
                select_vis_token_num = pred_score_vis.sum()
                keep_indexs = torch.cat(
                    (
                        select_token_idx.squeeze(0)[:v_token_start + select_vis_token_num],
                        cluster_idx,
                        select_token_idx.squeeze(0)[v_token_start + select_vis_token_num:]
                    )
                )
                select_and_merge_token = torch.cat(
                    (
                        select_token[:, :v_token_start + select_vis_token_num, :],
                        merge_sparse_token,
                        select_token[:, v_token_start + select_vis_token_num:, :]
                    ),
                    dim=1
                )
                layer_outputs = (select_and_merge_token, layer_outputs[1])
                v_token_num = pred_score_vis.sum() + cluster_num

            else:
                keep_indexs = torch.where(policy == 1)[1]
                select_token_idx = keep_indexs.unsqueeze(0)
                layer_outputs = (batch_index_select(layer_outputs[0], select_token_idx),
                                 layer_outputs[1])
                v_token_num = pred_score_vis.sum()

            text_token_start = v_token_start + v_token_num
            position_ids = keep_indexs.unsqueeze(0)
            new_output = layer_outputs
            cache_position = position_ids.squeeze(0)

            if attention_mask is not None:
                attention_mask = attention_mask[:, :, keep_indexs, keep_indexs]
            new_pe0 = position_embeddings[0][:, keep_indexs, :].clone()
            new_pe1 = position_embeddings[1][:, keep_indexs, :].clone()
            position_embeddings = (new_pe0, new_pe1)

            pruning_paras['v_token_num'] = v_token_num
            pruning_paras['text_token_start'] = text_token_start

            pruning_paras['position_ids'] = position_ids
            pruning_paras['cache_position'] = cache_position
            pruning_paras['position_embeddings'] = position_embeddings
            pruning_paras['attention_mask'] = attention_mask

            return new_output

        @prefill_wrapper
        def read_parameter_hook(module, args, kwargs, pruning_paras):
            kwargs['position_ids'] = pruning_paras['position_ids']
            kwargs['attention_mask'] = pruning_paras['attention_mask']
            kwargs['cache_position'] = pruning_paras['cache_position']
            kwargs['position_embeddings'] = pruning_paras['position_embeddings']

            return args, kwargs

        if self.model.__class__.__name__ == 'LlavaHf':
            self.model.embed_tokens.register_forward_pre_hook(
                functools.partial(input_hook, pruning_paras=self.pruning_paras)
            )
        elif self.model.__class__.__name__ == 'Llava':
            self.model.vlm_model.prepare_inputs_labels_for_multimodal = MethodType(
                input_hook_llava(
                    self.model.vlm_model.prepare_inputs_labels_for_multimodal,
                    self.pruning_paras,
                    llava_next=self.special_config['vision_token_length'] is None
                ), self.model.vlm_model
            )

        if self.model.__class__.__name__ == 'LlavaHf':
            llama_model = self.model.model
        elif self.model.__class__.__name__ == 'Llava':
            llama_model = self.model.model.model
        llama_model.register_forward_pre_hook(
            functools.partial(register_module_paras, pruning_paras=self.pruning_paras),
            with_kwargs=True
        )

        sorted_pruning_locs = sorted(self.pruning_loc)
        total_layers = len(self.blocks)

        for block_idx in range(sorted_pruning_locs[0], total_layers):
            if block_idx in sorted_pruning_locs:
                if self.model.__class__.__name__ == 'LlavaHf':
                    self.blocks[block_idx].register_forward_pre_hook(
                        functools.partial(
                            update_output_attentions_hook,
                            pruning_paras=self.pruning_paras,
                            layer_idx=block_idx,
                        ),
                        with_kwargs=True
                    )
                elif self.model.__class__.__name__ == 'Llava':
                    self.blocks[block_idx].register_forward_pre_hook(
                        functools.partial(
                            update_kwargs_hook,
                            pruning_paras=self.pruning_paras,
                            layer_idx=block_idx,
                        ),
                        with_kwargs=True
                    )
                    self.blocks[block_idx].self_attn.register_forward_hook(
                        functools.partial(
                            get_attn_logits_hook,
                            pruning_paras=self.pruning_paras,
                            layer_idx=block_idx,
                        ),
                        with_kwargs=True
                    )
                self.blocks[block_idx].register_forward_hook(
                    functools.partial(
                        decoder_attn_hook,
                        pruning_paras=self.pruning_paras,
                        layer_idx=block_idx
                    ),
                    with_kwargs=True
                )
            else:
                self.blocks[block_idx].register_forward_pre_hook(
                    functools.partial(
                        read_parameter_hook,
                        pruning_paras=self.pruning_paras
                    ),
                    with_kwargs=True
                )


def update_list():
    global sparse_token_list_192, sparse_token_list_128, sparse_token_list_64
    global sparse_token_list_640, sparse_token_list_320, sparse_token_list_160
    global prune_flag, merge_flag, sparse_token_dict

    if layer_dict == {2: 0, 6: 1, 15: 2}:  # 2*576  4*300 10*200  16*110
        sparse_token_list_192 = [300, 200, 110]
        sparse_token_list_128 = [303, 110, 36]
        sparse_token_list_64 = [66, 30, 17]
        prune_flag, merge_flag = True, True
    elif prune_flag and merge_flag:
        sparse_token_list_192 = [180]
        sparse_token_list_128 = [114]
        sparse_token_list_64 = [48]
        sparse_token_list_640 = [0.1979]
        sparse_token_list_320 = [0.0833]
        sparse_token_list_160 = [0.0261]
    elif prune_flag:
        sparse_token_list_192 = [192]
        sparse_token_list_128 = [128]
        sparse_token_list_64 = [64]
        sparse_token_list_640 = [0.2222]
        sparse_token_list_320 = [0.1111]
        sparse_token_list_160 = [0.0555]
    elif merge_flag:
        sparse_token_list_192 = [149]
        sparse_token_list_128 = [78]
        sparse_token_list_64 = [7]
    else:
        raise RuntimeError(
            'Both prune_flag and merge_flag are False — sparseVLM is inactive.'
        )

    sparse_token_dict = {
        192: sparse_token_list_192,
        128: sparse_token_list_128,
        64: sparse_token_list_64,
        640: sparse_token_list_640,
        320: sparse_token_list_320,
        160: sparse_token_list_160
    }


def attn_postprocess_topk(
        self_attn_weights,
        v_token_start,
        v_token_num,
        text_token_start,
        t_token_idx,
        layer_idx,
        retained_tokens,
        reduction_ratio):
    '''
    self_attn_weights: [B, H, L, L]
    '''
    self_attn_weights = self_attn_weights.mean(1)  # B, L[Q], L[K]

    t_token_idx = t_token_idx[1] + text_token_start

    relation_vis_text = self_attn_weights[:, t_token_idx,
                                          v_token_start: v_token_start + v_token_num]  # B, L2, L1

    relation_vis_text = relation_vis_text.mean(1)  # B, L1

    relation_vis = relation_vis_text
    s_flag = True       # s_flag controls whether token merge is needed.
    if retained_tokens in [192, 128, 64]:
        sparse_token_list = sparse_token_dict[retained_tokens]
    else:
        sparse_token_list = sparse_token_dict[round((1 - reduction_ratio) * 2880)]
    retained_tokens_prune = sparse_token_list[layer_dict[layer_idx]]
    if retained_tokens_prune < 1:
        retained_tokens_prune = round(retained_tokens_prune * v_token_num)
    if v_token_num != 0:
        mask = torch.zeros_like(relation_vis, dtype=bool)
        _, indices = torch.topk(relation_vis, min(
            retained_tokens_prune, v_token_num - 1), dim=1)
        mask[0][indices] = 1
    else:
        mask = torch.ones_like(relation_vis_text, dtype=bool)
        s_flag = False
    return mask, s_flag, relation_vis_text


def batch_index_select(x, idx):
    if len(x.size()) == 4:
        B, H, N, C = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long,
                              device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B * N, H, C)[idx.reshape(-1)].reshape(B, H, N_new, C)
        return out
    elif len(x.size()) == 3:
        # in this condition
        B, N, C = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long,
                              device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B * N, C)[idx.reshape(-1)].reshape(B, N_new, C)
        return out
    elif len(x.size()) == 2:
        B, N = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long,
                              device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B * N)[idx.reshape(-1)].reshape(B, N_new)
        return out
    else:
        raise NotImplementedError


def index_points(points, idx):
    """Sample features following the index.
    Returns:
        new_points:, indexed points data, [B, S, C]

    Args:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(
        device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def cluster_and_merge(x, cluster_num):

    B, N, C = x.shape

    x1 = ein.rearrange(x, 'b l r -> b l () r')
    x2 = ein.rearrange(x, 'b l r -> b () l r')
    distance = (x1 - x2).norm(dim=-1, p=2)
    dist_matrix = distance / (C ** 0.5)
    # get local density
    dist_nearest, index_nearest = torch.topk(
        dist_matrix, k=cluster_num, dim=-1, largest=False)
    density = (-(dist_nearest ** 2).mean(dim=-1)).exp()
    # add a little noise to ensure no tokens have the same density.
    density = density + torch.rand(
        density.shape, device=density.device, dtype=density.dtype) * 1e-6

    # get distance indicator
    mask = density[:, None, :] > density[:, :, None]
    mask = mask.type(x.dtype)
    dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]
    dist, _ = (dist_matrix * mask +
               dist_max * (1 - mask)).min(dim=-1)

    # select clustering center according to score
    score = dist * density
    _, index_down = torch.topk(score, k=cluster_num, dim=-1)

    # assign tokens to the nearest center
    dist_matrix = index_points(dist_matrix, index_down)

    idx_cluster = dist_matrix.argmin(dim=1)

    # make sure cluster center merge to itself
    idx_batch = torch.arange(B, device=x.device)[
        :, None].expand(B, cluster_num)
    idx_tmp = torch.arange(cluster_num, device=x.device)[
        None, :].expand(B, cluster_num)
    idx_cluster[idx_batch.reshape(-1),
                index_down.reshape(-1)] = idx_tmp.reshape(-1)

    # merge tokens

    B, N, C = x.shape
    # device = dist_matrix.device
    # idx_token = torch.arange(N)[None, :].repeat(B, 1).to(device)
    # agg_weight = x.new_ones(B, N, 1)

    token_weight = x.new_ones(B, N, 1)
    # self_attn_weights = self_attn_weights.mean(1)
    # token_weight = self_attn_weights.sum(dim=1).exp().unsqueeze(2)
    # B_weight,N_weigh,C_weight = token_weight.shape
    # token_weight = token_weight.reshape(B_weight*N_weigh, C_weight)
    # [sparse_token_idx.reshape(-1)].reshape(B, N, 1)

    idx_batch = torch.arange(B, device=x.device)[:, None]
    idx = idx_cluster + idx_batch * cluster_num

    all_weight = token_weight.new_zeros(B * cluster_num, 1)
    all_weight.index_add_(dim=0, index=idx.reshape(B * N),
                          source=token_weight.reshape(B * N, 1))
    all_weight = all_weight + 1e-6
    norm_weight = token_weight / all_weight[idx]

    # average token features
    x_merged = x.new_zeros(B * cluster_num, C)
    source = x * norm_weight
    x_merged.index_add_(dim=0, index=idx.reshape(B * N),
                        source=source.reshape(B * N, C).type(x.dtype))
    x_merged = x_merged.reshape(B, cluster_num, C)

    return x_merged, index_down
