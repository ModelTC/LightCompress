import functools
import math
from types import MethodType

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

try:
    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.model.llava_arch import LlavaMetaForCausalLM
    from llava.model.multimodal_encoder.siglip_encoder import (
        SigLipVisionConfig, SigLipVisionModel)
    from llava.utils import rank0_print
except ImportError:
    pass

from llmc.utils.registry_factory import TOKEN_REDUCTION_REGISTRY

from .token_reduction_module import TokenReductionModule


def head_forward(self, hidden_state):
    batch_size = hidden_state.shape[0]
    probe = self.probe.repeat(batch_size, 1, 1)

    hidden_state, attn_weights = self.attention(probe, hidden_state, hidden_state)

    residual = hidden_state
    hidden_state = self.layernorm(hidden_state)
    hidden_state = residual + self.mlp(hidden_state)

    return hidden_state[:, 0], attn_weights


class SigLipVisionAbstract(nn.Module):
    def __init__(self, vision_tower, vision_tower_cfg, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.config = SigLipVisionConfig()

        self.vision_tower_name = vision_tower

        if not delay_load:
            rank0_print(f'Loading vision abstract: {vision_tower}')
            self.load_model()
        else:
            self.cfg_only = self.config

    def load_model(self, device_map=None):
        if self.is_loaded:
            rank0_print(
                '{} is already loaded, `load_model` called again, skipping.'.format(
                    self.vision_tower_name
                )
            )
            return

        self.vision_abstract = SigLipVisionModel.from_pretrained(
            self.vision_tower_name, device_map=device_map
        )

        del self.vision_abstract.vision_model.embeddings
        del self.vision_abstract.vision_model.encoder

        self.vision_abstract.requires_grad_(False)

        self.is_loaded = True

        self.vision_abstract.vision_model.head.__class__.forward = head_forward

    def forward(self, images):
        last_hidden_state = self.vision_abstract.vision_model.post_layernorm(images)
        pooled_output, attn_weights = self.vision_abstract.vision_model.head(
            last_hidden_state
        )
        return pooled_output, attn_weights

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        for p in self.vision_tower.parameters():
            return p.dtype

    @property
    def device(self):
        for p in self.vision_tower.parameters():
            return p.device

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def image_size(self):
        return self.config.image_size


def build_vision_abstract(vision_tower_cfg, **kwargs):
    vision_tower = getattr(
        vision_tower_cfg,
        'mm_vision_tower',
        getattr(vision_tower_cfg, 'vision_tower', None),
    )

    return SigLipVisionAbstract(
        vision_tower, vision_tower_cfg=vision_tower_cfg, **kwargs
    )


def get_attn_2dPool(frame_attn_weights, stride=2, pruning_paras=None):
    height = width = pruning_paras['num_patches_per_side']
    num_frames, _, num_tokens = frame_attn_weights.shape
    frame_attn_weights = frame_attn_weights.view(
        num_frames, 1, height, width
    ).contiguous()
    if pruning_paras['mm_spatial_pool_mode'] == 'bilinear':
        height, width = frame_attn_weights.shape[-2:]
        scaled_shape = [math.ceil(height / stride), math.ceil(width / stride)]
        frame_attn_weights = nn.functional.interpolate(
            frame_attn_weights, size=scaled_shape, mode='bilinear'
        )
    else:
        raise ValueError(
            f"Unexpected mm_spatial_pool_mode: {pruning_paras['mm_spatial_pool_mode']}"
        )
    frame_attn_weights = frame_attn_weights.view(-1)
    return frame_attn_weights


@TOKEN_REDUCTION_REGISTRY.register('FastVID')
class FastVID(TokenReductionModule):
    def __init__(self, config, model, blocks):
        super().__init__(config, model, blocks)
        self.add_sparse_config()
        self.register_reduction_modules()

    def add_sparse_config(self):
        special_config = self.config.get('special', {})
        vlm_model = self.model.vlm_model
        assert self.model.__class__.__name__ in ('Llava_OneVision',)
        if self.model.__class__.__name__ == 'Llava_OneVision':
            delay_load = getattr(vlm_model.config, 'delay_load', False)
            vision_abstract = build_vision_abstract(
                vlm_model.config, delay_load=delay_load
            )
            vision_abstract.to(device='cuda', dtype=torch.float16)
            special_config['vision_abstract'] = vision_abstract

        special_config['num_patches_per_side'] = (
            vlm_model.get_vision_tower().num_patches_per_side
        )
        special_config['mm_spatial_pool_mode'] = vlm_model.config.mm_spatial_pool_mode
        self.pruning_paras = special_config

    def register_reduction_modules(self):

        def make_hook_prepare_inputs_labels_for_multimodal(pruning_paras):
            def hook_prepare_inputs_labels_for_multimodal(
                self,
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                modalities=['image'],
                image_sizes=None,
            ):
                if 'image_token_start_index' not in pruning_paras:
                    token_indices = input_ids[0][attention_mask[0]] == IMAGE_TOKEN_INDEX
                    pruning_paras['image_token_start_index'] = torch.where(
                        token_indices
                    )[0].item()
                return self._original_prepare_inputs_labels_for_multimodal(
                    input_ids,
                    position_ids,
                    attention_mask,
                    None,
                    None,
                    images,
                    modalities,
                    image_sizes=image_sizes,
                )

            return hook_prepare_inputs_labels_for_multimodal

        if self.model.__class__.__name__ == 'Llava_OneVision':
            hook_fn = make_hook_prepare_inputs_labels_for_multimodal(self.pruning_paras)
            self.model.vlm_model._original_prepare_inputs_labels_for_multimodal = (
                self.model.vlm_model.prepare_inputs_labels_for_multimodal
            )
            self.model.vlm_model.prepare_inputs_labels_for_multimodal = MethodType(
                hook_fn, self.model.vlm_model
            )

        def vision_abstract_hook(module, args, kwargs, pruning_pars):
            image_features = args[0]
            frame_global_features, frame_attn_weights = pruning_pars['vision_abstract'](
                image_features
            )
            frame_attn_weights = get_attn_2dPool(
                frame_attn_weights, pruning_paras=pruning_pars
            )

            pruning_pars['frame_global_features'] = frame_global_features
            pruning_pars['frame_attn_weights'] = frame_attn_weights

            return args, kwargs

        def fastvid_hook(module, args, kwargs, pruning_paras):
            hidden_states = args[0]
            seq_length = hidden_states.shape[1]
            causal_mask = kwargs['attention_mask']
            position_embeddings = kwargs['position_embeddings']
            frame_global_features = pruning_paras['frame_global_features']
            frame_attn_weights = pruning_paras['frame_attn_weights']
            video_start_idx = pruning_paras['image_token_start_index']

            frame_num = frame_global_features.shape[0]
            video_token_len = frame_attn_weights.shape[0]

            # FastVID
            if seq_length > 1:
                device_type = hidden_states.device
                hidden_states_dim = hidden_states.shape[-1]
                frame_token_len = video_token_len // frame_num
                batchframe_indices = torch.arange(
                    frame_num, device=device_type
                ).unsqueeze(1)
                alltoken_indices = (
                    torch.arange(video_token_len, device=device_type).view(
                        frame_num, frame_token_len
                    )
                    + video_start_idx
                )

                video_hidden_states = hidden_states[
                    :, video_start_idx: video_start_idx + video_token_len, :
                ].squeeze(0)
                video_hidden_states = video_hidden_states.reshape(
                    frame_num, frame_token_len, -1
                )
                frame_attn_weights = frame_attn_weights.reshape(
                    frame_num, frame_token_len
                )

                # DySeg
                # frame_global_features = self.frame_global_features
                frame_global_features = (
                    frame_global_features
                    / frame_global_features.norm(dim=1, keepdim=True)
                )
                similarity_matrix = (
                    frame_global_features[:-1] * frame_global_features[1:]
                ).sum(dim=1)

                cut_indices_topk = torch.topk(
                    similarity_matrix, pruning_paras['DySeg_c'] - 1, largest=False
                ).indices
                cut_indices_cos = torch.nonzero(
                    similarity_matrix < pruning_paras['DySeg_tau'], as_tuple=False
                ).squeeze(1)
                cut_indices = (
                    torch.unique(torch.cat([cut_indices_topk, cut_indices_cos]))
                    .sort()
                    .values
                )
                padded = F.pad(cut_indices, (1, 1), value=-1)
                padded[-1] = frame_num - 1
                segment_sizes = padded.diff().tolist()

                # STPrune
                keep_indexs = ()
                keep_indexs += (torch.arange(video_start_idx, device=device_type),)
                keep_indexs += (
                    torch.arange(
                        video_start_idx + video_token_len,
                        seq_length,
                        device=device_type,
                    ),
                )
                start_tokens = hidden_states[0, :video_start_idx, :]
                end_tokens = hidden_states[0, video_start_idx + video_token_len:, :]
                final_tokens = [start_tokens, end_tokens]

                frame_retain_num = int(
                    frame_token_len * pruning_paras['retention_ratio']
                )

                frame_salient_num = frame_retain_num - int(
                    frame_retain_num * pruning_paras['STPrune_d']
                )
                # frm_salient_num_list = [frame_salient_num] * frame_num

                frm_context_num_list = torch.zeros(
                    frame_num, dtype=torch.int, device=device_type
                )
                frame_context_num = frame_retain_num - frame_salient_num

                # Compute Anchor Token Distribution
                offset = 0
                for seg_i_len in segment_sizes:
                    seg_context_num = frame_context_num * seg_i_len
                    temp_num = (
                        seg_i_len + pruning_paras['DTM_p'] - 1
                    ) // pruning_paras['DTM_p']
                    cur_frm_context_num = seg_context_num // temp_num

                    end = offset + seg_i_len
                    seg_indices = torch.arange(
                        seg_i_len - 1, -1, -1, device=device_type
                    )
                    mask = seg_indices % pruning_paras['DTM_p'] == 0

                    frm_context_num_list[offset:end][mask] = cur_frm_context_num
                    offset = end

                # ATS
                salient_indexes = torch.topk(
                    frame_attn_weights, frame_salient_num, dim=1
                ).indices

                batch_indices = batchframe_indices.expand(-1, frame_salient_num)
                salient_tokens = video_hidden_states[batch_indices, salient_indexes]
                salient_global_indexes = alltoken_indices[
                    batch_indices, salient_indexes
                ]

                final_tokens.append(salient_tokens.view(-1, hidden_states_dim))
                keep_indexs += (salient_global_indexes.view(-1),)

                # Parallel Density Score Computation
                all_indices = (
                    torch.arange(frame_token_len, device=device_type)
                    .unsqueeze(0)
                    .expand(frame_num, -1)
                )
                all_indices_mask = torch.ones_like(all_indices, dtype=torch.bool)
                all_indices_mask.scatter_(1, salient_indexes, False)
                filtered_indices = all_indices[all_indices_mask].view(
                    frame_num, frame_token_len - frame_salient_num
                )

                batch_indices = batchframe_indices.expand(
                    -1, frame_token_len - frame_salient_num
                )
                token_filtered = video_hidden_states[batch_indices, filtered_indices]
                alltoken_filtered_indices = alltoken_indices[
                    batch_indices, filtered_indices
                ]

                tmp_frm_hidden_states = token_filtered
                dist_matrix = torch.cdist(
                    tmp_frm_hidden_states.float(), tmp_frm_hidden_states.float()
                ) / (hidden_states_dim**0.5)

                dist_nearest, index_nearest = torch.topk(
                    dist_matrix, k=4, dim=-1, largest=False
                )
                density = (-(dist_nearest**2).mean(dim=-1)).exp()
                density = (
                    density
                    + torch.rand(density.shape, device=device_type, dtype=density.dtype)
                    * 1e-6
                )

                density_mask = density[:, None, :] > density[:, :, None]
                density_mask = density_mask.type(tmp_frm_hidden_states.dtype)
                dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]
                dist_0, index_parent = (
                    dist_matrix * density_mask + dist_max * (1 - density_mask)
                ).min(dim=-1)

                density_score = dist_0 * density

                sampled_indexs = torch.topk(
                    density_score, k=frame_context_num, dim=-1
                ).indices

                # DTM for Single-Frame Segment
                batch_indices = batchframe_indices.expand(-1, frame_context_num)
                frm_context_tokens = token_filtered[batch_indices, sampled_indexs]
                frm_context_global_indexes = alltoken_filtered_indices[
                    batch_indices, sampled_indexs
                ]

                to_be_merge_tokens = token_filtered / token_filtered.norm(
                    dim=-1, keepdim=True
                )
                merge_target_tokens = to_be_merge_tokens[batch_indices, sampled_indexs]

                similarity = torch.bmm(
                    to_be_merge_tokens, merge_target_tokens.transpose(1, 2)
                )
                assign_one_hot = torch.zeros(
                    frame_num,
                    frame_token_len - frame_salient_num,
                    frame_context_num,
                    dtype=token_filtered.dtype,
                    device=device_type,
                )
                assign_one_hot.scatter_(2, similarity.argmax(dim=2).unsqueeze(-1), 1)

                avg_weights = (1 / (assign_one_hot.sum(dim=1).unsqueeze(-1) + 1)).clamp(
                    min=pruning_paras['DTM_alpha']
                )

                counts = assign_one_hot.sum(dim=1).clamp(min=1).unsqueeze(-1)
                aggregated_hidden = (
                    torch.bmm(assign_one_hot.transpose(1, 2), token_filtered) / counts
                )

                frm_context_tokens = (
                    avg_weights * frm_context_tokens
                    + (1 - avg_weights) * aggregated_hidden
                )

                context_for_frame_mask = frm_context_num_list == frame_context_num
                # context_for_frame_num = context_for_frame_mask.sum()

                context_for_frame_tokens = frm_context_tokens[context_for_frame_mask]
                context_for_frame_global_indexes = frm_context_global_indexes[
                    context_for_frame_mask
                ]

                final_tokens.append(
                    context_for_frame_tokens.view(-1, hidden_states_dim)
                )
                keep_indexs += (context_for_frame_global_indexes.view(-1),)

                # DTM for Multi-Frame Segment
                idx_seg_start = 0
                for seg_i_len in segment_sizes:
                    if seg_i_len > 1:
                        cur_seg_context_num_list = frm_context_num_list[
                            idx_seg_start: idx_seg_start + seg_i_len
                        ]
                        cur_seg_context_num = cur_seg_context_num_list[-1]

                        cur_seg_target_mask = (
                            cur_seg_context_num_list > frame_context_num
                        )
                        cur_seg_target_num = cur_seg_target_mask.sum()

                        cur_seg_density_score = density_score[
                            idx_seg_start: idx_seg_start + seg_i_len
                        ]
                        cur_seg_density_score = cur_seg_density_score[
                            cur_seg_target_mask
                        ]

                        cur_seg_token_filtered = token_filtered[
                            idx_seg_start: idx_seg_start + seg_i_len
                        ]
                        cur_seg_token_target = cur_seg_token_filtered[
                            cur_seg_target_mask
                        ]
                        cur_seg_token_filtered = cur_seg_token_filtered.view(
                            1, -1, hidden_states_dim
                        ).expand(cur_seg_target_num, -1, -1)

                        cur_seg_alltoken_indices = alltoken_filtered_indices[
                            idx_seg_start: idx_seg_start + seg_i_len
                        ]
                        cur_seg_alltoken_indices = cur_seg_alltoken_indices[
                            cur_seg_target_mask
                        ]

                        sampled_indexs = torch.topk(
                            cur_seg_density_score, k=cur_seg_context_num, dim=-1
                        ).indices
                        batch_indices = batchframe_indices[:cur_seg_target_num].expand(
                            -1, cur_seg_context_num
                        )
                        cur_context_tokens = cur_seg_token_target[
                            batch_indices, sampled_indexs
                        ]
                        cur_context_global_indexes = cur_seg_alltoken_indices[
                            batch_indices, sampled_indexs
                        ]

                        to_be_merge_tokens = (
                            cur_seg_token_filtered
                            / cur_seg_token_filtered.norm(dim=-1, keepdim=True)
                        )
                        merge_target_tokens = (
                            cur_context_tokens
                            / cur_context_tokens.norm(dim=-1, keepdim=True)
                        )

                        similarity = torch.bmm(
                            to_be_merge_tokens, merge_target_tokens.transpose(1, 2)
                        )
                        assign_one_hot = torch.zeros(
                            cur_seg_target_num,
                            to_be_merge_tokens.shape[1],
                            cur_seg_context_num,
                            dtype=token_filtered.dtype,
                            device=device_type,
                        )
                        assign_one_hot.scatter_(
                            2, similarity.argmax(dim=2).unsqueeze(-1), 1
                        )

                        avg_weights = (
                            1 / (assign_one_hot.sum(dim=1).unsqueeze(-1) + 1)
                        ).clamp(min=pruning_paras['DTM_alpha'])

                        counts = assign_one_hot.sum(dim=1).clamp(min=1).unsqueeze(-1)
                        aggregated_hidden = (
                            torch.bmm(
                                assign_one_hot.transpose(1, 2), cur_seg_token_filtered
                            )
                            / counts
                        )

                        cur_context_tokens = (
                            avg_weights * cur_context_tokens
                            + (1 - avg_weights) * aggregated_hidden
                        )

                        final_tokens.append(
                            cur_context_tokens.view(-1, hidden_states_dim)
                        )
                        keep_indexs += (cur_context_global_indexes.view(-1),)

                    idx_seg_start += seg_i_len

                hidden_states = torch.cat(final_tokens, dim=0)
                keep_indexs = torch.cat(keep_indexs, dim=0)

                sorted_indexs = torch.argsort(keep_indexs)
                hidden_states = hidden_states[sorted_indexs].unsqueeze(0)
                keep_indexs = keep_indexs[sorted_indexs]

                if causal_mask is not None:
                    kwargs['attention_mask'].fill_(
                        causal_mask[
                            :, :, : hidden_states.shape[1], : hidden_states.shape[1]
                        ]
                    )

                with torch.inference_mode():
                    kwargs['position_ids'].resize_as_(keep_indexs.unsqueeze(0)).copy_(
                        keep_indexs.unsqueeze(0).clone()
                    )
                    kwargs['cache_position'].resize_as_(keep_indexs).copy_(
                        keep_indexs.clone()
                    )

                    new_pe0 = position_embeddings[0][:, keep_indexs, :].clone()
                    new_pe1 = position_embeddings[1][:, keep_indexs, :].clone()
                    position_embeddings[0].resize_as_(new_pe0).copy_(new_pe0)
                    position_embeddings[1].resize_as_(new_pe0).copy_(new_pe1)

                    args[0].resize_as_(hidden_states).copy_(hidden_states.clone())

            ##############################################################

            return (hidden_states,), kwargs

        self.model.vlm_model.get_model().mm_projector.register_forward_pre_hook(
            functools.partial(vision_abstract_hook, pruning_pars=self.pruning_paras),
            with_kwargs=True,
        )

        self.blocks[0].register_forward_pre_hook(
            functools.partial(fastvid_hook, pruning_paras=self.pruning_paras),
            with_kwargs=True,
        )
