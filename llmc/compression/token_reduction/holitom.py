import functools
import math
import random
import re
from abc import ABC
from types import MethodType
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

try:
    from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
    from llava.mm_utils import get_anyres_image_grid_shape
    from llava.model.llava_arch import LlavaMetaForCausalLM, unpad_image
    from llava.model.multimodal_encoder.siglip_encoder import (
        SigLipEncoder, SigLipVisionTower)
    from llava.utils import rank0_print
    from transformers.modeling_outputs import (BaseModelOutput,
                                               BaseModelOutputWithPooling)
except ImportError:
    pass

from llmc.utils.registry_factory import TOKEN_REDUCTION_REGISTRY

from .token_reduction_module import TokenReductionModule


def SigLipEncoder_forward(
    self,
    inputs_embeds,
    attention_mask: Optional[torch.Tensor] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
):
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    encoder_states = () if output_hidden_states else None
    all_attentions = () if output_attentions else None

    hidden_states = inputs_embeds
    for encoder_layer in self.layers:
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)
        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                encoder_layer.__call__,
                hidden_states,
                attention_mask,
                output_attentions,
            )
        else:
            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask,
                output_attentions=output_attentions,
            )

        hidden_states = layer_outputs[0]

        if output_attentions:
            all_attentions = all_attentions + (layer_outputs[1],)

    if output_hidden_states:
        encoder_states = encoder_states + (hidden_states,)

    if not return_dict:
        return tuple(
            v for v in [hidden_states, encoder_states, all_attentions] if v is not None
        )
    return BaseModelOutput(
        last_hidden_state=hidden_states,
        hidden_states=encoder_states,
        attentions=all_attentions,
    )


def SigLipVisionTower_forward(self, images):
    if type(images) is list:
        image_features = []
        for image in images:
            image_forward_out = self.vision_tower(
                image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                output_hidden_states=True,
            )
            image_feature = image_forward_out.hidden_states[-1].to(image.dtype)
            assert image_features.shape[-2] == 729
            image_features.append(image_feature)
    else:
        import os

        wrapper = os.environ.get('WRAPPER')
        if wrapper in ['visionzip']:
            image_forward_outs = self.vision_tower(
                images.to(device=self.device, dtype=self.dtype),
                output_hidden_states=True,
                output_attentions=True,
            )
            attn_weights = image_forward_outs.attentions[-1]
            hidden_states = image_forward_outs.hidden_states[-1]
            metric = self.vision_tower.vision_model.encoder.layers[-1].metric
            return (
                hidden_states,
                attn_weights.mean(dim=1).mean(dim=1),
                metric,
                images.dtype,
            )
        if wrapper in ['holitom']:
            image_forward_outs = self.vision_tower(
                images.to(device=self.device, dtype=self.dtype),
                output_hidden_states=True,
                output_attentions=True,
            )
            attn_weights = image_forward_outs.attentions[-1]
            hidden_states = image_forward_outs.hidden_states[-1]
            return (
                hidden_states,
                attn_weights.mean(dim=1).mean(dim=1),
                None,
                images.dtype,
            )
        else:
            image_forward_outs = self.vision_tower(
                images.to(device=self.device, dtype=self.dtype),
                output_hidden_states=True,
            )
            image_features = image_forward_outs.hidden_states[-1].to(images.dtype)
            assert image_features.shape[-2] == 729

    return image_features


class LlavaMetaForCausalLM_holitom(ABC):

    def encode_images(self, images):
        image_features, _ = self.get_model().get_vision_tower()(images)
        # image_features = self.get_model().vision_resampler(image_features, images=images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def encode_images_multi(self, images):
        image_features, attn_weights, metric, images_dtype = (
            self.get_model().get_vision_tower()(images)
        )
        # image_features = self.get_model().vision_resampler(image_features, images=images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features, attn_weights, metric, images_dtype

    def cluster_dpc_knn(self, x, cluster_num, k=7):
        with torch.no_grad():
            batch_size, seq_len, embed_dim = x.shape

            dist_matrix = torch.cdist(x.float(), x.float()) / (
                embed_dim**0.5
            )  # (batch_size, seq_len, seq_len)

            # get local density
            dist_nearest, index_nearest = torch.topk(
                dist_matrix, k, dim=-1, largest=False
            )  # (batch_size, seq_len, k)
            density = (-(dist_nearest**2).mean(dim=-1)).exp()  # (batch_size, seq_len)
            # add a little noise to ensure no tokens have the same density.
            density = (
                density
                + torch.rand(density.shape, device=density.device, dtype=density.dtype)
                * 1e-6
            )

            # get distance indicator
            mask = (density[:, None, :] > density[:, :, None]).type(x.dtype)
            dist_max = dist_matrix.flatten(1).max(dim=-1).values[:, None, None]
            dist, index_parent = (dist_matrix * mask + dist_max * (1 - mask)).min(
                dim=-1
            )

            # select the cluster center according to the score
            score = dist * density
            _, index_center = score.topk(cluster_num, dim=-1)

            return index_center, dist_matrix

    def select_static_windows(self, feature_sim, batch_size, tau, max_window_size):
        # pruned_static_count[s,e]
        pruned_static_count = torch.zeros(
            (batch_size, batch_size), device=feature_sim.device
        )
        for start in range(0, batch_size):
            for end in range(start + 1, batch_size):
                static_feature_count = (
                    torch.all(feature_sim[start:end, :] > tau, dim=0).sum().item()
                )
                pruned_static_count[start, end] = static_feature_count * (
                    end - start
                )  # window_len = end - start + 1

        dp = torch.zeros(batch_size, device=pruned_static_count.device)
        prev = torch.zeros(
            batch_size, dtype=torch.long, device=pruned_static_count.device
        )
        # [prev[i], i]

        for i in range(batch_size):
            max_val = dp[i - 1] if i > 0 else 0
            best_j = i

            for window_size in range(2, min(i + 1, max_window_size) + 1):
                j = i - window_size
                current_val = (dp[j] if j >= 0 else 0) + pruned_static_count[
                    j + 1, i
                ]  # [-, j] + [j+1, i]
                if current_val > max_val:
                    max_val = current_val
                    best_j = j + 1

            dp[i] = max_val
            prev[i] = best_j  # [best_j, i]

        selected_frames = []
        i = batch_size - 1
        while i >= 0:
            selected_frames.append((prev[i].item(), i))
            i = prev[i].item() - 1

        selected_frames = selected_frames[::-1]
        total_reduced = dp[-1].item()

        return selected_frames, total_reduced

    def merge_tokens_by_clustering(
        self, feat, target_indices, dist_matrix, cluster_num, Beta
    ):
        batch_size, seq_len, embed_dim = feat.shape
        all_indices = torch.arange(seq_len, device=feat.device)
        all_indices = all_indices.unsqueeze(0).expand(
            batch_size, -1
        )  # (batch_size, seq_len)
        non_target_indices = torch.zeros(
            (batch_size, seq_len - cluster_num), dtype=torch.long, device=feat.device
        )
        for b in range(batch_size):
            non_target_mask = ~torch.isin(all_indices[b], target_indices[b])
            non_target_indices[b] = all_indices[b][non_target_mask]
        # non_target_indices (batch_size, seq_len-cluster_num)

        non_target_feat = torch.gather(
            feat,
            dim=1,
            index=non_target_indices.unsqueeze(-1).expand(-1, -1, feat.size(-1)),
        )  # (batch_size, seq_len-cluster_num, embed_dim)

        dist_matrix = torch.gather(
            dist_matrix,
            dim=1,
            index=non_target_indices.unsqueeze(-1).expand(-1, -1, dist_matrix.size(-1)),
        )  # (batch_size, seq_len-cluster_num, seq_len)
        dist_matrix = torch.gather(
            dist_matrix,
            dim=2,
            index=target_indices.unsqueeze(1).expand(-1, dist_matrix.size(1), -1),
        )  # (batch_size, seq_len-cluster_num, cluster_num)

        idx_cluster = torch.argmin(
            dist_matrix, dim=-1
        )  # (batch_size, seq_len-cluster_num)

        cluster_tokens = []
        for b in range(batch_size):
            batch_tokens = []
            for i in range(cluster_num):
                mask = idx_cluster[b] == i
                if mask.any():
                    cluster_features = non_target_feat[b][mask]
                    import os

                    if os.environ.get('NO_BETA', '0') == '0':
                        # rank0_print("USE_BETA")
                        cluster_means = cluster_features.mean(dim=0)
                        batch_tokens.append(
                            Beta * feat[b][target_indices[b][i]]
                            + (1 - Beta) * cluster_means
                        )
                    else:
                        # rank0_print("NO_BETA")
                        all_features = torch.cat(
                            [
                                feat[b][target_indices[b][i]].unsqueeze(0),
                                cluster_features,
                            ],
                            dim=0,
                        )
                        batch_tokens.append(all_features.mean(dim=0))
                else:
                    batch_tokens.append(feat[b][target_indices[b][i]])
            cluster_tokens.append(torch.stack(batch_tokens))
        cluster_tokens = torch.stack(
            cluster_tokens
        )  # shape: (batch_size, cluster_num, embed_dim)

        return cluster_tokens

    def merge_tokens_by_attention_density(
        self, feat, attn, pos, retain_ratio, D, Beta, K
    ):
        batch_size, seq_len, embed_dim = feat.shape
        dominant_num = round(math.ceil(seq_len * retain_ratio) * (1 - D))
        contextual_num = math.ceil(seq_len * retain_ratio) - dominant_num

        # Dominant Visual Tokens
        if dominant_num > 0:
            all_indices = attn.topk(dominant_num, dim=1).indices
            mask = torch.ones_like(
                feat[:, :, 0], dtype=torch.bool, device=feat.device
            ).scatter_(
                1, all_indices, False
            )  # (batch_size, seq_len) False means retained tokens
            # finally, (batch_size, dominant_num, embed_dim) compare with feat
            dominant_tokens = feat.masked_select(~mask.unsqueeze(-1)).view(
                batch_size, dominant_num, embed_dim
            )
            dominant_pos = pos.masked_select(~mask).view(batch_size, dominant_num)
        else:
            mask = torch.ones_like(feat[:, :, 0], dtype=torch.bool, device=feat.device)
            dominant_tokens = torch.empty(
                (batch_size, 0, embed_dim), device=feat.device
            )
            dominant_pos = torch.empty((batch_size, 0), device=feat.device)

        # Contextual Visual Tokens
        if contextual_num > 0:
            # Filter
            # feat_filtered: (batch_size, seq_len-dominant_num, embed_dim)
            feat_filtered = feat.masked_select(mask.unsqueeze(-1)).view(
                batch_size, seq_len - dominant_num, embed_dim
            )
            contextual_pos = pos.masked_select(mask.unsqueeze(-1)).view(
                batch_size, seq_len - dominant_num
            )
            target_indices, dist_matrix = self.cluster_dpc_knn(
                feat_filtered, contextual_num, k=min(K, contextual_num)
            )
            target_indices = torch.sort(target_indices, dim=-1)[0]
            contextual_pos = torch.stack(
                [contextual_pos[b][target_indices[b]] for b in range(batch_size)]
            )  # (batch_size, contextual_num)
            # target_indices (batch_size, contextual_num)
            # dist_matrix (batch_size, seq_len-dominant_num, seq_len-dominant_num)
            # assign tokens to the nearest center

            contextual_tokens = self.merge_tokens_by_clustering(
                feat_filtered, target_indices, dist_matrix, contextual_num, Beta
            )
        else:
            contextual_tokens = torch.empty(
                (batch_size, 0, embed_dim), device=feat.device
            )
            contextual_pos = torch.empty((batch_size, 0), device=feat.device)

        image_feat = []
        image_pos = []
        for b in range(batch_size):
            batch_tokens = torch.cat([dominant_tokens[b], contextual_tokens[b]], dim=0)
            batch_pos = torch.cat([dominant_pos[b], contextual_pos[b]], dim=0)
            image_feat.append(batch_tokens)
            image_pos.append(batch_pos)
        image_feat = torch.stack(
            image_feat
        )  # shape: (batch_size, dominant_num + contextual_num, embed_dim)
        image_pos = torch.stack(image_pos)

        return image_feat, image_pos

    def merge_tokens_by_density(self, feat, pos, retain_ratio, Beta, K):
        batch_size, seq_len, embed_dim = feat.shape
        cluster_num = round(seq_len * retain_ratio)
        if cluster_num > 0:
            target_indices, dist_matrix = self.cluster_dpc_knn(
                feat, cluster_num, k=min(K, cluster_num)
            )
            target_indices = torch.sort(target_indices, dim=-1)[0]
            image_pos = torch.stack(
                [pos[b][target_indices[b]] for b in range(batch_size)]
            )

            cluster_tokens = self.merge_tokens_by_clustering(
                feat, target_indices, dist_matrix, cluster_num, Beta
            )
            image_feat = cluster_tokens
        else:
            image_feat = torch.empty((batch_size, 0, embed_dim), device=feat.device)
            image_pos = torch.empty((batch_size, 0), device=feat.device)

        return image_feat, image_pos

    def add_newline_token(self, feat, pos, grid_size, newline_token):
        row_pos = pos // grid_size
        expanded_feat_list = []
        for cur_feat, cur_row_pos in zip(feat, row_pos):
            expanded_feat = []
            for row in range(grid_size):
                find_row_feat = cur_feat[cur_row_pos == row]
                if len(find_row_feat) > 0:
                    expanded_feat.append(
                        torch.cat((find_row_feat, newline_token), dim=0)
                    )
                else:
                    expanded_feat.append(find_row_feat)
            batch_feat = torch.cat(expanded_feat, dim=0)
            expanded_feat_list.append(batch_feat)

        image_feat = torch.cat(expanded_feat_list, dim=0)
        return image_feat

    def holitom(
        self,
        static_feat,
        dynamic_feat,
        dynamic_attn,
        static_pos,
        dynamic_pos,
        window_size,
        retain_ratio,
        D,
        Beta,
        K,
        images_dtype,
        mm_newline_position,
    ):
        newline_token = (
            self.model.image_newline[None].to(static_feat.device)
            if mm_newline_position == 'grid'
            else None
        )
        grid_size = int(math.sqrt(dynamic_feat.shape[1] + static_feat.shape[0]))

        if window_size == 1:
            dynamic_feat, dynamic_pos = self.merge_tokens_by_attention_density(
                dynamic_feat, dynamic_attn, dynamic_pos, retain_ratio, D, Beta, K
            )
            if mm_newline_position != 'grid':
                feat = dynamic_feat.flatten(0, 1)
            else:
                dynamic_pos, sorted_indices = torch.sort(dynamic_pos, dim=1)
                dynamic_feat = torch.gather(
                    dynamic_feat,
                    1,
                    sorted_indices.unsqueeze(-1).expand(-1, -1, dynamic_feat.shape[-1]),
                )
                dynamic_feat = self.add_newline_token(
                    dynamic_feat, dynamic_pos, grid_size, newline_token
                )

                feat = dynamic_feat

            return feat.to(images_dtype)
        else:
            dynamic_feat, dynamic_pos = self.merge_tokens_by_attention_density(
                dynamic_feat, dynamic_attn, dynamic_pos, retain_ratio, D, Beta, K
            )
            static_feat, static_pos = self.merge_tokens_by_density(
                static_feat.unsqueeze(0), static_pos, retain_ratio, Beta, K
            )
            if mm_newline_position != 'grid':
                feat = torch.cat(
                    [static_feat.flatten(0, 1), dynamic_feat.flatten(0, 1)]
                )
            else:
                first_dynamic_feat = dynamic_feat[0:1, :]
                first_dynamic_pos = dynamic_pos[0:1, :]
                first_feat = torch.cat(
                    [static_feat, first_dynamic_feat], dim=1
                )  # (batch_size, first_frame_tokens, embed_dim)
                first_pos = torch.cat([static_pos, first_dynamic_pos], dim=1)

                # Sort tokens by their original positions
                first_pos, first_sorted_indices = torch.sort(first_pos, dim=1)
                first_feat = torch.gather(
                    first_feat,
                    1,
                    first_sorted_indices.unsqueeze(-1).expand(
                        -1, -1, first_feat.shape[-1]
                    ),
                )

                first_feat = self.add_newline_token(
                    first_feat, first_pos, grid_size, newline_token
                )

                other_feat = dynamic_feat[1:, :]
                other_pos = dynamic_pos[1:, :]
                other_pos, other_sorted_indices = torch.sort(other_pos, dim=1)
                other_feat = torch.gather(
                    other_feat,
                    1,
                    other_sorted_indices.unsqueeze(-1).expand(
                        -1, -1, other_feat.shape[-1]
                    ),
                )
                other_feat = self.add_newline_token(
                    other_feat, other_pos, grid_size, newline_token
                )

                feat = torch.cat([first_feat, other_feat])

            return feat.to(images_dtype)

    def get_static_dynamic_features(
        self, image_feat, attn_weights, selected_frames, feature_sim, tau
    ):
        # attn_weights: (batch_size, seq_len)
        batch_size, seq_len, embed_dim = image_feat.shape
        static_feat_list, dynamic_feat_list, _, dynamic_attn_list = [], [], [], []
        static_pos_list, dynamic_pos_list = [], []
        for start, end in selected_frames:
            all_indices = torch.arange(seq_len, device=image_feat.device).unsqueeze(
                0
            )  # (1, seq_len)
            if start == end:
                static_feat_list.append(
                    torch.empty((0, embed_dim), device=image_feat.device)
                )
                # static_attn_list.append(torch.empty((0,), device=attn_weights.device))
                dynamic_feat_list.append(image_feat[start: end + 1])
                dynamic_attn_list.append(attn_weights[start: end + 1])

                static_pos_list.append(
                    torch.empty((0, seq_len), device=image_feat.device)
                )
                dynamic_pos_list.append(all_indices)
            else:
                windows_size = end - start + 1
                mask = torch.all(feature_sim[start:end, :] > tau, dim=0)
                static_feat = image_feat[start: end + 1, mask]
                # static_attn = attn_weights[start:end+1, mask]
                dynamic_feat = image_feat[start: end + 1, ~mask]
                dynamic_attn = attn_weights[start: end + 1, ~mask]

                static_feat_list.append(static_feat.mean(dim=0))
                # static_attn_list.append(static_attn.mean(dim=0))
                dynamic_feat_list.append(dynamic_feat)
                dynamic_attn_list.append(dynamic_attn)

                static_pos_list.append(all_indices[:, mask].expand(1, -1))
                dynamic_pos_list.append(all_indices[:, ~mask].expand(windows_size, -1))

        return (
            static_feat_list,
            dynamic_feat_list,
            _,
            dynamic_attn_list,
            static_pos_list,
            dynamic_pos_list,
        )

    def prepare_inputs_labels_for_multimodal(
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
        import os

        vision_tower = self.get_vision_tower()
        # rank_print(modalities)
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                None,
                labels,
            )

        if isinstance(modalities, str):
            modalities = [modalities]

        # import pdb; pdb.set_trace()
        if type(images) is list or images.ndim == 5:
            mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
            image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
            mm_newline_position = getattr(
                self.config, 'mm_newline_position', 'one_token'
            )

            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]

            video_idx_in_batch = []
            for _ in range(len(modalities)):
                if modalities[_] == 'video':
                    video_idx_in_batch.append(_)

            images_list = []
            for image in images:
                if image.ndim == 4:
                    images_list.append(image)
                else:
                    images_list.append(image.unsqueeze(0))

            concat_images = torch.cat([image for image in images_list], dim=0)
            split_sizes = [image.shape[0] for image in images_list]
            encoded_image_features, attn_weights, _, images_dtype = (
                self.encode_images_multi(concat_images)
            )
            retain_ratio = self.pruning_paras.get('RETAIN_RATIO', 0.1)
            # C = int(os.environ.get("C", 8))
            # tau = float(os.environ.get("T", 0.8))
            tau = self.pruning_paras.get('T', 0.1)
            # P = int(os.environ.get("P", 4))
            Beta = float(os.environ.get('BETA', 0.6))
            D = float(os.environ.get('D', 0))
            K = int(os.environ.get('K', 7))
            max_window_size = int(os.environ.get('MAX_WINDOW_SIZE', 1024))
            # NO_BETA = os.environ.get('NO_BETA', '1')
            # rank0_print(f"retain_ratio: {retain_ratio},
            # tau: {tau}, Beta: {Beta}, D: {D}, K: {K},
            # max_window_size: {max_window_size}, NO_BETA: {NO_BETA}")
            # image_features,all_faster_video_features =
            # self.encode_multimodals(concat_images, video_idx_in_batch, split_sizes)

            # This is a list, each element is [num_images, patch * patch, dim]
            # rank_print(f"Concat images : {concat_images.shape}")
            encoded_image_features = torch.split(encoded_image_features, split_sizes)
            image_features = []
            for idx, image_feat in enumerate(encoded_image_features):
                if idx in video_idx_in_batch:
                    # [modify]
                    # image_features.append(self.get_2dPool(image_feat))
                    # image_feat: (batch_size, seq_len, embed_dim)
                    # attn_weights: (batch_size, seq_len)
                    pooled_image_feat = self.get_2dPool(
                        image_feat
                    )  # (batch_size, seq_len', embed_dim)
                    attn_weights = attn_weights.unsqueeze(-1)
                    attn_weights = self.get_2dPool(attn_weights)
                    attn_weights = attn_weights.squeeze(-1)  # (batch_size, seq_len')

                    batch_size, seq_len, embed_dim = pooled_image_feat.shape

                    pooled_image_feat_normed = torch.nn.functional.normalize(
                        pooled_image_feat, p=2, dim=-1
                    )
                    feature_sim = torch.nn.functional.cosine_similarity(
                        pooled_image_feat_normed[:-1],
                        pooled_image_feat_normed[1:],
                        dim=-1,
                    )  # （batch_size-1, seq_len')

                    selected_frames, total_reduced = self.select_static_windows(
                        feature_sim, batch_size, tau, max_window_size
                    )
                    # rank0_print(f"Selected frames: {selected_frames}")
                    # rank0_print(f"Total reduced features: {total_reduced}")

                    total_tokens = batch_size * seq_len
                    retain_ratio = min(
                        retain_ratio / ((total_tokens - total_reduced) / total_tokens),
                        1,
                    )
                    # rank0_print(f"After static pruning, retain ratio: {retain_ratio}")

                    (
                        static_feat,
                        dynamic_feat,
                        _,
                        dynamic_attn,
                        static_pos,
                        dynamic_pos,
                    ) = self.get_static_dynamic_features(
                        pooled_image_feat,
                        attn_weights,
                        selected_frames,
                        feature_sim,
                        tau,
                    )

                    segment_features = []
                    for idx, (start, end) in enumerate(selected_frames):
                        window_size = end - start + 1
                        segment_features.append(
                            self.holitom(
                                static_feat[idx],
                                dynamic_feat[idx],
                                dynamic_attn[idx],
                                static_pos[idx],
                                dynamic_pos[idx],
                                window_size,
                                retain_ratio,
                                D,
                                Beta,
                                K,
                                images_dtype,
                                mm_newline_position,
                            )
                        )
                    image_features.append(torch.cat(segment_features, dim=0))

                else:
                    image_features.append(image_feat)
            # image_features =
            # self.encode_multimodals(concat_images, video_idx_in_batch, split_sizes)
            # rank_print(f"Encoded image feats : {[x.shape for x in image_features]}")
            # image_features = torch.split(image_features, split_sizes, dim=0)

            if mm_patch_merge_type == 'flat':
                image_features = [x.flatten(0, 1) for x in image_features]

            elif mm_patch_merge_type.startswith('spatial'):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    # FIXME: now assume the image is square, and split to 2x2 patches
                    # num_patches = h * w, where h = w = sqrt(num_patches)
                    # currently image_feature is a tensor of shape (4, num_patches, hidden_size)
                    # we want to first unflatten it to (2, 2, h, w, hidden_size)
                    # rank0_print("At least we are reaching here")
                    # import pdb; pdb.set_trace()
                    if image_idx in video_idx_in_batch:  # video operations
                        # rank0_print("Video")
                        if mm_newline_position == 'grid':
                            new_image_features.append(image_feature)
                        elif mm_newline_position == 'frame':
                            # Frame-wise
                            image_feature = self.add_token_per_frame(image_feature)

                            new_image_features.append(image_feature.flatten(0, 1))

                        elif mm_newline_position == 'one_token':
                            # one-token
                            # image_feature = image_feature.flatten(0, 1)
                            if 'unpad' in mm_patch_merge_type:
                                image_feature = torch.cat(
                                    (
                                        image_feature,
                                        self.model.image_newline[None].to(
                                            image_feature.device
                                        ),
                                    ),
                                    dim=0,
                                )
                            new_image_features.append(image_feature)
                        elif mm_newline_position == 'no_token':
                            new_image_features.append(image_feature.flatten(0, 1))
                        else:
                            raise ValueError(
                                f'Unexpected mm_newline_position: {mm_newline_position}'
                            )
                    elif (
                        image_feature.shape[0] > 1
                    ):  # multi patches and multi images operations
                        # rank0_print("Single-images")
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]

                        if 'anyres_max' in image_aspect_ratio:
                            matched_anyres_max_num_patches = re.match(
                                r'anyres_max_(\d+)', image_aspect_ratio
                            )
                            if matched_anyres_max_num_patches:
                                max_num_patches = int(
                                    matched_anyres_max_num_patches.group(1)
                                )

                        if (
                            image_aspect_ratio == 'anyres'
                            or 'anyres_max' in image_aspect_ratio
                        ):
                            if hasattr(self.get_vision_tower(), 'image_size'):
                                vision_tower_image_size = (
                                    self.get_vision_tower().image_size
                                )
                            else:
                                raise ValueError(
                                    'vision_tower_image_size is not found in the vision tower.'
                                )
                            try:
                                num_patch_width, num_patch_height = (
                                    get_anyres_image_grid_shape(
                                        image_sizes[image_idx],
                                        self.config.image_grid_pinpoints,
                                        vision_tower_image_size,
                                    )
                                )
                            except Exception as e:
                                rank0_print(f'Error: {e}')
                                num_patch_width, num_patch_height = 2, 2
                            image_feature = image_feature.view(
                                num_patch_height, num_patch_width, height, width, -1
                            )
                        else:
                            image_feature = image_feature.view(2, 2, height, width, -1)

                        if 'maxpool2x2' in mm_patch_merge_type:
                            image_feature = image_feature.permute(
                                4, 0, 2, 1, 3
                            ).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = nn.functional.max_pool2d(image_feature, 2)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        elif (
                            'unpad' in mm_patch_merge_type
                            and 'anyres_max' in image_aspect_ratio
                            and matched_anyres_max_num_patches
                        ):
                            unit = image_feature.shape[2]
                            image_feature = image_feature.permute(
                                4, 0, 2, 1, 3
                            ).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(
                                image_feature, image_sizes[image_idx]
                            )
                            c, h, w = image_feature.shape
                            times = math.sqrt(h * w / (max_num_patches * unit**2))
                            if times > 1.1:
                                image_feature = image_feature[None]
                                image_feature = nn.functional.interpolate(
                                    image_feature,
                                    [int(h // times), int(w // times)],
                                    mode='bilinear',
                                )[0]
                            image_feature = torch.cat(
                                (
                                    image_feature,
                                    self.model.image_newline[:, None, None]
                                    .expand(*image_feature.shape[:-1], 1)
                                    .to(image_feature.device),
                                ),
                                dim=-1,
                            )
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        elif 'unpad' in mm_patch_merge_type:
                            image_feature = image_feature.permute(
                                4, 0, 2, 1, 3
                            ).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(
                                image_feature, image_sizes[image_idx]
                            )
                            image_feature = torch.cat(
                                (
                                    image_feature,
                                    self.model.image_newline[:, None, None]
                                    .expand(*image_feature.shape[:-1], 1)
                                    .to(image_feature.device),
                                ),
                                dim=-1,
                            )
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(
                                0, 2, 1, 3, 4
                            ).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        if 'nobase' in mm_patch_merge_type:
                            pass
                        else:
                            image_feature = torch.cat(
                                (base_image_feature, image_feature), dim=0
                            )
                        new_image_features.append(image_feature)
                    else:  # single image operations
                        image_feature = image_feature[0]
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = torch.cat(
                                (image_feature, self.model.image_newline[None]), dim=0
                            )

                        new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(
                    f'Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}'
                )
        else:
            image_features = self.encode_images(images)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(
            self.config, 'mm_use_im_start_end', False
        ):
            raise NotImplementedError
        # rank_print(f"Total images : {len(image_features)}")

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(
                0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
            )
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        # _input_ids = input_ids
        input_ids = [
            cur_input_ids[cur_attention_mask]
            for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)
        ]
        labels = [
            cur_labels[cur_attention_mask]
            for cur_labels, cur_attention_mask in zip(labels, attention_mask)
        ]

        new_input_embeds = []
        new_labels = []
        if (
            self.pruning_paras.get('HOLITOM_k', None) is not None
            and self.pruning_paras.get('HOLITOM_r', None) is not None
        ):
            # [modified]
            image_token_posi = []
            prompt_len = []
        cur_image_idx = 0
        # rank_print("Inserting Images embedding")
        for batch_idx, cur_input_ids in enumerate(input_ids):
            if (
                self.pruning_paras.get('HOLITOM_k', None) is not None
                and self.pruning_paras.get('HOLITOM_r', None) is not None
            ):
                # [modified]
                # record image position for further dropping
                image_index = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[
                    0
                ].tolist()
                if image_index == []:
                    image_token_posi.append(-1)
                else:
                    image_token_posi.append(image_index[0])

                # record input instruction length in inference mode
                if not self.training:
                    if image_index == []:
                        prompt_len.append(cur_input_ids.shape[0])
                    else:
                        prompt_len.append(
                            cur_input_ids.shape[0] - 1
                        )  # consider image place holder

            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            # rank0_print(num_images)
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat(
                    [cur_input_embeds_1, cur_image_features[0:0]], dim=0
                )
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = (
                [-1]
                + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist()
                + [cur_input_ids.shape[0]]
            )
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(
                    cur_input_ids[
                        image_token_indices[i] + 1: image_token_indices[i + 1]
                    ]
                )
                cur_labels_noim.append(
                    cur_labels[image_token_indices[i] + 1: image_token_indices[i + 1]]
                )
            # [modify]
            # text_token_count = sum([x.shape[0] for x in cur_labels_noim])
            # vision_token_count = len(image_features[cur_image_idx])
            # rank0_print(f"Batch {batch_idx}:
            # Text tokens: {text_token_count} Original Vision tokens: {vision_token_count}")

            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(
                torch.cat(cur_input_ids_noim)
            )
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    try:
                        cur_image_features = image_features[cur_image_idx]
                    except IndexError:
                        cur_image_features = image_features[cur_image_idx - 1]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(
                        torch.full(
                            (cur_image_features.shape[0],),
                            IGNORE_INDEX,
                            device=cur_labels.device,
                            dtype=cur_labels.dtype,
                        )
                    )

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            # import pdb; pdb.set_trace()
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        if (
            self.pruning_paras.get('HOLITOM_k', None) is not None
            and self.pruning_paras.get('HOLITOM_r', None) is not None
        ):
            # [modified]
            self.model.image_token_posi = image_token_posi
            self.model.prompt_len = prompt_len
            self.model.image_tokens = [
                image_feature.shape[0] for image_feature in image_features
            ]

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(
            self.config, 'tokenizer_model_max_length', None
        )
        # rank_print("Finishing Inserting")

        new_input_embeds = [
            x[:tokenizer_model_max_length]
            for x, modality in zip(new_input_embeds, modalities)
        ]
        new_labels = [
            x[:tokenizer_model_max_length]
            for x, modality in zip(new_labels, modalities)
        ]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full(
            (batch_size, max_len),
            IGNORE_INDEX,
            dtype=new_labels[0].dtype,
            device=new_labels[0].device,
        )
        attention_mask = torch.zeros(
            (batch_size, max_len),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        position_ids = torch.zeros(
            (batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device
        )
        # rank0_print("Prepare pos id")

        for i, (cur_new_embed, cur_new_labels) in enumerate(
            zip(new_input_embeds, new_labels)
        ):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == 'left':
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                            cur_new_embed,
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(
                        0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                    )
            else:
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            cur_new_embed,
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(
                        0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                    )

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
        # rank0_print("tokenizer padding")

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None
        if getattr(self.config, 'use_pos_skipping', False) and self.training:
            position_ids = (
                torch.arange(new_input_embeds.size(1), device=new_input_embeds.device)
                .unsqueeze(0)
                .to(new_input_embeds.device)
            )
            split_position = random.randint(0, new_input_embeds.size(1))
            left_add = random.randint(0, self.config.pos_skipping_range)
            right_add = random.randint(left_add, self.config.pos_skipping_range)
            position_ids[:, :split_position] += left_add
            position_ids[:, split_position:] += right_add
        # import pdb; pdb.set_trace()
        # rank0_print("Finish preparing")
        return (
            None,
            position_ids,
            attention_mask,
            past_key_values,
            new_input_embeds,
            new_labels,
        )


@TOKEN_REDUCTION_REGISTRY.register('HoliTom')
class HoliTom(TokenReductionModule):
    def __init__(self, config, model, blocks):
        super().__init__(config, model, blocks)
        self.add_sparse_config()
        self.register_reduction_modules()

    def add_sparse_config(self):
        special_config = self.config.get('special', {})
        self.model.model.pruning_paras = special_config
        self.model.model.model.pruning_paras = special_config

        if self.model.__class__.__name__ == 'Llava_OneVision':
            SigLipEncoder.forward = SigLipEncoder_forward
            SigLipVisionTower.forward = SigLipVisionTower_forward

            LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal = (
                LlavaMetaForCausalLM_holitom.prepare_inputs_labels_for_multimodal
            )
            LlavaMetaForCausalLM.encode_images = (
                LlavaMetaForCausalLM_holitom.encode_images
            )
            LlavaMetaForCausalLM.encode_images_multi = (
                LlavaMetaForCausalLM_holitom.encode_images_multi
            )

            LlavaMetaForCausalLM.holitom = LlavaMetaForCausalLM_holitom.holitom
            LlavaMetaForCausalLM.cluster_dpc_knn = (
                LlavaMetaForCausalLM_holitom.cluster_dpc_knn
            )
            LlavaMetaForCausalLM.select_static_windows = (
                LlavaMetaForCausalLM_holitom.select_static_windows
            )
            LlavaMetaForCausalLM.get_static_dynamic_features = (
                LlavaMetaForCausalLM_holitom.get_static_dynamic_features
            )
            LlavaMetaForCausalLM.merge_tokens_by_attention_density = (
                LlavaMetaForCausalLM_holitom.merge_tokens_by_attention_density
            )
            LlavaMetaForCausalLM.merge_tokens_by_density = (
                LlavaMetaForCausalLM_holitom.merge_tokens_by_density
            )
            LlavaMetaForCausalLM.merge_tokens_by_clustering = (
                LlavaMetaForCausalLM_holitom.merge_tokens_by_clustering
            )
            LlavaMetaForCausalLM.add_newline_token = (
                LlavaMetaForCausalLM_holitom.add_newline_token
            )

            if (
                self.special_config.get('HOLITOM_k', None) is not None
                and self.special_config.get('HOLITOM_r', None) is not None
            ):
                from functools import partial

                from transformers.cache_utils import Cache, DynamicCache
                from transformers.modeling_flash_attention_utils import \
                    FlashAttentionKwargs
                from transformers.modeling_outputs import \
                    BaseModelOutputWithPast
                from transformers.processing_utils import Unpack

                def qwen_forward(
                    self,
                    input_ids: Optional[torch.LongTensor] = None,
                    attention_mask: Optional[torch.Tensor] = None,
                    position_ids: Optional[torch.LongTensor] = None,
                    past_key_values: Optional[Cache] = None,
                    inputs_embeds: Optional[torch.FloatTensor] = None,
                    use_cache: Optional[bool] = None,
                    output_attentions: Optional[bool] = None,
                    output_hidden_states: Optional[bool] = None,
                    cache_position: Optional[torch.LongTensor] = None,
                    **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
                ) -> BaseModelOutputWithPast:
                    output_attentions = (
                        output_attentions
                        if output_attentions is not None
                        else self.config.output_attentions
                    )
                    output_hidden_states = (
                        output_hidden_states
                        if output_hidden_states is not None
                        else self.config.output_hidden_states
                    )
                    use_cache = (
                        use_cache if use_cache is not None else self.config.use_cache
                    )

                    if (input_ids is None) ^ (inputs_embeds is not None):
                        raise ValueError(
                            'You must specify exactly one of input_ids or inputs_embeds'
                        )

                    if self.gradient_checkpointing and self.training and use_cache:
                        logger.warning_once(
                            '`use_cache=True` is incompatible with gradient checkpointing.' +
                            'Setting `use_cache=False`.'
                        )
                        use_cache = False

                    # TODO (joao): remove this exception in v4.56 --
                    # it exists for users that try to pass a legacy cache
                    if not isinstance(past_key_values, (type(None), Cache)):
                        raise ValueError(
                            'The `past_key_values` should be either a `Cache` object or `None`.'
                        )

                    if inputs_embeds is None:
                        inputs_embeds = self.embed_tokens(input_ids)

                    if use_cache and past_key_values is None:
                        past_key_values = DynamicCache()

                    if cache_position is None:
                        past_seen_tokens = (
                            past_key_values.get_seq_length()
                            if past_key_values is not None
                            else 0
                        )
                        cache_position = torch.arange(
                            past_seen_tokens,
                            past_seen_tokens + inputs_embeds.shape[1],
                            device=inputs_embeds.device,
                        )

                    if position_ids is None:
                        position_ids = cache_position.unsqueeze(0)

                    causal_mask = self._update_causal_mask(
                        attention_mask,
                        inputs_embeds,
                        cache_position,
                        past_key_values,
                        output_attentions,
                    )

                    hidden_states = inputs_embeds

                    # create position embeddings to be shared across the decoder layers
                    position_embeddings = self.rotary_emb(hidden_states, position_ids)

                    # decoder layers
                    all_hidden_states = () if output_hidden_states else None
                    all_self_attns = () if output_attentions else None

                    HOLITOM_k = self.pruning_paras.get('HOLITOM_k', 3)
                    HOLITOM_r = self.pruning_paras.get('HOLITOM_r', 0.5)
                    HOLITOM_image_token_start_index = self.image_token_posi[0]
                    HOLITOM_image_token_length = self.image_tokens[0]
                    seq_length_with_past = past_seen_tokens + inputs_embeds.shape[1]

                    for layer_idx, decoder_layer in enumerate(
                        self.layers[: self.config.num_hidden_layers]
                    ):
                        if output_hidden_states:
                            all_hidden_states += (hidden_states,)

                        if self.gradient_checkpointing and self.training:
                            layer_outputs = self._gradient_checkpointing_func(
                                partial(decoder_layer.__call__, **flash_attn_kwargs),
                                hidden_states,
                                causal_mask,
                                position_ids,
                                past_key_values,
                                output_attentions,
                                use_cache,
                                cache_position,
                                position_embeddings,
                            )
                        else:
                            if layer_idx < HOLITOM_k:
                                pass
                            elif layer_idx == HOLITOM_k and position_ids.size(1) > 1:
                                # compute pruned tokens, generate fastv sign
                                last_layer_attention = layer_outputs[1]
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
                                        HOLITOM_image_token_start_index:
                                        HOLITOM_image_token_start_index
                                        + HOLITOM_image_token_length
                                    ]
                                # get the indexes of the top ATTENTION_RANK tokens
                                top_attention_rank_index = (
                                    last_layer_attention_avg_last_tok_image.topk(
                                        round(
                                            HOLITOM_image_token_length * (1 - HOLITOM_r)
                                        )
                                    ).indices
                                    + HOLITOM_image_token_start_index
                                )
                                # print("Before merge:", HOLITOM_image_token_length, "After merge:",
                                # round(HOLITOM_image_token_length*(1-HOLITOM_r)))

                                device = hidden_states.device
                                # [modified]
                                all_indices = torch.arange(
                                    HOLITOM_image_token_length, device=device
                                )
                                non_topk_mask = ~torch.isin(
                                    all_indices,
                                    top_attention_rank_index
                                    - HOLITOM_image_token_start_index,
                                )
                                non_topk_indices = (
                                    all_indices[non_topk_mask]
                                    + HOLITOM_image_token_start_index
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

                                for b in range(hidden_states.size(0)):
                                    for i in range(len(non_topk_indices)):
                                        non_topk_idx = non_topk_indices[i]
                                        most_similar_topk_idx = (
                                            top_attention_rank_index[
                                                sim_max_index[b, i]
                                            ]
                                        )
                                        hidden_states[b, most_similar_topk_idx, :] = (
                                            hidden_states[b, most_similar_topk_idx, :]
                                            + hidden_states[b, non_topk_idx, :]
                                        ) / 2
                                # [modified]

                                # keep index
                                keep_indexes = torch.cat(
                                    (
                                        torch.arange(
                                            HOLITOM_image_token_start_index,
                                            device=device,
                                        ),
                                        top_attention_rank_index,
                                        torch.arange(
                                            HOLITOM_image_token_start_index
                                            + HOLITOM_image_token_length,
                                            seq_length_with_past,
                                            device=device,
                                        ),
                                    )
                                )
                                # sort index
                                keep_indexes = keep_indexes.sort().values
                                # update seq length
                                new_seq_length = keep_indexes.shape[0]
                                # filter hidden states

                                hidden_states = hidden_states[
                                    :, keep_indexes, :
                                ]
                                # lead the cuda error in the
                                # second iteration of decoding layeridx 3
                                # update position ids
                                position_ids = keep_indexes.unsqueeze(0)

                                position_embeddings = self.rotary_emb(
                                    hidden_states, position_ids
                                )

                                cache_position = cache_position[:new_seq_length]

                            if layer_idx == HOLITOM_k - 1:
                                output_attentions = True
                            else:
                                output_attentions = False

                            layer_outputs = decoder_layer(
                                hidden_states,
                                attention_mask=causal_mask,
                                position_ids=position_ids,
                                past_key_value=past_key_values,
                                output_attentions=output_attentions,
                                use_cache=use_cache,
                                cache_position=cache_position,
                                position_embeddings=position_embeddings,
                                **flash_attn_kwargs,
                            )

                        hidden_states = layer_outputs[0]

                        # if output_attentions:
                        #     all_self_attns += (layer_outputs[1],)

                    hidden_states = self.norm(hidden_states)

                    # add hidden states from the last decoder layer
                    if output_hidden_states:
                        all_hidden_states += (hidden_states,)

                    return BaseModelOutputWithPast(
                        last_hidden_state=hidden_states,
                        past_key_values=past_key_values if use_cache else None,
                        hidden_states=all_hidden_states,
                        attentions=all_self_attns,
                    )

                from transformers.models.qwen2.modeling_qwen2 import Qwen2Model

                Qwen2Model.forward = qwen_forward

    def register_reduction_modules(self):
        pass
