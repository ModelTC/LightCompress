import functools
import math
from functools import wraps
from types import MethodType
from typing import Optional, Tuple

import torch
import torch.nn as nn

from llmc.utils.registry_factory import TOKEN_REDUCTION_REGISTRY

from .token_reduction_module import TokenReductionModule
from .utils import get_anyres_image_grid_shape, prefill_wrapper, unpad_image


@TOKEN_REDUCTION_REGISTRY.register('VisPruner')
class VisPruner(TokenReductionModule):
    def __init__(self, config, model, blocks):
        super().__init__(config, model, blocks)
        self.add_sparse_config()
        self.register_reduction_modules()

    def add_sparse_config(self):
        self.special_config['select_layer'] = self.model.pruning_config.get(
            'select_layer', -1
        )
        self.special_config['select_feature'] = self.model.pruning_config.get(
            'select_feature', None
        )

        self.pruning_paras = self.special_config

    def register_reduction_modules(self):

        def change_images_hook(fn, pruning_paras):
            @wraps(fn)
            def wrapper(self, *args, **kwargs):
                images = args[5]
                input_ids = args[0]
                vision_tower = self.get_vision_tower()

                if vision_tower is None or images is None or input_ids.shape[1] == 1:
                    return fn(*args, **kwargs)

                if images.ndim == 5:
                    args = list(args)
                    concat_images = torch.cat([image for image in images], dim=0)
                    args[5] = concat_images.unsqueeze(dim=0).unsqueeze(dim=0)
                    pruning_paras['image_sizes'] = kwargs['image_sizes']
                    pruning_paras['num_patches_per_side'] = vision_tower.num_patches_per_side
                    if hasattr(vision_tower, 'image_size'):
                        pruning_paras['vision_tower_image_size'] = vision_tower.image_size
                    else:
                        pruning_paras['vision_tower_image_size'] = None
                    pruning_paras['image_newline'] = self.model.image_newline

                    return fn(*tuple(args), **kwargs)
                else:
                    return fn(*args, **kwargs)
            return wrapper

        def update_output_attentions_hook(module, args, kwargs):
            args = list(args)
            if args[0].ndim == 6:
                args[0] = args[0].squeeze(dim=0).squeeze(dim=0)
            kwargs['output_attentions'] = True
            return tuple(args), kwargs

        def store_attention_hook(module, inps, outs, pruning_paras):
            image_attentions = outs.attentions[pruning_paras['select_layer']]
            if pruning_paras['select_feature'] == 'patch':
                image_attentions = image_attentions[:, :, 0, 1:]
            elif pruning_paras['select_feature'] == 'cls_patch':
                image_attentions = image_attentions
                raise ValueError(f"Unexpected select feature: {pruning_paras['select_feature']}")

            pruning_paras['image_attentions'] = image_attentions.to(inps[0].dtype)

        def get_index_masks_hook(module, args, pruning_paras):
            image_features = args[0]
            image_attentions = pruning_paras['image_attentions']

            B, N, C = image_features.shape
            device = image_features.device
            index_masks = torch.ones(B, N, dtype=torch.bool, device=device)
            visual_token_num = round(N * (1 - self.special_config['prune_ratio']))  # T
            important_ratio = self.pruning_paras['important_ratio']  # r
            important_token_num = int(visual_token_num * important_ratio)  # T_imp = T * r
            diverse_token_num = visual_token_num - important_token_num  # T_div = T * (1 - r)

            # [VisPruner] Select important tokens using attention scores
            image_attentions = image_attentions.mean(dim=1)  # (B, N)
            token_indices = image_attentions.argsort(dim=-1, descending=True)  # (B, N)
            important_indices = token_indices[:, :important_token_num]  # (B, T_imp)
            residual_indices = token_indices[:, important_token_num:]  # (B, N - T_imp)

            # [VisPruner] Remove duplicate tokens by iterative matching and pruning
            image_normalized = image_features / image_features.norm(dim=-1, keepdim=True)
            while diverse_token_num > 0:
                R = residual_indices.shape[1]
                r = min(8, R - diverse_token_num)
                if r <= 0:
                    break

                residual_tokens = image_normalized[
                    torch.arange(B).unsqueeze(-1).expand(-1, R),
                    residual_indices
                ]  # (B, R, C)
                a, b = residual_tokens[..., ::2, :], residual_tokens[..., 1::2, :]  # (B, R // 2, C)
                scores = a @ b.transpose(-1, -2)  # (B, R // 2, R // 2)
                scores = scores.max(dim=-1).values  # (B, R // 2)

                distinct_indices = scores.argsort(dim=-1, descending=True)[:, r:]  # (B, R // 2 - r)
                residual_indices = torch.cat([
                    residual_indices[..., ::2][
                        torch.arange(B).unsqueeze(-1).expand(-1, R // 2 - r),
                        distinct_indices
                    ],
                    residual_indices[..., 1::2]
                ], dim=-1)  # (B, R - r)

            if diverse_token_num > 0:
                selected_indices = torch.cat([important_indices, residual_indices], dim=-1)
            else:
                selected_indices = important_indices  # (B, T)
            index_masks = torch.zeros(B, N, dtype=torch.bool, device=device)
            index_masks.scatter_(1, selected_indices, True)

            pruning_paras['index_masks'] = index_masks

        def prune_hook(module, inputs, outputs, pruning_paras, model_config):
            image_features = outputs
            index_masks = pruning_paras['index_masks']

            if image_features.shape[0] == 1:
                return image_features[index_masks].unsqueeze(0)

            image_sizes = pruning_paras['image_sizes']
            split_sizes = [image_features.shape[0]]
            image_features = torch.split(image_features, split_sizes, dim=0)
            index_masks = torch.split(index_masks, split_sizes, dim=0)
            # 'spatial_unpad', 'anyres'
            mm_patch_merge_type = getattr(model_config, 'mm_patch_merge_type', 'flat')
            # mm_patch_merge_type = mm_patch_merge_type.replace('_unpad', '')
            image_aspect_ratio = getattr(model_config, 'image_aspect_ratio', 'square')

            if mm_patch_merge_type == 'flat':
                image_features = [x.flatten(0, 1) for x in image_features]
                index_masks = [x.flatten(0, 1) for x in index_masks]
                image_features = [x[m] for x, m in zip(image_features, index_masks)]
            elif mm_patch_merge_type.startswith('spatial'):
                new_image_features = []
                for image_idx, (image_feature, index_mask) in enumerate(
                    zip(image_features, index_masks)
                ):
                    if image_feature.shape[0] > 1:
                        base_image_feature, base_index_mask = image_feature[0], index_mask[0]
                        image_feature, index_mask = image_feature[1:], index_mask[1:]
                        height = width = pruning_paras['num_patches_per_side']
                        assert height * width == base_image_feature.shape[0]

                        if image_aspect_ratio == 'anyres':
                            if pruning_paras['vision_tower_image_size'] is not None:
                                vision_tower_image_size = pruning_paras['vision_tower_image_size']
                            else:
                                raise ValueError(
                                    'vision_tower_image_size is not found in the vision tower.'
                                )
                            try:
                                num_patch_width, num_patch_height = get_anyres_image_grid_shape(
                                    image_sizes[image_idx],
                                    model_config.image_grid_pinpoints,
                                    vision_tower_image_size
                                )
                            except Exception:
                                num_patch_width, num_patch_height = 2, 2
                            image_feature = image_feature.view(
                                num_patch_height, num_patch_width, height, width, -1
                            )
                            index_mask = index_mask.view(
                                num_patch_height, num_patch_width, height, width
                            )
                        else:
                            raise NotImplementedError

                        if 'maxpool2x2' in mm_patch_merge_type:
                            raise NotImplementedError
                        elif 'unpad' in mm_patch_merge_type and 'anyres_max' in image_aspect_ratio:
                            raise NotImplementedError
                        elif 'unpad' in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((
                                image_feature,
                                pruning_paras['image_newline'][:, None, None].expand(
                                    *image_feature.shape[:-1], 1
                                ).to(image_feature.device)
                            ), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                            index_mask = index_mask.permute(0, 2, 1, 3).contiguous().unsqueeze(0)
                            index_mask = index_mask.flatten(1, 2).flatten(2, 3)
                            index_mask = unpad_image(index_mask, image_sizes[image_idx])
                            index_mask = torch.cat((
                                index_mask,
                                torch.ones(
                                    *index_mask.shape[:-1], 1, dtype=torch.bool
                                ).to(index_mask.device)
                            ), dim=-1)
                            index_mask = index_mask.flatten(1, 2).squeeze(0)
                            image_feature = image_feature[index_mask]
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                            index_mask = index_mask.permute(0, 2, 1, 3).contiguous()
                            index_mask = index_mask.flatten(0, 3)
                            image_feature = image_feature[index_mask]
                        if 'nobase' in mm_patch_merge_type:
                            raise NotImplementedError
                        else:
                            base_image_feature = base_image_feature[base_index_mask]
                            image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                    else:  # single image operations
                        image_feature = image_feature[0]
                        index_mask = index_mask[0]
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = torch.cat((
                                image_feature,
                                pruning_paras['image_newline'][None]
                            ), dim=0)
                            index_mask = torch.cat((
                                index_mask,
                                torch.ones(1, dtype=torch.bool).to(index_mask.device)
                            ), dim=0)
                        image_feature = image_feature[index_mask]
                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(
                    f'Unexpected mm_patch_merge_type: {model_config.mm_patch_merge_type}'
                )
            return image_features

        if self.model.__class__.__name__ != 'Qwen2_5VL':
            self.model.vlm_model.prepare_inputs_labels_for_multimodal = MethodType(
                change_images_hook(
                    self.model.vlm_model.prepare_inputs_labels_for_multimodal,
                    self.pruning_paras
                ),
                self.model.vlm_model
            )

            self.model.vision_model.vision_tower.register_forward_pre_hook(
                update_output_attentions_hook,
                with_kwargs=True
            )

            self.model.vision_model.vision_tower.register_forward_hook(
                functools.partial(store_attention_hook, pruning_paras=self.pruning_paras),
            )

            self.model.vision_projector.register_forward_pre_hook(
                functools.partial(get_index_masks_hook, pruning_paras=self.pruning_paras),
            )

            self.model.vision_projector.register_forward_hook(
                functools.partial(
                    prune_hook,
                    pruning_paras=self.pruning_paras,
                    model_config=self.model.vlm_model_config
                ),
            )

        def get_metric(fn, pruning_paras):
            @wraps(fn)
            def wrapper(self, *args, **kwargs):
                return fn(self, *args, pruning_paras=pruning_paras, **kwargs)
            return wrapper

        def merger_hook(module, inputs, kwargs, layer_outs, pruning_paras):
            with torch.no_grad():
                attn_mean = pruning_paras['attn_logits'].mean(dim=0)  # 16 1120, 1120 -> 1120, 1120
                window_index, _ = module.get_window_index(kwargs['grid_thw'])
                reverse_indices = torch.argsort(window_index)  # [280]
                attn_mean = attn_mean.sum(dim=0)  # 1120, 1120 -> 1120
                attn_mean = attn_mean.view(attn_mean.shape[0] // 4, -1).mean(dim=-1)  # 1120 -> 280
                attn_mean = attn_mean[reverse_indices]
                pruning_paras['attn_mean'] = attn_mean
            return layer_outs

        @prefill_wrapper
        def get_input_ids_hook(module, input_args, pruning_paras):
            pruning_paras['input_ids'] = input_args[0]
            return input_args

        def prune_qwenv25vl_hook(module, args, kwargs, pruning_paras):
            # only support bs=1
            if kwargs['position_ids'].shape[-1] == 1:
                return args, kwargs
            inputs_embeds = kwargs['inputs_embeds']

            img_mask = (pruning_paras['input_ids'] == pruning_paras['vision_token_index'])[0]
            img_idx = torch.nonzero(img_mask, as_tuple=True)[0]  # [280]
            image_features = inputs_embeds[:, img_idx, :]

            B, N, C = image_features.shape
            device = image_features.device
            visual_token_num = round(N * (1 - self.special_config['prune_ratio']))  # T
            important_ratio = self.pruning_paras['important_ratio']  # r
            important_token_num = int(visual_token_num * important_ratio)  # T_imp = T * r
            if (N - important_token_num) % 2 != 0:
                important_token_num += 1
            diverse_token_num = visual_token_num - important_token_num  # T_div = T * (1 - r)

            # [VisPruner] Select important tokens using attention scores
            image_attentions = pruning_paras['attn_mean'].unsqueeze(0)  # (B, N)
            token_indices = image_attentions.argsort(dim=-1, descending=True)  # (B, N)
            important_indices = token_indices[:, :important_token_num]  # (B, T_imp)
            residual_indices = token_indices[:, important_token_num:]  # (B, N - T_imp)

            # [VisPruner] Remove duplicate tokens by iterative matching and pruning
            image_normalized = image_features / image_features.norm(dim=-1, keepdim=True)
            while diverse_token_num > 0:
                R = residual_indices.shape[1]
                r = min(8, R - diverse_token_num)
                if r <= 0:
                    break

                residual_tokens = image_normalized[
                    torch.arange(B).unsqueeze(-1).expand(-1, R),
                    residual_indices
                ]  # (B, R, C)
                a, b = residual_tokens[..., ::2, :], residual_tokens[..., 1::2, :]  # (B, R // 2, C)
                scores = a @ b.transpose(-1, -2)  # (B, R // 2, R // 2)
                scores = scores.max(dim=-1).values  # (B, R // 2)

                distinct_indices = scores.argsort(dim=-1, descending=True)[:, r:]  # (B, R // 2 - r)
                residual_indices = torch.cat([
                    residual_indices[..., ::2][
                        torch.arange(B).unsqueeze(-1).expand(-1, R // 2 - r),
                        distinct_indices
                    ],
                    residual_indices[..., 1::2]
                ], dim=-1)  # (B, R - r)

            if diverse_token_num > 0:
                selected_indices = torch.cat([important_indices, residual_indices], dim=-1)
            else:
                selected_indices = important_indices  # (B, T)
            index_masks = torch.zeros(B, N, dtype=torch.bool, device=device)
            index_masks.scatter_(1, selected_indices, True)
            if img_idx.numel() > 0:
                first, last = img_idx[0].item(), img_idx[-1].item()
                img_mask[first: last + 1] = ~index_masks[0]
                img_mask = ~img_mask

            kwargs['position_ids'] = kwargs['position_ids'][:, :, img_mask]
            kwargs['attention_mask'] = kwargs['attention_mask'][:, img_mask]
            kwargs['inputs_embeds'] = inputs_embeds[:, img_mask]

            return args, kwargs

        if self.model.__class__.__name__ == 'Qwen2_5VL':
            self.blocks[-1].attn.forward = MethodType(
                get_metric(Qwen2_5_VLVisionAttention_forward, self.pruning_paras),
                self.blocks[-1].attn
            )
            self.model.vision_model.register_forward_hook(
                functools.partial(
                    merger_hook,
                    pruning_paras=self.pruning_paras,
                ),
                with_kwargs=True
            )
            self.model.embed_tokens.register_forward_pre_hook(
                functools.partial(get_input_ids_hook, pruning_paras=self.pruning_paras)
            )
            self.model.language_model.register_forward_pre_hook(
                functools.partial(
                    prune_qwenv25vl_hook,
                    pruning_paras=self.pruning_paras,
                ),
                with_kwargs=True
            )


def Qwen2_5_VLVisionAttention_forward(
    self,
    hidden_states: torch.Tensor,
    pruning_paras,
    cu_seqlens: torch.Tensor,
    rotary_pos_emb: Optional[torch.Tensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import \
        apply_rotary_pos_emb_vision
    head_dim = self.qkv.in_features // self.num_heads
    seq_length = hidden_states.shape[0]
    q, k, v = self.qkv(hidden_states).reshape(
        seq_length, 3, self.num_heads, -1
    ).permute(1, 0, 2, 3).unbind(0)
    if position_embeddings is None:
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
    else:
        cos, sin = position_embeddings
    q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

    attention_mask = torch.full(
        [1, seq_length, seq_length], torch.finfo(q.dtype).min, device=q.device, dtype=q.dtype
    )
    for i in range(1, len(cu_seqlens)):
        attention_mask[..., cu_seqlens[i - 1]: cu_seqlens[i], cu_seqlens[i - 1]: cu_seqlens[i]] = 0

    q = q.transpose(0, 1)
    k = k.transpose(0, 1)
    v = v.transpose(0, 1)
    attn_weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(head_dim)
    attn_weights = attn_weights + attention_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
    attn_output = torch.matmul(attn_weights, v)
    attn_output = attn_output.transpose(0, 1)
    attn_output = attn_output.reshape(seq_length, -1)
    attn_output = self.proj(attn_output)
    pruning_paras['attn_logits'] = attn_weights
    return attn_output
