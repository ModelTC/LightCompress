import ast
import re
from functools import wraps
from typing import List, Tuple, Union

import torch
from loguru import logger
from transformers.models.clip.modeling_clip import CLIPEncoderLayer

try:
    from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
except ImportError:
    pass
import random


def prefill_wrapper(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # for the decoding stage
        if len(args) > 1:
            input_args = args[1]
            if hasattr(input_args[0], 'shape') and input_args[0].shape[1] == 1:
                return None
        return func(*args, **kwargs)
    return wrapper


def prefill_wrapper_model(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # for the decoding stage
        if len(args) > 1:
            input_args = args[2]['inputs_embeds']
            if hasattr(input_args, 'shape') and input_args.shape[1] == 1:
                return None
        return func(*args, **kwargs)
    return wrapper


def parse_r(num_layers: int, r: Union[List[int], Tuple[int, float], int]) -> List[int]:
    """Copy from the TOME. https://github.com/facebookresearch/ToMe.

    Process a constant r or r schedule into a list for use internally.

    r can take the following forms:
     - int: A constant number of tokens per layer.
     - Tuple[int, float]: A pair of r, inflection.
       Inflection describes there the the reduction / layer should trend
       upward (+1), downward (-1), or stay constant (0). A value of (r, 0)
       is as providing a constant r. (r, -1) is what we describe in the paper
       as "decreasing schedule". Any value between -1 and +1 is accepted.
     - List[int]: A specific number of tokens per layer. For extreme granularity.
    """
    inflect = 0
    if isinstance(r, list):
        if len(r) < num_layers:
            r = r + [0] * (num_layers - len(r))
        return list(r)
    elif isinstance(r, tuple):
        r, inflect = r

    min_val = int(r * (1.0 - inflect))
    max_val = 2 * r - min_val
    step = (max_val - min_val) / (num_layers - 1)

    return [int(min_val + step * i) for i in range(num_layers)]


def make_tome_class(transformer_class):
    class VisionZipTransformer(transformer_class):
        """
        Modifications:
        - Initialize r
        """
        def forward(self, *args, **kwargs) -> torch.Tensor:
            self._info['r'] = parse_r(len(self.vision_model.encoder.layers), self.r)
            # self._info["r"] = self.r
            return super().forward(*args, **kwargs)

    return VisionZipTransformer


def apply_info(model, dominant_num, contextual_num):

    VisionZipTransformer = make_tome_class(model.__class__)

    model.__class__ = VisionZipTransformer
    model.r = [0 for i in range(22)] + [1] + [0]

    model._info = {
        'r': [model.r],
        'dominant': dominant_num,
        'contextual': contextual_num,
    }
    for module in model.modules():
        if isinstance(module, CLIPEncoderLayer):
            module.self_attn.k_proj._info = model._info


def add_post_hook_to_get_2dPool(model, post_hook_fn, pruning_paras):
    original_fn = model.get_2dPool

    def wrapped_fn(*args, **kwargs):
        result = original_fn(*args, **kwargs)
        return post_hook_fn(result, pruning_paras)

    model.get_2dPool = wrapped_fn


def select_best_resolution(original_size, possible_resolutions):

    original_width, original_height = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float('inf')

    for width, height in possible_resolutions:
        # Calculate the downscaled size to keep the aspect ratio
        scale = min(width / original_width, height / original_height)
        downscaled_width = int(original_width * scale)
        downscaled_height = int(original_height * scale)

        # Calculate effective and wasted resolutions
        effective_resolution = min(
            downscaled_width * downscaled_height,
            original_width * original_height
        )
        wasted_resolution = (width * height) - effective_resolution

        if (effective_resolution > max_effective_resolution) or (
            effective_resolution == max_effective_resolution and
            wasted_resolution < min_wasted_resolution
        ):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height)

    return best_fit


def get_anyres_image_grid_shape(image_size, grid_pinpoints, patch_size):
    """Calculate the shape of the image patch grid after the preprocessing for
    images of any resolution.

    Args:
        image_size (tuple): The size of the input image in the format (width, height).
        grid_pinpoints (str): A string representation of a list of possible resolutions.
        patch_size (int): The size of each image patch.

    Returns:
        tuple: The shape of the image patch grid in the format (width, height).
    """
    if isinstance(grid_pinpoints, str) and 'x' in grid_pinpoints:
        assert patch_size in [224, 336, 384, 448, 512], (
            'patch_size should be in [224, 336, 384, 448, 512]'
        )
        # Use regex to extract the range from the input string
        matches = re.findall(r'\((\d+)x(\d+)\)', grid_pinpoints)
        range_start = tuple(map(int, matches[0]))
        range_end = tuple(map(int, matches[-1]))
        # Generate a matrix of tuples
        # from (range_start[0], range_start[1]) to (range_end[0], range_end[1])
        grid_pinpoints = [
            (i, j)
            for i in range(range_start[0], range_end[0] + 1)
            for j in range(range_start[1], range_end[1] + 1)
        ]
        # Multiply all elements by patch_size
        grid_pinpoints = [[dim * patch_size for dim in pair] for pair in grid_pinpoints]
    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)
    width, height = select_best_resolution(image_size, possible_resolutions)
    return width // patch_size, height // patch_size


def unpad_image(tensor, original_size):
    """Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of the image (height, width).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    # Compute aspect ratios
    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    # Determine padding size and direction
    if original_aspect_ratio > current_aspect_ratio:
        # Padding was added to the height
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding: current_height - padding, :]
    else:
        # Padding was added to the width
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding: current_width - padding]

    return unpadded_tensor


def prepare_inputs_labels_for_multimodal_with_index_masks(
    self, input_ids, position_ids, attention_mask, past_key_values, labels,
    images, modalities=['image'], image_sizes=None
):
    vision_tower = self.get_vision_tower()
    if vision_tower is None or images is None or input_ids.shape[1] == 1:
        return input_ids, position_ids, attention_mask, past_key_values, None, labels, None

    if isinstance(modalities, str):
        modalities = [modalities]

    if type(images) is list or images.ndim == 5:
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
        encoded_image_features = self.encode_images(concat_images)
        index_masks = vision_tower.index_masks
        encoded_image_features = torch.split(encoded_image_features, split_sizes)
        index_masks = torch.split(index_masks, split_sizes)
        image_features = []
        for idx, image_feat in enumerate(encoded_image_features):
            if idx in video_idx_in_batch:
                image_features.append(self.get_2dPool(image_feat))
            else:
                image_features.append(image_feat)
        mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
        # mm_patch_merge_type = mm_patch_merge_type.replace('_unpad', '')
        image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')

        if mm_patch_merge_type == 'flat':
            image_features = [x.flatten(0, 1) for x in image_features]
            index_masks = [x.flatten(0, 1) for x in index_masks]
            image_features = [x[m] for x, m in zip(image_features, index_masks)]
        elif mm_patch_merge_type.startswith('spatial'):
            new_image_features = []
            for image_idx, (image_feature, index_mask) in enumerate(
                zip(image_features, index_masks)
            ):
                if image_idx in video_idx_in_batch:  # video operations
                    raise NotImplementedError
                elif image_feature.shape[0] > 1:

                    base_image_feature, base_index_mask = image_feature[0], index_mask[0]
                    image_feature, index_mask = image_feature[1:], index_mask[1:]
                    height = width = self.get_vision_tower().num_patches_per_side
                    assert height * width == base_image_feature.shape[0]

                    if image_aspect_ratio == 'anyres':
                        if hasattr(self.get_vision_tower(), 'image_size'):
                            vision_tower_image_size = self.get_vision_tower().image_size
                        else:
                            raise ValueError('vision_tower_image_size is not found.')
                        try:
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(
                                image_sizes[image_idx],
                                self.config.image_grid_pinpoints,
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
                        image_feature = torch.cat(
                            (
                                image_feature,
                                self.model.image_newline[
                                    :, None, None
                                ].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
                            ), dim=-1
                        )
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
                        pass
                    else:
                        base_image_feature = base_image_feature[base_index_mask]
                        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                    new_image_features.append(image_feature)
                else:  # single image operations
                    image_feature = image_feature[0]
                    index_mask = index_mask[0]
                    if 'unpad' in mm_patch_merge_type:
                        image_feature = torch.cat((
                            image_feature,
                            self.model.image_newline[None].to(image_feature.device)
                        ), dim=0)
                        index_mask = torch.cat((
                            index_mask,
                            torch.ones(1, dtype=torch.bool).to(index_mask.device)
                        ), dim=0)
                    image_feature = image_feature[index_mask]
                    new_image_features.append(image_feature)
            image_features = new_image_features
        else:
            raise ValueError(f'Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}')
    else:
        image_features = self.encode_images(images)
        image_features = image_features[index_masks].unsqueeze(0)
    vision_tower.index_masks = []
    vtoken_length = image_features[0].shape[0]
    # TODO: image start / end is not implemented here to support pretraining.
    if (
        getattr(self.config, 'tune_mm_mlp_adapter', False) and
        getattr(self.config, 'mm_use_im_start_end', False)
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
            0, input_ids.shape[1],
            dtype=torch.long, device=input_ids.device
        )
    if labels is None:
        labels = torch.full_like(input_ids, IGNORE_INDEX)

    # remove the padding using attention_mask -- FIXME
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
    cur_image_idx = 0
    # rank_print("Inserting Images embedding")
    for batch_idx, cur_input_ids in enumerate(input_ids):
        num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
        # rank0_print(num_images)
        if num_images == 0:
            cur_image_features = image_features[cur_image_idx]
            cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
            cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
            new_input_embeds.append(cur_input_embeds)
            new_labels.append(labels[batch_idx])
            cur_image_idx += 1
            continue

        image_token_indices = [-1] + \
            torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
        cur_input_ids_noim = []
        cur_labels = labels[batch_idx]
        cur_labels_noim = []
        for i in range(len(image_token_indices) - 1):
            cur_input_ids_noim.append(
                cur_input_ids[image_token_indices[i] + 1: image_token_indices[i + 1]]
            )
            cur_labels_noim.append(
                cur_labels[image_token_indices[i] + 1: image_token_indices[i + 1]]
            )
        split_sizes = [x.shape[0] for x in cur_labels_noim]
        cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
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
                        device=cur_labels.device, dtype=cur_labels.dtype
                    )
                )

        cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

        cur_new_input_embeds = torch.cat(cur_new_input_embeds)
        cur_new_labels = torch.cat(cur_new_labels)

        new_input_embeds.append(cur_new_input_embeds)
        new_labels.append(cur_new_labels)

    # Truncate sequences to max length as image embeddings can make the sequence longer
    tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
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
        device=new_labels[0].device
    )
    attention_mask = torch.zeros(
        (batch_size, max_len),
        dtype=attention_mask.dtype,
        device=attention_mask.device
    )
    position_ids = torch.zeros(
        (batch_size, max_len),
        dtype=position_ids.dtype, device=position_ids.device
    )
    # rank0_print("Prepare pos id")

    for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
        cur_len = cur_new_embed.shape[0]
        if getattr(self.config, 'tokenizer_padding_side', 'right') == 'left':
            new_input_embeds_padded.append(
                torch.cat(
                    (
                        torch.zeros(
                            (max_len - cur_len, cur_new_embed.shape[1]),
                            dtype=cur_new_embed.dtype, device=cur_new_embed.device
                        ),
                        cur_new_embed
                    ), dim=0
                )
            )
            if cur_len > 0:
                new_labels_padded[i, -cur_len:] = cur_new_labels
                attention_mask[i, -cur_len:] = True
                position_ids[i, -cur_len:] = torch.arange(
                    0, cur_len,
                    dtype=position_ids.dtype, device=position_ids.device
                )
        else:
            new_input_embeds_padded.append(
                torch.cat(
                    (
                        cur_new_embed,
                        torch.zeros(
                            (max_len - cur_len, cur_new_embed.shape[1]),
                            dtype=cur_new_embed.dtype, device=cur_new_embed.device
                        )
                    ), dim=0
                )
            )
            if cur_len > 0:
                new_labels_padded[i, :cur_len] = cur_new_labels
                attention_mask[i, :cur_len] = True
                position_ids[i, :cur_len] = torch.arange(
                    0, cur_len,
                    dtype=position_ids.dtype, device=position_ids.device
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
        position_ids = torch.arange(
            new_input_embeds.size(1),
            device=new_input_embeds.device
        ).unsqueeze(0).to(new_input_embeds.device)
        split_position = random.randint(0, new_input_embeds.size(1))
        left_add = random.randint(0, self.config.pos_skipping_range)
        right_add = random.randint(left_add, self.config.pos_skipping_range)
        position_ids[:, :split_position] += left_add
        position_ids[:, split_position:] += right_add
    # rank0_print("Finish preparing")
    # print(vtoken_length)
    return None, position_ids, attention_mask, past_key_values, \
        new_input_embeds, new_labels, vtoken_length
