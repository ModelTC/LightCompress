import ast
import re
from functools import wraps
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.models.clip.modeling_clip import CLIPEncoderLayer


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
