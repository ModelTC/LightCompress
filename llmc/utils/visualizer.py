import os

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from loguru import logger
from PIL import Image, ImageDraw
from torchvision.transforms import ToPILImage

try:
    import matplotlib.pyplot as plt
except Exception:
    logger.warning(
        'Can not import matplotlib. '
        'If you need it, please install.'
    )


def to_pil_image(
    image_tensor,
    mean=[0.48145466, 0.4578275, 0.40821073],
    std=[0.26862954, 0.26130258, 0.27577711]
):
    img = image_tensor.cpu().numpy().transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
    img = img * std + mean
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(img)


def save_image(image_tensor, mean, std, save_path):
    img = to_pil_image(image_tensor)
    Image.fromarray(img).save(save_path)


def visualize_kept_patches(
    image,
    keep_idx,
    mean=[0.48145466, 0.4578275, 0.40821073],
    std=[0.26862954, 0.26130258, 0.27577711],
    patch_size=14,
    darken_ratio=0.3,
    save_path=None,
):
    assert image.ndim == 3 and image.shape[0] == 3, \
        f'Expected image of shape [3, H, W], got {image.shape}'
    # save_image(image,mean,std,save_path)

    _, H, W = image.shape  # 3 336 336
    device = image.device
    num_patches_h = H // patch_size  # 24
    num_patches_w = W // patch_size   # 24
    total_patches = num_patches_h * num_patches_w

    patch_mask = torch.zeros(total_patches, dtype=torch.bool, device=device)
    patch_mask[keep_idx] = True
    patch_mask = patch_mask.view(num_patches_h, num_patches_w)

    mask = patch_mask.repeat_interleave(patch_size, dim=0).repeat_interleave(patch_size, dim=1)
    mask = mask.unsqueeze(0)  # shape [1, H, W]

    # Darken image
    masked_image = image * (mask + (~mask) * darken_ratio)

    save_image(masked_image, mean, std, save_path)


def grid_show(to_shows, cols, save_path=None, dpi=100):
    rows = (len(to_shows) - 1) // cols + 1
    it = iter(to_shows)
    fig, axs = plt.subplots(rows, cols, figsize=(rows * 8.5, cols * 2))
    for i in range(rows):
        for j in range(cols):
            try:
                image, title = next(it)
            except StopIteration:
                image = np.zeros_like(to_shows[0][0])
                title = 'pad'
            axs[i, j].imshow(image)
            axs[i, j].set_title(title)
            axs[i, j].set_yticks([])
            axs[i, j].set_xticks([])
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
    plt.close()

# def visualize_head(att_map):
#     ax = plt.gca()
#     # Plot the heatmap
#     im = ax.imshow(att_map)
#     # Create colorbar
#     cbar = ax.figure.colorbar(im, ax=ax)
#     plt.show()


def visualize_heads(att_map, cols, save_path):
    to_shows = []
    att_map = att_map.squeeze().detach().cpu().numpy()
    for i in range(att_map.shape[0]):
        to_shows.append((att_map[i], f'Head {i}'))
    average_att_map = att_map.mean(axis=0)
    to_shows.append((average_att_map, 'Head Average'))
    grid_show(to_shows, cols=cols, save_path=save_path)


def gray2rgb(image):
    return np.repeat(image[..., np.newaxis], 3, 2)


def cls_padding(image, mask, cls_weight, grid_size):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)

    image = np.array(image)

    H, W = image.shape[:2]
    delta_H = int(H / grid_size[0])
    delta_W = int(W / grid_size[1])

    padding_w = delta_W
    padding_h = H
    padding = np.ones_like(image) * 255
    padding = padding[:padding_h, :padding_w]

    padded_image = np.hstack((padding, image))
    padded_image = Image.fromarray(padded_image)
    draw = ImageDraw.Draw(padded_image)
    draw.text((int(delta_W / 4), int(delta_H / 4)), 'CLS', fill=(0, 0, 0))

    mask = mask / max(np.max(mask), cls_weight)
    cls_weight = cls_weight / max(np.max(mask), cls_weight)

    if len(padding.shape) == 3:
        padding = padding[:, :, 0]
        padding[:, :] = np.min(mask)
    mask_to_pad = np.ones((1, 1)) * cls_weight
    mask_to_pad = Image.fromarray(mask_to_pad)
    mask_to_pad = mask_to_pad.resize((delta_W, delta_H))
    mask_to_pad = np.array(mask_to_pad)

    padding[:delta_H, :delta_W] = mask_to_pad
    padded_mask = np.hstack((padding, mask))
    padded_mask = padded_mask

    meta_mask = np.zeros((padded_mask.shape[0], padded_mask.shape[1], 4))
    meta_mask[delta_H:, 0:delta_W, :] = 1

    return padded_image, padded_mask, meta_mask


def visualize_grid_to_grid_with_cls(
    att_map,
    grid_index,
    image,
    grid_size=14,
    alpha=0.6,
    save_path=None
):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)

    attention_map = att_map[grid_index]
    cls_weight = attention_map[0]

    mask = attention_map[1:].reshape(grid_size[0], grid_size[1])
    mask = Image.fromarray(mask).resize((image.size))

    padded_image, padded_mask, meta_mask = cls_padding(image, mask, cls_weight, grid_size)

    if grid_index != 0:  # adjust grid_index since we pad our image
        grid_index = grid_index + (grid_index - 1) // grid_size[1]

    grid_image = highlight_grid(padded_image, [grid_index], (grid_size[0], grid_size[1] + 1))

    fig, ax = plt.subplots(1, 2, figsize=(10, 7))
    fig.tight_layout()

    ax[0].imshow(grid_image)
    ax[0].axis('off')

    ax[1].imshow(grid_image)
    ax[1].imshow(padded_mask, alpha=alpha, cmap='rainbow')
    ax[1].imshow(meta_mask)
    ax[1].axis('off')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()


def visualize_grid_to_grid(att_map, grid_index, image, grid_size=14, alpha=0.6, save_path=None):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)

    image = to_pil_image(image)
    grid_image = highlight_grid(image, [grid_index], grid_size)

    mask = att_map[grid_index].reshape(grid_size[0], grid_size[1])
    mask = mask.cpu().numpy().astype(np.float32)
    mask = Image.fromarray(mask).resize((image.size))

    fig, ax = plt.subplots(1, 2, figsize=(10, 7))
    fig.tight_layout()

    ax[0].imshow(grid_image)
    ax[0].axis('off')

    ax[1].imshow(grid_image)
    ax[1].imshow(mask / np.max(mask), alpha=alpha, cmap='rainbow')
    ax[1].axis('off')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()


def highlight_grid(image, grid_indexes, grid_size=14):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)

    W, H = image.size
    h = H / grid_size[0]
    w = W / grid_size[1]
    image = image.copy()
    for grid_index in grid_indexes:
        x, y = np.unravel_index(grid_index, (grid_size[0], grid_size[1]))
        a = ImageDraw.ImageDraw(image)
        a.rectangle(
            [(y * w, x * h), (y * w + w, x * h + h)],
            fill=None, outline='red', width=2
        )
    return image
