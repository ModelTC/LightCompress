import os

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from PIL import Image, ImageDraw

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
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

    if not save_path.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf')):
        os.makedirs(save_path, exist_ok=True)
        base_path = os.path.join(save_path, '{:04d}_visprunerP.png')
        idx = 0
        while os.path.exists(base_path.format(idx)):
            idx += 1
        save_path = base_path.format(idx)

    else:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    img.save(save_path)


def visualize_kept_patches(
    image,
    keep_idx=None,
    mean=[0.48145466, 0.4578275, 0.40821073],
    std=[0.26862954, 0.26130258, 0.27577711],
    patch_size=14,
    darken_ratio=0.8,
    save_path=None,
):
    # save_image(image, mean, std, save_path)
    # return
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

    # white
    prune_mask = ~mask
    white_tensor = torch.ones_like(image)
    masked_image = image * (1 - darken_ratio * prune_mask.float()) + \
        white_tensor * (darken_ratio * prune_mask.float())

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


def visualize_attention(attention, grid_size=24, save_path=None):

    if hasattr(attention, 'detach'):
        attention = attention.detach().cpu().numpy()

    H, W = attention.shape
    new_H = H // grid_size * grid_size
    new_W = W // grid_size * grid_size
    attention = attention[:new_H, :new_W]

    blocks = attention.reshape(new_H // grid_size, grid_size, new_W // grid_size, grid_size)
    block_means = blocks.mean(axis=(1, 3))

    mask = np.triu(np.ones_like(block_means, dtype=bool), k=1)

    plt.figure(figsize=(10, 10))
    sns.heatmap(block_means, mask=mask, cmap='viridis', square=True, cbar=True)

    ticks = np.arange(0, block_means.shape[0], 1)
    labels = ['' for i in ticks]
    plt.xticks(ticks=ticks, labels=labels, rotation=90)
    plt.yticks(ticks=ticks, labels=labels)

    plt.title('Attention Map')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def visualize_attention_v2(attention, grid_size=24, save_path=None):

    if hasattr(attention, 'detach'):
        attention = attention.detach().cpu().numpy()

    # 分区
    block_ranges = []

    # SYS: 2 blocks

    sys_splits = [0, 17, 35]
    for i in range(len(sys_splits) - 1):
        block_ranges.append((sys_splits[i], sys_splits[i + 1]))
    # IMG: 24 blocks of size 24
    for i in range(24):
        start = 35 + i * 24
        end = start + 24
        block_ranges.append((start, end))

    # INS: 6 blocks
    ins_splits = [611 + i * 91 for i in range(7)]  # 611 + 6 * 91 = 1157 → crop to 1155
    ins_splits[-1] = 1155
    for i in range(len(ins_splits) - 1):
        block_ranges.append((ins_splits[i], ins_splits[i + 1]))

    # 对每个 block pair 求平均
    num_blocks = len(block_ranges)
    block_attention = np.zeros((num_blocks, num_blocks))
    for i in range(num_blocks):
        i_start, i_end = block_ranges[i]
        for j in range(num_blocks):
            j_start, j_end = block_ranges[j]
            block = attention[i_start:i_end, j_start:j_end]
            block_attention[31 - i, j] = block.mean()

    mask = np.triu(np.ones_like(block_attention, dtype=bool), k=1)
    plt.figure(figsize=(10, 10))
    block_attention = block_attention / block_attention.max(axis=1, keepdims=True)
    sns.heatmap(block_attention, mask=mask, cmap='viridis', square=True, cbar=True)
    # sns.heatmap(block_attention, cmap='viridis', square=True, cbar=True)

    section_labels = ['SYS', 'IMG', 'INS']
    section_boundaries = [2, 26, 32]  # block_ranges 分别为2个SYS，24个IMG，6个INS
    ticks = np.arange(0, num_blocks)
    plt.xticks(ticks=ticks, labels=[''] * num_blocks)
    plt.yticks(ticks=ticks, labels=[''] * num_blocks)
    plt.xticks(ticks=section_boundaries, labels=section_labels, fontsize=12)
    plt.yticks(ticks=section_boundaries, labels=section_labels, fontsize=12)
    plt.title('Attention Map')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def visualize_cosin_token(token_embedding, save_path=None):

    plt.rcParams['font.size'] = 15

    x = token_embedding[0, 14: 14 + 196 * 4, :]
    x_norm = F.normalize(x, p=2, dim=1)
    similarity_matrix = x_norm @ x_norm.T

    sim_np = similarity_matrix.cpu().numpy()
    sim_np = np.triu(sim_np, k=1)
    valid_sim = sim_np[sim_np > 0]
    vmin = np.percentile(valid_sim, 90)  # 10% min

    plt.subplots(figsize=(10, 10))
    sns.heatmap(similarity_matrix.cpu().numpy(), cmap='Reds', vmin=vmin, vmax=1)

    start = 0
    step = 196
    ticks = np.arange(start, 196 * 5, step)
    plt.xticks(ticks, ticks)
    plt.yticks(ticks, ticks)

    plt.title('')
    plt.xlabel('')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig(save_path, format='pdf')
    plt.rcdefaults()
    plt.close()


def visualize_cosin_token_32p(token_embedding, save_path=None):

    plt.rcParams['font.size'] = 20

    all_tokens = token_embedding[0, 14:14 + 196 * 32, :]
    x_norm = F.normalize(all_tokens, p=2, dim=1)
    similarity_matrix = x_norm @ x_norm.T
    sim_np = similarity_matrix.cpu().numpy()
    sim_np = np.triu(sim_np, k=1)
    valid_sim = sim_np[sim_np > 0]
    vmin = np.percentile(valid_sim, 90)  # 10% min

    group_size = 4
    num_groups = 8
    tokens_per_group = 196 * group_size
    step = 196

    fig, axs = plt.subplots(2, 4, figsize=(22, 10))  # 2x4排布
    axs = axs.flatten()

    for i in range(num_groups):
        x = all_tokens[i * tokens_per_group: (i + 1) * tokens_per_group, :]
        x_norm = F.normalize(x, p=2, dim=1)
        similarity_matrix = x_norm @ x_norm.T

        ax = axs[i]
        sns.heatmap(
            similarity_matrix.cpu().numpy(), cmap='Reds',
            vmin=vmin, vmax=1, ax=ax, cbar=False
        )

        ticks = np.arange(0, tokens_per_group, step)
        labels = np.arange(i * tokens_per_group, (i + 1) * tokens_per_group, step)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(labels, rotation=0)
        ax.set_yticklabels(labels)
        start_frame = i * group_size
        end_frame = (i + 1) * group_size - 1
        ax.set_xlabel(f'Frame {start_frame}-{end_frame}', fontsize=17, labelpad=10)

    plt.tight_layout()
    # plt.savefig(save_path, format='pdf')
    # plt.savefig(save_path.replace('.pdf', '.svg'), format='svg', bbox_inches='tight')
    plt.savefig(save_path, dpi=300)
    plt.rcdefaults()
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
