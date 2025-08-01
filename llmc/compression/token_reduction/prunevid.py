import functools
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from loguru import logger
from torch import einsum

try:
    from llava.model.llava_arch import LlavaMetaForCausalLM
except ImportError:
    pass

from llmc.utils.registry_factory import TOKEN_REDUCTION_REGISTRY

from .token_reduction_module import TokenReductionModule
from .utils import add_post_hook_to_get_2dPool


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
    batch_indices = (
        torch.arange(B, dtype=torch.long)
        .to(device)
        .view(view_shape)
        .repeat(repeat_shape)
    )
    new_points = points[batch_indices, idx, :]
    return new_points


def cluster_dpc_knn(x, cluster_num, k=5, token_mask=None):
    """Cluster tokens with DPC-KNN algorithm.

    Return:
        idx_cluster (Tensor[B, N]): cluster index of each token.
        cluster_num (int): actual cluster number. The same with
            input cluster number
    Args:
        x: input token feature, [B, N, C]
        cluster_num (int): cluster number
        k (int): number of the nearest neighbor used for local density.
        token_mask (Tensor[B, N]): mask indicate the whether the token is
            padded empty token. Non-zero value means the token is meaningful,
            zero value means the token is an empty token. If set to None, all
            tokens are regarded as meaningful.
    """
    with torch.no_grad():
        B, N, C = x.shape

        dist_matrix = torch.cdist(x.float(), x.float()) / (C**0.5)

        if token_mask is not None:
            token_mask = token_mask > 0
            # in order to not affect the local density, the distance between empty tokens
            # and any other tokens should be the maximal distance.
            dist_matrix = dist_matrix * token_mask[:, None, :] + (
                dist_matrix.max() + 1
            ) * (~token_mask[:, None, :])

        # get local density

        dist_nearest, index_nearest = torch.topk(
            dist_matrix, k=k, dim=-1, largest=False
        )
        density = (-(dist_nearest**2).mean(dim=-1)).exp()
        # add a little noise to ensure no tokens have the same density.
        density = (
            density
            + torch.rand(density.shape, device=density.device, dtype=density.dtype)
            * 1e-6
        )

        if token_mask is not None:
            # the density of empty token should be 0
            density = density * token_mask

        # get distance indicator
        mask = density[:, None, :] > density[:, :, None]
        mask = mask.type(x.dtype)
        dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]
        dist, index_parent = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)

        # select clustering center according to score
        score = dist * density
        _, index_down = torch.topk(score, k=cluster_num, dim=-1)

        # # assign tokens to the nearest center
        dist_matrix = index_points(dist_matrix, index_down)

        idx_cluster = dist_matrix.argmin(dim=1)

        # make sure cluster center merge to itself
        idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, cluster_num)
        idx_tmp = torch.arange(cluster_num, device=x.device)[None, :].expand(
            B, cluster_num
        )
        idx_cluster[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)
    return idx_cluster, cluster_num


def refine_clusters(cluster_idx):
    """根据给定的聚类结果，对每个批次进行精炼处理。

    Args:
        cluster_idx: Tensor of shape (B, N)，每个元素是聚类的索引。

    Returns:
        refined_cluster_idx: Tensor of shape (B, N)，精炼后的聚类结果。
    """
    import torch

    B, N = cluster_idx.shape
    refined_cluster_idx = cluster_idx.clone()
    for b in range(B):
        clusters = torch.unique(cluster_idx[b])
        segment_info = {}
        # 步骤1：对于每个 cluster，找到其所有的连续片段
        for cluster_label in clusters:
            indices = (cluster_idx[b] == cluster_label).nonzero(as_tuple=True)[0]
            if indices.numel() == 0:
                continue
            # 找到连续片段
            segments = []
            start = indices[0].item()
            prev = indices[0].item()
            for idx in indices[1:]:
                idx = idx.item()
                if idx == prev + 1:
                    prev = idx
                else:
                    # 新的片段
                    segments.append((start, prev))
                    start = idx
                    prev = idx
            # 添加最后一个片段
            segments.append((start, prev))
            segment_info[cluster_label.item()] = segments

        # 步骤2：保留每个 cluster 中最长的片段，其余片段需要重新归类
        for cluster_label, segments in segment_info.items():
            # 找到最长的片段长度
            max_length = 0
            for start, end in segments:
                length = end - start + 1
                if length > max_length:
                    max_length = length
            # 如果最长的片段长度为1，且只有长度为1的片段，该 cluster 需要移除
            if max_length == 1:
                for start, end in segments:
                    refined_cluster_idx[b, start: end + 1] = -1  # -1表示需要重新归类
                continue
            # 保留最长的片段，重新归类其他片段
            for start, end in segments:
                length = end - start + 1
                if length == max_length:
                    continue  # 保留最长的片段
                else:
                    refined_cluster_idx[b, start: end + 1] = -1  # 需要重新归类

        # 步骤3：对于需要重新归类的片段，按照左右邻居最长的片段的 cluster 进行归类
        idx = 0
        while idx < N:
            if refined_cluster_idx[b, idx] == -1:
                # 找到需要重新归类的片段
                start = idx
                while idx < N and refined_cluster_idx[b, idx] == -1:
                    idx += 1
                end = idx - 1
                # 找到左侧和右侧的邻居 cluster 及其片段长度
                left_cluster_label = None
                left_length = 0
                if start > 0:
                    left_label = refined_cluster_idx[b, start - 1].item()
                    # 左侧片段长度
                    l_idx = start - 1
                    while l_idx >= 0 and refined_cluster_idx[b, l_idx] == left_label:
                        l_idx -= 1
                    left_length = start - l_idx - 1
                    left_cluster_label = left_label
                right_cluster_label = None
                right_length = 0
                if end < N - 1:
                    right_label = refined_cluster_idx[b, end + 1].item()
                    # 右侧片段长度
                    r_idx = end + 1
                    while r_idx < N and refined_cluster_idx[b, r_idx] == right_label:
                        r_idx += 1
                    right_length = r_idx - end - 1
                    right_cluster_label = right_label
                # 选择片段长度较长的邻居 cluster 进行归类，若长度相同，选择左侧
                if left_length > right_length:
                    new_label = left_cluster_label
                elif right_length > left_length:
                    new_label = right_cluster_label
                else:
                    new_label = (
                        left_cluster_label
                        if left_cluster_label is not None
                        else right_cluster_label
                    )
                # 如果左右邻居都不存在，默认归类为 cluster 0
                if new_label is None:
                    new_label = 0
                # 重新归类
                refined_cluster_idx[b, start: end + 1] = new_label
            else:
                idx += 1
    return refined_cluster_idx


def segment_lengths(tensor):
    # 获取设备信息（CPU 或 GPU）
    device = tensor.device
    B, N = tensor.shape

    # 列表用于存储每个视频的段长度
    segment_lengths_list = []
    max_segments = 0  # 记录最大段数

    for i in range(B):
        seq = tensor[i]
        # 计算值发生变化的位置
        change_points = torch.where(seq[1:] != seq[:-1])[0] + 1
        # 包含起始和结束位置
        boundaries = torch.cat(
            [
                torch.tensor([0], device=device),
                change_points,
                torch.tensor([N], device=device),
            ]
        )
        # 计算每个段的长度
        lengths = boundaries[1:] - boundaries[:-1]
        segment_lengths_list.append(lengths)
        max_segments = max(max_segments, lengths.numel())

    # 初始化结果张量，填充为0
    result = torch.zeros((B, max_segments), dtype=torch.long, device=device)
    # 将每个视频的段长度填入结果张量
    for i in range(B):
        lengths = segment_lengths_list[i]
        result[i, : lengths.numel()] = lengths

    return result


def compute_cluster_vectors(image_key_vectors, cluster_key_idx, num_cluster):
    """
    Args:
        image_key_vectors: Tensor of shape (B, L, D), the feature vectors
        cluster_key_idx: Tensor of shape (B, L), cluster indices for each vector
        num_cluster: int, the total number of clusters

    Returns:
        cluster_vectors: Tensor of shape (B, num_cluster, D), the averaged features for each cluster
    """
    # image_key_vectors: (B, L, D)
    # cluster_key_idx: (B, L)
    # num_cluster: integer, number of clusters

    B, L, D = image_key_vectors.shape

    # Step 1: 将cluster_key_idx进行one-hot编码
    # 得到的cluster_key_idx_onehot形状为 (B, L, num_cluster)
    cluster_key_idx_onehot = F.one_hot(cluster_key_idx, num_classes=num_cluster).to(
        dtype=image_key_vectors.dtype
    )

    # Step 2: 计算每个cluster的特征和
    # 首先调整cluster_key_idx_onehot的维度，使其变为 (B, num_cluster, L)
    cluster_key_idx_onehot_t = cluster_key_idx_onehot.permute(0, 2, 1)

    # 然后通过矩阵乘法计算每个cluster的特征和，得到的cluster_sums形状为 (B, num_cluster, D)
    cluster_sums = torch.bmm(cluster_key_idx_onehot_t, image_key_vectors)

    # Step 3: 计算每个cluster的元素数量
    # cluster_counts形状为 (B, num_cluster)
    cluster_counts = cluster_key_idx_onehot.sum(dim=1)

    # Step 4: 计算每个cluster的平均特征
    # 先避免除以0，将cluster_counts中为0的值替换为1
    cluster_counts_nonzero = cluster_counts.clone()
    cluster_counts_nonzero[cluster_counts_nonzero == 0] = 1

    # 计算平均值，结果cluster_features形状为 (B, num_cluster, D)
    cluster_features = cluster_sums / cluster_counts_nonzero.unsqueeze(-1)

    # Step 5: 对于没有元素的cluster，将其特征设置为0
    zero_mask = (cluster_counts == 0).unsqueeze(-1)  # (B, num_cluster, 1)
    cluster_features = cluster_features.masked_fill(zero_mask, 0)

    return cluster_features  # (B, num_cluster, D)


def spatial_merge_tokens(feature, num_cluster, k):
    cluster_idx, _ = cluster_dpc_knn(feature, cluster_num=num_cluster, k=k)
    feature = compute_cluster_vectors(feature, cluster_idx, num_cluster=num_cluster)
    return feature


def merge_frames_dynamic(frames, pruning_paras, k=7):
    # B, L, C = frames.shape
    B = 1
    num_frames, L, C = frames.shape
    threshold = pruning_paras['taus']
    cluster_ratio = pruning_paras['cluster_ratios']
    temporal_segment_ratio = pruning_paras['temporal_segment_ratios']
    frames = frames.view(B, num_frames, L, C)  # B T L C
    idx_clusters, _ = cluster_dpc_knn(
        frames.mean(dim=2), cluster_num=int(num_frames * temporal_segment_ratio), k=k
    )
    idx_clusters = refine_clusters(idx_clusters)
    window_list = segment_lengths(idx_clusters)

    static_features = []
    dynamic_features = []
    static_sizes = []
    dynamic_sizes = []

    start_idx = 0
    for window_size in window_list[0]:  # 假设window_list的形状为(B, S)
        # 获取当前window的帧
        current_frames = frames[:, start_idx: start_idx + window_size, :, :]  # B W L C

        # 计算相似度
        frames_normed = F.normalize(current_frames, p=2, dim=-1)
        frames_sim = einsum('b w l c, b t l c -> b w t l', frames_normed, frames_normed)
        frames_sim = (frames_sim.sum(dim=-2) - 1).sum(dim=-2) / (
            window_size * (window_size - 1)
        )  # B L

        # 创建mask
        mask = frames_sim > threshold
        mask_expand = mask.view(B, 1, L, 1).expand(-1, window_size, -1, C)  # B W L C

        # 处理静态特征
        static_mask = mask_expand
        static_feat = (
            torch.masked_select(current_frames, static_mask)
            .view(B, window_size, -1, C)
            .mean(dim=1)
        )
        if static_feat.shape[1] > 14:
            static_feat = spatial_merge_tokens(
                static_feat, num_cluster=int(static_feat.shape[1] * cluster_ratio), k=7
            )
        static_features.append(static_feat)
        static_sizes.append(static_feat.shape[1])

        # 处理动态特征
        dynamic_mask = ~mask_expand
        dynamic_feat = torch.masked_select(current_frames, dynamic_mask).view(
            B, window_size, -1, C
        )
        dynamic_window_list = []
        for i in range(window_size):
            dynamic_feat_window = dynamic_feat[:, i, :, :]
            if dynamic_feat_window.shape[1] > 14:
                dynamic_feat_window = spatial_merge_tokens(
                    dynamic_feat_window,
                    num_cluster=int(dynamic_feat_window.shape[1] * cluster_ratio),
                    k=7,
                )
            dynamic_window_list.append(dynamic_feat_window)
        dynamic_feat = torch.cat(dynamic_window_list, dim=1)
        # dynamic_feat = torch.masked_select(current_frames, dynamic_mask).view(B, -1, C)

        dynamic_features.append(dynamic_feat)
        dynamic_sizes.append(dynamic_feat.shape[1])

        start_idx += window_size

    # 合并所有特征
    final_features = []
    for static_feature, dynamic_feature in zip(static_features, dynamic_features):
        final_features.append(static_feature)
        final_features.append(dynamic_feature)
    final_features = torch.cat(final_features, dim=1)

    # window_sizes = window_list[0].tolist()  # 转换为列表形式

    return final_features
    # return final_features, static_sizes, dynamic_sizes, window_sizes


@TOKEN_REDUCTION_REGISTRY.register('PruneVid')
class PruneVid(TokenReductionModule):
    def __init__(self, config, model, blocks):
        super().__init__(config, model, blocks)
        self.register_reduction_modules()

    def register_reduction_modules(self):

        if isinstance(self.model.model, LlavaMetaForCausalLM):
            add_post_hook_to_get_2dPool(
                self.model.model, merge_frames_dynamic, self.special_config
            )
