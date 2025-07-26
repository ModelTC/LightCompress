import functools

import torch

from llmc.utils.registry_factory import TOKEN_REDUCTION_REGISTRY

from .token_reduction_module import TokenReductionModule


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

        def update_output_attentions_hook(module, args, kwargs):
            kwargs['output_attentions'] = True

        def store_attention_hook(module, inps, outs, pruning_paras):
            image_attentions = outs.attentions[pruning_paras['select_layer']]
            if pruning_paras['select_feature'] == 'patch':
                image_attentions = image_attentions[:, :, 0, 1:]
            elif pruning_paras['select_feature'] == 'cls_patch':
                image_attentions = image_attentions
                raise ValueError(f'Unexpected select feature: {self.select_feature}')

            pruning_paras['image_attentions'] = image_attentions.to(inps[0].dtype)

        def get_index_masks_hook(module, args, pruning_paras):
            image_features = args[0]
            image_attentions = pruning_paras['image_attentions']

            B, N, C = image_features.shape
            device = image_features.device
            index_masks = torch.ones(B, N, dtype=torch.bool, device=device)

            visual_token_num = round(
                self.special_config['vision_token_length'] * (
                    1 - self.special_config['prune_ratio']
                )
            )  # T
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

        def prune_hook(module, inputs, outputs, pruning_paras):
            image_features = outputs
            index_masks = pruning_paras['index_masks']
            return image_features[index_masks].unsqueeze(0)

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
            functools.partial(prune_hook, pruning_paras=self.pruning_paras),
        )
