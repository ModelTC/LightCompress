
from functools import wraps


class TokenReductionModule:
    def __init__(self, config, model, blocks):
        self.config = config
        self.model = model
        self.blocks = blocks
        self.set_sparse_config()

    def set_sparse_config(self):
        self.special_config = self.config.get('special', {})
        self.special_config['is_video_model'] = self.model.pruning_config['is_video_model']
        # vision_token can be image or video
        if self.special_config['is_video_model']:
            self.special_config['vision_token_index'] = self.model.pruning_config[
                'video_token_index'
            ]
            self.special_config['vision_token_length'] = self.model.pruning_config[
                'video_token_length'
            ]
        else:
            self.special_config['vision_token_index'] = self.model.pruning_config.get(
                'image_token_index', None
            )
            self.special_config['vision_token_start_index'] = self.model.pruning_config.get(
                'vision_token_start_index', None
            )
            self.special_config['vision_token_length'] = self.model.pruning_config.get(
                'image_token_length', None
            )

    def register_reduction_modules(self):
        pass

    def vtoken_length_for_llava_hook(self, fn, pruning_paras):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            if args[0].shape[1] == 1:
                return fn(*args, **kwargs)

            message = (
                'To obtain the vision_token_length for LLaVA-1.6, you should append '
                '`image_features[0].shape[0]` to the return value of the function '
                '`prepare_inputs_labels_for_multimodal`, and modify the related code accordingly.'
            )
            outs = fn(*args, **kwargs)
            assert len(outs) == 7, message
            pruning_paras['vision_token_length'] = outs[-1]
            return outs
        return wrapper
