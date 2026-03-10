import gc
import os

import numpy as np
import torch
from diffusers.utils import export_to_video, load_image
from loguru import logger

from llmc.utils import seed_all
from llmc.utils.registry_factory import MODEL_REGISTRY

from .eval_base import BaseEval


class VideoGenerateEval(BaseEval):

    def __init__(self, model, config):
        super().__init__(model, config)
        self.output_video_path = self.eval_cfg.get('output_video_path', None)
        assert self.output_video_path is not None
        os.makedirs(self.output_video_path, exist_ok=True)
        self.target_height = self.eval_cfg.get('target_height', 480)
        self.target_width = self.eval_cfg.get('target_width', 832)
        self.num_frames = self.eval_cfg.get('num_frames', 81)
        self.guidance_scale = self.eval_cfg.get('guidance_scale', 5.0)
        self.guidance_scale_2 = self.eval_cfg.get('guidance_scale_2', None)
        self.fps = self.eval_cfg.get('fps', 15)

    @torch.no_grad()
    def eval(self, model_llmc, eval_pos):
        seed_all(self.config.base.seed + int(os.environ['RANK']))
        model_llmc.Pipeline.to('cuda')
        eval_res = self.eval_func(
            model_llmc,
            self.testenc,
            self.eval_dataset_bs,
            eval_pos,
        )

        model_llmc.Pipeline.to('cpu')
        gc.collect()
        torch.cuda.empty_cache()
        return eval_res

    def pre_process(self, model, image_path):
        image = load_image(image_path)
        max_area = self.target_height * self.target_width
        aspect_ratio = image.height / image.width
        mod_value = model.Pipeline.vae_scale_factor_spatial * model.model.config.patch_size[1]
        height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
        width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
        image = image.resize((width, height))
        return image, width, height

    @torch.no_grad()
    def t2v_eval(self, model, testenc, bs, eval_pos):
        assert bs == 1, 'Only support eval bs=1'

        for i, data in enumerate(testenc):
            pipe_kw = {
                'prompt': data['prompt'],
                'negative_prompt': data['negative_prompt'],
                'height': self.target_height,
                'width': self.target_width,
                'num_frames': self.num_frames,
                'guidance_scale': self.guidance_scale,
            }
            if self.guidance_scale_2 is not None:
                pipe_kw['guidance_scale_2'] = self.guidance_scale_2
            output = model.Pipeline(**pipe_kw).frames[0]
            export_to_video(
                output,
                os.path.join(self.output_video_path, f'{eval_pos}_output_{i}.mp4'),
                fps=self.fps,
            )

        return None

    @torch.no_grad()
    def i2v_eval(self, model, testenc, bs, eval_pos):
        for i, data in enumerate(testenc):
            image, width, height = self.pre_process(model, data['image'])

            pipe_kw = {
                'image': image,
                'prompt': data['prompt'],
                'negative_prompt': data['negative_prompt'],
                'height': height,
                'width': width,
                'num_frames': self.num_frames,
                'guidance_scale': self.guidance_scale,
            }
            if self.guidance_scale_2 is not None:
                pipe_kw['guidance_scale_2'] = self.guidance_scale_2
            output = model.Pipeline(**pipe_kw).frames[0]

            export_to_video(
                output,
                os.path.join(self.output_video_path, f'{eval_pos}_output_{i}.mp4'),
                fps=self.fps,
            )

        return None

    @torch.no_grad()
    def eval_func(self, model, testenc, bs, eval_pos):
        assert bs == 1, 'Evaluation only supports batch size = 1.'
        assert self.model_type in ['WanT2V', 'WanI2V', 'Wan2T2V'], (
            f"Unsupported model type '{self.model_type}'.\n"
            'Only Wan video generation models (WanT2V, WanI2V, Wan2T2V) are supported.'
        )
        if self.eval_dataset_name == 't2v':
            return self.t2v_eval(model, testenc, bs, eval_pos)
        elif self.eval_dataset_name == 'i2v':
            return self.i2v_eval(model, testenc, bs, eval_pos)
        else:
            raise Exception(f'Unsupported eval dataset: {self.eval_dataset_name}')
