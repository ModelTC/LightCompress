import gc
import copy
import inspect
import os
import sys
from collections import defaultdict
from types import SimpleNamespace

import torch
import torch.nn as nn
from diffusers import AutoencoderKLWan, WanPipeline
from loguru import logger

from llmc.compression.quantization.module_utils import LlmcWanTransformerBlock
from llmc.utils.registry_factory import MODEL_REGISTRY

from .base_model import BaseModel


class WanOfficialPipelineAdapter:
    """Adapter that exposes Wan-Video/Wan2.2 official t2v runtime as a Pipeline-like interface."""

    def __init__(
        self,
        runner,
        sample_solver='unipc',
        sampling_steps=40,
        sample_shift=12.0,
        offload_model=True,
    ):
        self.runner = runner
        # Keep the same expert naming semantics as existing LLMC Wan2.2 flow:
        # transformer -> high-noise expert, transformer_2 -> low-noise expert.
        self.transformer = runner.high_noise_model
        self.transformer_2 = runner.low_noise_model
        self.sample_solver = sample_solver
        self.sampling_steps = sampling_steps
        self.sample_shift = sample_shift
        self.offload_model = offload_model
        self._is_wan_official = True

    @staticmethod
    def _tensor_to_frames(video):
        if video is None:
            return []
        if not torch.is_tensor(video):
            return video

        video = video.detach().cpu()
        if video.dim() != 4:
            raise ValueError(f'Unexpected official Wan video shape: {tuple(video.shape)}')

        # Accept [C, F, H, W] and convert to [F, C, H, W].
        if video.shape[0] in (1, 3):
            video = video.permute(1, 0, 2, 3)

        if video.dtype.is_floating_point:
            if video.min().item() < 0:
                video = (video.clamp(-1, 1) + 1.0) / 2.0
            else:
                video = video.clamp(0, 1)
            video = (video * 255).round().to(torch.uint8)
        elif video.dtype != torch.uint8:
            video = video.to(torch.uint8)

        return [frame.permute(1, 2, 0).contiguous().numpy() for frame in video]

    def to(self, device):  # noqa: ARG002
        # Keep the same API as diffusers pipeline; official runner manages model movement itself.
        return self

    def __call__(
        self,
        prompt,
        negative_prompt='',
        height=480,
        width=832,
        num_frames=81,
        guidance_scale=5.0,
        guidance_scale_2=None,
        **kwargs,
    ):
        if isinstance(prompt, (list, tuple)):
            prompt = prompt[0]
        if isinstance(negative_prompt, (list, tuple)):
            negative_prompt = negative_prompt[0]

        # Official Wan2.2 guide_scale order: (low_noise, high_noise).
        guide_scale_low = guidance_scale if guidance_scale_2 is None else guidance_scale_2
        guide_scale_high = guidance_scale

        sampling_steps = kwargs.get(
            'num_inference_steps',
            kwargs.get('sampling_steps', self.sampling_steps)
        )
        sample_shift = kwargs.get('sample_shift', self.sample_shift)
        sample_solver = kwargs.get('sample_solver', self.sample_solver)
        seed = kwargs.get('seed', -1)
        offload_model = kwargs.get('offload_model', self.offload_model)

        video = self.runner.generate(
            input_prompt=prompt,
            size=(width, height),
            frame_num=num_frames,
            shift=sample_shift,
            sample_solver=sample_solver,
            sampling_steps=sampling_steps,
            guide_scale=(guide_scale_low, guide_scale_high),
            n_prompt=negative_prompt if negative_prompt is not None else '',
            seed=seed,
            offload_model=offload_model,
        )
        return SimpleNamespace(frames=[self._tensor_to_frames(video)])


@MODEL_REGISTRY
class Wan2T2V(BaseModel):
    """Wan2.2-T2V with MoE: two experts (high-noise + low-noise), same block structure as Wan2.1."""

    def __init__(self, config, device_map=None, use_cache=False):
        super().__init__(config, device_map, use_cache)
        if 'calib' in config:
            self.calib_bs = config.calib.bs
            self.sample_steps = config.calib.sample_steps
            self.target_height = config.calib.get('target_height', 480)
            self.target_width = config.calib.get('target_width', 832)
            self.num_frames = config.calib.get('num_frames', 81)
            self.guidance_scale = config.calib.get('guidance_scale', 5.0)
            self.guidance_scale_2 = config.calib.get('guidance_scale_2', 3.0)
        else:
            self.sample_steps = None

    @staticmethod
    def _normalize_hf_repo_path(model_path):
        hf_prefix = 'https://huggingface.co/'
        if not isinstance(model_path, str) or not model_path.startswith(hf_prefix):
            return model_path
        repo_path = model_path[len(hf_prefix):].strip('/')
        for marker in ['/tree/', '/blob/', '/resolve/']:
            if marker in repo_path:
                repo_path = repo_path.split(marker, maxsplit=1)[0]
        return repo_path

    @staticmethod
    def _has_diffusers_layout(model_path):
        if not isinstance(model_path, str):
            return False
        return (
            os.path.isdir(model_path)
            and os.path.isfile(os.path.join(model_path, 'model_index.json'))
            and os.path.isdir(os.path.join(model_path, 'transformer'))
            and os.path.isdir(os.path.join(model_path, 'vae'))
        )

    @staticmethod
    def _has_wan22_native_layout(model_path):
        if not isinstance(model_path, str):
            return False
        return (
            os.path.isdir(model_path)
            and os.path.isfile(os.path.join(model_path, 'configuration.json'))
            and os.path.isdir(os.path.join(model_path, 'high_noise_model'))
            and os.path.isdir(os.path.join(model_path, 'low_noise_model'))
        )

    def _import_official_wan(self):
        def _import_impl():
            from wan.configs import t2v_A14B
            from wan.text2video import WanT2V as WanOfficialT2V

            return t2v_A14B, WanOfficialT2V

        try:
            return _import_impl()
        except Exception as e:
            repo_path = self.config.model.get('wan2_repo_path', None)
            if repo_path and os.path.isdir(repo_path):
                if repo_path not in sys.path:
                    sys.path.insert(0, repo_path)
                try:
                    return _import_impl()
                except Exception as e2:
                    logger.warning(
                        f'Failed to import official Wan2.2 from wan2_repo_path={repo_path}: {e2}'
                    )
            logger.warning(
                'Failed to import official Wan2.2 runtime (wan package). '
                f'Falling back to diffusers import path. import_error={e}'
            )
            return None, None

    def _try_build_official_wan_pipeline(self):
        normalized_model_path = self._normalize_hf_repo_path(self.model_path)
        if not self._has_wan22_native_layout(normalized_model_path):
            return False
        if self.config.model.get('force_diffusers', False):
            logger.info('force_diffusers=True, skip official Wan2.2 import backend.')
            return False

        t2v_A14B, WanOfficialT2V = self._import_official_wan()
        if t2v_A14B is None or WanOfficialT2V is None:
            return False

        wan_config = copy.deepcopy(t2v_A14B)
        # Keep official defaults unless explicitly overridden by llmc config.
        if self.config.model.get('sample_steps', None) is not None:
            wan_config.sample_steps = self.config.model.sample_steps
        if self.config.model.get('sample_shift', None) is not None:
            wan_config.sample_shift = self.config.model.sample_shift
        if self.config.model.get('boundary', None) is not None:
            wan_config.boundary = self.config.model.boundary

        runner = WanOfficialT2V(
            config=wan_config,
            checkpoint_dir=normalized_model_path,
            device_id=int(os.environ.get('LOCAL_RANK', 0)),
            rank=int(os.environ.get('RANK', 0)),
            t5_fsdp=False,
            dit_fsdp=False,
            use_sp=False,
            t5_cpu=self.config.model.get('t5_cpu', False),
            init_on_cpu=self.config.model.get('init_on_cpu', True),
            convert_model_dtype=self.config.model.get('convert_model_dtype', False),
        )
        self.Pipeline = WanOfficialPipelineAdapter(
            runner=runner,
            sample_solver=self.config.model.get('sample_solver', 'unipc'),
            sampling_steps=self.config.model.get(
                'sampling_steps', getattr(wan_config, 'sample_steps', 40)
            ),
            sample_shift=self.config.model.get(
                'sample_shift', getattr(wan_config, 'sample_shift', 12.0)
            ),
            offload_model=self.config.model.get('offload_model', True),
        )
        self.pipeline_model_path = normalized_model_path
        self.pipeline_source = 'wan_official'
        self.use_official_wan = True
        logger.info(
            f'Loaded Wan2.2 via official Wan runtime from native checkpoint: {normalized_model_path}'
        )
        return True

    def _resolve_pipeline_model_path(self):
        explicit_diffusers_path = self.config.model.get('diffusers_path', None)
        if explicit_diffusers_path is not None:
            resolved_path = self._normalize_hf_repo_path(explicit_diffusers_path)
            logger.info(f'Use explicit Wan2.2 diffusers_path: {resolved_path}')
            return resolved_path

        raw_model_path = self.model_path
        normalized_path = self._normalize_hf_repo_path(raw_model_path)

        if normalized_path != raw_model_path:
            logger.info(
                f'Normalize Wan2.2 model path from URL to repo id: {normalized_path}'
            )

        if self._has_diffusers_layout(normalized_path):
            return normalized_path

        if self._has_wan22_native_layout(normalized_path):
            local_diffusers_candidate = normalized_path.rstrip('/\\') + '-Diffusers'
            if self._has_diffusers_layout(local_diffusers_candidate):
                logger.info(
                    'Detected native Wan2.2 checkpoint. '
                    f'Use local diffusers directory: {local_diffusers_candidate}'
                )
                return local_diffusers_candidate
            logger.warning(
                'Detected native Wan2.2 checkpoint layout '
                f'({normalized_path}) but no local diffusers export found. '
                'Fallback to official diffusers repo: Wan-AI/Wan2.2-T2V-A14B-Diffusers. '
                'You can set model.diffusers_path to override this behavior.'
            )
            return 'Wan-AI/Wan2.2-T2V-A14B-Diffusers'

        if normalized_path.rstrip('/\\').endswith('Wan2.2-T2V-A14B'):
            mapped_path = normalized_path.rstrip('/\\') + '-Diffusers'
            logger.info(
                f'Map Wan2.2 native repo/path to diffusers pipeline source: {mapped_path}'
            )
            return mapped_path

        return normalized_path

    def build_model(self):
        self.use_official_wan = False
        if self._try_build_official_wan_pipeline():
            self.find_llmc_model()
            self.find_blocks()
            logger.info(
                'Wan2.2 MoE official backend loaded: blocks=%s(+%s)',
                len(self.Pipeline.transformer.blocks),
                (
                    len(self.Pipeline.transformer_2.blocks)
                    if hasattr(self.Pipeline, 'transformer_2')
                    and self.Pipeline.transformer_2 is not None
                    else 0
                ),
            )
            logger.info('Model: %s', self.model)
            return

        self.pipeline_model_path = self._resolve_pipeline_model_path()
        vae = AutoencoderKLWan.from_pretrained(
            self.pipeline_model_path,
            subfolder='vae',
            torch_dtype=torch.float32,
            use_safetensors=True,
        )
        # Wan2.2: one pipeline, two transformer experts (transformer + transformer_2).
        # Pipeline switches by SNR; both use WanTransformer3DModel with same block layout as Wan2.1.
        self.Pipeline = WanPipeline.from_pretrained(
            self.pipeline_model_path,
            vae=vae,
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
        )
        self.find_llmc_model()
        # Wrap both experts with LlmcWanTransformerBlock (same as Wan2.1 per-block layout).
        for block_idx, block in enumerate(self.Pipeline.transformer.blocks):
            new_block = LlmcWanTransformerBlock.new(block)
            self.Pipeline.transformer.blocks[block_idx] = new_block
        if hasattr(self.Pipeline, 'transformer_2') and self.Pipeline.transformer_2 is not None:
            for block_idx, block in enumerate(self.Pipeline.transformer_2.blocks):
                new_block = LlmcWanTransformerBlock.new(block)
                self.Pipeline.transformer_2.blocks[block_idx] = new_block
            self.num_transformer_blocks = len(self.Pipeline.transformer.blocks)
            self.blocks = list(self.Pipeline.transformer.blocks) + list(self.Pipeline.transformer_2.blocks)
            logger.info(
                'Wan2.2 MoE: both experts wrapped (high-noise + low-noise, 80 blocks total).'
            )
        else:
            self.blocks = list(self.Pipeline.transformer.blocks)
            self.num_transformer_blocks = len(self.blocks)
            logger.info('Wan2.2: single transformer wrapped (40 blocks).')
        logger.info('Model: %s', self.model)

    def find_llmc_model(self):
        self.model = self.Pipeline.transformer

    def find_blocks(self):
        self.blocks = list(self.Pipeline.transformer.blocks)
        self.num_transformer_blocks = len(self.blocks)
        if hasattr(self.Pipeline, 'transformer_2') and self.Pipeline.transformer_2 is not None:
            self.blocks += list(self.Pipeline.transformer_2.blocks)

    def _expert_name_from_block_idx(self, block_idx):
        if block_idx < self.num_transformer_blocks:
            return 'transformer'
        return 'transformer_2'

    def get_blockwise_input(self, block_idx, fallback_input):
        if not hasattr(self, 'blockwise_inputs'):
            return fallback_input
        return self.blockwise_inputs[self._expert_name_from_block_idx(block_idx)]

    def set_blockwise_input(self, block_idx, block_input):
        if not hasattr(self, 'blockwise_inputs'):
            return
        self.blockwise_inputs[self._expert_name_from_block_idx(block_idx)] = block_input

    def get_catcher(self, first_block_input):
        sample_steps = self.sample_steps

        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module
                self.signature = inspect.signature(module.forward)
                self.step = 0

            def forward(self, *args, **kwargs):
                params = list(self.signature.parameters.keys())
                for i, arg in enumerate(args):
                    if i > 0:
                        kwargs[params[i]] = arg
                first_block_input['data'].append(args[0])
                first_block_input['kwargs'].append(kwargs)
                self.step += 1
                if self.step == sample_steps:
                    raise ValueError
                else:
                    return self.module(*args)

        return Catcher

    @torch.no_grad()
    def collect_first_block_input(self, calib_data, padding_mask=None):
        first_block_input = {
            'transformer': defaultdict(list),
            'transformer_2': defaultdict(list),
        }
        sample_steps = self.sample_steps

        class Catcher(nn.Module):
            def __init__(self, module, expert_name):
                super().__init__()
                self.module = module
                self.signature = inspect.signature(module.forward)
                self.expert_name = expert_name

            def _to_cpu(self, x):
                if torch.is_tensor(x):
                    return x.detach().cpu()
                if isinstance(x, tuple):
                    return tuple(self._to_cpu(t) for t in x)
                return x

            def forward(self, *args, **kwargs):
                params = list(self.signature.parameters.keys())
                for i, arg in enumerate(args):
                    if i > 0:
                        kwargs[params[i]] = arg
                cur_num = len(first_block_input[self.expert_name]['data'])
                if cur_num < sample_steps:
                    first_block_input[self.expert_name]['data'].append(
                        args[0].detach().cpu() if torch.is_tensor(args[0]) else args[0]
                    )
                    first_block_input[self.expert_name]['kwargs'].append(
                        {k: self._to_cpu(v) for k, v in kwargs.items()}
                    )
                if all(len(first_block_input[name]['data']) >= sample_steps for name in first_block_input):
                    raise ValueError
                return self.module(*args)

        first_block = self.Pipeline.transformer.blocks[0]
        self.Pipeline.transformer.blocks[0] = Catcher(first_block, 'transformer')
        first_block_2 = None
        if hasattr(self.Pipeline, 'transformer_2') and self.Pipeline.transformer_2 is not None:
            first_block_2 = self.Pipeline.transformer_2.blocks[0]
            self.Pipeline.transformer_2.blocks[0] = Catcher(first_block_2, 'transformer_2')

        self.Pipeline.to('cuda')
        for data in calib_data:
            try:
                pipe_kw = {
                    'prompt': data['prompt'],
                    'negative_prompt': data['negative_prompt'],
                    'height': self.target_height,
                    'width': self.target_width,
                    'num_frames': self.num_frames,
                    'guidance_scale': self.guidance_scale,
                }
                if hasattr(self, 'guidance_scale_2'):
                    pipe_kw['guidance_scale_2'] = self.guidance_scale_2
                self.Pipeline(**pipe_kw)
            except ValueError:
                pass
            gc.collect()
            torch.cuda.empty_cache()

        self.Pipeline.transformer.blocks[0] = self.Pipeline.transformer.blocks[0].module
        if first_block_2 is not None:
            self.Pipeline.transformer_2.blocks[0] = self.Pipeline.transformer_2.blocks[0].module
        self.Pipeline.to('cpu')

        assert len(first_block_input['transformer']['data']) > 0, 'Catch transformer input data failed.'
        if hasattr(self.Pipeline, 'transformer_2') and self.Pipeline.transformer_2 is not None:
            assert len(first_block_input['transformer_2']['data']) > 0, \
                'Catch transformer_2 input data failed.'

        self.blockwise_inputs = first_block_input
        self.first_block_input = self.blockwise_inputs['transformer']
        self.n_samples = sum(len(v['data']) for v in self.blockwise_inputs.values())
        logger.info(
            'Retrieved Wan2.2 calibration samples: transformer=%s, transformer_2=%s.',
            len(self.blockwise_inputs['transformer']['data']),
            len(self.blockwise_inputs['transformer_2']['data']),
        )

    def get_padding_mask(self):
        return None

    def has_bias(self):
        return True

    def __str__(self):
        return '\nWan2.2 MoE Model:\n%s\nTotal params: ~27B (14B active per step)' % (
            str(self.model),
        )

    def get_layernorms_in_block(self, block):
        if hasattr(block, 'affine_norm1'):
            return {
                'affine_norm1': block.affine_norm1,
                'norm2': block.norm2,
                'affine_norm3': block.affine_norm3,
            }
        return {
            'norm1': block.norm1,
            'norm3': block.norm3,
            'norm2': block.norm2,
        }

    def get_subsets_in_block(self, block):
        if not hasattr(block, 'attn1'):
            # Official Wan2.2 native block layout:
            #   self_attn/qkv/o, cross_attn/qkv/o, ffn[0|2], modulation.
            return [
                {
                    'layers': {
                        'self_attn.q': block.self_attn.q,
                        'self_attn.k': block.self_attn.k,
                        'self_attn.v': block.self_attn.v,
                    },
                    # Official Wan2.2 uses non-affine norm1/norm2 by default.
                    # Skip trans-based scale folding to avoid invalid ln.weight operations.
                    'prev_op': [None],
                    'input': ['self_attn.q'],
                    'inspect': block.self_attn,
                    'has_kwargs': True,
                    'do_trans': False,
                    'sub_keys': {
                        'seq_lens': 'seq_lens',
                        'grid_sizes': 'grid_sizes',
                        'freqs': 'freqs',
                    },
                },
                {
                    'layers': {
                        'cross_attn.q': block.cross_attn.q,
                    },
                    'prev_op': [None],
                    'input': ['cross_attn.q'],
                    'inspect': block.cross_attn,
                    'has_kwargs': True,
                    'do_trans': False,
                    'sub_keys': {
                        'context': 'context',
                        'context_lens': 'context_lens',
                    },
                },
                {
                    'layers': {
                        'ffn.0': block.ffn[0],
                    },
                    'prev_op': [None],
                    'input': ['ffn.0'],
                    'inspect': block.ffn,
                    'has_kwargs': False,
                    'do_trans': False,
                },
            ]
        return [
            {
                'layers': {
                    'attn1.to_q': block.attn1.to_q,
                    'attn1.to_k': block.attn1.to_k,
                    'attn1.to_v': block.attn1.to_v,
                },
                'prev_op': [block.affine_norm1],
                'input': ['attn1.to_q'],
                'inspect': block.attn1,
                'has_kwargs': True,
                'sub_keys': {'rotary_emb': 'rotary_emb'},
            },
            {
                'layers': {
                    'attn2.to_q': block.attn2.to_q,
                },
                'prev_op': [block.norm2],
                'input': ['attn2.to_q'],
                'inspect': block.attn2,
                'has_kwargs': True,
                'sub_keys': {'encoder_hidden_states': 'encoder_hidden_states'},
            },
            {
                'layers': {
                    'ffn.net.0.proj': block.ffn.net[0].proj,
                },
                'prev_op': [block.affine_norm3],
                'input': ['ffn.net.0.proj'],
                'inspect': block.ffn,
                'has_kwargs': True,
            },
        ]

    def find_embed_layers(self):
        pass

    def get_embed_layers(self):
        pass

    def get_layers_except_blocks(self):
        pass

    def skip_layer_name(self):
        pass
