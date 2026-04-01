# Wan2.2 视频生成模型量化指南

## 概述

本仓库为 **Wan2.2-T2V** 提供的现成示例是 **4-bit AWQ 模拟量化**（`configs/quantization/video_gen/wan2_2_t2v/awq_w_a.yaml`）。

Wan2.2 为 **MoE 双专家**：高噪声（`transformer`）与低噪声（`transformer_2`），校准与块级量化会覆盖两条支路。保存侧默认示例为 `save_fake`，推理对接需按你的推理栈自行对齐。

**模型示例（原生 checkpoint 布局）**：[Wan-AI/Wan2.2-T2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B)

## Wan2.2 相对 Wan2.1 的要点

| 项目 | 说明 |
|------|------|
| 注册名 | `Wan2T2V` |
| 结构 | 双专家 MoE，非单路 DiT |
| 推理后端 | 优先官方 `wan` + 原生目录；可按 YAML 注释回退 Diffusers |
| CFG | `guidance_scale`（高噪声）与 `guidance_scale_2`（低噪声），与官方双引导一致 |

## 量化配置示例

`awq_w_a.yaml` 中 `quant` 段与仓库一致，例如：

```yaml
quant:
    video_gen:
        method: Awq
        weight:
            quant_type: hif4
            bit: 4
            symmetric: True
            granularity: per_channel
            group_size: -1
        act:
            quant_type: hif4
            bit: 4
            symmetric: True
            granularity: per_token
        special:
            trans: True
            trans_version: v2
            weight_clip: True
            clip_sym: True
```

## 运行步骤

### 1. 环境

```bash
export llmc=/path/to/LightCompress
export PYTHONPATH=$llmc:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
```

原生布局需要能 `import wan`，通常：

```bash
pip install -e /path/to/Wan2.2
```

或在 YAML 里设置 `wan2_repo_path: /path/to/Wan2.2`。

### 2. 校准数据

与 Wan2.1 T2V 相同，文本 prompt 文件目录，例如：

```
assets/wan_t2v/calib/
├── prompt_1.txt
├── prompt_2.txt
└── ...
```

配置中 `calib.name: t2v`，`calib.path` 指向该目录。

### 3. 修改 `awq_w_a.yaml`

必改：

- `model.path`：Wan2.2 权重路径  
- `calib.path` / `eval.path`：校准与评估数据  
- `save.save_path`：输出目录  

可选（见 YAML 注释）：

- `use_cpu_to_save_cuda_mem_for_catcher: True`：校准显存紧张时减轻峰值  
- `allow_diffusers_fallback: True`：无法用官方后端时回退 Diffusers  

双引导示例：

```yaml
calib:
    guidance_scale: 4.0      # high_noise
    guidance_scale_2: 3.0    # low_noise
eval:
    guidance_scale: 4.0
    guidance_scale_2: 3.0
```

### 4. 启动量化

```bash
torchrun \
  --nnodes 1 \
  --nproc_per_node 1 \
  --rdzv_id $RANDOM \
  --rdzv_backend c10d \
  --rdzv_endpoint 127.0.0.1:29500 \
  ${llmc}/llmc/__main__.py \
  --config ${llmc}/configs/quantization/video_gen/wan2_2_t2v/awq_w_a.yaml \
  --task_id wan22_awq_int4
```

`scripts/run_llmc.sh` 中把 `model_name=wan2_2_t2v`、`task_name=awq_w_a` 等与上述 YAML 对齐即可（需按本机修改脚本里的 Python 路径等）。

## 参数速查

| 区域 | 说明 |
|------|------|
| `model.type` | `Wan2T2V` |
| `quant.video_gen.method` | `Awq` |
| `weight` / `act` | `bit: 4`（具体 `quant_type` 以 YAML 为准） |
| `save` | 示例 `save_fake: True` 与 `save_path` |

## 常见问题

- **OOM**：减小 `sample_steps`、`num_frames`、分辨率；`bs: 1`；可开 `use_cpu_to_save_cuda_mem_for_catcher`。  
- **无法 `import wan`**：安装官方仓库或配置 `wan2_repo_path`。  
- **画质下降**：增加/多样化校准 prompt；在支持范围内微调 `special` 与校准规模。  

## 参考

- `configs/quantization/video_gen/wan2_2_t2v/awq_w_a.yaml`  
- `llmc/models/wan2_2_t2v.py`  
- 其它精度（如 FP8、INT8）可参考 `docs/wan2.1_quantization_guide.md` 的思路，自行新增 `wan2_2_t2v` 下 YAML 并替换 `model.type` 与路径。
