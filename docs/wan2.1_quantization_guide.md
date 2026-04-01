# Wan2.1 视频生成模型量化指南

## 概述

llmc 框架现已全面支持 Wan2.1 系列视频生成模型的量化，并提供真正量化的 INT8/FP8 权重导出，与 lightx2v 推理框架兼容。

## 支持的模型类型

- **WanI2V**: Image-to-Video (图像到视频)
- **WanT2V**: Text-to-Video (文本到视频)

## 支持的量化方法

### FP8 量化 (推荐)

**配置文件**: `configs/quantization/video_gen/wan_i2v/smoothquant_w_a_fp8.yaml`

**特点**:
- 使用 E4M3 FP8 格式 (8-bit 浮点数，4位指数，3位尾数)
- SmoothQuant 算法，平衡权重和激活的量化难度
- 适合 GPU 推理，性能损失小

**量化配置**:
```yaml
quant:
    video_gen:
        method: SmoothQuant
        weight:
            quant_type: float-quant
            bit: e4m3          # FP8 E4M3 格式
            symmetric: True
            granularity: per_channel
            use_qtorch: True
        act:
            quant_type: float-quant
            bit: e4m3          # FP8 E4M3 格式
            symmetric: True
            granularity: per_token
            use_qtorch: True
        special:
            alpha: 0.75        # SmoothQuant 平衡参数
```

### INT8 量化

#### 1. RTN (Round-to-Nearest)
**配置文件**: `configs/quantization/video_gen/wan_i2v/rtn_w_a.yaml`

**特点**:
- 最简单的量化方法
- 直接四舍五入到最近的量化级别
- 速度快，精度略低

#### 2. AWQ (Activation-aware Weight Quantization)
**配置文件**: `configs/quantization/video_gen/wan_i2v/awq_w_a.yaml`

**特点**:
- 基于激活分布优化权重量化
- 保护重要通道，减少精度损失
- 需要校准数据

#### 3. SmoothQuant
**配置文件**: `configs/quantization/video_gen/wan_i2v/smoothquant_w_a.yaml`

**特点**:
- 平衡权重和激活的量化难度
- 数学上等价于平滑激活异常值
- 通常提供最佳精度

### LoRA 模型量化

支持对 LoRA 适配器模型的量化：
- `smoothquant_w_a_int8_lora.yaml`
- `rtn_w_a_lora.yaml`

## 运行步骤

### 1. 准备环境

```bash
# 设置 llmc 路径
export llmc=/path/to/llmc
export PYTHONPATH=$llmc:$PYTHONPATH

# 设置 GPU
export CUDA_VISIBLE_DEVICES=0
```

### 2. 准备校准数据

为 I2V 模型准备校准数据：
```
assets/wan_i2v/calib/
├── image_1.jpg
├── image_2.jpg
└── ...
```

为 T2V 模型准备校准数据：
```
assets/wan_t2v/calib/
├── prompt_1.txt
├── prompt_2.txt
└── ...
```

### 3. 修改配置文件

编辑对应的 YAML 配置文件，设置：
- `model.path`: Wan2.1 模型路径
- `calib.path`: 校准数据路径
- `save.save_path`: 量化模型保存路径

**示例 (FP8 量化)**:
```yaml
base:
    seed: 42
model:
    type: WanI2V
    path: /path/to/wan2.1-i2v-model  # 修改为你的模型路径
    torch_dtype: auto
calib:
    name: i2v
    download: False
    path: /path/to/calibration/data  # 修改为校准数据路径
    sample_steps: 40
    bs: 1
    target_height: 480
    target_width: 832
    num_frames: 81
    guidance_scale: 5.0
save:
    save_lightx2v: True
    save_path: /path/to/save/quantized/model  # 修改为保存路径
```

### 4. 运行量化

#### 使用脚本运行 (推荐)

```bash
# 运行 FP8 量化 (I2V)
./run_llmc.sh wan_i2v_fp8

# 运行 INT8 RTN 量化 (I2V)
./run_llmc.sh wan_i2v_int8_rtn

# 运行 INT8 AWQ 量化 (I2V)
./run_llmc.sh wan_i2v_int8_awq

# 运行 INT8 SmoothQuant 量化 (I2V)
./run_llmc.sh wan_i2v_int8_smoothquant

# 运行 T2V 模型量化
./run_llmc.sh wan_t2v_int8_rtn
./run_llmc.sh wan_t2v_int8_awq
./run_llmc.sh wan_t2v_int8_smoothquant
```

#### 直接运行命令

```bash
torchrun \
--nnodes 1 \
--nproc_per_node 1 \
--rdzv_id $RANDOM \
--rdzv_backend c10d \
--rdzv_endpoint 127.0.0.1:29500 \
${llmc}/llmc/__main__.py \
--config configs/quantization/video_gen/wan_i2v/smoothquant_w_a_fp8.yaml \
--task_id my_quant_task
```

### 5. 监控进度

```bash
# 查看日志
tail -f wan_i2v_fp8.log

# 查看进程
ps aux | grep __main__.py
```

### 6. 停止任务

```bash
# 使用保存的 PID 文件
xargs kill -9 < wan_i2v_fp8.pid
```

## 配置参数说明

### 模型配置
- `type`: 模型类型 (`WanI2V` 或 `WanT2V`)
- `path`: 模型权重路径
- `torch_dtype`: 数据类型 (`auto`, `bfloat16`, `float32`)

### 校准配置
- `sample_steps`: 采样步数 (通常 20-40)
- `bs`: 批大小 (通常 1，视频生成显存占用大)
- `target_height`: 目标视频高度 (默认 480)
- `target_width`: 目标视频宽度 (默认 832)
- `num_frames`: 视频帧数 (默认 81)
- `guidance_scale`: CFG 引导强度 (默认 5.0)

### 量化配置
- `method`: 量化方法 (`RTN`, `Awq`, `SmoothQuant`)
- `weight.bit`: 权重位宽 (8, e4m3)
- `act.bit`: 激活位宽 (8, e4m3)
- `granularity`: 量化粒度 (`per_channel`, `per_token`)
- `special.alpha`: SmoothQuant 平衡参数 (0.5-1.0)

## 在 lightx2v 中使用量化模型

### 1. 配置 lightx2v

编辑 `lightx2v/configs/quantization/wan_i2v.json`:
```json
{
    "infer_steps": 40,
    "target_video_length": 81,
    "target_height": 480,
    "target_width": 832,
    "dit_quantized_ckpt": "/path/to/quantized/model",
    "dit_quantized": true,
    "dit_quant_scheme": "int8-vllm"
}
```

对于 FP8 模型，设置 `"dit_quant_scheme": "fp8"`。

### 2. 运行推理

```bash
python -m lightx2v.infer \
--model_cls wan2.1 \
--task i2v \
--model_path /path/to/original/model \
--config_json configs/quantization/wan_i2v.json \
--prompt "Your prompt here" \
--image_path /path/to/input/image.jpg \
--save_result_path output.mp4
```

## 性能建议

1. **FP8 vs INT8**:
   - FP8: 精度更高，适合对质量要求高的场景
   - INT8: 压缩率更高，适合对速度要求高的场景

2. **量化方法选择**:
   - 快速原型: RTN
   - 平衡精度和速度: SmoothQuant
   - 最高精度: AWQ

3. **校准数据**:
   - 使用 10-50 个样本
   - 覆盖典型使用场景
   - I2V: 使用多样化图像
   - T2V: 使用多样化文本描述

4. **资源需求**:
   - GPU: 建议 24GB+ 显存
   - 校准时间: 30分钟 - 2小时 (取决于数据量)
   - 存储空间: 量化后模型约原模型 25-50% 大小

## 故障排除

### 显存不足
- 减小 `bs` 到 1
- 减小 `num_frames`
- 减小 `target_height` 和 `target_width`

### 量化精度损失过大
- 尝试 SmoothQuant 方法
- 增加校准数据数量
- 调整 `alpha` 参数 (0.5-1.0)

### lightx2v 兼容性问题
- 确保使用 `save_lightx2v: True`
- 检查 `dit_quant_scheme` 设置
- 确认量化模型路径正确

## 参考

- lightx2v 文档: [lightx2v 项目地址]
- llmc 框架: [llmc 项目地址]
- Wan2.1 模型: [模型地址]
