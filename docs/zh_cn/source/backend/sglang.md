# Sglang量化推理

[Sglang](https://github.com/sgl-project/sglang) 是一个快速服务的大型语言模型和视觉语言模型框架。通过共同设计后端运行时和前端语言，它可以使你与模型的交互更加快速和可控。


## 1.1 环境准备

要使用 Sglang 进行量化推理，首先需要安装并配置 Sglang 环境：
```bash
pip install --upgrade pip
pip install "sglang[all]"

# Install FlashInfer CUDA kernels
pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/
```

## 1.2 量化格式

与 [**VLLM**](https://llmc-zhcn.readthedocs.io/en/latest/backend/vllm.html) 相同。

## 1.3 使用LLMC量化模型


### 1.3.1 校准数据

在本章节中，我们使用**Plieval**和**Wikitext**两个学术数据集作为校准数据，有关于校准数据的下载和预处理请参考[章节](https://llmc-zhcn.readthedocs.io/en/latest/configs.html)。

在实际使用中，建议应使用真实部署场景的数据进行离线量化校准。


### 1.3.2 量化算法的选择

**W8A16**

在 W8A16 的量化设置下，大语言模型的精度通常不会出现明显问题。在这种情况下，我们建议使用最简单的 RTN（Round to Nearest）算法，该算法不需要额外的校准步骤，运行速度较快。

具体实现可以参考 RTN W8A16 的权重量化 [配置文件](https://github.com/ModelTC/llmc/tree/main/configs/quantization/backend/sglang/rtn_w8a16.yml)

```yaml
# configs/quantization/backend/sglang/rtn_w8a16.yml
quant:
    method: RTN
    weight:
        bit: 8
        symmetric: True
        granularity: per_group
        group_size: 128
        need_pack: True
```
请注意，在此步骤中需要将 `need_pack` 参数设置为 `True`, 这会将8-bit的权重`打包`为`torch.int32`的格式供Sglang直接加载推理。

**W4A16**

在 W4A16 的量化设置下，RTN（Round to Nearest）不能保证精度无问题，因此需要使用一些高阶量化算法来维持模型的精度。在这种情况下，我们建议使用 **LLMC** 中的 AWQ 算法.


具体实现可以参考 AWQ W4A16 的权重量化 [配置文件](https://github.com/ModelTC/llmc/tree/main/configs/quantization/backend/sglang/awq_w4a16.yml)

```yaml
# configs/quantization/backend/sglang/awq_w4a16.yml
quant:
    method: Awq
    weight:
        bit: 4
        symmetric: True
        granularity: per_group
        group_size: 128
        need_pack: True
    special:
        trans: True
        trans_version: v2
        weight_clip: True
    quant_out: True  
```
请注意，在此步骤中需要将 `need_pack` 参数设置为 `True`, 这会将4-bit的权重`打包`为`torch.int32`的格式存储，供**SGlang**直接加载推理。


此外，如果 AWQ 无法满足精度需求，我们建议使用 [章节](https://llmc-zhcn.readthedocs.io/en/latest/practice/awq_omni.html)介绍的 **AWQ+OmniQuant 组合算法** 来进一步提升精度。在此也给出相应的[配置文件](https://github.com/ModelTC/llmc/tree/main/configs/quantization/backend/sglang/w4a16_combin)


**W8A8**

在 W8A8 的量化设置下，我们同样建议使用 AWQ 算法。AWQ 在大多数情况下的表现优于 SmoothQuant 和 OS+，能够提供更好的量化精度。

具体的实现可以参考 AWQ W8A8 的 [配置文件](https://github.com/ModelTC/llmc/tree/main/configs/quantization/backend/sglang/awq_w8a8.yml)。

```yaml
# configs/quantization/backend/sglang/awq_w8a8.yml
quant:
    method: Awq
    weight:
        bit: 8
        symmetric: True
        granularity: per_channel
        group_size: -1
    act:
        bit: 8
        symmetric: True
        granularity: per_token
    special:
        trans: True
        trans_version: v2
        weight_clip: True
    quant_out: True 
```

此外，如果 AWQ 无法满足精度需求，我们建议使用 [章节](https://llmc-zhcn.readthedocs.io/en/latest/practice/quarot_gptq.html) 介绍的 **Quarot+GPTQ 组合算法** 来进一步提升精度。在此也给出相应的[配置文件](https://github.com/ModelTC/llmc/tree/main/configs/quantization/backend/sglang/w8a8_combin)

**FP8-Dynamic**

在 FP8 的量化中，**LLMC** 支持权重per-channel，激活动态per-token的量化，在这种情况下，使用RTN（Round to Nearest）算法就足够了。然而，我们仍然建议使用AWQ算法以获得更好的量化精度。具体的实现可以参考AWQ FP8的[配置文件](https://github.com/ModelTC/llmc/tree/main/configs/quantization/backend/vllm/fp8/awq_fp8.yml)。

```yaml
# configs/quantization/backend/vllm/fp8/awq_fp8.yml
quant:
    method: Awq
    quant_type: float_quant
    weight:
        # Support ["e4m3", "e5m2"]
        bit: e4m3
        symmetric: True
        granularity: per_channel
        use_qtorch: True
    act:
        # Support ["e4m3", "e5m2"]
        bit: e4m3
        symmetric: True
        granularity: per_token
        use_qtorch: True
    special:
        trans: True
        trans_version: v2
        weight_clip: True
    quant_out: True
```

请确保将 `quant_type` 设置为 `float_quant`，表示浮点量化。同时，将 `use_qtorch` 设置为 `True`，因为 `LLMC` 的浮点量化实现依赖 [QPyTorch](https://github.com/Tiiiger/QPyTorch) 库中的部分功能。

您可以使用以下命令来安装 [QPyTorch](https://github.com/Tiiiger/QPyTorch)：

```bash
pip install qtorch
```

**FP8-Static**

在 FP8 的量化中，**LLMC** 同时也支持权重per-tensor，激活静态per-tensor的量化，在这种情况下，我们建议使用AWQ算法，调整下激活的范围，可以参考AWQ FP8静态量化的[配置文件](https://github.com/ModelTC/llmc/tree/main/configs/quantization/backend/vllm/fp8/awq_fp8_static.yml)。

```yaml
# configs/quantization/backend/vllm/fp8/awq_fp8_static.yml
quant:
    method: Awq
    quant_type: float-quant
    weight:
        # Support ["e4m3", "e5m2"]
        bit: e4m3
        symmetric: True
        granularity: per_tensor
        use_qtorch: True
    act:
        # Support ["e4m3", "e5m2"]
        bit: e4m3
        symmetric: True
        granularity: per_tensor
        use_qtorch: True
        static: True
```


### 1.3.3 真实量化模型导出

```yaml
save:
    save_sgl: True
    save_path: /path/to/save_for_sglang_rtn_w8a16/
```
请注意，务必将 `save_sgl` 设置为 `True`。对于 **W4A16** 和 **W8A16** 的量化设置，LLMC 会将权重打包为 `torch.int32` 形式导出，便于 SGlang 直接加载，并且会同时导出量化参数。

对于 **W8A8** 的量化设置，LLMC 会将权重量化为 `torch.int8` 形式导出，便于 SGlang 直接加载，同时也会导出相关的量化参数。


### 1.3.4 运行LLMC

修改运行脚本中的配置文件路径并运行：

```bash
# scripts/run_llmc.sh
llmc=llmc_path
export PYTHONPATH=$llmc:$PYTHONPATH

task_name=rtn_for_sglang
config=${llmc}/configs/quantization/backend/sglang/rtn_w8a16.yml
```
等LLMC运行结束后，真实量化的模型就会存储在`save.save_path`路径

## 1.4 使用Sglang推理模型

### 1.4.1 推理服务

下面的命令显式使用 http://localhost:10000，与调用服务的请求地址保持一致。`model_path`替换成`save.save_path`路径下保存的`量化模型`即可。

启动服务：

```
python -m sglang.launch_server --model-path model_path --port 10000
```

调用服务：

```
curl http://localhost:10000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Once upon a time,",
    "sampling_params": {
      "max_new_tokens": 16,
      "temperature": 0
    }
  }'
```

同时，我们构建了一个使用 **Sglang** 进行推理的[示例](https://github.com/ModelTC/llmc/blob/main/examples/backend/sglang/infer_with_sglang.py)。

```bash
cd examples/backend/sglang

python infer_with_sglang.py
```