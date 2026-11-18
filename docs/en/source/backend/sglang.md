
# SGLang Quantized Inference

[SGLang](https://github.com/sgl-project/sglang) is a fast-serving framework for large language models and vision-language models. By co-designing the backend runtime and frontend language, it makes interactions with models faster and more controllable.

## 1.1 Environment Setup

To use SGLang for quantized inference, you first need to install and configure the SGLang environment:
```bash
pip install --upgrade pip
pip install "sglang[all]"

# Install FlashInfer CUDA kernels
pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/
```

## 1.2 Quantization Format

Same as [**VLLM**](https://llmc-en.readthedocs.io/en/latest/backend/vllm.html).

## 1.3 Using LLMC for Model Quantization

### 1.3.1 Calibration Data

In this section, we use the **Plieval** and **Wikitext** academic datasets as calibration data. For downloading and preprocessing calibration data, please refer to [this section](https://llmc-en.readthedocs.io/en/latest/configs.html).

For real use cases, it is recommended to use real deployment scenario data for offline quantization calibration.

### 1.3.2 Choosing a Quantization Algorithm

**W8A16**

Under the W8A16 quantization setting, the accuracy of large language models generally does not show significant issues. In this case, we recommend using the simplest RTN (Round to Nearest) algorithm, which does not require additional calibration steps and runs quickly.

The specific implementation can be found in the RTN W8A16 weight quantization [configuration file](https://github.com/ModelTC/llmc/tree/main/configs/quantization/backend/sglang/rtn_w8a16.yml).

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
Please note that in this step, the `need_pack` parameter must be set to `True`, which will "pack" the 8-bit weights into the `torch.int32` format for SGLang to directly load for inference.

**W4A16**

Under the W4A16 quantization setting, RTN (Round to Nearest) cannot ensure accuracy, so higher-order quantization algorithms are required to maintain model accuracy. In this case, we recommend using the **AWQ** algorithm from **LLMC**.

The specific implementation can be found in the AWQ W4A16 weight quantization [configuration file](https://github.com/ModelTC/llmc/tree/main/configs/quantization/backend/sglang/awq_w4a16.yml).

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
Please note that in this step, the `need_pack` parameter must be set to `True`, which will "pack" the 4-bit weights into the `torch.int32` format for **SGlang** to directly load for inference.

Additionally, if AWQ does not meet accuracy requirements, we recommend using the **AWQ + OmniQuant** combined algorithm as introduced in [this section](https://llmc-en.readthedocs.io/en/latest/practice/awq_omni.html) to further improve accuracy. The corresponding [configuration file](https://github.com/ModelTC/llmc/tree/main/configs/quantization/backend/sglang/w4a16_combin) is also provided.

**W8A8**

Under the W8A8 quantization setting, we also recommend using the AWQ algorithm. AWQ generally outperforms SmoothQuant and OS+ in most cases, providing better quantization accuracy.

The specific implementation can be found in the AWQ W8A8 [configuration file](https://github.com/ModelTC/llmc/tree/main/configs/quantization/backend/sglang/awq_w8a8.yml).

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

Additionally, if AWQ does not meet accuracy requirements, we recommend using the **Quarot + GPTQ** combined algorithm as introduced in [this section](https://llmc-en.readthedocs.io/en/latest/practice/quarot_gptq.html) to further improve accuracy. The corresponding [configuration file](https://github.com/ModelTC/llmc/tree/main/configs/quantization/backend/sglang/w8a8_combin) is also provided.


**FP8-Dynamic**

In FP8 quantization, **LLMC** supports weight quantization per-channel and activation quantization dynamically per-token. In this case, the RTN (Round to Nearest) algorithm is sufficient. However, we recommend using the AWQ algorithm for better quantization accuracy. For implementation details, refer to the AWQ FP8 [configuration file](https://github.com/ModelTC/llmc/tree/main/configs/quantization/backend/sglang/fp8/awq_fp8.yml).

```yaml
# configs/quantization/backend/sglang/fp8/awq_fp8.yml
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

Ensure that `quant_type` is set to `float_quant` to indicate floating-point quantization. Additionally, set `use_qtorch` to `True`, as **LLMC**'s FP8 implementation depends on certain functionalities from the [QPyTorch](https://github.com/Tiiiger/QPyTorch) library.

Install [QPyTorch](https://github.com/Tiiiger/QPyTorch) with the following command:

```bash
pip install qtorch
```

**FP8-Static**

In FP8 quantization, **LLMC** also supports weight quantization per-tensor and activation quantization statically per-tensor. In this case, we recommend using the AWQ algorithm while adjusting the activation ranges. Refer to the AWQ FP8 static quantization [configuration file](https://github.com/ModelTC/llmc/tree/main/configs/quantization/backend/sglang/fp8/awq_fp8_static.yml).

```yaml
# configs/quantization/backend/sglang/fp8/awq_fp8_static.yml
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

### 1.3.3 Exporting Real Quantized Model

```yaml
save:
    save_sgl: True
    save_path: /path/to/save_for_sglang_rtn_w8a16/
```
Please note that you must set `save_sgl` to `True`. For **W4A16** and **W8A16** quantization settings, LLMC will "pack" the weights into `torch.int32` format for direct loading by SGlang, while also exporting the quantization parameters.

For the **W8A8** quantization setting, LLMC will quantize the weights into `torch.int8` format for direct loading by SGlang, and export the relevant quantization parameters as well.

### 1.3.4 Running LLMC

Modify the configuration file path in the script and run:

```bash
# scripts/run_llmc.sh
llmc=llmc_path
export PYTHONPATH=$llmc:$PYTHONPATH

task_name=rtn_for_sglang
config=${llmc}/configs/quantization/backend/sglang/rtn_w8a16.yml
```
Once LLMC finishes running, the quantized model will be stored at the `save.save_path` location.

## 1.4 Using Sglang for Inference

### 1.4.1 Inference Service

The command below explicitly starts the server at http://localhost:10000, matching the request URL. Replace `model_path` with the `quantized model` saved under the `save.save_path`.

Start the service:

```bash
python -m sglang.launch_server --model-path model_path --port 10000
```

Call the service:

```bash
curl http://localhost:10000/generate   -H "Content-Type: application/json"   -d '{
    "text": "Once upon a time,",
    "sampling_params": {
      "max_new_tokens": 16,
      "temperature": 0
    }
  }'
```

Additionally, we have built an [example](https://github.com/ModelTC/llmc/blob/main/examples/backend/sglang/infer_with_sglang.py) that uses **SGLang** for inference.

```bash
cd examples/backend/sglang

python infer_with_sglang.py