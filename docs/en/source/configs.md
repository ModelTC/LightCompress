# Configs' brief description

All configurations can be found [here](https://github.com/ModelTC/llmc/tree/main/configs)

Here's a brief config example

```
base:
    seed: &seed 42 # Set random seed
model:
    type: model_type # Type of the model
    path: model path # Path to the model
    tokenizer_mode: fast # Type of the model's tokenizer
    torch_dtype: auto # Data type of the model
calib:
    name: pileval # Name of the calibration dataset
    download: False # Whether to download the calibration dataset online
    path: calib data path # Path to the calibration dataset
    n_samples: 512 # Number of samples in the calibration dataset
    bs: 1 # Batch size for the calibration dataset
    seq_len: 512 # Sequence length for the calibration dataset
    preproc: pileval_smooth # Preprocessing method for the calibration dataset
    seed: *seed # Random seed for the calibration dataset
eval:
    eval_pos: [pretrain, transformed, fake_quant] # Evaluation points
    name: wikitext2 # Name of the evaluation dataset
    download: False # Whether to download the evaluation dataset online
    path: eval data path # Path to the evaluation dataset
    bs: 1 # Batch size for the evaluation dataset
    seq_len: 2048 # Sequence length for the evaluation dataset
    eval_token_consist: False # Whether to evaluate the consistency of tokens between the quantized and original models
quant:
    method: SmoothQuant # Compression method
    weight:
        bit: 8 # Number of quantization bits for weights
        symmetric: True # Whether weight quantization is symmetric
        granularity: per_channel # Granularity of weight quantization
    act:
        bit: 8 # Number of quantization bits for activations
        symmetric: True # Whether activation quantization is symmetric
        granularity: per_token # Granularity of activation quantization
    speical: # Special parameters required for the quantization algorithm. Refer to the comments in the configuration file and the original paper for usage.
save:
    save_vllm: False # Whether to save the real quantized model for VLLM inference
    save_sgl: False # Whether to save the real quantized model for Sglang inference
    save_autoawq: False # Whether to save the real quantized model for AutoAWQ inference
    save_mlcllm: False # Whether to save the real quantized model for MLC-LLM inference
    save_trans: False # Whether to save the model after weight transformation
    save_fake: False # Whether to save the fake quantized weights
    save_path: /path/to/save # Save path
```

# Configs' detailed description

## base

<font color=792ee5> base.seed </font>

Set Random Seed, which is used to set all random seeds for the entire frame

## model

<font color=792ee5> model.type </font>

The type of model, which can support Llama, Qwen2, Llava, Gemma2 and other models, you can check all the models supported by llmc from [here](https://github.com/ModelTC/llmc/blob/main/llmc/models/__init__.py).

<font color=792ee5> model.path </font>

Currently, LLMC only supports models in Hugging Face format, and you can use the following code to check whether the model can be loaded normally.

```
from transformers import AutoModelForCausalLM, AutoConfig


model_path = # model path
model_config = AutoConfig.from_pretrained(
    model_path, trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    config=model_config,
    trust_remote_code=True,
    torch_dtype="auto",
    low_cpu_mem_usage=True,
)

print(model)
```
If the above code does not load the model you give, may be:

1. Your model format is not hugging face format

2. Your version of tansformers is too low and you can execute `pip install transformers --upgrade` to upgrade it.

Before llmc runs, make sure that the above code can load your model successfully, otherwise llmc will not be able to load your model.

<font color=792ee5> model.tokenizer_mode </font>

Choose whether to use a Slow or Fast tokenizer

<font color=792ee5> model.torch_dtype </font>

You can set the data types of model weights:

1. auto

2. torch.float16

3. torch.bfloat16

3. torch.float32

where auto will follow the original data type setting of the weight file

## calib

<font color=792ee5> calib.name </font>

The name of the calibration dataset. Currently supported by the following types of calibration datasets:

1. pileval

2. wikitext2

3. c4

4. ptb

5. custom

where custom indicates the use of user-defined calibration datasets, refer to the [Custom Calibration Dataset section](https://llmc-en.readthedocs.io/en/latest/advanced/custom_dataset.html) of the advanced usage document for specific instructions

<font color=792ee5> calib.download </font>

Indicates whether the calibration dataset needs to be downloaded online at runtime

If you set True, you do not need to set calib.path, llmc will automatically download the dataset online

If you set False, you need to set calib.path, and llmc will read the dataset from this address, and you don't need to run llmc on the Internet

<font color=792ee5> calib.path </font>

If calib.download is set to False, you need to set calib.path, which indicates the path where the calibration dataset is stored

The data stored in this path must be a dataset in arrow format

To download the dataset in Arrow format from Hugging Face, you can use the following code
```
from datasets import load_dataset
calib_dataset = load_dataset(...)
calib_dataset.save_to_disk(...)
```
Load datasets in that format can be used
```
from datasets import load_from_disk
data = load_from_disk(...)
```
The LLMC has provided a download script for the above dataset

The calibration dataset can be downloaded [here](https://github.com/ModelTC/llmc/blob/main/tools/download_calib_dataset.py).

The execution command is `python download_calib_dataset.py --save_path [calib dataset save path]`

The test dataset can be downloaded [here](https://github.com/ModelTC/llmc/blob/main/tools/download_eval_dataset.py).

 The execution command is `python download_eval_dataset.py --save_path [eval dataset save path]`

If you want to use more datasets, you can refer to the download method of the arrow format dataset above and modify it yourself

<font color=792ee5> calib.n_samples </font>

Select n_samples pieces of data for calibration

<font color=792ee5> calib.bs </font>

Set the calibration data to calib.bs as the batch size, if it is -1, all the data is packaged into a batch of data

<font color=792ee5> calib.seq_len </font>

The sequence length of the calibration data

<font color=792ee5> calib.preproc </font>

The preprocessing methods of calibration data are currently implemented by llmc in a variety of preprocessing methods

1. wikitext2_gptq

2. ptb_gptq

3. c4_gptq

4. pileval_awq

5. pileval_smooth

6. pileval_omni

7. general

8. random_truncate_txt

With the exception of general, the rest of the preprocessing can be found [here](https://github.com/ModelTC/llmc/blob/main/llmc/data/dataset/specified_preproc.py)

general is implemented in the general_preproc function in the [base_dataset](https://github.com/ModelTC/llmc/blob/main/llmc/data/dataset/base_dataset.py)

<font color=792ee5> calib.seed </font>

The random seed in the data preprocessing follows the base.seed setting by default


## eval

<font color=792ee5> eval.eval_pos </font>

Indicates the eval positions, and currently supports three positions that can be evaluated

1. pretrain

2. transformed

3. fake_quant

eval_pos need to give a list, the list can be empty, and an empty list means that no tests are being performed

<font color=792ee5> eval.name </font>

The name of the eval dataset is supported by the following types of test datasets:

1. wikitext2

2. c4

3. ptb

For details about how to download the test dataset, see calib.name calibration dataset

<font color=792ee5> eval.download </font>

Indicates whether the eval dataset needs to be downloaded online at runtime, see calib.download

<font color=792ee5> eval.path </font>

Refer to calib.path

<font color=792ee5> eval.bs </font>

Eval batch size

<font color=792ee5> eval.seq_len </font>

The sequence length of the eval data

<font color=792ee5> eval.inference_per_block </font>

If your model is too large and the gpu memory of a single card cannot cover the entire model during the eval, then you need to open the inference_per_block for inference, and at the same time, on the premise of not exploding the gpu memory, appropriately increase the bs to improve the inference speed.

Here's a config example
```
bs: 10
inference_per_block: True
```

<font color=792ee5> Eval multiple datasets at the same time </font>

LLMC also supports the simultaneous evaluation of multiple datasets

Below is an example of evaluating a single wikitext2 dataset

```
eval:
    name: wikitext2
    path: wikitext2 path
```

Here's an example of evaluating multiple datasets

```
eval:
    name: [wikitext2, c4, ptb]
    path: The common upper directory of these data sets
```

It should be noted that the names of multiple dataset evaluations need to be represented in the form of a list, and the following directory rules need to be followed


- upper-level directory
    - wikitext2
    - c4
    - ptb

If you use the LLMC [download script](https://github.com/ModelTC/llmc/blob/main/tools/download_eval_dataset.py) directly, the shared upper-level directory is the `--save_path` specified dataset storage path




## quant

<font color=792ee5> quant.method </font>

The names of the quantization algorithms used, and all the quantization algorithms supported by the LLMC, can be viewed [here](https://github.com/ModelTC/llmc/blob/main/llmc/compression/quantization/__init__.py).


<font color=792ee5> quant.weight </font>

Quantization settings for weights

<font color=792ee5> quant.weight.bit </font>

The quantized number of bits of the weight

<font color=792ee5> quant.weight.symmetric </font>

Quantitative symmetry of weights

<font color=792ee5> quant.weight.granularity </font>

The quantification granularity of the weights supports the following granularities

1. per tensor

2. per channel

3. per group

<font color=792ee5> quant.act </font>

Activated quantization settings

<font color=792ee5> quant.act.bit </font>

Activated quantized bit digits

<font color=792ee5> quant.act.symmetric </font>

Quantified symmetry or not

<font color=792ee5> quant.act.granularity </font>

The quantization granularity of the activation supports the following granularities

1. per tensor

2. per token

3. per head

If quant.method is set to RTN, activating quantization can support static per tensor settings, and the following is a W8A8 configuration that activates static per tensor quantization

```
quant:
    method: RTN
    weight:
        bit: 8
        symmetric: True
        granularity: per_channel
    act:
        bit: 8
        symmetric: True
        granularity: per_tensor
        static: True
```

## sparse

<font color=792ee5> sparse.method </font>

The name of the sparsification algorithm used. This includes both [model sparsification](https://github.com/ModelTC/LightCompress/blob/main/llmc/compression/sparsification/__init__.pyn) and [reduction](https://github.com/ModelTC/LightCompress/blob/main/llmc/compression/token_reduction/__init__.py) of visual tokens. All supported algorithms can be found in the corresponding files.

It’s worth noting that for model sparsification, you need to specify the exact algorithm name, whereas for token reduction, you only need to set it to `TokenReduction` first, and then specify the exact algorithm under `special`.

```yaml
sparse:
    method: Wanda
```

```yaml
sparse:
    method: TokenReduction
    special:
        method: FastV
```

## save

<font color=792ee5> save.save_vllm</font>

Whether to save as a [VLLM](https://github.com/vllm-project/vllm) inference backend-supported real quantized model.

When this option is enabled, the saved model weights will significantly shrink (real quantization), and it can be directly loaded for inference using the VLLM backend. This improves inference speed and reduces memory usage. For more details on the [VLLM](https://github.com/vllm-project/vllm) inference backend, refer to [this section](https://llmc-en.readthedocs.io/en/latest/backend/vllm.html#).

<font color=792ee5> save.save_sgl</font>

Whether to save as a [Sglang](https://github.com/sgl-project/sglang) inference backend-supported real quantized model.

When this option is enabled, the saved model weights will significantly shrink (real quantization), and it can be directly loaded for inference using the [Sglang](https://github.com/sgl-project/sglang) backend. This improves inference speed and reduces memory usage. For more details on the [Sglang](https://github.com/sgl-project/sglang) inference backend, refer to [this section](https://llmc-en.readthedocs.io/en/latest/backend/sglang.html).

<font color=792ee5> save.save_autoawq</font>

Whether to save as an [AutoAWQ](https://github.com/casper-hansen/AutoAWQ) inference backend-supported real quantized model.

When this option is enabled, the saved model weights will significantly shrink (real quantization), and it can be directly loaded for inference using the [AutoAWQ](https://github.com/casper-hansen/AutoAWQ) backend. This improves inference speed and reduces memory usage. For more details on the [AutoAWQ](https://github.com/casper-hansen/AutoAWQ) inference backend, refer to [this section](https://llmc-en.readthedocs.io/en/latest/backend/autoawq.html).

<font color=792ee5> save.save_mlcllm</font>

Whether to save as an [MLC-LLM](https://github.com/mlc-ai/mlc-llm) inference backend-supported real quantized model.

When this option is enabled, the saved model weights will significantly shrink (real quantization), and it can be directly loaded for inference using the [MLC-LLM](https://github.com/mlc-ai/mlc-llm) backend. This improves inference speed and reduces memory usage. For more details on the [MLC-LLM](https://github.com/mlc-ai/mlc-llm) inference backend, refer to [this section](https://llmc-en.readthedocs.io/en/latest/backend/mlcllm.html).

<font color=792ee5> save.save_trans</font>

Whether to save the adjusted model weights.

The saved weights are adjusted to be more suitable for quantization, possibly containing fewer outliers. They are still saved in fp16/bf16 format (with the same file size as the original model). When deploying the model in the inference engine, the engine's built-in `naive quantization` needs to be used to achieve quantized inference.

Unlike `save_vllm` and similar options, this option requires the inference engine to perform real quantization, while `llmc` provides a floating-point model weight that is more suitable for quantization.

For example, the `save_trans` models exported by algorithms such as `SmoothQuant, Os+, AWQ, and Quarot` have `fewer outliers` and are more suitable for quantization.


<font color=792ee5> save.save_fake</font>

Whether to save the fake quantized model.

<font color=792ee5> save.save_path</font>

The path where the model is saved. This path must be a new, non-existent directory, otherwise, LLMC will terminate the run and issue an appropriate error message.