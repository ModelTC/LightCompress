# 配置的简要说明

所有的配置均可以在[这里](https://github.com/ModelTC/llmc/tree/main/configs)找到，具体地，包括[量化算法](https://github.com/ModelTC/llmc/tree/main/configs/quantization/methods)，[量化实践以及方法组合技](https://github.com/ModelTC/llmc/tree/main/configs/quantization/combination), 以及[推理后端](https://github.com/ModelTC/llmc/tree/main/configs/quantization/backend)相关的配置

下面的是一个简要的配置例子

```
base:
    seed: &seed 42 # 设置随机种子
model:
    type: model_type # 模型的类型
    path: model path # 模型的路径
    tokenizer_mode: fast # 模型的tokenizer类型
    torch_dtype: auto # 模型的dtype
calib:
    name: pileval # 校准数据集名
    download: False # 校准数据集是否在线下载
    path: calib data path # 校准数据集路径
    n_samples: 512 # 校准数据集的数量
    bs: 1 # 校准数据集的batch size
    seq_len: 512 # 校准数据集的长度
    preproc: pileval_smooth # 校准数据集的预处理方式
    seed: *seed # 校准数据集的随机种子
eval:
    eval_pos: [pretrain, transformed, fake_quant] # 评测的位点
    name: wikitext2 # 评测数据集的名字
    download: False # 评测数据集是否在线下载
    path: eval data path # 评测数据集的路径
    bs: 1 # 评测数据集的batch size
    seq_len: 2048 # 评测数据集的长度
    eval_token_consist: False # 是否评测量化模型和原始模型输出token的一致性
quant:
    method: SmoothQuant # 压缩方法
    weight:
        bit: 8 # 权重的量化bit数
        symmetric: True # 权重量化是否是对称量化
        granularity: per_channel # 权重量化的粒度
    act:
        bit: 8 # 激活的量化bit数
        symmetric: True # 激活量化是否是对称量化
        granularity: per_token # 激活量化的粒度
    speical: # 量化算法需要的特殊参数，可参照每个算法的配置文件的注释以及原论文掌握其用法
save:
    save_vllm: False # 是否保存真实量化的模型，以供VLLM推理
    save_sgl: False # 是否保存真实量化的模型，以供Sglang推理
    save_autoawq: False # 是否保存真实量化的模型，以供AutoAWQ推理
    save_mlcllm: False # 是否保存真实量化的模型，以供MLC-LLM推理
    save_trans: False # 是否保存权重变换之后的模型
    save_fake: False # 是否保存伪量化的权重
    save_path: /path/to/save # 保存路径
```

# 配置的详细说明

## base

<font color=792ee5> base.seed </font>

设置随机种子，用于整个框架的所有随机种子的设定

## model

<font color=792ee5> model.type </font>

模型的类型，可支持Llama,Qwen2,Llava,Gemma2等模型，可以从[这里](https://github.com/ModelTC/llmc/blob/main/llmc/models/__init__.py)查看llmc支持的所有模型

<font color=792ee5> model.path </font>

模型的权重路径，llmc目前只支持hugging face格式的模型，可以用以下的代码检测是否可以正常load

```
from transformers import AutoModelForCausalLM, AutoConfig


model_path = # 模型的权重路径
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
如果上述代码不可以load你所给的模型，可能原因有

1. 你的模型格式并不是hugging face格式

2. 你的tansformers版本太低了，可以执行`pip install transformers --upgrade`升级

llmc运行之前确保上述代码能加载成功你的模型，否则llmc也无法加载你的模型

<font color=792ee5> model.tokenizer_mode </font>

选择使用slow还是fast的tokenizer

<font color=792ee5> model.torch_dtype </font>

设置模型权重的数据类型，可以选择以下几种类型

1. auto

2. torch.float16

3. torch.bfloat16

3. torch.float32

其中auto将跟随权重文件原本的数据类型设置


## calib

<font color=792ee5> calib.name </font>

校准数据集的名称，目前支持以下几种校准数据集

1. pileval

2. wikitext2

3. c4

4. ptb

5. custom

其中custom表示使用用户自定义的校准数据集，具体使用说明参考文档的进阶用法的[自定义校准数据集章节](https://llmc-zhcn.readthedocs.io/en/latest/advanced/custom_dataset.html)

<font color=792ee5> calib.download </font>

表示该校准数据集是否需要运行时在线下载

如果设置True，则无需设置calib.path，llmc会自动联网下载数据集

如果设置False，则需要设置calib.path，llmc会从该地址读取数据集，全程也无需联网运行llmc

<font color=792ee5> calib.path </font>

如果设置calib.download为False，则需要设置calib.path，表示存储校准数据集的路径

其中该路径存储的数据，需要是arrow格式的数据集

从hugging face下载arrow格式的数据集，可以使用以下代码
```
from datasets import load_dataset
calib_dataset = load_dataset(数据集名)
calib_dataset.save_to_disk(保存路径)
```
加载该格式的数据集可以使用
```
from datasets import load_from_disk
data = load_from_disk(数据集路径)
```
llmc已经提供了上述数据集的下载脚本

校准数据集可以在[这里](https://github.com/ModelTC/llmc/blob/main/tools/download_calib_dataset.py)下载

执行命令是`python download_calib_dataset.py --save_path [校准数据集保存路径]`

测试数据集可以在[这里](https://github.com/ModelTC/llmc/blob/main/tools/download_eval_dataset.py)下载

执行命令是`python download_eval_dataset.py --save_path [测试数据集保存路径]`

如果用户想用更多的数据集，就可以参考上面的arrow格式数据集的下载方式，自行修改

<font color=792ee5> calib.n_samples </font>

选择n_samples条数据用于校准

<font color=792ee5> calib.bs </font>

将校准数据以calib.bs为batch size进行打包，如果设置为-1，表示将全部数据打包成一个batch数据

<font color=792ee5> calib.seq_len </font>

校准数据的长度

<font color=792ee5> calib.preproc </font>

校准数据的预处理方式，目前llmc实现了多种预处理方式

1. wikitext2_gptq

2. ptb_gptq

3. c4_gptq

4. pileval_awq

5. pileval_smooth

6. pileval_omni

7. general

8. random_truncate_txt

除了general，其余预处理均可以在[这里](https://github.com/ModelTC/llmc/blob/main/llmc/data/dataset/specified_preproc.py)找到实现方式

general在[base_dataset](https://github.com/ModelTC/llmc/blob/main/llmc/data/dataset/base_dataset.py)中的general_preproc函数中实现

<font color=792ee5> calib.seed </font>

数据预处理中的随机种子，默认跟随base.seed的设置


## eval
llmc默认支持评测量化模型的困惑度(PPL), 以及量化模型和原始模型输出token的一致性。此外还支持通过harness和opencompass评测下游任务的精度（可见[评测章节v1](https://llmc-zhcn.readthedocs.io/en/latest/advanced/model_test_v1.md)和[v2](https://llmc-zhcn.readthedocs.io/en/latest/advanced/model_test_v2.md)）

<font color=792ee5> eval.eval_pos </font>

表示评测PPL的位点，目前支持三个位点可以被评测

1. pretrain

2. transformed

3. fake_quant

eval_pos需要给一个列表，列表可以为空，空列表表示不进行测试

<font color=792ee5> eval.name </font>

测试数据集的名称，目前支持以下几种测试数据集

1. wikitext2

2. c4

3. ptb

测试数据集下载方式参考calib.name校准数据集

<font color=792ee5> eval.download </font>

表示该测试据集是否需要运行时在线下载，参考calib.download

<font color=792ee5> eval.path </font>

参考calib.path

<font color=792ee5> eval.bs </font>

测试的batch size

<font color=792ee5> eval.seq_len </font>

测试的数据长度

<font color=792ee5> eval.inference_per_block </font>

llmc仅支持单卡运行，如果你的模型太大，在测试的时候，单张卡的显存放不下整个模型，那么就需要打开inference_per_block，使用per block进行推理测试，同时在不爆显存的前提下，适当提高bs以提高推理速度

下面的是一个配置例子
```
bs: 10
inference_per_block: True
```

<font color=792ee5> 同时测试多个数据集 </font>

llmc也支持同时评测多个数据集的PPL

下面是评测单个wikitext2数据集的例子

```
eval:
    name: wikitext2
    path: wikitext2的数据集路径
```

下面是评测多个数据集的例子

```
eval:
    name: [wikitext2, c4, ptb]
    path: 这几个数据集的共有上层目录
```

需要注意的是，多个数据集评测的name需要以列表形式表示，同时需要遵循以下目录规则

- 共有上层目录
    - wikitext2
    - c4
    - ptb

如果直接使用llmc的[下载脚本](https://github.com/ModelTC/llmc/blob/main/tools/download_eval_dataset.py)，则共有上层目录就是`--save_path`所指定的数据集保存路径

<font color=792ee5> eval.eval_token_consist </font>

表示是否评测量化模型和原始模型输出token的一致性，取值范围[0,1], 越接近1越说明量化模型的性能越接近原始模型

## quant

<font color=792ee5> quant.method </font>

使用的量化算法名，llmc支持的所有量化算法可以在[这里](https://github.com/ModelTC/llmc/blob/main/llmc/compression/quantization/__init__.py)查看

<font color=792ee5> quant.weight </font>

权重的量化设置

<font color=792ee5> quant.weight.bit </font>

权重的量化bit数

<font color=792ee5> quant.weight.symmetric </font>

权重的量化对称与否

<font color=792ee5> quant.weight.granularity </font>

权重的量化粒度，支持以下粒度

1. per_tensor

2. per_channel

3. per_group

<font color=792ee5> quant.weight.group_size </font>

当权重是per-group量化时，其表示group的大小

<font color=792ee5> quant.weight.ste </font>

在权重量化的取整过程中，是否用直通估计器(straight-through estimator)来使round函数可以产生梯度以便进行反向传播

<font color=792ee5> quant.weight.calib </font>

权重的校准方法，默认采用minmax，除此之外，llmc还支持learnable，mse两种方法，可能会取得更好的结果


<font color=792ee5> quant.act </font>

激活的量化设置

<font color=792ee5> quant.act.bit </font>

激活的量化bit数字

<font color=792ee5> quant.act.symmetric </font>

激活的量化对称与否

<font color=792ee5> quant.act.granularity </font>

激活的量化粒度，支持以下粒度

1. per tensor

2. per token

3. per head

<font color=792ee5> quant.act.ste </font>

在激活量化的取整过程中，是否用直通估计器(straight-through estimator)来使round函数可以产生梯度以便进行反向传播

<font color=792ee5> quant.act.calib </font>

激活的校准方法，默认采用minmax，且只支持minmax

其中如果quant.method设置的为RTN，激活量化可以支持静态per tensor设置，下面是，权重静态per-channel量化，激活静态per tensor量化的配置和激活动态per token 8bit量化的配置

```
quant:
    method: RTN
    # 静态per-channel量化
    weight:
        bit: 8
        symmetric: True
        granularity: per_channel

    # 静态per-tensor量化
    act:
        bit: 8
        symmetric: True
        granularity: per_tensor
        static: True
```

```
quant:
    method: RTN
    #静态per-channel量化
    weight:
        bit: 8
        symmetric: True
        granularity: per_channel

    # 动态per-tensor量化
    act:
        bit: 8
        symmetric: True
        granularity: per_token
```

## sparse

<font color=792ee5> sparse.method </font>

使用的稀疏化算法名，这包含对[模型的稀疏化](https://github.com/ModelTC/LightCompress/blob/main/llmc/compression/sparsification/__init__.pyn)和对视觉token的[reduction](https://github.com/ModelTC/LightCompress/blob/main/llmc/compression/token_reduction/__init__.py)，所有支持算法可以在文件中查看。

值得说明的是针对模型稀疏化，需要指定具体的算法名称，而token reduction只需要先指定为`TokenReduction`，在`special`中继续指定具体的算法。

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

<font color=792ee5> save.save_vllm </font>

是否保存为[VLLM](https://github.com/vllm-project/vllm)推理后端支持的真实量化模型

当开启该选项时，你会发现保存的模型权重显著变小(真实量化)，同时可以通过VLLM后端来直接加载推理，提高推理速度以及降低显存占用，有关于[VLLM](https://github.com/vllm-project/vllm)推理后端的内容见[该章节](https://llmc-zhcn.readthedocs.io/en/latest/backend/vllm.html)

<font color=792ee5> save.save_sgl </font>

是否保存为[Sglang](https://github.com/sgl-project/sglang)推理后端支持的真实量化

当开启该选项时，你会发现保存的模型权重显著变小(真实量化)，同时可以通过[Sglang](https://github.com/sgl-project/sglang)后端来直接加载推理，提高推理速度以及降低显存占用，有关于[Sglang](https://github.com/sgl-project/sglang)推理后端的内容见[该章节](https://llmc-zhcn.readthedocs.io/en/latest/backend/sglang.html)


<font color=792ee5> save.save_autoawq </font>

是否保存为[AutoAWQ](https://github.com/casper-hansen/AutoAWQ)推理后端支持的真实量化模型

当开启该选项时，你会发现保存的模型权重显著变小(真实量化)，同时可以通过[AutoAWQ](https://github.com/casper-hansen/AutoAWQ)后端来直接加载推理，提高推理速度以及降低显存占用，有关于[AutoAWQ](https://github.com/casper-hansen/AutoAWQ)推理后端的内容见[该章节](https://llmc-zhcn.readthedocs.io/en/latest/backend/autoawq.html)

<font color=792ee5> save.save_mlcllm </font>

是否保存为[MLC-LLM](https://github.com/mlc-ai/mlc-llm)推理后端支持的真实量化模型

当开启该选项时，你会发现保存的模型权重显著变小(真实量化)，同时可以通过[MLC-LLM](https://github.com/mlc-ai/mlc-llm)后端来直接加载推理，提高推理速度以及降低显存占用，有关于[MLC-LLM](https://github.com/mlc-ai/mlc-llm)推理后端的内容见[该章节](https://llmc-zhcn.readthedocs.io/en/latest/backend/mlcllm.html)


<font color=792ee5> save.save_trans </font>

是否保存调整之后的模型权重

保存的该权重，是经过调整之后的更适合量化的权重，其可能包含更少的离群值，其还是以fp16/bf16的格式保存(权重文件大小与原始模型保持一致)，在推理引擎中部署的时候，需要开启推理引擎的`naive量化`功能，即可实现量化推理。

与`save_vllm`等不同的是，其需要该推理引擎来完成真实量化，而`llmc`提供一个更适合量化的浮点模型权重。

例如`SmoothQuant/Os+/AWQ/Quarot`等算法导出的`save_trans`模型，其具有`更少的outliers`，更适合量化。

<font color=792ee5> save.save_fake </font>

是否保存伪量化的模型

<font color=792ee5> save.save_path </font>

保存模型的路径，该路径需要是一个不存在的新的目录路径，否则llmc会终止运行，并发出相应的错误提示
