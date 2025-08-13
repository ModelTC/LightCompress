# Token Reduction

目前LightCompress支持对主流的多模态大语言模型进行token reduction，配置十分简单，即插即用。

下面是一个配置的例子

```yaml
base:
    seed: &seed 42
model:
    type: Llava
    path: model path
    torch_dtype: auto
eval:
    eval_pos: [pretrain, transformed]
    type: vqa
    name: [gqa, mmbench_en_dev, mme]
    bs: 1
    inference_per_block: False
sparse:
    method: TokenReduction
    special:
        method: FastV
        pruning_loc: 3
        rate: 0.778
save:
    save_trans: False
    save_fake: False
    save_path: /path/to/save/
```

配置文件中包含三大核心内容，包括：

1. `model`
在模型选择上，可以选择LLaVA，LLaVA-NeXT，Qwen2.5VL以及LLaVA OneVision等，这些模型涵盖了图像任务和视频任务，详细的模型支持列表可以查阅[文件](https://github.com/ModelTC/LightCompress/blob/main/llmc/models/__init__.py)，未来LightCompress也会支持更多的模型。

2. `eval`
首先，在`eval_pos`参数的选择上，`pretrain`表示原始保留所有视觉token的模型，`transformed`表示应用相应算法进行token reduction的模型。LightCompress接入了[lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval)进行各种下游数据集测评，需要将`type`指定为`vqa`，`name`中的下游测评数据集参考lmms-eval[文档](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/current_tasks.md)中的命名方式。

3. `sparse`
`method`需要首先指定为TokenReduction，在`special`中继续指定具体的算法以及相关的一些超参数。由于每个算法对应的超参数不同，详细的可以参考[配置文件](https://github.com/ModelTC/LightCompress/tree/main/configs/sparsification/methods)。


## 结合量化

LightCompress也支持同时使用token reduction和量化的极致压缩方案，首先需要选择量化算法存储一个`fake_qunat`模型，可以参考量化板块的文档。其次加载这个模型并在`quant`下加入`token_reduction`字段即可。

```yaml
quant:
    method: RTN
    weight:
        bit: 4
        symmetric: False
        granularity: per_group
        group_size: 128
    special:
        actorder: True
        static_groups: True
        percdamp: 0.01
        blocksize: 128
        true_sequential: True
    quant_out: True
    token_reduction:    
        method: FastV
        special:
            pruning_loc: 3
            rate: 0.778
```