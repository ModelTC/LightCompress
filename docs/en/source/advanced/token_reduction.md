

# Token Reduction

LightCompress currently supports token reduction for mainstream multimodal large language models. Configuration is very simple—plug and play.

Here is an example configuration

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

The configuration file contains three core sections, including:

1. **`model`**  
   For model selection, you can choose LLaVA, LLaVA-NeXT, Qwen2.5VL, and LLaVA OneVision, etc. These models cover both image and video tasks. For the detailed list of supported models, see the file. LightCompress will support more models in the future.

2. **`eval`**  
   For the `eval_pos` parameter:  
   - `pretrain` denotes the original model that keeps all visual tokens.  
   - `transformed` denotes the model with token reduction applied.  
   LightCompress integrates lmms-eval to evaluate various downstream datasets. Set `type` to `vqa`, and specify the datasets in `name` following the naming conventions in the lmms-eval documentation.

3. **`sparse`**  
   Set `method` to `TokenReduction` first, and then specify the concrete algorithm and related hyperparameters under `special`. Since each algorithm has different hyperparameters, refer to the configuration files for details.

## Combining Quantization

LightCompress also supports an extreme compression scheme that combines token reduction with quantization. First, choose a quantization algorithm to save a `fake_qunat` model (see the quantization section of the docs). Then load this model and add the `token_reduction` field under `quant`.

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