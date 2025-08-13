# Vit 量化和 img 数据集

llmc目前支持对Vit模型使用图像数据集进行校准并量化

## Vit 量化

下面是一个配置的例子

```yaml
model:
    type: Vit
    path: /models/vit-base-patch16-224
    torch_dtype: auto
calib:
    name: imagenet
    type: img
    download: False
    path: img calib datasets path
    n_samples: 32
    bs: 1
    seq_len: 512      # Useless arguments for vit
    preproc: img_general
    seed: *seed
eval:
    eval_pos: [pretrain, fake_quant]
    name: imagenet
    type: acc         # acc: accracy 
    download: False
    path: img datasets path
    seq_len: 2048     # Useless arguments for vit
    bs: 1
    inference_per_block: False
    eval_token_consist: False
```

## img 数据集
img数据集格式要求：img数据集目录下存在图像

img数据集格式示例:
```
images/
├── image1.jpg
├── image2.jpg
├── image3.jpg
└── ... (other images)
```
