# VLM 量化和 custom_mm 数据集

llmc目前支持对VLM模型使用图像-文本数据集进行校准并量化

## VLM 量化
当前支持的模型如下：
1. llava

2. intervl2

3. llama3.2

4. qwen2vl

更多的vlm正在实现中

下面是一个配置的例子，可以参考GitHub上的[校准数据集模板](https://github.com/user-attachments/files/18433608/general_custom_data_examples.zip)。

```yaml
model:
    type: Llava
    path: model path
    tokenizer_mode: slow
    torch_dtype: auto
calib:
    name: custom_mm
    download: False
    path: calib data path
    apply_chat_template: True
    add_answer: True # Defalut is False. If set it to Ture, calib data will add answers.
    n_samples: 8
    bs: -1
    seq_len: 512
    padding: True
```

## custom_mm 数据集
custom_mm 数据集格式如下：
```
custom_mm-datasets/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   ├── image3.jpg
│   └── ... (other images)
└── img_qa.json
```

img_qa.json 格式示例：
```json
[
    {
        "image": "images/0a3035bfca2ab920.jpg",
        "question": "Is this an image of Ortigia? Please answer yes or no.",
        "answer": "Yes"
    },
    {
        "image": "images/0a3035bfca2ab920.jpg",
        "question": "Is this an image of Montmayeur castle? Please answer yes or no.",
        "answer": "No"
    },
    {
        "image": "images/0ab2ed007db301d5.jpg",
        "question": "Is this a picture of Highgate Cemetery? Please answer yes or no.",
        "answer": "Yes"
    }
]
```
"answer" 可以不需要

custom_mm数据集中可以存在仅有文本的校准数据（当前llama3.2除外）

## VLM 测评

llmc接入了[lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval)进行各种下游数据集测评，在config的eval中需要指定type为vqa，name中的下游测评数据集参考lmms-eval的标准。

```
eval:
    type: vqa
    name: [mme] # vqav2, gqa, vizwiz_vqa, scienceqa, textvqa
```
