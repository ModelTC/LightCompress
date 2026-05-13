<div align="center" style="font-family: charter;">
<h1> LightCompress：迈向准确且高效的AIGC大模型压缩 </h1>

<img src="./imgs/llmc.png" alt="llmc" width="75%" />

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/ModelTC/LightCompress)
[![arXiv](https://img.shields.io/badge/LLMC-2405.06001-b31b1b)](https://arxiv.org/abs/2405.06001)
[![Discord Banner](https://img.shields.io/discord/1139835312592392214?logo=discord&logoColor=white)](https://discord.com/invite/NfJzbkK3jY)
[![QQ](https://img.shields.io/badge/QQ-EB1923?logo=tencent-qq&logoColor=white)](http://qm.qq.com/cgi-bin/qm/qr?_wv=1027&k=I9IGPWWj8uuRXWH3_ELWjouf6gkIMgUl&authKey=GA3WbFAsm90ePJf%2FCbc7ZyXXq4ShQktlBaLxgqS5yuSPAsr3%2BDKMRdosUiLYoilO&noverify=0&group_code=526192592)
[![Doc](https://img.shields.io/badge/docs-English-99cc2)](https://llmc-en.readthedocs.io/en/latest/)
[![Doc](https://img.shields.io/badge/文档-中文-99cc2)](https://llmc-zhcn.readthedocs.io/en/latest/)&#160;

**\[ [English](README.md) | 中文 \]**

</div>

> **📢 提示**: 本仓库原名为 **LLMC**，现已更名为 **LightCompress**。

**LightCompress** 是一个开箱即用的工具，专为压缩AIGC大模型(LLM, VLM, Diffusion ...)设计，利用最先进的压缩算法提高效率并减少模型体积，同时不影响预测精度。你可以通过以下命令下载可以运行LightCompress的docker镜像，中国大陆用户推荐使用阿里云docker。

```shell
# Docker Hub: https://hub.docker.com/r/llmcompression/llmc
docker pull llmcompression/llmc:pure-latest

# 阿里云镜像: registry.cn-hangzhou.aliyuncs.com/yongyang/llmcompression:[tag]
docker pull registry.cn-hangzhou.aliyuncs.com/yongyang/llmcompression:pure-latest
```

**社区**： [Discord 服务器](https://discord.com/invite/NfJzbkK3jY)、[腾讯 QQ 群](http://qm.qq.com/cgi-bin/qm/qr?_wv=1027&k=I9IGPWWj8uuRXWH3_ELWjouf6gkIMgUl&authKey=GA3WbFAsm90ePJf%2FCbc7ZyXXq4ShQktlBaLxgqS5yuSPAsr3%2BDKMRdosUiLYoilO&noverify=0&group_code=526192592)。

**文档**： [English](https://llmc-en.readthedocs.io/en/latest/)、[中文](https://llmc-zhcn.readthedocs.io/en/latest/)。

> **推荐 Python 版本**：建议本地开发和安装使用 **Python 3.11**。这与项目的 Docker 镜像和 CI 配置保持一致，并且对当前依赖集合而言通常比 Python 3.12 更稳定。

## :fire: 最新动态

- **2025年8月13日:** 🚀 我们已开源针对 **视觉语言模型（VLMs）** 的压缩方案，支持共计超过 **20 种算法**，涵盖 **token reduction** 和 **quantization**。此次发布为多模态任务提供了灵活、即插即用的压缩策略。具体请参阅[文档](https://llmc-zhcn.readthedocs.io/en/latest/advanced/token_reduction.html)。

- **2025年5月12日：** 🔥 我们现已全面支持 **`Wan2.1`** 系列视频生成模型的量化，并支持导出真实量化的 **INT8/FP8** 权重，兼容 [lightx2v](https://github.com/ModelTC/lightx2v) 推理框架。详情请参考 [lightx2v 使用文档](https://llmc-zhcn.readthedocs.io/en/latest/backend/lightx2v.html)。

- **2025年2月7日:** 🔥 我们现已全面支持 **`DeepSeekv3`**、**`DeepSeek-R1`** 和 **`DeepSeek-R1-zero`** 等 671B 大规模 **`MOE`** 模型的量化。 您可以直接加载 `FP8` 权重，无需额外转换，使用单张 80G 显存的 GPU 即可运行 `AWQ` 和 `RTN` 量化，同时还支持导出真实量化的 **INT4/INT8** 权重

- **2024年11月20日:** 🔥 我们现已全面支持✨`DeepSeekv2(2.5)`等`MOE`模型以及✨`Qwen2VL`、`Llama3.2`等`VLM`模型的量化。支持的量化方案包括✅整型量化、✅浮点量化，以及✅AWQ、✅GPTQ、✅SmoothQuant 和 ✅Quarot 等先进算法。

- **2024年11月12日:** 🔥 我们新增对各种模型和算法的💥`激活静态 per-tensor量化`支持，涵盖✅整型量化和✅浮点量化，进一步优化性能和效率。同时支持导出`✨真实量化模型`，并使用 [VLLM](https://github.com/vllm-project/vllm)和[SGLang](https://github.com/sgl-project/sglang)后端进行推理加速，具体请参阅[VLLM文档](https://llmc-zhcn.readthedocs.io/en/latest/backend/vllm.html)和[SGLang文档](https://llmc-zhcn.readthedocs.io/en/latest/backend/sglang.html)。

- **2024年9月26日:** 🔥 我们现在支持从🚀 `LLMC`导出💥 `FP8 量化（E4M3，E5M2）`模型到一些先进的推理后端，例如[VLLM](https://github.com/vllm-project/vllm)和[SGLang](https://github.com/sgl-project/sglang)。关于详细使用方法，请参阅[VLLM文档](https://llmc-zhcn.readthedocs.io/en/latest/backend/vllm.html)和[SGLang文档](https://llmc-zhcn.readthedocs.io/en/latest/backend/sglang.html)。

<details close>
<summary>更早动态</summary>

- **2024年9月24日:** 🔥 我们正式发布了 ✨`Llama-3.1-405B` 的 ✅INT4 和 ✅INT8 模型，这些模型通过 🚀`LLMC` 使用 `save_lightllm` 模式进行量化。你可以在[此处](https://huggingface.co/Dongz/llama31-405b-quant)下载模型参数。

- **2024年9月23日:** 🔥 我们现在支持从 🚀`LLMC` 导出 ✨`真正量化的(INT4, INT8)` 模型到先进推理后端，例如 [VLLM](https://github.com/vllm-project/vllm), [SGLang](https://github.com/sgl-project/sglang), [AutoAWQ](https://github.com/casper-hansen/AutoAWQ), 和 [MLC-LLM](https://github.com/mlc-ai/mlc-llm) 用于量化推理部署，从而实现 ✨`减少内存使用` 和 ✨`加快推理速度`。
  详细使用方法，请参考 [VLLM 文档](https://llmc-zhcn.readthedocs.io/en/latest/backend/vllm.html)、[SGLang 文档](https://llmc-zhcn.readthedocs.io/en/latest/backend/sglang.html)、[AutoAWQ 文档](https://llmc-zhcn.readthedocs.io/en/latest/backend/autoawq.html) 和 [MLC-LLM 文档](https://llmc-zhcn.readthedocs.io/en/latest/backend/mlcllm.html)。

- **2024年9月9日:** 🔥 我们提供了一些最佳实践配置，帮助提升性能（参见最佳实践[此处](https://llmc-zhcn.readthedocs.io/en/latest/)）。

- **2024年9月3日:** 🔥 我们支持通过[opencompass](https://github.com/open-compass/opencompass) 评估 🚀`LLMC` 模型。请参考此[文档](https://llmc-zhcn.readthedocs.io/en/latest/advanced/model_test_v2.html)试用！

- **2024年8月22日:** 🔥我们支持许多小型语言模型，包括当前SOTA的 [SmolLM](https://huggingface.co/collections/HuggingFaceTB/smollm-6695016cad7167254ce15966)(参见[支持的模型列表](#supported-model-list))。

- **2024年8月22日:** 🔥此外，我们还支持通过我们修改的 [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) 进行下游任务评估 🤗。具体操作，用户可以先采用 `save_trans` 模式（参见 [配置](https://llmc-zhcn.readthedocs.io/en/latest/configs.html) 中的 `save` 部分）保存权重修改后的模型。在获得转换模型后，可以直接参考 [run_lm_eval.sh](scripts/run_lm_eval.sh) 对量化模型进行评估。更多细节请见[此处](https://llmc-zhcn.readthedocs.io/en/latest/advanced/model_test_v1.html)。

- **2024年7月23日:** 🍺🍺🍺 我们发布了全新的基准论文：

  [**LLMC: Benchmarking Large Language Model Quantization with a Versatile Compression Toolkit**](https://arxiv.org/abs/2405.06001v2)。

  [Ruihao Gong\*](https://xhplus.github.io/), [Yang Yong\*](https://github.com/helloyongyang), [Shiqiao Gu\*](https://github.com/gushiqiao), [Yushi Huang\*](https://github.com/Harahan), [Chengtao Lv](https://scholar.google.com/citations?user=r8vseSUAAAAJ&hl=en), [Yunchen Zhang](https://scholar.google.com/citations?user=glkWFyUAAAAJ&hl=en), [Xianglong Liu📧](https://xlliu-beihang.github.io/), [Dacheng Tao](https://scholar.google.com/citations?user=RwlJNLcAAAAJ&hl=en)

  (\* 表示同等贡献，📧 表示通讯作者。)

- **2024年7月16日:** 🔥我们现在支持 Wanda/Naive（幅度）进行 LLM 稀疏化和逐层混合比特量化！

- **2024年7月14日:** 🔥我们现在支持基于旋转的量化 QuaRot！

- **2024年5月17日:** 🚀 我们现在支持一些先进的大型模型，例如 LLaVA、Mixtral、LLaMA V3 和 Qwen V2。快来试试吧！

- **2024年5月13日:** 🍺🍺🍺 我们发布了量化基准论文：

  [**LLM-QBench: A Benchmark Towards the Best Practice for Post-training Quantization of Large Language Models**](https://arxiv.org/abs/2405.06001)。

  [Ruihao Gong\*](https://xhplus.github.io/), [Yang Yong\*](https://github.com/helloyongyang), [Shiqiao Gu\*](https://github.com/gushiqiao), [Yushi Huang\*](https://github.com/Harahan), [Yunchen Zhang](https://scholar.google.com/citations?user=glkWFyUAAAAJ&hl=en), [Xianglong Liu📧](https://xlliu-beihang.github.io/), [Dacheng Tao](https://scholar.google.com/citations?user=RwlJNLcAAAAJ&hl=en)

  (\* 表示同等贡献，📧 表示通讯作者。)

  <div align=center>
   <img src="./imgs/best_practice.png" alt="comp" width="800" />
  </div>

  我们模块化且公平地基准测试了量化技术，考虑了校准成本、推理效率和量化准确性。在多种模型和数据集上进行了近600次实验，得出了三个关于校准数据、算法管道和量化配置选择的有见地的结论。基于这些结论，设计了一种LLM后训练量化管道的最佳实践，以在各种场景下实现最佳的准确性和效率平衡。

- **2024年3月7日:** 🚀 我们发布了一个功能强大且高效的LLM压缩工具的量化部分。值得注意的是，我们的基准论文即将发布😊。

</details>

## 🚀 亮点功能

- 💥**综合算法支持**: 提供广泛的 ✨`SOTA压缩算法` 支持，包括 ✅量化、✅混合精度量化 和 ✅稀疏化，同时保持与原始仓库一致的精度。我们还提供 ✨`量化最佳实践`（参见✨`最佳实践` 章节[此处](https://llmc-zhcn.readthedocs.io/en/latest/)），确保最佳性能和效率。

- 💥**支持的格式**: 支持 ✨`量化`（整型和浮点）和 ✨`稀疏化`，具体包括 ✅权重激活量化、✅权重量化、✅混合精度量化，以及 ✅结构化 和 ✅非结构化稀疏化。

- 💥**广泛模型支持**: 支持多种 ✨`LLM模型`，包括 ✅LLama、✅Mistral、✅InternLM2、✅Qwen2 等，以及 ✅MOE(DeepSeekv3, Deepseek-R1) 和 ✅VLM(Llama3.2-vision, Qwen2-vl) 模型（参见[支持的模型列表](#supported-model-list)）。

- 💥**多后端兼容性**: 无缝集成多个后端，增强部署灵活性。多种量化设置和模型格式兼容广泛的后端和硬件平台，例如 ✅VLLM、✅Sglang、✅LightLLM、✅MLC-LLM 和 ✅AutoAWQ，使其高度灵活（参见✨`推理后端` 章节 [此处](https://llmc-zhcn.readthedocs.io/en/latest/)）。

- 💥**性能效率**: 支持大规模LLM的量化，例如 ✨`Llama3.1-405B` 和 ✨`DeepSeek-R1-671B`，并可在 `单个 A100/H100/H800 GPU` 上评估 PPL。

## ⚙️ 快速上手

请参阅 🚀`快速入门`章节[此处](https://llmc-zhcn.readthedocs.io/en/latest/)。

## :robot: 支持的模型

- ✅ [BLOOM](https://huggingface.co/bigscience/bloom)
- ✅ [LLaMA](https://github.com/facebookresearch/llama)
- ✅ [LLaMA V2](https://huggingface.co/meta-llama)
- ✅ [StarCoder](https://github.com/bigcode-project/starcoder)
- ✅ [OPT](https://huggingface.co/docs/transformers/model_doc/opt)

<details>
<summary>更多模型</summary>

- ✅ [Falcon](https://huggingface.co/docs/transformers/model_doc/falcon)
- ✅ [InternLM2](https://huggingface.co/internlm)
- ✅ [Mistral](https://huggingface.co/docs/transformers/model_doc/mistral)
- ✅ [LLaMA V3](https://huggingface.co/meta-llama)
- ✅ [Mixtral](https://huggingface.co/docs/transformers/model_doc/mixtral)
- ✅ [Qwen V2](https://github.com/QwenLM/Qwen2)
- ✅ [LLaVA](https://github.com/haotian-liu/LLaVA)
- ✅ [InternLM2.5](https://huggingface.co/internlm)
- ✅ [StableLM](https://github.com/Stability-AI/StableLM)
- ✅ [Gemma2](https://huggingface.co/docs/transformers/main/en/model_doc/gemma2)
- ✅ [Phi2](https://huggingface.co/microsoft/phi-2)
- ✅ [Phi 1.5](https://huggingface.co/microsoft/phi-1_5)
- ✅ [MiniCPM](https://github.com/OpenBMB/MiniCPM)
- ✅ [SmolLM](https://huggingface.co/collections/HuggingFaceTB/smollm-6695016cad7167254ce15966)
- ✅ [DeepSeekv2.5](https://huggingface.co/deepseek-ai/DeepSeek-V2.5)
- ✅ [LLaMA V3.2 Vision](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision)
- ✅ [Qwen MOE](https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B)
- ✅ [Qwen2-VL](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)
- ✅ [InternVL2](https://huggingface.co/OpenGVLab/InternVL2-2B)

</details>

您可参考 `llmc/models/*.py` 添加自定义模型。

## :bus: 支持的后端

- ✅ [VLLM](https://github.com/vllm-project/vllm)
- ✅ [LightLLM](https://github.com/ModelTC/lightllm)
- ✅ [Sglang](https://github.com/sgl-project/sglang)
- ✅ [MLC-LLM](https://github.com/mlc-ai/mlc-llm)
- ✅ [AutoAWQ](https://github.com/casper-hansen/AutoAWQ)

## 💡 支持的算法

### 量化

- ✅ Naive
- ✅ [AWQ](https://arxiv.org/abs/2306.00978)
- ✅ [GPTQ](https://arxiv.org/abs/2210.17323)
- ✅ [SmoothQuant](https://arxiv.org/abs/2211.10438)
- ✅ [OS+](https://arxiv.org/abs/2304.09145)

<details>
<summary>更多算法</summary>

- ✅ [OmniQuant](https://arxiv.org/abs/2308.13137)
- ✅ [NormTweaking](https://arxiv.org/abs/2309.02784)
- ✅ [AdaDim](https://arxiv.org/pdf/2309.15531.pdf)
- ✅ [QUIK](https://arxiv.org/abs/2310.09259)
- ✅ [SpQR](https://arxiv.org/abs/2306.03078)
- ✅ [DGQ](https://arxiv.org/abs/2310.04836)
- ✅ [OWQ](https://arxiv.org/abs/2306.02272)
- ✅ [LLM.int8()](https://arxiv.org/abs/2208.07339)
- ✅ [HQQ](https://mobiusml.github.io/hqq_blog/)
- ✅ [QuaRot](https://arxiv.org/abs/2404.00456)
- ✅ [SpinQuant](https://arxiv.org/abs/2405.16406) **([见此分支](https://github.com/ModelTC/llmc/tree/dev_spinquant))**
- ✅ [TesseraQ](https://arxiv.org/abs/2410.19103)

</details>

### 剪枝

- ✅ Naive(Magnitude)
- ✅ [Wanda](https://arxiv.org/abs/2306.11695)
- ✅ [ShortGPT](https://arxiv.org/abs/2403.03853)

## 🤝 致谢

本项目参考了以下仓库：

- [mit-han-lab/llm-awq](https://github.com/mit-han-lab/llm-awq)
- [mit-han-lab/smoothquant](https://github.com/mit-han-lab/smoothquant)
- [OpenGVLab/OmniQuant](https://github.com/OpenGVLab/OmniQuant)
- [IST-DASLab/gptq](https://github.com/IST-DASLab/gptq)
- [ModelTC/Outlier_Suppression_Plus](https://github.com/ModelTC/Outlier_Suppression_Plus)

<details>
<summary>更多相关实现</summary>

- [IST-DASLab/QUIK](https://github.com/IST-DASLab/QUIK)
- [Vahe1994/SpQR](https://github.com/Vahe1994/SpQR)
- [ilur98/DGQ](https://github.com/ilur98/DGQ)
- [xvyaward/owq](https://github.com/xvyaward/owq)
- [TimDettmers/bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
- [mobiusml/hqq](https://github.com/mobiusml/hqq)
- [spcl/QuaRot](https://github.com/spcl/QuaRot)
- [locuslab/wanda](https://github.com/locuslab/wanda)
- [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [facebookresearch/SpinQuant](https://github.com/facebookresearch/SpinQuant)
- [Intelligent-Computing-Lab-Yale/TesseraQ](https://github.com/Intelligent-Computing-Lab-Yale/TesseraQ)

</details>

## 🌟 Star 历史

[![Star History Chart](https://api.star-history.com/svg?repos=ModelTC/llmc&type=Timeline)](https://star-history.com/#ModelTC/llmc&Timeline)

## ✏️ 引用

如果您觉得本工具包或相关论文对您的研究有帮助，请引用：

```
@inproceedings{DBLP:conf/emnlp/GongYGHLZT024,
  author    = {Ruihao Gong and Yang Yong and Shiqiao Gu and Yushi Huang and Chengtao Lv and Yunchen Zhang and Dacheng Tao and Xianglong Liu},
  title     = {LLMC: Benchmarking Large Language Model Quantization with a Versatile Compression Toolkit},
  booktitle = {EMNLP (Industry Track)},
  year      = {2024},
  pages     = {132--152},
  url       = {https://aclanthology.org/2024.emnlp-industry.12}
}
```
