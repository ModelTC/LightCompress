.. llmc documentation master file, created by
   sphinx-quickstart on Mon Jun 24 10:56:49 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

欢迎使用大模型压缩工具llmc!
================================

llmc是一个用于大模型压缩的工具，支持多种模型和多种压缩算法。

github链接: https://github.com/ModelTC/llmc

arxiv链接: https://arxiv.org/abs/2405.06001

.. toctree::
   :maxdepth: 2
   :caption: 快速入门

   quickstart.md


.. toctree::
   :maxdepth: 2
   :caption: 配置说明

   configs.md


.. toctree::
   :maxdepth: 2
   :caption: 进阶用法

   advanced/model_test_v1.md
   advanced/model_test_v2.md
   advanced/custom_dataset.md
   advanced/Vit_quant&img_dataset.md
   advanced/VLM_quant&img-txt_dataset.md
   advanced/mix_bits.md
   advanced/sparsification.md
   advanced/token_reduction.md

.. toctree::
   :maxdepth: 2
   :caption: 量化最佳实践

   practice/awq.md
   practice/awq_omni.md
   practice/quarot_gptq.md

.. toctree::
   :maxdepth: 2
   :caption: 量化推理后端

   backend/vllm.md
   backend/sglang.md
   backend/autoawq.md
   backend/mlcllm.md