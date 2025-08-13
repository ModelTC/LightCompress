.. llmc documentation master file, created by
   sphinx-quickstart on Mon Jun 24 10:56:49 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to llmc's documentation!
================================

llmc is a tool for large model compression, supporting a variety of models and a variety of compression algorithms.

github: https://github.com/ModelTC/llmc

arxiv: https://arxiv.org/abs/2405.06001

.. toctree::
   :maxdepth: 2
   :caption: Quick Start

   quickstart.md


.. toctree::
   :maxdepth: 2
   :caption: Configs

   configs.md

.. toctree::
   :maxdepth: 2
   :caption: Advanced

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
   :caption: Best Practice

   practice/awq.md
   practice/awq_omni.md
   practice/quarot_gptq.md

.. toctree::
   :maxdepth: 2
   :caption: Backbend

   backend/vllm.md
   backend/sglang.md
   backend/autoawq.md
   backend/mlcllm.md
