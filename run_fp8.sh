#!/bin/bash
export PATH=/mnt/lm_data_afs/wangzining/charles/miniconda3/envs/llmc/bin:$PATH
export PYTHON=/mnt/lm_data_afs/wangzining/charles/miniconda3/envs/llmc/bin/python
export PIP=/mnt/lm_data_afs/wangzining/charles/miniconda3/envs/llmc/bin/pip
export HF_ENDPOINT=https://hf-mirror.com

cd /mnt/lm_data_afs/wangzining/charles/lab/llmc


model_name=thinking_model   
method_name=fp8             
dataset_name=wikitext
# ==============================

log_name=${model_name}_${method_name}_${dataset_name}
rm -rf ./save_for_vllm/${log_name}/

llmc=.
export PYTHONPATH=$llmc:$PYTHONPATH


config=${llmc}/configs/quantization/backend/vllm/fp8/thinking_model_fp8.yml

nnodes=1
nproc_per_node=4  


find_unused_port() {
    while true; do
        port=$(shuf -i 10000-60000 -n 1)
        if ! ss -tuln | grep -q ":$port "; then
            echo "$port"
            return 0
        fi
    done
}
UNUSED_PORT=$(find_unused_port)
MASTER_ADDR=127.0.0.1
MASTER_PORT=$UNUSED_PORT
task_id=$UNUSED_PORT

echo "开始执行任务，日志将保存在 ${log_name}.log"

torchrun \
--nnodes $nnodes \
--nproc_per_node $nproc_per_node \
--rdzv_id $task_id \
--rdzv_backend c10d \
--rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
${llmc}/llmc/__main__.py --config $config --task_id $task_id | tee ${log_name}.log
