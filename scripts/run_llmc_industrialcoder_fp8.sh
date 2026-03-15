#!/usr/bin/env bash
model_name=industrialcoder
method_name=rtn_int_gptq
dataset_name=wikitext

log_name=${model_name}_${method_name}_${dataset_name}
rm -rf ./save_for_vllm/${log_name}/
llmc=.
export PYTHONPATH=$llmc:$PYTHONPATH
config=${llmc}/configs/quantization/backend/vllm/fp8/${log_name}.yml
nnodes=1
nproc_per_node=8

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


torchrun \
--nnodes $nnodes \
--nproc_per_node $nproc_per_node \
--rdzv_id $task_id \
--rdzv_backend c10d \
--rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
${llmc}/llmc/__main__.py --config $config --task_id $task_id |tee ${log_name}.log 
