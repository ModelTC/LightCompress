export PATH=/mnt/lm_data_afs/wangzining/charles/miniconda3/envs/llmc/bin:$PATH
export PYTHON=/mnt/lm_data_afs/wangzining/charles/miniconda3/envs/llmc/bin/python
export PIP=/mnt/lm_data_afs/wangzining/charles/miniconda3/envs/llmc/bin/pip
export HF_ENDPOINT=https://hf-mirror.com
cd /mnt/lm_data_afs/wangzining/charles/lab/llmc
# model_name=wan_t2v
model_name=wan2_2_t2v
task_name=awq_w_a
# task_name=awq_w_a_s
log_name=${model_name}_${task_name}
rm -rf ./save_for_lightx2v/${model_name}/${task_name}/original
llmc=.
export PYTHONPATH=$llmc:$PYTHONPATH
config=${llmc}/configs/quantization/video_gen/${model_name}/${task_name}.yaml
nnodes=1
nproc_per_node=1

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