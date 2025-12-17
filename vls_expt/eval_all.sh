#!/bin/bash
MODEL="Qwen3-VL-2B-Instruct"
mkdir -p ./outputs/test/
echo "🚀 Starting Distributed Parallel Evaluations. $MODEL"

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
#  --nproc_per_node=4 \
#  --master_port=29500 \
#  ./vls_expt/run_vls.py \
#  --config ./vls_expt/my_qwen_config.json \
#  --visual_alpha=0 \
#  --work-dir ./outputs/${MODEL}/${MODEL}_Base \
#  > ./outputs/${MODEL}/${MODEL}_Base.txt &

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --nproc_per_node=4 \
  --master_port=29501 \
  ./vls_expt/run_vls.py \
  --config ./vls_expt/my_qwen_config.json \
  --visual_alpha=0.0 \
  --work-dir ./outputs/${MODEL}/${MODEL}_VGD \
  > ./outputs/${MODEL}/${MODEL}_Base.txt &

wait
echo "✅ Evaluations Completed."