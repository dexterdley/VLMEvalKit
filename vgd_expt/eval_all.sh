#!/bin/bash
export NCCL_P2P_DISABLE=1
export SPLIT_THINK=True

MODELS=(
  #"Qwen3-VL-2B-Instruct"
  #"Qwen3-VL-8B-Instruct"
  #Qwen3-VL-2B-Thinking
  #Qwen3-VL-8B-Thinking
  "InternVL3_5-2B"
  "Gemma3-4B"
)
for MODEL in "${MODELS[@]}"
do
    mkdir -p ./outputs/${MODEL}

    rm -rf ./outputs/${MODEL}/${MODEL}_Base/
    rm -rf ./outputs/${MODEL}/${MODEL}_VGD/

    echo "🚀 Starting Distributed Parallel Evaluations. $MODEL"

    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
      --nproc_per_node=4 \
      --master_port=29500 \
      ./vgd_expt/run_vgd.py \
      --config ./vgd_expt/my_qwen_config.json \
      --visual_alpha=0 \
      --model=${MODEL} \
      --work-dir ./outputs/${MODEL}/${MODEL}_Base \
      >> ./outputs/${MODEL}/${MODEL}_Base.txt &

    CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun \
      --nproc_per_node=4 \
      --master_port=29501 \
      ./vgd_expt/run_vgd.py \
      --config ./vgd_expt/my_qwen_config.json \
      --visual_alpha=1.5 \
      --model=${MODEL} \
      --work-dir ./outputs/${MODEL}/${MODEL}_VGD \
      >> ./outputs/${MODEL}/${MODEL}_VGD.txt &

      wait     
      echo "✅ Finished $MODEL"
done
echo "✅ Evaluations Completed."