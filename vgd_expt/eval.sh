#!/bin/bash
export NCCL_P2P_DISABLE=1
export SPLIT_THINK=True

#CUDA_VISIBLE_DEVICES=0 python ./vgd_expt/run_vgd.py --config ./vgd_expt/my_qwen_config.json --visual_alpha=0.0 > ./outputs/Qwen3-VL-8B-Instruct/Qwen3-VL-8B-Instruct_Base.txt
#CUDA_VISIBLE_DEVICES=0 python ./vgd_expt/run_vgd.py --config ./vgd_expt/my_qwen_config.json --visual_alpha=1.5 > ./outputs/Qwen3-VL-8B-Instruct/Qwen3-VL-8B-Instruct_VGD.txt

MODELS=(
  #"Qwen3-VL-2B-Thinking"
  #"Qwen3-VL-8B-Thinking"
  #"Gemma3-4B"
  #"Qwen3-VL-2B-Instruct"
  #"Qwen3-VL-8B-Instruct"
  #"Qwen2.5-VL-7B-Instruct"
  "InternVL3_5-2B"
  "InternVL3_5-8B"
)
for SEED in 42 55 69
do
  for MODEL in "${MODELS[@]}"
  do
      mkdir -p ./outputs/${MODEL}

      rm -rf ./outputs/${MODEL}/${MODEL}_VCD_${SEED}/

      echo "🚀 Starting Distributed Parallel Evaluations. $MODEL"
      
      CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun \
        --nproc_per_node=4 \
        --master_port=29501 \
        ./vgd_expt/run_vgd.py \
        --config ./vgd_expt/my_qwen_config.json \
        --visual_alpha=1.75 \
        --model=${MODEL} \
        --seed=${SEED} \
        --work-dir ./outputs/${MODEL}/${MODEL}_VGD_${SEED} \
        >> ./outputs/${MODEL}/${MODEL}_VGD_${SEED}.txt &

      wait     
      echo "✅ Finished $MODEL"
  done
done
echo "✅ Main Evaluations Completed."
wait
echo "✅ Evaluations Completed."