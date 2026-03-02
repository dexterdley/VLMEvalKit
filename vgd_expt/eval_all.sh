#!/bin/bash
export NCCL_P2P_DISABLE=1
export SPLIT_THINK=True

# CUDA_VISIBLE_DEVICES=7 python ./vgd_expt/run_vgd.py --config ./vgd_expt/my_qwen_config.json --visual_alpha=2.0 --model="InternVL3_5-2B"

MODELS=(
  #"Qwen3-VL-2B-Thinking"
  #"Qwen3-VL-8B-Thinking"
  #"Gemma3-4B"
  #"Qwen3-VL-2B-Instruct"
  "Qwen3-VL-8B-Instruct"
  "Qwen2.5-VL-7B-Instruct"
  "InternVL3_5-2B"
  "InternVL3_5-8B"
  "llava_next_vicuna_7b"
)
for SEED in 42 55 69
do
  for MODEL in "${MODELS[@]}"
  do
      mkdir -p ./outputs/${MODEL}

      #rm -rf ./outputs/${MODEL}/${MODEL}_Base_${SEED}/
      #rm -rf ./outputs/${MODEL}/${MODEL}_VGD_${SEED}/
      #rm -rf ./outputs/${MODEL}/${MODEL}_VCD_${SEED}/
      #rm -rf ./outputs/${MODEL}/${MODEL}_ICD_${SEED}/
      #rm -rf ./outputs/${MODEL}/${MODEL}_OPERA_${SEED}/
      rm -rf ./outputs/${MODEL}/${MODEL}_VORD_${SEED}/

      echo "🚀 Starting Distributed Parallel Evaluations. $MODEL"
      
      
      CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
        --nproc_per_node=4 \
        --master_port=29505 \
        ./vgd_expt/run_vgd.py \
        --config ./vgd_expt/my_qwen_config.json \
        --vord_margin=0.1 \
        --model=${MODEL} \
        --seed=${SEED} \
        --work-dir ./outputs/${MODEL}/${MODEL}_VORD_${SEED} \
        >> ./outputs/${MODEL}/${MODEL}_VORD_${SEED}.txt &

      wait     
      echo "✅ Finished $MODEL"
  done
done
echo "✅ Main Evaluations Completed."