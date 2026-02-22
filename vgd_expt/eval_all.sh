#!/bin/bash
export NCCL_P2P_DISABLE=1
export SPLIT_THINK=True

MODELS=(
  #"Qwen3-VL-2B-Thinking"
  #"Qwen3-VL-8B-Thinking"
  #"Gemma3-4B"
  #"Qwen3-VL-2B-Instruct"
  "Qwen3-VL-8B-Instruct"
  "Qwen2.5-VL-7B-Instruct"
  "InternVL3_5-2B"
  "InternVL3_5-8B"
)
for SEED in 42 55 69
do
  for MODEL in "${MODELS[@]}"
  do
      mkdir -p ./outputs/${MODEL}

      rm -rf ./outputs/${MODEL}/${MODEL}_Base_${SEED}/
      rm -rf ./outputs/${MODEL}/${MODEL}_VGD_${SEED}/
      rm -rf ./outputs/${MODEL}/${MODEL}_VCD_${SEED}/
      rm -rf ./outputs/${MODEL}/${MODEL}_ICD_${SEED}/
      rm -rf ./outputs/${MODEL}/${MODEL}_OPERA_${SEED}/

      echo "🚀 Starting Distributed Parallel Evaluations. $MODEL"
      
      CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
        --nproc_per_node=4 \
        --master_port=29500 \
        ./vgd_expt/run_vgd.py \
        --config ./vgd_expt/my_qwen_config.json \
        --visual_alpha=0 \
        --model=${MODEL} \
        --seed=${SEED} \
        --work-dir ./outputs/${MODEL}/${MODEL}_Base_${SEED} \
        >> ./outputs/${MODEL}/${MODEL}_Base_${SEED}.txt &
      
      CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun \
        --nproc_per_node=4 \
        --master_port=29501 \
        ./vgd_expt/run_vgd.py \
        --config ./vgd_expt/my_qwen_config.json \
        --visual_alpha=2.0 \
        --model=${MODEL} \
        --seed=${SEED} \
        --work-dir ./outputs/${MODEL}/${MODEL}_VGD_${SEED} \
        >> ./outputs/${MODEL}/${MODEL}_VGD_${SEED}.txt &

      CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
        --nproc_per_node=4 \
        --master_port=29502 \
        ./vgd_expt/run_vgd.py \
        --config ./vgd_expt/my_qwen_config.json \
        --vcd_alpha=1.0 \
        --model=${MODEL} \
        --seed=${SEED} \
        --work-dir ./outputs/${MODEL}/${MODEL}_VCD_${SEED} \
        >> ./outputs/${MODEL}/${MODEL}_VCD_${SEED}.txt &

      CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun \
        --nproc_per_node=4 \
        --master_port=29503 \
        ./vgd_expt/run_vgd.py \
        --config ./vgd_expt/my_qwen_config.json \
        --icd_alpha=1.0 \
        --model=${MODEL} \
        --seed=${SEED} \
        --work-dir ./outputs/${MODEL}/${MODEL}_ICD_${SEED} \
        >> ./outputs/${MODEL}/${MODEL}_ICD_${SEED}.txt &

      CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
        --nproc_per_node=4 \
        --master_port=29500 \
        ./vgd_expt/run_vgd.py \
        --config ./vgd_expt/my_qwen_config.json \
        --opera_alpha=1.0 \
        --opera_alpha=50 \
        --model=${MODEL} \
        --seed=${SEED} \
        --work-dir ./outputs/${MODEL}/${MODEL}_OPERA_${SEED} \
        >> ./outputs/${MODEL}/${MODEL}_OPERA_${SEED}.txt &

      wait     
      echo "✅ Finished $MODEL"
  done
done
echo "✅ Main Evaluations Completed."