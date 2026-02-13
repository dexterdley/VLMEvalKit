#!/bin/bash
export NCCL_P2P_DISABLE=1
export SPLIT_THINK=True

# CUDA_VISIBLE_DEVICES=7 python ./vgd_expt/run_vgd.py --config ./vgd_expt/my_qwen_config.json --visual_alpha=2.0 --model="InternVL3_5-2B"

MODELS=(
  #Qwen3-VL-2B-Thinking
  Qwen3-VL-4B-Thinking
  #Qwen3-VL-8B-Thinking
  #"Qwen3-VL-2B-Instruct"
  #"Qwen3-VL-8B-Instruct"
  #"Qwen2.5-VL-7B-Instruct"
  #"InternVL3_5-2B"
  #"Gemma3-4B"
)
for SEED in 42 55 69
do
  for MODEL in "${MODELS[@]}"
  do
      mkdir -p ./outputs/${MODEL}
      rm -rf ./outputs/${MODEL}/${MODEL}_Reasoning_${SEED}/

      for ALPHA in 0 1.25 1.5 1.75 2.0 2.5
      do
        echo "🚀 Starting Distributed Parallel Evaluations. Model: $MODEL | Alpha: $ALPHA | Seed: $SEED"

        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 torchrun \
          --nproc_per_node=6 \
          --master_port=29502 \
          ./vgd_expt/run_vgd.py \
          --config ./vgd_expt/my_qwen_config.json \
          --visual_alpha=$ALPHA \
          --model=${MODEL} \
          --seed=${SEED} \
          --work-dir ./outputs/${MODEL}/${MODEL}_Reasoning_${SEED} \
          >> ./outputs/${MODEL}/${MODEL}_Reasoning_${SEED}.txt &

          wait     
          echo "✅ Finished $MODEL"
        done
    done
done
echo "✅ Evaluations on Reasoning Completed."