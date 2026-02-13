#!/bin/bash
MODEL="Qwen3-VL-2B-Instruct"
mkdir -p ./outputs/$MODEL/
echo "🚀 Starting Evaluations on $MODEL"

#CUDA_VISIBLE_DEVICES=0 python ./vgd_expt/run_vgd.py --config ./vgd_expt/my_qwen_config.json --visual_alpha=0.0 > ./outputs/Qwen3-VL-8B-Instruct/Qwen3-VL-8B-Instruct_Base.txt
#CUDA_VISIBLE_DEVICES=0 python ./vgd_expt/run_vgd.py --config ./vgd_expt/my_qwen_config.json --visual_alpha=1.5 > ./outputs/Qwen3-VL-8B-Instruct/Qwen3-VL-8B-Instruct_VGD.txt

# 1. Run Base (Alpha 0.0) on GPU 0 in the background
CUDA_VISIBLE_DEVICES=0 python ./vgd_expt/run_vgd.py \
  --config ./vgd_expt/my_qwen_config.json \
  --visual_alpha=0.0 \
  --model=${MODEL} \
  --work-dir ./outputs/${MODEL}/${MODEL}_Base \
  > ./outputs/${MODEL}/${MODEL}_Base.txt &

# 2. Run VGD (Alpha 1.5) on GPU 1 in the background
CUDA_VISIBLE_DEVICES=1 python ./vgd_expt/run_vgd.py \
  --config ./vgd_expt/my_qwen_config.json \
  --visual_alpha=1.5 \
  --model=${MODEL} \
  --work-dir ./outputs/${MODEL}/${MODEL}_VGD \
  > ./outputs/${MODEL}/${MODEL}_VGD.txt &

# 3. Wait for both background processes to finish
wait
echo "✅ Evaluations Completed."