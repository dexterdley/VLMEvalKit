#!/bin/bash
# torchrun --nproc-per-node=8 run.py --data MMBench_DEV_EN MME SEEDBench_IMG --model idefics_80b_instruct qwen_chat mPLUG-Owl2 --verbose
echo "🚀 Starting Evaluations"

# 1. Create the output directory first!
mkdir -p ./outputs/Qwen3-VL-8B-Instruct/

#CUDA_VISIBLE_DEVICES=0 python ./vls_expt/run_vls.py --config ./vls_expt/my_qwen_config.json --visual_alpha=0.0 > ./outputs/Qwen3-VL-8B-Instruct/Qwen3-VL-8B-Instruct_Base.txt
#CUDA_VISIBLE_DEVICES=0 python ./vls_expt/run_vls.py --config ./vls_expt/my_qwen_config.json --visual_alpha=1.5 > ./outputs/Qwen3-VL-8B-Instruct/Qwen3-VL-8B-Instruct_VGD.txt

# 1. Run Base (Alpha 0.0) on GPU 0 in the background
CUDA_VISIBLE_DEVICES=0 python ./vls_expt/run_vls.py \
  --config ./vls_expt/my_qwen_config.json \
  --visual_alpha=0.0 \
  --work-dir ./outputs/Qwen3-VL-8B-Instruct/Qwen3-VL-8B-Instruct_Base \
  > ./outputs/Qwen3-VL-8B-Instruct/Qwen3-VL-8B-Instruct_Base.txt &

# 2. Run VGD (Alpha 1.5) on GPU 1 in the background
CUDA_VISIBLE_DEVICES=1 python ./vls_expt/run_vls.py \
  --config ./vls_expt/my_qwen_config.json \
  --visual_alpha=1.5 \
  --work-dir ./outputs/Qwen3-VL-8B-Instruct/Qwen3-VL-8B-Instruct_VGD \
  > ./outputs/Qwen3-VL-8B-Instruct/Qwen3-VL-8B-Instruct_VGD.txt &

# 3. Wait for both background processes to finish
wait

echo "✅ Evaluations Completed."