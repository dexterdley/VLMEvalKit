#!/bin/bash

# 1. Create the output directory first!
mkdir -p ./outputs/Qwen3-VL-8B-Instruct/

CUDA_VISIBLE_DEVICES=0 python ./vls_expt/run_vls.py --config ./vls_expt/my_qwen_config.json --visual_alpha=0.0 > ./outputs/Qwen3-VL-8B-Instruct/Qwen3-VL-8B-Instruct_Base.txt
CUDA_VISIBLE_DEVICES=0 python ./vls_expt/run_vls.py --config ./vls_expt/my_qwen_config.json --visual_alpha=1.5 > ./outputs/Qwen3-VL-8B-Instruct/Qwen3-VL-8B-Instruct_VGD.txt