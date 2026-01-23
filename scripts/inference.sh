#!/bin/bash

# 设置CUDA设备（如果有多个GPU，可以指定使用哪个）
export PYTHONPATH=$PYTHONPATH:$(pwd)

export CUDAVISIABLE=1

mkdir -p logs
log_file=logs/train_$(date +%Y%m%d_%H%M%S).log
config_path=/home/zxd/zxd/Huawei/ResamplerDit/backups/ResamplerDit3/exps/eps14/logs/config.yaml
mode=inference

# 基础训练命令
python main.py \
    --config $config_path \
    --mode $mode

# tail -f $log_file
