#!/bin/bash
# scripts/train.sh
# 训练脚本

# 设置CUDA设备（如果有多个GPU，可以指定使用哪个）
export PYTHONPATH=$PYTHONPATH:$(pwd)

mkdir -p logs
log_file=logs/train_$(date +%Y%m%d_%H%M%S).log

# # 基础训练命令
nohup python main.py \
    mode=train \
    data.device=cuda:0 \
    diffusion.prediction_type=x \
    exp_dir=exps/x1 \
    train.epochs=5000 \
    msg=train流程第一次测试 \
    > $log_file 2>&1 &

tail -f $log_file

# # 基础训练命令
# python main.py \
#     mode=train \
#     exp_dir=exps/exp_001 \
#     train.epochs=10000