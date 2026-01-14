#!/bin/bash
# scripts/train.sh
# 训练脚本

# 设置CUDA设备（如果有多个GPU，可以指定使用哪个）
export PYTHONPATH=$PYTHONPATH:$(pwd)

mkdir -p logs
log_file=logs/train_$(date +%Y%m%d_%H%M%S).log

# 基础训练命令
nohup python main.py \
    mode=train \
    exp_dir=exps/v2 \
    data.device=cuda:1 \
    \
    diffusion.prediction_type=v \
    diffusion.betas.schedule=cosine \
    \
    resampler.hidden_dim=384 \
    resampler.num_heads=4 \
    resampler.depth=4 \
    \
    dit.num_heads=4 \
    dit.depth=6 \
    \
    train.epochs=10000 \
    train.batch_size=128 \
    train.weight_decay=5e-2 \
    train.cfg_drop_rate=0.5 \
    msg=中等模型resampler和dit，去掉co-sorting，v预测，cfg_drop_rate=0.5 \
    > $log_file 2>&1 &

# tail -f $log_file

# # 基础训练命令
# python main.py \
#     mode=train \
#     exp_dir=exps/exp_001 \
#     train.epochs=10000