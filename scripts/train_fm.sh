#!/bin/bash
# scripts/train.sh
# 训练脚本

# 设置CUDA设备（如果有多个GPU，可以指定使用哪个）
export PYTHONPATH=$PYTHONPATH:$(pwd)

mkdir -p logs
log_file=logs/train_$(date +%Y%m%d_%H%M%S).log

# 基础训练命令
nohup python main_fm.py \
    mode=train \
    exp_dir=exps/fm4 \
    data.device=cuda:2 \
    \
    diffusion.small_weight=0.2 \
    \
    resampler.latent_cond_len=224 \
    resampler.hidden_dim=768 \
    resampler.num_heads=4 \
    resampler.depth=4 \
    resampler.dropout=0.2 \
    \
    dit.num_heads=8 \
    dit.depth=12 \
    dit.mlp_ratio=4.0 \
    dit.dropout=0.2 \
    \
    train.epochs=5000 \
    train.batch_size=128 \
    train.weight_decay=5e-2 \
    train.cfg_drop_rate=0.1 \
    train.ema_rate=0.9999 \
    train.cond_noise_factor=0.01 \
    \
    lr_scheduler.max_lr=1e-4 \
    lr_scheduler.start_lr=1e-6 \
    lr_scheduler.eta_min=1e-6 \
    lr_scheduler.warmup_ratio=0.1 \
    \
    msg=轻量化fm \
    > $log_file 2>&1 &

# tail -f $log_file

# # 基础训练命令
# python main.py \
#     mode=train \
#     exp_dir=exps/exp_001 \
#     train.epochs=10000