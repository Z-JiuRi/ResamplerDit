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
    exp_dir=exps/eps16 \
    data.device=cuda:0 \
    \
    diffusion.prediction_type=eps \
    diffusion.betas.scheduler_type=linear \
    diffusion.snr_gamma=5.0 \
    diffusion.small_weight=0.3 \
    \
    resampler.latent_cond_len=128 \
    resampler.hidden_dim=1024 \
    resampler.num_heads=8 \
    resampler.depth=4 \
    resampler.dropout=0.2 \
    \
    dit.num_heads=8 \
    dit.depth=12 \
    dit.mlp_ratio=4.0 \
    dit.dropout=0.2 \
    \
    train.epochs=10000 \
    train.batch_size=128 \
    train.weight_decay=5e-2 \
    train.cfg_drop_rate=0.1 \
    train.ema_rate=0.999 \
    train.cond_noise_factor=0.01 \
    train.grad_accum_steps=4 \
    \
    lr_scheduler.type=cosine_warmup \
    lr_scheduler.max_lr=5e-4 \
    lr_scheduler.start_lr=1e-6 \
    lr_scheduler.eta_min=1e-6 \
    lr_scheduler.warmup_ratio=0.1 \
    \
    msg=14同款，不带梯度累计 \
    > $log_file 2>&1 &

# tail -f $log_file

# # 基础训练命令
# python main.py \
#     mode=train \
#     exp_dir=exps/exp_001 \
#     train.epochs=10000