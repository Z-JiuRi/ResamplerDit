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
    exp_dir=exps/eps7 \
    data.device=cuda:1 \
    \
    diffusion.prediction_type=eps \
    diffusion.betas.scheduler_type=cosine \
    diffusion.snr_gamma=5.0 \
    diffusion.small_weight=0.1 \
    \
    resampler.latent_cond_len=224 \
    resampler.hidden_dim=512 \
    resampler.num_heads=8 \
    resampler.depth=8 \
    resampler.dropout=0.3 \
    \
    dit.num_heads=8 \
    dit.depth=8 \
    dit.mlp_ratio=4.0 \
    dit.dropout=0.3 \
    \
    train.epochs=10000 \
    train.batch_size=108 \
    train.weight_decay=5e-2 \
    train.cfg_drop_rate=0.2 \
    train.ema_rate=0.9999 \
    train.cond_noise_factor=0.01 \
    lr_scheduler.warmup_ratio=0.1 \
    \
    msg=1 \
    > $log_file 2>&1 &

# tail -f $log_file

# # 基础训练命令
# python main.py \
#     mode=train \
#     exp_dir=exps/exp_001 \
#     train.epochs=10000