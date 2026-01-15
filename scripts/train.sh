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
    exp_dir=exps/eps17 \
    data.device=cuda:0 \
    \
    diffusion.prediction_type=eps \
    diffusion.betas.scheduler_type=cosine \
    diffusion.snr_gamma=5.0 \
    diffusion.small_weight=0.5 \
    \
    resampler.latent_cond_len=64 \
    resampler.hidden_dim=256 \
    resampler.num_heads=4 \
    resampler.depth=4 \
    resampler.dropout=0.2 \
    \
    dit.num_heads=4 \
    dit.depth=4 \
    dit.mlp_ratio=4.0 \
    dit.dropout=0.2 \
    \
    train.epochs=10000 \
    train.batch_size=120 \
    train.weight_decay=5e-2 \
    train.cfg_drop_rate=0.2 \
    train.ema_rate=0.9999 \
    train.cond_noise_factor=0.01 \
    lr_scheduler.warmup_ratio=0.1 \
    \
    msg=大型模型resampler和dit,用改进版co-sorting，v,snr加权，重写了sort和归一化，随机分割数据集，跟李测试集，大小样本平均加权 \
    > $log_file 2>&1 &

# tail -f $log_file

# # 基础训练命令
# python main.py \
#     mode=train \
#     exp_dir=exps/exp_001 \
#     train.epochs=10000