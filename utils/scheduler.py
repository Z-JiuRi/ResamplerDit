import math
import torch
from typing import List, Dict
from torch.optim import Optimizer
from torch.optim import lr_scheduler


def get_lr_scheduler(
    optimizer: Optimizer,
    **kwargs
):
    """
    根据 kwargs['scheduler_type'] 返回对应的学习率调度器 (lr scheduler)。

    支持的 scheduler_type：
    - "cosine"              : CosineAnnealingLR，余弦退火
    - "cosine_warm_restart" : CosineAnnealingWarmRestarts，带周期性重启的余弦退火
    - "cosine_warmup"       : 先线性预热，再余弦退火（自定义 LambdaLR）
    - "const"               : 保持不变
    """

    scheduler_type = kwargs['scheduler_type'].lower()

    # CosineAnnealingLR：余弦退火
    if scheduler_type == "cosine":
        """
        调度方式：
        - 在 [0, T_max] 内按照余弦函数从初始 lr 平滑下降到 eta_min：
          lr_t = eta_min + (lr_0 - eta_min) * (1 + cos(pi * t / T_max)) / 2

        参数：
        - T_max (int)    ：一个完整余弦周期的长度（通常是总 epoch 数）
        - eta_min (float)：最小学习率，默认 0.0
        """
        T_max: int = kwargs.get("T_max", 50)
        eta_min: float = kwargs.get("eta_min", 0.0)
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)


    # CosineAnnealingWarmRestarts：带重启的余弦退火
    elif scheduler_type == "cosine_warm_restart":
        """
        调度方式：
        - 使用余弦退火，但会周期性地重启；
        - 周期长度从 T_0 开始，每次乘以 T_mult；
        - 每个周期内从 lr_0 退火到 eta_min，然后重置回 lr_0 再退火。

        参数：
        - T_0 (int)     ：第一个周期长度
        - T_mult (int)  ：每次重启后周期长度放大倍数，默认 2
        - lr_ratio (float)：每个周期 lr 与初始 lr 的比例，默认 1.0
        - eta_min (float)：每个周期的最低 lr，默认 0.0
        """
        T_0: int = kwargs.get("T_0", 10)
        T_mult: int = kwargs.get("T_mult", 2)
        lr_ratio: float = kwargs.get("lr_ratio", 1.0)
        eta_min: float = kwargs.get("eta_min", 0.0)
        return lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min
        ) * lr_ratio


    # 自定义：预热 + 余弦退火 (cosine_warmup)
    elif scheduler_type == "cosine_warmup":
        """
        调度方式：
        - 阶段 1：预热 (warmup)
            前 warmup_epochs 个 epoch 内，
            lr 从 start_lr 线性上升到 base_lr（optimizer 当前 lr）。
        - 阶段 2：余弦退火 (cosine)
            之后的 (max_epochs - warmup_epochs) 个 epoch 内，
            使用余弦退火从 base_lr 下降到 eta_min。

        参数：
        - warmup_epochs (int)    ：预热 epoch 数
        - max_epochs (int)       ：总 epoch 数（包含预热）
        - start_lr (float)：预热起始 lr，默认 0.0
        - eta_min (float)        ：余弦退火的最低 lr，默认 0.0

        用法：
        - 一般在每个 epoch 结束后调用 scheduler.step()。
        """
        warmup_epochs = kwargs.get("warmup_epochs", 5)
        max_epochs = kwargs.get("max_epochs", 100)
        eta_min = kwargs.get("eta_min", 0.0)

        # Check if we are using steps instead of epochs
        # If max_steps is provided in kwargs, we use it as max_duration
        # and warmup_steps as warmup_duration
        if "max_steps" in kwargs:
            max_duration = kwargs["max_steps"]
            warmup_duration = kwargs.get("warmup_steps", 0)
        else:
            max_duration = max_epochs
            warmup_duration = warmup_epochs

        # 假设所有 param_group 的 lr 相同
        base_lr = optimizer.param_groups[0]["lr"]
        start_lr = kwargs.get("start_lr", 0.0)

        def lr_lambda(current_step: int):
            # 阶段 1：线性预热
            if current_step < warmup_duration:
                warmup_progress = current_step / max(1, warmup_duration)
                lr = start_lr + (base_lr - start_lr) * warmup_progress
                return lr / base_lr  # 转成倍率

            # 阶段 2：余弦退火
            cos_step = current_step - warmup_duration
            cos_total = max_duration - warmup_duration
            cos_total = max(1, cos_total)

            cos_factor = 0.5 * (1 + math.cos(math.pi * cos_step / cos_total))
            lr = eta_min + (base_lr - eta_min) * cos_factor
            return lr / base_lr

        return lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


    # 自定义：保持不变 (const)
    elif scheduler_type == "const":
        """
        调度方式：
        - 保持 lr 不变，不进行任何调整。

        参数：
        - 无

        用法：
        - 一般在每个 epoch 结束后调用 scheduler.step()。
        """
        return lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
    
    else:
        raise ValueError(f"Unknown lr scheduler type: {scheduler_type}")