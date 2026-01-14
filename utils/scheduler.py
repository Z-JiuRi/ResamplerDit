import math
import torch
from typing import List, Dict
from torch.optim import Optimizer
from torch.optim import lr_scheduler


# def get_lr_scheduler(
#     optimizer: Optimizer,
#     **kwargs
# ):
#     """
#     根据 kwargs['scheduler_type'] 返回对应的学习率调度器 (lr scheduler)。

#     支持的 scheduler_type：
#     - "step"                : StepLR，按固定间隔阶梯式衰减
#     - "multi_step"          : MultiStepLR，在指定 epoch 列表处阶梯式衰减
#     - "exponential"         : ExponentialLR，按固定比例每个 step/epoch 指数衰减
#     - "cosine"              : CosineAnnealingLR，余弦退火
#     - "cosine_warm_restart" : CosineAnnealingWarmRestarts，带周期性重启的余弦退火
#     - "reduce_on_plateau"   : ReduceLROnPlateau，指标长期不提升时降低 lr
#     - "cosine_warmup"       : 先线性预热，再余弦退火（自定义 LambdaLR）
#     - "custom_multi_step"   : 多步自定义，指定到某些 epoch 时 lr 变为某个绝对值
#     - "const"               : 保持不变
#     """

#     scheduler_type = kwargs['scheduler_type'].lower()

#     # 1. StepLR：固定间隔阶梯衰减
#     if scheduler_type == "step":
#         """
#         调度方式：
#         - 每隔 step_size 个 step/epoch，将学习率乘以 gamma：
#           lr_t = lr_0 * (gamma ** floor(t / step_size))

#         参数：
#         - step_size (int)  ：衰减间隔
#         - gamma (float)    ：每次衰减倍率，默认 0.5
#         """
#         step_size: int = kwargs.get("step_size", 30)
#         gamma: float = kwargs.get("gamma", 0.5)
#         return lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

#     # 2. MultiStepLR：在多处阶梯衰减
#     elif scheduler_type == "multi_step":
#         """
#         调度方式：
#         - 在 milestones 指定的 epoch/step 上，将 lr 乘以 gamma（可以多次）

#         参数：
#         - milestones (List[int])：衰减的 epoch/step 列表（必须提供）
#         - gamma (float)         ：每次衰减倍率，默认 0.5
#         """
#         milestones: List[int] = kwargs["milestones"]
#         gamma: float = kwargs.get("gamma", 0.5)
#         return lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

#     # 3. ExponentialLR：指数衰减
#     elif scheduler_type == "exponential":
#         """
#         调度方式：
#         - 每次调用 scheduler.step() 时，将 lr 乘以 gamma：
#           lr_t = lr_0 * (gamma ** t)

#         参数：
#         - gamma (float)：衰减因子，0<gamma<1 时衰减
#         """
#         gamma: float = kwargs.get("gamma", 0.95)
#         return lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

#     # 4. CosineAnnealingLR：余弦退火
#     elif scheduler_type == "cosine":
#         """
#         调度方式：
#         - 在 [0, T_max] 内按照余弦函数从初始 lr 平滑下降到 eta_min：
#           lr_t = eta_min + (lr_0 - eta_min) * (1 + cos(pi * t / T_max)) / 2

#         参数：
#         - T_max (int)    ：一个完整余弦周期的长度（通常是总 epoch 数）
#         - eta_min (float)：最小学习率，默认 0.0
#         """
#         T_max: int = kwargs.get("T_max", 50)
#         eta_min: float = kwargs.get("eta_min", 0.0)
#         return lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

#     # 5. CosineAnnealingWarmRestarts：带重启的余弦退火
#     elif scheduler_type == "cosine_warm_restart":
#         """
#         调度方式：
#         - 使用余弦退火，但会周期性地重启；
#         - 周期长度从 T_0 开始，每次乘以 T_mult；
#         - 每个周期内从 lr_0 退火到 eta_min，然后重置回 lr_0 再退火。

#         参数：
#         - T_0 (int)     ：第一个周期长度
#         - T_mult (int)  ：每次重启后周期长度放大倍数，默认 2
#         - lr_ratio (float)：每个周期 lr 与初始 lr 的比例，默认 1.0
#         - eta_min (float)：每个周期的最低 lr，默认 0.0
#         """
#         T_0: int = kwargs.get("T_0", 10)
#         T_mult: int = kwargs.get("T_mult", 2)
#         lr_ratio: float = kwargs.get("lr_ratio", 1.0)
#         eta_min: float = kwargs.get("eta_min", 0.0)
#         return lr_scheduler.CosineAnnealingWarmRestarts(
#             optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min
#         ) * lr_ratio



#     # 6. ReduceLROnPlateau：指标停滞时降低学习率
#     elif scheduler_type == "reduce_on_plateau":
#         """
#         调度方式：
#         - 根据监控指标（如 val_loss）变化情况来调整 lr；
#         - 当指标在 patience 个 epoch 内没有明显改善时，将 lr 乘以 factor；
#         - 使用方式：每个 epoch 结束时调用 scheduler.step(metric_value)。

#         参数（常用）：
#         - mode (str)      ："min" 或 "max"，默认 "min"
#         - factor (float)  ：每次降低 lr 的倍率，如 0.5
#         - patience (int)  ：容忍多少个 epoch 不提升
#         - threshold (float)：认为“有提升”的最小变化
#         - min_lr (float)  ：lr 下界
#         """
#         mode: str = kwargs.get("mode", "min")
#         factor: float = kwargs.get("factor", 0.5)
#         patience: int = kwargs.get("patience", 10)
#         threshold: float = kwargs.get("threshold", 1e-4)
#         min_lr: float = kwargs.get("min_lr", 0.0)

#         return lr_scheduler.ReduceLROnPlateau(
#             optimizer,
#             mode=mode,
#             factor=factor,
#             patience=patience,
#             threshold=threshold,
#             min_lr=min_lr,
#         )

#     # 7. 自定义：预热 + 余弦退火 (cosine_warmup)
#     elif scheduler_type == "cosine_warmup":
#         """
#         调度方式：
#         - 阶段 1：预热 (warmup)
#             前 warmup_epochs 个 epoch 内，
#             lr 从 start_lr 线性上升到 base_lr（optimizer 当前 lr）。
#         - 阶段 2：余弦退火 (cosine)
#             之后的 (max_epochs - warmup_epochs) 个 epoch 内，
#             使用余弦退火从 base_lr 下降到 eta_min。

#         参数：
#         - warmup_epochs (int)    ：预热 epoch 数
#         - max_epochs (int)       ：总 epoch 数（包含预热）
#         - start_lr (float)：预热起始 lr，默认 0.0
#         - eta_min (float)        ：余弦退火的最低 lr，默认 0.0

#         用法：
#         - 一般在每个 epoch 结束后调用 scheduler.step()。
#         """
#         warmup_epochs = kwargs.get("warmup_epochs", 5)
#         max_epochs = kwargs.get("max_epochs", 100)
#         eta_min = kwargs.get("eta_min", 0.0)

#         # Check if we are using steps instead of epochs
#         # If max_steps is provided in kwargs, we use it as max_duration
#         # and warmup_steps as warmup_duration
#         if "max_steps" in kwargs:
#             max_duration = kwargs["max_steps"]
#             warmup_duration = kwargs.get("warmup_steps", 0)
#         else:
#             max_duration = max_epochs
#             warmup_duration = warmup_epochs

#         # 假设所有 param_group 的 lr 相同
#         base_lr = optimizer.param_groups[0]["lr"]
#         start_lr = kwargs.get("start_lr", 0.0)

#         def lr_lambda(current_step: int):
#             # 阶段 1：线性预热
#             if current_step < warmup_duration:
#                 warmup_progress = current_step / max(1, warmup_duration)
#                 lr = start_lr + (base_lr - start_lr) * warmup_progress
#                 return lr / base_lr  # 转成倍率

#             # 阶段 2：余弦退火
#             cos_step = current_step - warmup_duration
#             cos_total = max_duration - warmup_duration
#             cos_total = max(1, cos_total)

#             cos_factor = 0.5 * (1 + math.cos(math.pi * cos_step / cos_total))
#             lr = eta_min + (base_lr - eta_min) * cos_factor
#             return lr / base_lr

#         return lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

#     # 8. 自定义：多步自定义绝对 lr (custom_multi_step)
#     elif scheduler_type == "custom_multi_step":
#         """
#         调度方式：
#         - 用户指定一个字典：lr_milestones = {epoch: lr_value, ...}
#         - 当 current_epoch >= 某个 epoch 时，lr 变为对应的 lr_value；
#         - 如果有多个满足条件的 epoch，取“最大且不超过当前 epoch”的那个；
#         - 例如 lr_milestones = {10: 0.01, 30: 0.001}：
#             - 0 <= epoch < 10：lr = base_lr（optimizer 初始 lr）
#             - 10 <= epoch < 30：lr = 0.01
#             - epoch >= 30    ：lr = 0.001

#         参数：
#         - lr_milestones (Dict[int, float])：
#             key   : epoch 编号（从 0 开始的整数）
#             value : 该 epoch 及之后使用的绝对 lr 值

#         注意：
#         - 这是“绝对值调度”，不是按比例乘法；
#         - 通过 LambdaLR 实现，需要用 base_lr 把绝对 lr 转成倍率。
#         """
#         lr_milestones: Dict[int, float] = kwargs["lr_milestones"]
#         if not lr_milestones:
#             raise ValueError("`custom_multi_step` requires non-empty `lr_milestones` dict.")

#         # 初始 lr 作为 base_lr
#         base_lr = optimizer.param_groups[0]["lr"]

#         # 先把 milestone 的 epoch 排序，方便查找
#         sorted_epochs = sorted(lr_milestones.keys())

#         def lr_lambda(current_epoch: int):
#             # 找到最后一个 <= current_epoch 的 milestone
#             target_lr = base_lr  # 默认用初始 lr
#             for e in sorted_epochs:
#                 if current_epoch >= e:
#                     target_lr = lr_milestones[e]
#                 else:
#                     break
#             # 转成倍率
#             return target_lr / base_lr

#         return lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    
#     # 9. 自定义：保持不变 (const)
#     elif scheduler_type == "const":
#         """
#         调度方式：
#         - 保持 lr 不变，不进行任何调整。

#         参数：
#         - 无

#         用法：
#         - 一般在每个 epoch 结束后调用 scheduler.step()。
#         """
#         return lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
    
#     else:
#         raise ValueError(f"Unknown lr scheduler type: {scheduler_type}")
    
    
import math
import torch
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
from torch.optim import Optimizer
from torch.optim import lr_scheduler

def get_lr_scheduler(optimizer: Optimizer, **kwargs):
    """
    根据 kwargs['scheduler_type'] 返回对应的学习率调度器。
    """
    scheduler_type = kwargs['scheduler_type'].lower()
    
    # 获取总步数，这对于合并最后一个周期的逻辑至关重要
    total_steps = kwargs.get("max_steps", kwargs.get("max_epochs", 100))

    # ... [Const, Cosine, CosineWarmRestart, CustomMultiStep, CosineWarmup 保持不变] ...
    # 为了节省篇幅，这里省略这部分代码，与上一版本一致，
    # 请直接使用上一段回复中的代码，重点看下面的 restart_warm_cosine 修改。

    # =========================================================================
    # 1-5. 其他调度器 (保持原样，此处省略以聚焦重点)
    # =========================================================================
    if scheduler_type == "const":
        return lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
    elif scheduler_type == "cosine":
        eta_min = kwargs.get("eta_min", 0.0)
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=eta_min)
    elif scheduler_type == "cosine_warm_restart":
        T_0 = kwargs.get("T_0", 10)
        T_mult = kwargs.get("T_mult", 2)
        eta_min = kwargs.get("eta_min", 0.0)
        return lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min)
    elif scheduler_type == "custom_multi_step":
        lr_milestones = kwargs.get("lr_milestones", {})
        base_lr = optimizer.param_groups[0]["lr"]
        sorted_steps = sorted(lr_milestones.keys())
        def lr_lambda(current_step):
            target_lr = base_lr
            for s in sorted_steps:
                if current_step >= s: target_lr = lr_milestones[s]
                else: break
            return target_lr / base_lr
        return lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    elif scheduler_type == "cosine_warmup":
        warmup_steps = kwargs.get("warmup_steps", kwargs.get("warmup_epochs", 0))
        eta_min = kwargs.get("eta_min", 0.0)
        base_lr = optimizer.param_groups[0]["lr"]
        start_lr = kwargs.get("start_lr", 0.0)
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                alpha = current_step / max(1, warmup_steps)
                return (start_lr + (base_lr - start_lr) * alpha) / base_lr
            progress = current_step - warmup_steps
            total_decay_steps = max(1, total_steps - warmup_steps)
            progress = min(progress, total_decay_steps)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress / total_decay_steps))
            return (eta_min + (base_lr - eta_min) * cosine_decay) / base_lr
        return lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # =========================================================================
    # 6. [修改] Restart Warm Cosine：尾部合并策略
    # =========================================================================
    elif scheduler_type == "restart_warm_cosine":
        """
        逻辑：
        1. Global Warmup.
        2. 预计算所有周期的起止点。
        3. 如果最后一个自然周期结束时还没到 total_steps，将剩余步数合并进最后一个周期。
        """
        warmup_steps = kwargs.get("warmup_steps", 0)
        T_0 = kwargs.get("T_0", 50)
        T_mult = kwargs.get("T_mult", 2)
        restart_ratio = kwargs.get("restart_ratio", 1.0)
        eta_min = kwargs.get("eta_min", 0.0)
        
        base_lr = optimizer.param_groups[0]["lr"]
        start_lr = kwargs.get("start_lr", 0.0)

        # --- 预计算周期表 (Pre-calculate Cycle Schedule) ---
        # 结构: List of (start_step, end_step, peak_scale, cycle_duration)
        cycles = []
        
        available_steps = total_steps - warmup_steps
        current_t = T_0
        current_scale = 1.0
        used_steps = 0
        
        # 1. 生成自然周期
        while used_steps < available_steps:
            # 如果当前是自然周期的长度
            if used_steps + current_t <= available_steps:
                # 这是一个完整的自然周期
                cycles.append({
                    "start": warmup_steps + used_steps,
                    "end": warmup_steps + used_steps + current_t,
                    "scale": current_scale,
                    "duration": current_t
                })
                used_steps += current_t
                current_t = int(current_t * T_mult)
                current_scale *= restart_ratio
            else:
                # 剩下的步数不足一个完整的自然周期
                # 跳出循环，交给下面的合并逻辑处理
                break
        
        # 2. 处理剩余步数 (Merge Logic)
        remaining = total_steps - (warmup_steps + used_steps)
        
        if cycles:
            # 情况 A: 之前至少有一个完整周期 -> 将剩余步数加到最后一个周期
            last_cycle = cycles[-1]
            last_cycle["end"] += remaining      # 结束点延后
            last_cycle["duration"] += remaining # 周期拉长
        else:
            # 情况 B: 连第一个周期 T_0 都放不下 -> 第一个周期就是全部可用时间
            if available_steps > 0:
                cycles.append({
                    "start": warmup_steps,
                    "end": total_steps,
                    "scale": 1.0,
                    "duration": available_steps
                })

        def lr_lambda(current_step):
            # 1. 预热阶段
            if current_step < warmup_steps:
                alpha = current_step / max(1, warmup_steps)
                lr = start_lr + (base_lr - start_lr) * alpha
                return lr / base_lr

            # 2. 周期衰减阶段
            # 查找当前 step 落在哪个周期内
            # 由于周期数量很少（通常 < 20），直接遍历即可
            target_cycle = None
            for cycle in cycles:
                if cycle["start"] <= current_step < cycle["end"]:
                    target_cycle = cycle
                    break
            
            # 保护：如果超出所有周期（理论上不应发生，除非 total_steps 设置错误），返回最小lr
            if target_cycle is None:
                return eta_min / base_lr

            # 计算周期内进度
            step_in_cycle = current_step - target_cycle["start"]
            duration = target_cycle["duration"]
            scale = target_cycle["scale"]
            
            # 进度归一化 [0, 1]
            # 使用 min 确保不会因为浮点误差导致 cos 输入越界
            progress = min(step_in_cycle, duration)
            
            # 余弦衰减
            decay_factor = 0.5 * (1 + math.cos(math.pi * progress / duration))
            
            current_peak_lr = base_lr * scale
            target_lr = eta_min + (current_peak_lr - eta_min) * decay_factor
            
            return target_lr / base_lr

        return lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
