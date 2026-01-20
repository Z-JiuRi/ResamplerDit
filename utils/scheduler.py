import math
import torch
from typing import List, Dict
from torch.optim import Optimizer
from torch.optim import lr_scheduler


import math
import torch
from torch.optim.lr_scheduler import _LRScheduler
import bisect

class CosineAnnealingRestartAtPoints(_LRScheduler):
    def __init__(self, optimizer, restart_points, max_steps, decay_ratio=1.0, eta_min=0, last_epoch=-1):
        """
        Args:
            optimizer (Optimizer): PyTorch 优化器
            restart_points (list[float]): 重启点位的比例列表，例如 [0.1, 0.3, 0.8]
            max_steps (int): 总训练步数 (tot_step)
            decay_ratio (float): 每次重启后，LR 上限的衰减比例 (0 < ratio <= 1)
            eta_min (float): 最小学习率 (默认为 0)
            last_epoch (int): 上一个 epoch/step 的索引
        """
        self.max_steps = max_steps
        self.decay_ratio = decay_ratio
        self.eta_min = eta_min
        
        # 处理重启点：排序并转换为绝对步数
        # 自动添加 0 和 max_steps 以形成完整的区间段
        # 例如输入 [0.1], max=100 -> [0, 10, 100]
        points = [0.0] + sorted(restart_points) + [1.0]
        # 去重并转换为整数步数
        self.milestones = sorted(list(set([int(p * max_steps) for p in points])))
        
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # 1. 确定当前处于哪个区间 (使用 bisect 快速查找)
        # bisect_right 返回插入点，减 1 得到当前区间的左边界索引
        current_step = self.last_epoch
        
        # 防止 step 超出范围导致的索引越界
        if current_step >= self.milestones[-1]:
             idx = len(self.milestones) - 2
        else:
            idx = bisect.bisect_right(self.milestones, current_step) - 1
            
        # 2. 获取当前区间的起止点
        start_step = self.milestones[idx]
        end_step = self.milestones[idx + 1]
        
        # 3. 计算当前区间的进度 (0.0 到 1.0)
        segment_len = end_step - start_step
        if segment_len == 0: # 避免除以0
            progress = 1.0
        else:
            progress = (current_step - start_step) / segment_len
            
        # 4. 计算当前的衰减系数 (Decay Ratio)
        # 第0段(idx=0)不衰减，第1段(idx=1)衰减一次，以此类推
        current_decay = self.decay_ratio ** idx

        # 5. 应用 Cosine 公式
        return [
            self.eta_min + (base_lr * current_decay - self.eta_min) *
            (1 + math.cos(math.pi * progress)) / 2
            for base_lr in self.base_lrs
        ]


def get_lr_scheduler(
    optimizer: Optimizer,
    **kwargs
):
    """
    根据 kwargs['scheduler_type'] 返回对应的学习率调度器 (lr scheduler)。

    支持的 scheduler_type：
    - "cosine"              : CosineAnnealingLR，余弦退火
    - "cosine_restart" : CosineAnnealingWarmRestarts，带周期性重启的余弦退火
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
    elif scheduler_type == "cosine_restart":
        """
        调度方式：
        - 使用余弦退火，在指定比例位置重启
        - 每个周期内从 lr 退火到 eta_min，然后重置并衰减
        
        参数：
        - restart_points (list): 重启点位置，用总步数的比例表示，例如 [0.1, 0.3, 0.8]
        - lr_ratio (float): 每次重启时峰值学习率的衰减因子，默认 1.0（不衰减）
        - eta_min (float): 每个周期的最低 lr，默认 0.0
        - total_steps (int): 总训练步数
        """
        restart_points: list = kwargs.get("restart_points", [0.1, 0.3, 0.8])
        lr_ratio: float = kwargs.get("lr_ratio", 1.0)
        eta_min: float = kwargs.get("eta_min", 0.0)
        total_steps: int = kwargs.get("total_steps", 100)
        
        class CosineAnnealingRestartAtPoints(lr_scheduler._LRScheduler):
            def __init__(self, optimizer, restart_points, total_steps, lr_ratio=1.0, eta_min=0.0, last_epoch=-1):
                """
                Args:
                    optimizer (Optimizer): 优化器
                    restart_points (list): 重启点比例列表，如 [0.1, 0.3, 0.8]
                    total_steps (int): 总训练步数
                    lr_ratio (float): 每次重启时峰值学习率的衰减因子
                    eta_min (float): 每个周期的最小学习率
                """
                # 验证输入
                if not restart_points:
                    raise ValueError("restart_points 不能为空")
                if not all(0 < p < 1 for p in restart_points):
                    raise ValueError("restart_points 必须在 (0, 1) 范围内")
                if not sorted(restart_points) == restart_points:
                    raise ValueError("restart_points 必须按升序排列")
                
                self.restart_points = restart_points
                self.total_steps = total_steps
                self.lr_ratio = lr_ratio
                self.eta_min = eta_min
                
                # 计算实际的重启步数
                self.restart_steps = [int(p * total_steps) for p in restart_points]
                # 确保最后一个重启点在总步数之前
                if self.restart_steps[-1] >= total_steps:
                    raise ValueError("最后一个重启点必须在总步数之前")
                
                # 计算每个周期的长度
                self.period_lengths = []
                prev_step = 0
                for step in self.restart_steps:
                    self.period_lengths.append(step - prev_step)
                    prev_step = step
                # 最后一个周期到训练结束
                self.period_lengths.append(total_steps - prev_step)
                
                self.num_periods = len(self.period_lengths)
                self.current_period = 0  # 当前周期索引
                self.steps_in_period = 0  # 当前周期内的步数
                
                # 保存初始学习率作为基准
                self.base_lrs = [group['lr'] for group in optimizer.param_groups]
                self.current_peak_lrs = self.base_lrs.copy()  # 当前周期的峰值学习率
                
                super().__init__(optimizer, last_epoch)
                
                # 初始化状态
                if last_epoch != -1:
                    self._initialize_state(last_epoch)
            
            def _initialize_state(self, last_epoch):
                """根据给定的epoch初始化状态"""
                self.current_period = 0
                cumulative_steps = 0
                
                # 找到当前epoch所在的周期
                for i, length in enumerate(self.period_lengths):
                    if last_epoch < cumulative_steps + length:
                        self.current_period = i
                        self.steps_in_period = last_epoch - cumulative_steps
                        break
                    cumulative_steps += length
                
                # 更新当前周期的峰值学习率（应用衰减）
                self.current_peak_lrs = [base_lr * (self.lr_ratio ** self.current_period) 
                                    for base_lr in self.base_lrs]
            
            def get_lr(self):
                """计算当前步数的学习率"""
                current_length = self.period_lengths[self.current_period]
                
                # 如果周期长度为0，避免除零错误
                if current_length <= 0:
                    return [self.eta_min for _ in self.base_lrs]
                
                # 计算余弦退火
                # 使用 (1 + cos(π * progress)) / 2 的公式
                # progress 从 0 到 1
                progress = self.steps_in_period / current_length
                
                return [
                    self.eta_min + (peak_lr - self.eta_min) * 
                    (1 + math.cos(math.pi * progress)) / 2
                    for peak_lr in self.current_peak_lrs
                ]
            
            def step(self, epoch=None):
                """更新调度器状态"""
                if epoch is None:
                    epoch = self.last_epoch + 1
                
                self.last_epoch = epoch
                
                # 计算累积步数以确定当前周期
                cumulative_steps = 0
                for i, length in enumerate(self.period_lengths):
                    if epoch < cumulative_steps + length:
                        new_period = i
                        steps_in_new_period = epoch - cumulative_steps
                        break
                    cumulative_steps += length
                
                # 检查是否切换到新周期
                if new_period != self.current_period:
                    self.current_period = new_period
                    self.steps_in_period = steps_in_new_period
                    # 更新峰值学习率（应用衰减）
                    self.current_peak_lrs = [base_lr * (self.lr_ratio ** self.current_period) 
                                        for base_lr in self.base_lrs]
                else:
                    self.steps_in_period = steps_in_new_period
                
                # 更新优化器的学习率
                for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
                    param_group['lr'] = lr
            
            def get_current_period(self):
                """获取当前周期信息"""
                return {
                    'period': self.current_period,
                    'total_periods': self.num_periods,
                    'steps_in_period': self.steps_in_period,
                    'period_length': self.period_lengths[self.current_period],
                    'current_peak_lr': self.current_peak_lrs[0] if self.current_peak_lrs else None,
                    'next_restart_at': None if self.current_period >= len(self.restart_steps) 
                        else self.restart_steps[self.current_period]
                }
        
        import math
        return CosineAnnealingRestartAtPoints(
            optimizer, 
            restart_points=restart_points,
            total_steps=total_steps,
            lr_ratio=lr_ratio, 
            eta_min=eta_min
        )


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