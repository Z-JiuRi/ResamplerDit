# models/flow_matching.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import math

class RectifiedFlow(nn.Module):
    def __init__(
        self,
        denoiser: nn.Module,
        num_train_timesteps: int = 1000,
        snr_gamma: float = None,
        small_weight: float = 0.3,
    ):
        """
        Rectified Flow (Flow Matching) 核心类
        
        Args:
            denoiser: DiT 模型
            num_train_timesteps: 用于将连续时间 t [0, 1] 缩放到模型接受的范围 (例如 0-1000)
            snr_gamma: Min-SNR 加权阈值 (可选，虽然 FM 通常不需要，但保留接口)
            small_weight: 小矩阵 (A1, B2) 的 Loss 权重
        """
        super().__init__()
        self.denoiser = denoiser
        self.num_train_timesteps = num_train_timesteps
        self.snr_gamma = snr_gamma
        self.small_weight = small_weight

    def transport(self, x_0, t=None, noise=None):
        """
        前向过程 (Transport): x_t = (1 - t) * x_0 + t * x_1
        构建从数据到噪声的直线轨迹。
        """
        b = x_0.shape[0]
        device = x_0.device
        
        # 1. 采样时间 t \in [0, 1]
        if t is None:
            t = torch.rand((b,), device=device)
        
        # 2. 采样噪声 x_1 (Standard Gaussian)
        if noise is None:
            x_1 = torch.randn_like(x_0)
        
        # 3. 扩展维度以便广播 (B, 1, 1)
        t_expand = t.view(b, *([1] * (len(x_0.shape) - 1)))
        
        # 4. 插值生成 x_t
        # x_t = (1 - t) * x_0 + t * x_1
        x_t = (1 - t_expand) * x_0 + t_expand * x_1
        
        # 5. 计算目标速度 v_target
        # ODE: dx/dt = v(x, t) = x_1 - x_0
        v_target = x_1 - x_0
        
        return t, x_t, v_target

    def get_model_input_time(self, t):
        """
        将连续时间 t [0, 1] 映射到模型原本设计的时间步范围 [0, num_train_timesteps]
        """
        return t * self.num_train_timesteps

    def loss_fn(self, x_0, cond, layer_ids=None, matrix_ids=None, return_pred=False):
        """
        计算 Flow Matching Loss
        """
        # 1. 执行 Transport 过程
        t, x_t, v_target = self.transport(x_0)
        
        # 2. 缩放时间步并输入模型
        t_scaled = self.get_model_input_time(t)
        
        # 3. 模型预测速度场 v_pred
        # 注意：这里我们假设 DiT 的输出就是 velocity
        v_pred = self.denoiser(x_t, t_scaled, cond, layer_ids=layer_ids, matrix_ids=matrix_ids)
        
        # 4. 计算 MSE Loss (Per Token)
        loss_per_token = F.mse_loss(v_pred, v_target, reduction='none').mean(dim=2)
        
        # 5. 应用矩阵权重 (保留原有逻辑)
        # A1 (小): layer=0, mat=0 | B2 (小): layer=1, mat=1
        is_small = ((layer_ids == 0) & (matrix_ids == 0)) | \
                   ((layer_ids == 1) & (matrix_ids == 1))
        is_large = ~is_small
        
        loss_small = (loss_per_token * is_small.float()).sum(dim=1) / (is_small.float().sum(dim=1) + 1e-8)
        loss_large = (loss_per_token * is_large.float()).sum(dim=1) / (is_large.float().sum(dim=1) + 1e-8)
        
        # Batch Loss
        loss_batch = self.small_weight * loss_small + (1 - self.small_weight) * loss_large
        
        # 6. Min-SNR 加权 (可选)
        # 在 FM 中通常不需要，或者权重恒为 1
        if self.snr_gamma is not None:
            # 这里的 SNR 计算对于 FM 来说没有直接对应的物理意义，
            # 但如果想模仿 DDPM 的加权策略，可以使用类似逻辑。
            # 为了简单起见，且遵循 FM 的常规做法，这里暂时不应用 SNR 加权，
            # 或者你可以实现特定的 FM 加权策略。
            # 目前仅返回均值。
            pass

        loss = loss_batch.mean()

        # Metrics
        with torch.no_grad():
            cos_per_token = F.cosine_similarity(v_pred, v_target, dim=2)
            cos = cos_per_token.mean()
            cos_small = (cos_per_token * is_small.float()).sum() / (is_small.float().sum() + 1e-8)
            cos_large = (cos_per_token * is_large.float()).sum() / (is_large.float().sum() + 1e-8)

        loss_dict = {
            'loss': loss,
            'cos': cos,
            'cos_small': cos_small,
            'cos_large': cos_large
        }
        
        if return_pred:
            loss_dict['pred'] = v_pred
            loss_dict['target'] = v_target
            
        return loss_dict

    @torch.no_grad()
    def sample(self, cond, shape, steps=25, layer_ids=None, matrix_ids=None, cfg_scale=1.0, uncond_cond=None):
        """
        Euler Solver 采样
        x_{t+dt} = x_t + v(x_t, t) * dt
        """
        b = shape[0]
        device = cond.device
        
        # 1. 初始化噪声 x_0 (t=0 在代码逻辑里对应 t=0.0，但在 OT-Flow 定义里 x_0 是数据)
        # 我们的 transport 是 x_t = (1-t)x_0 + t*x_1
        # 所以当 t=1 时是纯噪声 x_1，当 t=0 时是数据 x_0。
        # 因此我们从 t=1 开始，向 t=0 演化。
        x = torch.randn(shape, device=device)
        
        # 2. 时间步离散化 (从 1.0 到 0.0)
        times = torch.linspace(1.0, 0.0, steps + 1, device=device)
        dt = 1.0 / steps # 步长 (注意方向)
        
        iterator = tqdm(range(steps), desc='Euler Sampling', leave=False)
        
        for i in iterator:
            t_curr = times[i]
            # t_next = times[i+1]
            
            # 构造 Batch 时间输入
            t_batch = torch.full((b,), t_curr, device=device, dtype=torch.float32)
            t_scaled = self.get_model_input_time(t_batch) # 缩放到 [0, 1000]
            
            # 模型预测 v
            if cfg_scale > 1.0 and uncond_cond is not None:
                x_in = torch.cat([x] * 2)
                t_in = torch.cat([t_scaled] * 2)
                c_in = torch.cat([uncond_cond, cond])
                l_in = torch.cat([layer_ids] * 2) if layer_ids is not None else None
                m_in = torch.cat([matrix_ids] * 2) if matrix_ids is not None else None
                
                v_out = self.denoiser(x_in, t_in, c_in, layer_ids=l_in, matrix_ids=m_in)
                v_uncond, v_cond = v_out.chunk(2)
                v_pred = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                v_pred = self.denoiser(x, t_scaled, cond, layer_ids=layer_ids, matrix_ids=matrix_ids)

            # Euler Update
            # dx/dt = v_pred  =>  dx = v_pred * dt
            # 因为时间是从 1 -> 0，dt 应该是负的 (-1/steps)
            # 公式: x_{t-1} = x_t + v_pred * (t_{next} - t_{curr})
            # t_{next} - t_{curr} = -dt
            x = x + v_pred * (-dt)
            
        return x

    def forward(self, x_0, cond, layer_ids=None, matrix_ids=None, return_pred=False):
        return self.loss_fn(x_0, cond, layer_ids, matrix_ids, return_pred)