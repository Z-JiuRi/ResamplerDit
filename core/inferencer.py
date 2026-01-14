import torch
import os
import numpy as np
from models.ddpm import GaussianDiffusion
from models.dit import DiT
from models.resampler import PerceiverResampler

import torch.nn as nn

class Inferencer:
    def __init__(self, cfg, checkpoint_path=None, use_ema=True):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Init Models (Consistent with Trainer)
        self.resampler = PerceiverResampler(
            input_dim=cfg.data.cond_shape[1],
            hidden_dim=cfg.resampler.hidden_dim,
            cond_len=cfg.data.cond_shape[0],
            latent_cond_len=cfg.resampler.latent_cond_len,
            num_heads=cfg.resampler.num_heads,
            depth=cfg.resampler.depth
        ).to(self.device)
        
        self.dit = DiT(
            input_dim=cfg.data.token_size,
            hidden_dim=cfg.resampler.hidden_dim,
            depth=cfg.dit.depth,
            num_heads=cfg.dit.num_heads,
            max_len=cfg.data.max_len,
            mlp_ratio=cfg.dit.mlp_ratio
        ).to(self.device)
        
        self.diffusion = GaussianDiffusion(
            denoiser=self.dit,
            timesteps=cfg.diffusion.timesteps,
            beta_kwargs=cfg.diffusion.betas,
            prediction_type=cfg.diffusion.prediction_type,
            snr_gamma=cfg.diffusion.snr_gamma
        ).to(self.device)

        # Null cond placeholder (if needed for compatibility)
        self.null_cond = nn.Parameter(torch.zeros(1, cfg.resampler.latent_cond_len, cfg.resampler.hidden_dim, device=self.device))
        
        # Load Weights
        if checkpoint_path is None:
            checkpoint_path = os.path.join(self.cfg.exp_dir, "ckpts/best.pth")
            
        print(f"Loading checkpoint from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        
        if use_ema and 'ema_shadow' in ckpt:
            print("Loading EMA weights...")
            # Reconstruct EMA shadow mapping
            # Trainer used: 
            # self.ema = EMA(nn.ModuleDict({'resampler': ..., 'dit': ..., 'null_cond_wrapper': ...}))
            # Keys in ema_shadow will be like "resampler.latents", "dit.x_embedder.proj.weight"
            
            shadow = ckpt['ema_shadow']
            
            # Manually apply shadow to models
            model_dict = {
                'resampler': self.resampler,
                'dit': self.dit,
                'null_cond_wrapper': nn.ParameterList([self.null_cond])
            }
            
            # Helper to load into submodule
            for module_name, module in model_dict.items():
                for name, param in module.named_parameters():
                    full_name = f"{module_name}.{name}"
                    if full_name in shadow:
                        param.data.copy_(shadow[full_name].to(self.device))
                    else:
                        print(f"Warning: {full_name} not found in EMA shadow")
        else:
            print("Loading standard weights...")
            self.resampler.load_state_dict(ckpt['resampler'])
            self.dit.load_state_dict(ckpt['dit'])
            
        self.diffusion.eval()
        self.resampler.eval()
        self.dit.eval()
        
        # Load Stats
        self.stats = torch.load(cfg.data.stats_path)
        
        # Precompute IDs (Assume standard shape for all generations)
        # 这里需要根据 token 数量重新生成一次 IDs，逻辑与 DataLoader 相同
        # 为简化，可以在初始化时硬编码这些 IDs (因为生成长度固定为 66)
        self.num_tokens = 66 
        self.layer_ids, self.matrix_ids = self._make_fixed_ids()

    def _make_fixed_ids(self):
        # 需复用 DataLoader 中的逻辑生成 (66,) 的 tensor
        # 这里略写，实际代码请复制 DataLoader 中的 process_mat 逻辑生成一次即可
        pass 

    @torch.no_grad()
    def inference(self, cond_tensor, cond_mask=None):
        """
        cond_tensor: (1, 224, 512)
        """
        cond_tensor = cond_tensor.to(self.device)
        if cond_mask is not None:
            cond_mask = cond_mask.to(self.device)
            
        # 1. Condition
        cond_feats = self.resampler(cond_tensor, cond_mask)
        
        # 2. Sample
        shape = (1, self.num_tokens, self.cfg.data.token_size)
        # 构造 kwargs
        # 需要传入 layer_ids 等
        # ids = self.layer_ids.to(self.device).unsqueeze(0) ...
        
        samples = self.diffusion.sample(
            cond=cond_feats,
            shape=shape,
            layer_ids=self.layer_ids.to(self.device).unsqueeze(0),
            matrix_ids=self.matrix_ids.to(self.device).unsqueeze(0),
            target_mask=None # 生成时不 mask
        )
        
        # 3. Reconstruct
        # samples: (1, 66, 128) -> flatten -> split -> un-norm -> reshape
        return self._reconstruct_matrices(samples[0])

    def _reconstruct_matrices(self, flat_tokens):
        # 逆操作：
        # 1. Flatten all tokens -> (8448,)
        flat = flat_tokens.flatten()
        
        # 2. Slice & Reshape & Un-norm
        results = {}
        ptr = 0
        
        # Configs (Hardcoded for clarity based on your problem)
        shapes = [
            ('a1', (2, 64)), ('b1', (2048, 2)), 
            ('a2', (2, 2048)), ('b2', (64, 2))
        ]
        
        for name, shape in shapes:
            size = shape[0] * shape[1]
            data = flat[ptr : ptr + size].reshape(shape)
            ptr += size
            # Pad align skip (token alignment)
            # 在 DataLoader 里我们是按矩阵独立 Padding 的，所以这里也要处理
            # 实际上：需要复用 DataLoader 的 token 分割逻辑来反推 ptr 跳跃
            # 简易方案：每个矩阵占用的 chunks 数是固定的
            # a1: 128 / 128 = 1 chunk
            # b1: 4096 / 128 = 32 chunks
            # ...
            # 所以直接按 chunk 取即可
            
            # Un-normalize
            s = self.stats[name]
            data = data * s['std'] + s['mean']
            results[name] = data
            
        return results