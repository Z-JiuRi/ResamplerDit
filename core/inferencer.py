import torch
import os
import numpy as np
from models.ddpm import GaussianDiffusion
from models.dit import DiT
from models.resampler import PerceiverResampler

class Inferencer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Init Models
        self.resampler = PerceiverResampler(
            input_dim=cfg.data.svd_dim, embed_dim=cfg.model.embed_dim,
            latent_len=cfg.model.latent_length, depth=cfg.model.depth_resampler
        ).to(self.device)
        
        self.dit = DiT(
            input_dim=cfg.data.token_size, embed_dim=cfg.model.embed_dim,
            depth=cfg.model.depth_dit
        ).to(self.device)
        
        self.diffusion = GaussianDiffusion(self.dit, timesteps=cfg.diffusion.timesteps).to(self.device)
        
        # Load Weights
        if checkpoint_path is None:
            checkpoint_path = os.path.join(self.cfg.exp_dir, "checkpoints/best.ckpt")
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.resampler.load_state_dict(ckpt['resampler'])
        self.dit.load_state_dict(ckpt['dit'])
        self.diffusion.eval()
        self.resampler.eval()
        
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