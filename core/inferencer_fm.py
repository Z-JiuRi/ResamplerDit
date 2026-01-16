# core/inferencer.py
import torch
import torch.nn as nn
import os
import numpy as np
# [修改] 引入 RectifiedFlow
from models.flow_matching import RectifiedFlow
from models.dit import DiT
from models.resampler import PerceiverResampler
from utils.tools import load_config, seed_everything

class Inferencer:
    def __init__(self, cfg):
        self.cfg = load_config(cfg.inference.config)
        # 用 inference 的配置覆盖 train 的配置，以便在推理时调整 steps 等
        # 但要注意不要覆盖模型结构参数
        self.inference_cfg = cfg.inference 
        
        cfg = self.cfg
        self.device = torch.device(cfg.data.device)
        self.use_ema = self.inference_cfg.use_ema
        self.checkpoint_path = self.inference_cfg.checkpoint_path
        self.cond_path = self.inference_cfg.cond_path
        
        seed_everything(self.inference_cfg.seed if 'seed' in self.inference_cfg else 42)
        
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
        
        # [修改] 使用 RectifiedFlow
        self.diffusion = RectifiedFlow(
            denoiser=self.dit,
            num_train_timesteps=cfg.diffusion.timesteps,
            snr_gamma=None, # 推理时用不到
            small_weight=cfg.diffusion.small_weight
        ).to(self.device)

        self.null_cond = nn.Parameter(torch.zeros(1, cfg.resampler.latent_cond_len, cfg.resampler.hidden_dim, device=self.device))
        
        if not os.path.exists(self.checkpoint_path):
             raise FileNotFoundError(f"Checkpoint not found at {self.checkpoint_path}")

        print(f"Loading checkpoint from {self.checkpoint_path}")
        ckpt = torch.load(self.checkpoint_path, map_location=self.device)
        
        if self.use_ema and 'ema_shadow' in ckpt:
            print("Loading EMA weights...")
            shadow = ckpt['ema_shadow']
            model_dict = {
                'resampler': self.resampler,
                'dit': self.dit,
                'null_cond_wrapper': nn.ParameterList([self.null_cond])
            }
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
            if 'null_cond' in ckpt:
                self.null_cond.data = ckpt['null_cond'].data.to(self.device)
            
        self.diffusion.eval() # 这里的 eval 对 FM 的 sample 其实影响不大，主要影响 dropout
        self.resampler.eval()
        self.dit.eval()
        
        if not os.path.exists(cfg.data.stats_path):
             raise FileNotFoundError(f"Stats file not found at {cfg.data.stats_path}")
        self.stats = torch.load(cfg.data.stats_path, map_location='cpu')
        
        self.matrix_configs = [
            {'name': 'a1', 'shape': tuple(cfg.data.original_shapes.a1), 'layer_id': 0, 'matrix_id': 0},
            {'name': 'b1', 'shape': tuple(cfg.data.original_shapes.b1), 'layer_id': 0, 'matrix_id': 1},
            {'name': 'a2', 'shape': tuple(cfg.data.original_shapes.a2), 'layer_id': 1, 'matrix_id': 0},
            {'name': 'b2', 'shape': tuple(cfg.data.original_shapes.b2), 'layer_id': 1, 'matrix_id': 1},
        ]
        self.token_size = cfg.data.token_size
        self.layer_ids, self.matrix_ids, self.num_tokens = self._make_fixed_ids()
        
        print(f"Inferencer initialized. Total generation tokens: {self.num_tokens}")

    def _make_fixed_ids(self):
        # ... (保持原样，省略以节省空间) ...
        # (请直接复制之前的 _make_fixed_ids 实现，完全通用)
        layer_ids_list = []
        matrix_ids_list = []
        total_tokens = 0
        for config in self.matrix_configs:
            rows, cols = config['shape']
            num_elements = rows * cols
            num_chunks = num_elements // self.token_size
            total_tokens += num_chunks
            l_ids = torch.full((num_chunks,), config['layer_id'], dtype=torch.long)
            m_ids = torch.full((num_chunks,), config['matrix_id'], dtype=torch.long)
            layer_ids_list.append(l_ids)
            matrix_ids_list.append(m_ids)
        layer_ids = torch.cat(layer_ids_list)
        matrix_ids = torch.cat(matrix_ids_list)
        return layer_ids, matrix_ids, total_tokens

    @torch.no_grad()
    def inference(self):
        self.resampler.eval()
        self.dit.eval()
        
        cond_tensor = torch.load(self.cond_path, map_location=self.device)['cond'].unsqueeze(0)
        cond_mask = torch.load(self.cond_path, map_location=self.device)['mask'].unsqueeze(0)

        cond_feats = self.resampler(cond_tensor, cond_mask) 
        uncond_cond = self.null_cond.expand(cond_feats.shape[0], -1, -1)
        
        batch_size = cond_tensor.shape[0]
        layer_ids = self.layer_ids.unsqueeze(0).repeat(batch_size, 1).to(self.device)
        matrix_ids = self.matrix_ids.unsqueeze(0).repeat(batch_size, 1).to(self.device)
        
        shape = (batch_size, self.num_tokens, self.token_size)
        
        # [修改] 调用 FM 的 Euler Sampling
        # 从 inference_cfg 读取 steps, 默认为 25
        steps = getattr(self.inference_cfg, 'inference_steps', 25)
        
        samples = self.diffusion.sample(
            cond=cond_feats,
            shape=shape,
            steps=steps, # Euler steps
            layer_ids=layer_ids,
            matrix_ids=matrix_ids,
            cfg_scale=self.inference_cfg.cfg_scale,
            uncond_cond=uncond_cond
        ) 
        
        output_path = os.path.join(self.inference_cfg.output_dir, os.path.basename(self.cond_path))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(self._reconstruct_matrices(samples[0]), output_path)
        print(f"Output saved to {output_path}")

    def _reconstruct_matrices(self, flat_tokens):
        # ... (保持原样，与 DDPM 版本一致) ...
        # (请直接复制之前的 _reconstruct_matrices 实现)
        flat_data = flat_tokens.flatten()
        results = {}
        ptr = 0
        for config in self.matrix_configs:
            name = config['name']
            rows, cols = config['shape']
            num_elements = rows * cols
            matrix_flat = flat_data[ptr : ptr + num_elements]
            ptr += num_elements
            if 'b' in name:
                matrix_data = matrix_flat.reshape(cols, rows) 
                if name in self.stats:
                    stat = self.stats[name]
                    matrix_data = matrix_data * stat['std'] + stat['mean']
                matrix_data = matrix_data.T
            else:
                matrix_data = matrix_flat.reshape(rows, cols)
                if name in self.stats:
                    stat = self.stats[name]
                    matrix_data = matrix_data * stat['std'] + stat['mean']
            results[name] = matrix_data.cpu()
        return results