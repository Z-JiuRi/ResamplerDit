from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from torch.utils.data import random_split

from utils.tools import zscore

import logging
logger = logging.getLogger(__name__)

class LoRADataset(Dataset):
    def __init__(self, data_dir, stats_path, token_size=128):
        self.data_dir = Path(data_dir)
        self.cond_dir = self.data_dir / 'conditions'
        self.param_dir = self.data_dir / 'params'
        self.token_size = token_size
        self.stats_path = Path(stats_path)
        
        # 1. 索引所有文件
        self.samples = []
        # 确保目录存在
        if not self.cond_dir.exists():
            raise FileNotFoundError(f"Conditions dir not found: {self.cond_dir}")

        self.seeds = sorted(p.name for p in self.cond_dir.iterdir())
         
        for seed in self.seeds:
            c_path = self.cond_dir / seed
            p_path = self.param_dir / seed
            if not c_path.exists(): 
                logger.warning(f"Condition path not found skipping: {c_path}")
                continue
            if not p_path.exists(): 
                logger.warning(f"Param path not found skipping: {p_path}")
                continue
            
            files = sorted([p for p in c_path.iterdir() if p.name.endswith('.pth')])
            for f in files:
                data_id = f.stem # e.g. dataid_0
                self.samples.append({
                    'seed': seed,
                    'data_id': data_id,
                    'cond_path': f,
                    'param_path': p_path
                })
        
        # 2. 加载或计算统计量
        self.stats = self._prepare_statistics()
        
    def _canonicalize_lora(self, a, b):
        """
        对 LoRA 矩阵进行规范化对齐（Canonicalization）。
        解决排列对称性 (Permutation Symmetry) 和符号对称性 (Sign Symmetry)。
        Args:
            a: (Rank, Dim_in)  e.g., (2, 64)
            b: (Dim_out, Rank) e.g., (2048, 2)
        Returns:
            a_sorted, b_sorted
        """
        # 1. 计算每个 Rank 分量的“能量” (Energy)
        norm_a = torch.norm(a, p=2, dim=1)  # (Rank,)
        norm_b = torch.norm(b, p=2, dim=0)  # (Rank,)
        
        # 使用乘积作为排序依据
        energy = norm_a * norm_b 
        
        # 2. 按能量降序排列 (解决排列模糊性)
        sorted_indices = torch.argsort(energy, descending=True)
        
        a_sorted = a[sorted_indices]
        b_sorted = b[:, sorted_indices]
        
        # 3. 符号校正 (Sign Flipping) (解决符号模糊性)
        # 找到 A 每一行中绝对值最大值的索引
        max_abs_idx = torch.argmax(torch.abs(a_sorted), dim=1) # (Rank,)
        
        # Gather 取出这些位置的实际数值
        max_vals = a_sorted.gather(1, max_abs_idx.unsqueeze(1)).squeeze(1) # (Rank,)
        
        # 计算翻转系数
        signs = torch.sign(max_vals)
        signs[signs == 0] = 1.0 
        
        # 广播 signs 以便乘法
        a_final = a_sorted * signs.unsqueeze(1)
        b_final = b_sorted * signs.unsqueeze(0)
        
        return a_final, b_final

    def _prepare_statistics(self):
        if self.stats_path.exists():
            logger.info(f"Loaded stats from {self.stats_path}")
            return torch.load(self.stats_path, map_location='cpu')
        
        logger.info("Calculating global statistics (with Canonicalization & Transpose)...")
        sums = {'a1': 0., 'b1': 0., 'a2': 0., 'b2': 0.}
        sq_sums = {'a1': 0., 'b1': 0., 'a2': 0., 'b2': 0.}
        counts = {'a1': 0, 'b1': 0, 'a2': 0, 'b2': 0}
        
        for item in tqdm(self.samples):
            # 加载原始参数
            p_path = Path(item['param_path'])
            a1 = torch.load(p_path / f"{item['data_id']}_a1.pth", map_location='cpu')
            b1 = torch.load(p_path / f"{item['data_id']}_b1.pth", map_location='cpu')
            a2 = torch.load(p_path / f"{item['data_id']}_a2.pth", map_location='cpu')
            b2 = torch.load(p_path / f"{item['data_id']}_b2.pth", map_location='cpu')

            # 1. 规范化对齐
            a1, b1 = self._canonicalize_lora(a1, b1)
            a2, b2 = self._canonicalize_lora(a2, b2)

            # 2. 转置 B 矩阵 [Dim_out, Rank] -> [Rank, Dim_out]
            # 这样 flatten 后才是按 Rank 分组的
            b1 = b1.T
            b2 = b2.T

            # 统计
            for name, val in [('a1', a1), ('b1', b1), ('a2', a2), ('b2', b2)]:
                sums[name] += val.sum().item()
                sq_sums[name] += (val ** 2).sum().item()
                counts[name] += val.numel()
        
        stats = {}
        for k in sums.keys():
            mean = sums[k] / counts[k]
            var = (sq_sums[k] / counts[k]) - (mean ** 2)
            std = np.sqrt(max(var, 1e-8))
            stats[k] = {'mean': float(mean), 'std': float(std)}
            
        torch.save(stats, self.stats_path)
        logger.info(f"Saved stats to {self.stats_path}")
        return stats
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        
        # --- 1. Load Condition ---
        cond_data = torch.load(item['cond_path'], map_location='cpu')
        cond = cond_data['cond'] # (cond_len, cond_dim)
        cond_mask = cond_data['mask'] # (cond_len,) 0=pad
        
        # --- 2. Load Params & Co-sorting ---
        params = {}
        
        # 分别加载两组 LoRA
        a1 = torch.load(Path(item['param_path']) / f"{item['data_id']}_a1.pth", map_location='cpu')
        b1 = torch.load(Path(item['param_path']) / f"{item['data_id']}_b1.pth", map_location='cpu')
        a2 = torch.load(Path(item['param_path']) / f"{item['data_id']}_a2.pth", map_location='cpu')
        b2 = torch.load(Path(item['param_path']) / f"{item['data_id']}_b2.pth", map_location='cpu')

        # 1. 规范化 (Canonicalize)
        a1, b1 = self._canonicalize_lora(a1, b1)
        a2, b2 = self._canonicalize_lora(a2, b2)

        # 2. 转置 B 矩阵 (Transpose B)
        # B: [2048, 2] -> [2, 2048]
        b1 = b1.T 
        b2 = b2.T

        # 3. Z-Score 归一化 (使用基于转置数据的统计量)
        def process_group_data(a, b, suffix):
            stats_a = self.stats[f'a{suffix}']
            stats_b = self.stats[f'b{suffix}']
            a = zscore(a, stats_a['mean'], stats_a['std'])
            b = zscore(b, stats_b['mean'], stats_b['std'])
            return a, b

        params['a1'], params['b1'] = process_group_data(a1, b1, '1')
        params['a2'], params['b2'] = process_group_data(a2, b2, '2')
        
        # --- 3. Flatten & Tokenize ---
        # 顺序: A1, B1, A2, B2
        tokens_list = []    # (token_len, token_size)
        layer_ids   = []    # 0 for A1, B1, 1 for A2, B2
        matrix_ids  = []    # 0 for A, 1 for B
        
        def process_mat(mat, layer_id, matrix_id):
            flat = mat.flatten() # (N,)
            # 这里 flat 的顺序已经是 [Rank0_row, Rank1_row...]
            if flat.size(0) % self.token_size != 0:
                raise ValueError(f"Matrix size {flat.size(0)} not divisible by token_size {self.token_size}")
            
            chunks = flat.view(-1, self.token_size) # (mat_token_len, mat_token_size)
            tokens_list.append(chunks)
            layer_ids.append(torch.full((chunks.size(0),), layer_id))
            matrix_ids.append(torch.full((chunks.size(0),), matrix_id))
        
        process_mat(params['a1'], 0, 0)
        process_mat(params['b1'], 0, 1)
        process_mat(params['a2'], 1, 0)
        process_mat(params['b2'], 1, 1)
        
        target_tokens     = torch.cat(tokens_list, dim=0)       # (token_len, token_size)
        target_layer_ids  = torch.cat(layer_ids, dim=0).long()  # (token_len,)
        target_matrix_ids = torch.cat(matrix_ids, dim=0).long() # (token_len,)
        
        return {
            'cond': cond,                   # (cond_len, cond_dim)
            'cond_mask': cond_mask,         # (cond_len,)
            'tokens': target_tokens,        # (token_len, token_size)
            'layer_ids': target_layer_ids,  # (token_len,)
            'matrix_ids': target_matrix_ids # (token_len,)
        }

def get_dataloaders(cfg):
    tot_datasets = LoRADataset(cfg.data.data_dir, cfg.data.stats_path, cfg.data.token_size)
    
    tot_size = len(tot_datasets)
    train_size = int(cfg.data.train_ratio * tot_size)
    val_size  = tot_size - train_size
    
    # 按顺序划分
    # train_datasets = torch.utils.data.Subset(tot_datasets, range(train_size))
    # val_datasets = torch.utils.data.Subset(tot_datasets, range(train_size, len(tot_datasets)))
    
    train_datasets, val_datasets = random_split(
        tot_datasets, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg.train.seed) 
    )
    
    train_loader = DataLoader(train_datasets, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.data.num_workers)
    val_loader = DataLoader(val_datasets, batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.data.num_workers)
    
    return train_loader, val_loader