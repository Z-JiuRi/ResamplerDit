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
        self.seeds = sorted(p.name for p in self.cond_dir.iterdir())
         
        for seed in self.seeds:
            c_path = self.cond_dir / seed
            p_path = self.param_dir / seed
            if not c_path.exists(): 
                raise FileNotFoundError(f"Condition path not found: {c_path}")
            if not p_path.exists(): 
                raise FileNotFoundError(f"Param path not found: {p_path}")
            
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
        
    def _prepare_statistics(self):
        if self.stats_path.exists():
            logger.info(f"Loaded stats from {self.stats_path}")
            return torch.load(self.stats_path, map_location='cpu')
        
        logger.info("Calculating global statistics...")
        # 用于累积 sum 和 sq_sum
        sums = {'a1': 0., 'b1': 0., 'a2': 0., 'b2': 0.}
        sq_sums = {'a1': 0., 'b1': 0., 'a2': 0., 'b2': 0.}
        counts = {'a1': 0, 'b1': 0, 'a2': 0, 'b2': 0}
        
        for item in tqdm(self.samples):
            for suffix in ['a1', 'b1', 'a2', 'b2']:
                p_path = Path(item['param_path']) / f"{item['data_id']}_{suffix}.pth"
                val = torch.load(p_path, map_location='cpu')
                sums[suffix] += val.sum().item()
                sq_sums[suffix] += (val ** 2).sum().item()
                counts[suffix] += val.numel()
        
        stats = {}
        for k in sums.keys():
            mean = sums[k] / counts[k]
            var = (sq_sums[k] / counts[k]) - (mean ** 2)
            std = np.sqrt(max(var, 1e-8))
            stats[k] = {'mean': float(mean), 'std': float(std)}
            
        torch.save(stats, self.stats_path)
        logger.info(f"Saved stats to {self.stats_path}")
        return stats

    def _co_sort(self, a, b):
        """
        Co-sorting: A(r, dim), B(dim, r)
        Rank = a.shape[0] (e.g., 2)
        """
        # 计算 A 每一行的 L2 范数
        norms = torch.norm(a, p=2, dim=1)
        # 降序排列索引
        idx = torch.argsort(norms, descending=True)
        
        # 重排 A 的行
        a_sorted = a[idx]
        # 重排 B 的列 (对应 A 的行)
        b_sorted = b[:, idx]
        
        return a_sorted, b_sorted

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
        for group in ['1', '2']:
            a_path = Path(item['param_path']) / f"{item['data_id']}_a{group}.pth"
            b_path = Path(item['param_path']) / f"{item['data_id']}_b{group}.pth"
            a = torch.load(a_path, map_location='cpu')
            b = torch.load(b_path, map_location='cpu')
            
            a, b = self._co_sort(a, b)
            
            stats_a = self.stats[f'a{group}']
            stats_b = self.stats[f'b{group}']
            a = zscore(a, stats_a['mean'], stats_a['std'])
            b = zscore(b, stats_b['mean'], stats_b['std'])
            
            params[f'a{group}'] = a
            params[f'b{group}'] = b
        
        # --- 3. Flatten & Tokenize ---
        # 顺序: A1, B1, A2, B2
        tokens_list = []    # (token_len, token_size)
        layer_ids   = []    # 0 for A1, B1, 1 for A2, B2
        matrix_ids  = []    # 0 for A, 1 for B
        
        def process_mat(mat, layer_id, matrix_id):
            flat = mat.flatten() # (N,)
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
    
    train_size = int(cfg.data.train_ratio * len(tot_datasets))
    test_size = len(tot_datasets) - train_size
    train_datasets, test_datasets = random_split(tot_datasets, [train_size, test_size], generator=torch.Generator().manual_seed(cfg.train.seed))    
    
    train_loader = DataLoader(train_datasets, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.data.num_workers)
    test_loader = DataLoader(test_datasets, batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.data.num_workers)
    
    return train_loader, test_loader


# # 测试
# if __name__ == '__main__':
#     from matplotlib import pyplot as plt
#     def plot_single_heatmap(data, filename=None):
#         plt.figure(figsize=(10, 5))
#         im = plt.imshow(data, aspect='auto', cmap='seismic', 
#                         vmin=-data.abs().max(), vmax=data.abs().max())
#         plt.colorbar(im)
#         plt.title('Single Heatmap')
#         plt.tight_layout()
#         if filename:
#             plt.savefig(filename)
#         plt.close()
    
#     data_dir = '/home/zxd/zxd/Huawei/datasets/lora'
#     stats_path = './stats.pth'
#     token_size = 128
#     batch_size = 32
#     num_workers = 8
#     train_ratio = 0.9
#     seed = 42
    
#     train_loader, test_loader = get_dataloader(data_dir, stats_path, token_size, batch_size, num_workers, train_ratio, seed)
    
#     print(f"Train dataset size: {len(train_loader.dataset)}")
#     print(f"Test dataset size: {len(test_loader.dataset)}")
    
#     # 检查一个批次的数据
#     for batch in train_loader:
#         print(f"cond shape: {batch['cond'].shape}")      # 应该是 (32, 224, 512)
#         print(batch['cond'])
#         plot_single_heatmap(batch['cond'][0].T, 'cond_heatmap.png')
#         ##################################################
#         print(f"cond_mask shape: {batch['cond_mask'].shape}") # 应该是 (32, 224)
#         print(batch['cond_mask'])
#         plot_single_heatmap(batch['cond_mask'], 'cond_mask_heatmap.png')
#         ##################################################
#         print(f"tokens shape: {batch['tokens'].shape}, \
#             mean: {batch['tokens'].mean()}, \
#             std: {batch['tokens'].std()}, \
#             max: {batch['tokens'].max()}, \
#             min: {batch['tokens'].min()}")    # 应该是 (32, 66, 128)
#         print(batch['tokens'])
#         plot_single_heatmap(batch['tokens'][0], 'tokens_heatmap.png')
#         ##################################################
#         print(f"layer_ids shape: {batch['layer_ids'].shape}") # 应该是 (32, 66)
#         print(batch['layer_ids'])
#         plot_single_heatmap(batch['layer_ids'], 'layer_ids_heatmap.png')
#         ##################################################
#         print(f"matrix_ids shape: {batch['matrix_ids'].shape}")# 应该是 (32, 66)
#         print(batch['matrix_ids'])
#         plot_single_heatmap(batch['matrix_ids'], 'matrix_ids_heatmap.png')
#         break