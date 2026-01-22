# 代码变更报告 (HEAD vs eps14/29d8d09)
\n生成时间: Thu 22 Jan 2026 01:43:29 PM CST
\n## 变更概览
 configs/config.yaml     |  92 +++++++------
 configs/fm/config.yaml  |  67 ---------
 core/inferencer.py      |  63 ++++++---
 core/inferencer_fm.py   | 181 ------------------------
 core/trainer.py         |  97 ++++++++-----
 core/trainer_fm.py      | 275 ------------------------------------
 data/dataloader.py      | 218 ++++++++---------------------
 data/stats.pth          | Bin 984 -> 984 bytes
 main.py                 |  13 +-
 main_fm.py              |  16 ---
 models/ddpm.py          |  33 ++++-
 models/flow_matching.py | 180 ------------------------
 scripts/inference.sh    |  15 +-
 scripts/train.sh        |  47 +------
 scripts/train_fm.sh     |  52 -------
 utils/scheduler.py      | 360 +++++++++++++++++++++++++++++-------------------
 16 files changed, 491 insertions(+), 1218 deletions(-)
\n## 详细变更
\n### 1. 训练脚本 (scripts/train.sh)
```bash
diff --git a/scripts/train.sh b/scripts/train.sh
index f4c858c..b33b766 100644
--- a/scripts/train.sh
+++ b/scripts/train.sh
@@ -7,50 +7,13 @@ export PYTHONPATH=$PYTHONPATH:$(pwd)
 
 mkdir -p logs
 log_file=logs/train_$(date +%Y%m%d_%H%M%S).log
+config_path=configs/config.yaml
+mode=train
 
 # 基础训练命令
 nohup python main.py \
-    mode=train \
-    exp_dir=exps/eps14 \
-    data.device=cuda:0 \
-    \
-    diffusion.prediction_type=eps \
-    diffusion.betas.scheduler_type=linear \
-    diffusion.snr_gamma=5.0 \
-    diffusion.small_weight=0.3 \
-    \
-    resampler.latent_cond_len=128 \
-    resampler.hidden_dim=1024 \
-    resampler.num_heads=8 \
-    resampler.depth=4 \
-    resampler.dropout=0.2 \
-    \
-    dit.num_heads=8 \
-    dit.depth=12 \
-    dit.mlp_ratio=4.0 \
-    dit.dropout=0.2 \
-    \
-    train.epochs=10000 \
-    train.batch_size=224 \
-    train.weight_decay=5e-2 \
-    train.cfg_drop_rate=0.1 \
-    train.ema_rate=0.999 \
-    train.cond_noise_factor=0.01 \
-    train.grad_accum_steps=8 \
-    \
-    lr_scheduler.type=cosine_warmup \
-    lr_scheduler.max_lr=4e-4 \
-    lr_scheduler.start_lr=1e-5 \
-    lr_scheduler.eta_min=1e-5 \
-    lr_scheduler.warmup_ratio=0.1 \
-    \
-    msg=1 \
+    --config $config_path \
+    --mode $mode \
     > $log_file 2>&1 &
 
-# tail -f $log_file
-
-# # 基础训练命令
-# python main.py \
-#     mode=train \
-#     exp_dir=exps/exp_001 \
-#     train.epochs=10000
\ No newline at end of file
+tail -f $log_file
```
\n### 2. 核心训练器 (core/trainer.py)
```python
diff --git a/core/trainer.py b/core/trainer.py
index a21260f..640396a 100644
--- a/core/trainer.py
+++ b/core/trainer.py
@@ -74,18 +74,32 @@ class Trainer:
             weight_decay=cfg.train.weight_decay
         )
         
-        self.grad_accum_steps = self.cfg.train.grad_accum_steps
-        tot_steps = (len(self.train_loader) * cfg.train.epochs + self.grad_accum_steps - 1) // self.grad_accum_steps
+        # 创建学习率调度器
+        steps_per_epoch = len(self.train_loader)
+        tot_steps = steps_per_epoch * cfg.train.epochs
+        # if cfg.lr_scheduler.type == 'cosine_warmup':
+        #     kwargs = {
+        #         'scheduler_type': cfg.lr_scheduler.type,
+        #         'warmup_steps': cfg.lr_scheduler.warmup_ratio * tot_steps,
+        #         'max_steps': tot_steps,
+        #         'start_lr': cfg.lr_scheduler.start_lr,
+        #         'eta_min': cfg.lr_scheduler.eta_min
+        #     }
+        # self.scheduler = get_lr_scheduler(self.optimizer, **kwargs)
+        
+        
+        # from utils.scheduler import CosineAnnealingRestartAtPoints
+        # self.scheduler = CosineAnnealingRestartAtPoints(
+        #     optimizer=self.optimizer,
+        #     restart_points=cfg.lr_scheduler.restart_points,
+        #     max_steps=tot_steps,
+        #     decay_ratio=cfg.lr_scheduler.decay_ratio,
+        #     eta_min=cfg.lr_scheduler.eta_min
+        # )
+        
+        from torch.optim import lr_scheduler
+        self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=tot_steps, eta_min=cfg.lr_scheduler.eta_min)
         
-        if cfg.lr_scheduler.type == 'cosine_warmup':
-            kwargs = {
-                'scheduler_type': cfg.lr_scheduler.type,
-                'warmup_steps': cfg.lr_scheduler.warmup_ratio * tot_steps,
-                'max_steps': tot_steps,
-                'start_lr': cfg.lr_scheduler.start_lr,
-                'eta_min': cfg.lr_scheduler.eta_min
-            }
-        self.scheduler = get_lr_scheduler(self.optimizer, **kwargs)
         
         self.start_epoch = 1 
         self.step = 0
@@ -112,8 +126,6 @@ class Trainer:
             self.resampler.train()
             self.dit.train()
             tot_loss = 0
-            grad_norm = 0.0
-            self.optimizer.zero_grad()
             
             pbar = tqdm(self.train_loader, desc=f"[Train] Epoch {epoch}")
             for batch_idx, batch in enumerate(pbar):
@@ -141,37 +153,44 @@ class Trainer:
                 else:
                     loss_dict = self.diffusion(tokens, current_cond, layer_ids=layer_ids, matrix_ids=matrix_ids)
                 
-                loss = loss_dict['loss'] / self.grad_accum_steps
-                cos = loss_dict['cos']
+                loss = loss_dict['loss']
+                cos_sim = loss_dict['cos_sim']
+                norm_sim = loss_dict['norm_sim']
                 cos_small = loss_dict['cos_small']
                 cos_large = loss_dict['cos_large']
+                norm_small = loss_dict['norm_small']
+                norm_large = loss_dict['norm_large']
                 
+                # 2. Backpropagation
                 loss.backward()
-                should_update = ((batch_idx + 1) % self.grad_accum_steps == 0) or (batch_idx + 1 == len(self.train_loader))
-                if should_update:
-                    grad_norm = torch.nn.utils.clip_grad_norm_(
-                        list(self.resampler.parameters()) + list(self.dit.parameters()) + [self.null_cond], 
-                        self.cfg.train.grad_clip
-                    )
-                    self.optimizer.step()
-                    self.ema.update()
-                    self.scheduler.step()
-                    self.optimizer.zero_grad()
                 
-                tot_loss += loss_dict['loss'].item()
+                grad_norm = torch.nn.utils.clip_grad_norm_(
+                    list(self.resampler.parameters()) + list(self.dit.parameters()) + [self.null_cond], 
+                    self.cfg.train.grad_clip
+                )
+                self.optimizer.step()
+                self.ema.update()
+                self.scheduler.step()
+                self.optimizer.zero_grad()
                 
+                # 4. Logging
                 self.step += 1
-                self.writer.add_scalar('Train/Loss', loss_dict['loss'].item(), self.step)
+                self.writer.add_scalar('Train/Loss', loss.item(), self.step)
                 self.writer.add_scalar('Train/LR', self.scheduler.get_last_lr()[0], self.step)
-                self.writer.add_scalar('Train/Cos', cos.item(), self.step)
+                self.writer.add_scalar('Train/Cos_sim', cos_sim.item(), self.step)
+                self.writer.add_scalar('Train/Norm_sim', norm_sim.item(), self.step)
                 self.writer.add_scalar('Train/Cos_small', cos_small.item(), self.step)
                 self.writer.add_scalar('Train/Cos_large', cos_large.item(), self.step)
+                self.writer.add_scalar('Train/Norm_small', norm_small.item(), self.step)
+                self.writer.add_scalar('Train/Norm_large', norm_large.item(), self.step)
                 self.writer.add_scalar('Train/GradNorm', grad_norm, self.step)
                 
                 pbar.set_postfix({
                     'loss': f"{loss.item():.4e}",
                     'lr': f"{self.scheduler.get_last_lr()[0]:.4e}"
                 })
+                
+                tot_loss += loss.item()
             
             avg_loss = tot_loss / len(self.train_loader)
             
@@ -195,9 +214,12 @@ class Trainer:
         self.resampler.eval()
         self.dit.eval()
         tot_loss = 0
-        tot_cos = 0
+        tot_cos_sim = 0
+        tot_norm_sim = 0
         tot_cos_small = 0
         tot_cos_large = 0
+        tot_norm_small = 0
+        tot_norm_large = 0
         
         for batch_idx, batch in enumerate(self.val_loader):
             cond = batch['cond'].to(self.device)
@@ -217,20 +239,29 @@ class Trainer:
                 plot_gaussian(pred - target, self.exp_dir / "results" / "diff" / f"[Val]_{epoch}.png")
             
             tot_loss += loss_dict['loss'].item()
-            tot_cos += loss_dict['cos'].item()
+            tot_cos_sim += loss_dict['cos_sim'].item()
+            tot_norm_sim += loss_dict['norm_sim'].item()
             tot_cos_small += loss_dict['cos_small'].item()
             tot_cos_large += loss_dict['cos_large'].item()
+            tot_norm_small += loss_dict['norm_small'].item()
+            tot_norm_large += loss_dict['norm_large'].item()
             
         avg_loss = tot_loss / len(self.val_loader)
-        avg_cos = tot_cos / len(self.val_loader)
+        avg_cos_sim = tot_cos_sim / len(self.val_loader)
+        avg_norm_sim = tot_norm_sim / len(self.val_loader)
         avg_cos_small = tot_cos_small / len(self.val_loader)
         avg_cos_large = tot_cos_large / len(self.val_loader)
+        avg_norm_small = tot_norm_small / len(self.val_loader)
+        avg_norm_large = tot_norm_large / len(self.val_loader)
         
         logger.info(f"[Val] Epoch {epoch}: Loss={avg_loss:.4e}")
         self.writer.add_scalar('Val/Loss', avg_loss, epoch)
-        self.writer.add_scalar('Val/Cos', avg_cos, epoch)
+        self.writer.add_scalar('Val/Cos_sim', avg_cos_sim, epoch)
+        self.writer.add_scalar('Val/Norm_sim', avg_norm_sim, epoch)
         self.writer.add_scalar('Val/Cos_small', avg_cos_small, epoch)
         self.writer.add_scalar('Val/Cos_large', avg_cos_large, epoch)
+        self.writer.add_scalar('Val/Norm_small', avg_norm_small, epoch)
+        self.writer.add_scalar('Val/Norm_large', avg_norm_large, epoch)
         self.ema.restore()
         return avg_loss
 
@@ -279,4 +310,4 @@ class Trainer:
         if self.training:
             logger.info(f"Loaded checkpoint from {path}, resuming from epoch {self.start_epoch}")
         else:
-            logger.info(f"Loaded checkpoint from {path}")
+            logger.info(f"Loaded checkpoint from {path}")
\ No newline at end of file
```
\n### 3. 数据加载器 (data/dataloader.py)
```python
diff --git a/data/dataloader.py b/data/dataloader.py
index 49de817..173a7af 100644
--- a/data/dataloader.py
+++ b/data/dataloader.py
@@ -5,133 +5,30 @@ import numpy as np
 from tqdm import tqdm
 from torch.utils.data import random_split
 
-from utils.tools import zscore
 
 import logging
 logger = logging.getLogger(__name__)
 
 class LoRADataset(Dataset):
-    def __init__(self, data_dir, stats_path, token_size=128):
+    def __init__(self, data_dir, token_size=128):
         self.data_dir = Path(data_dir)
-        self.cond_dir = self.data_dir / 'conditions'
-        self.param_dir = self.data_dir / 'params'
         self.token_size = token_size
-        self.stats_path = Path(stats_path)
-        
-        # 1. 索引所有文件
         self.samples = []
-        # 确保目录存在
-        if not self.cond_dir.exists():
-            raise FileNotFoundError(f"Conditions dir not found: {self.cond_dir}")
-
-        self.seeds = sorted(p.name for p in self.cond_dir.iterdir())
+        
+        self.seeds = sorted(p.name for p in self.data_dir.iterdir())
          
         for seed in self.seeds:
-            c_path = self.cond_dir / seed
-            p_path = self.param_dir / seed
-            if not c_path.exists(): 
-                logger.warning(f"Condition path not found skipping: {c_path}")
-                continue
-            if not p_path.exists(): 
-                logger.warning(f"Param path not found skipping: {p_path}")
-                continue
-            
-            files = sorted([p for p in c_path.iterdir() if p.name.endswith('.pth')])
+            files = sorted([p for p in (self.data_dir / seed).iterdir() if p.name.endswith('.pth')])
             for f in files:
-                data_id = f.stem # e.g. dataid_0
+                data = torch.load(self.data_dir / seed / f)
                 self.samples.append({
-                    'seed': seed,
-                    'data_id': data_id,
-                    'cond_path': f,
-                    'param_path': p_path
+                    'a1': data['a1'],
+                    'b1': data['b1'],
+                    'a2': data['a2'],
+                    'b2': data['b2'],
+                    'cond': data['cond'],
+                    'cond_mask': data['mask']
                 })
-        
-        # 2. 加载或计算统计量
-        self.stats = self._prepare_statistics()
-        
-    def _canonicalize_lora(self, a, b):
-        """
-        对 LoRA 矩阵进行规范化对齐（Canonicalization）。
-        解决排列对称性 (Permutation Symmetry) 和符号对称性 (Sign Symmetry)。
-        Args:
-            a: (Rank, Dim_in)  e.g., (2, 64)
-            b: (Dim_out, Rank) e.g., (2048, 2)
-        Returns:
-            a_sorted, b_sorted
-        """
-        # 1. 计算每个 Rank 分量的“能量” (Energy)
-        norm_a = torch.norm(a, p=2, dim=1)  # (Rank,)
-        norm_b = torch.norm(b, p=2, dim=0)  # (Rank,)
-        
-        # 使用乘积作为排序依据
-        energy = norm_a * norm_b 
-        
-        # 2. 按能量降序排列 (解决排列模糊性)
-        sorted_indices = torch.argsort(energy, descending=True)
-        
-        a_sorted = a[sorted_indices]
-        b_sorted = b[:, sorted_indices]
-        
-        # 3. 符号校正 (Sign Flipping) (解决符号模糊性)
-        # 找到 A 每一行中绝对值最大值的索引
-        max_abs_idx = torch.argmax(torch.abs(a_sorted), dim=1) # (Rank,)
-        
-        # Gather 取出这些位置的实际数值
-        max_vals = a_sorted.gather(1, max_abs_idx.unsqueeze(1)).squeeze(1) # (Rank,)
-        
-        # 计算翻转系数
-        signs = torch.sign(max_vals)
-        signs[signs == 0] = 1.0 
-        
-        # 广播 signs 以便乘法
-        a_final = a_sorted * signs.unsqueeze(1)
-        b_final = b_sorted * signs.unsqueeze(0)
-        
-        return a_final, b_final
-
-    def _prepare_statistics(self):
-        if self.stats_path.exists():
-            logger.info(f"Loaded stats from {self.stats_path}")
-            return torch.load(self.stats_path, map_location='cpu')
-        
-        logger.info("Calculating global statistics (with Canonicalization & Transpose)...")
-        sums = {'a1': 0., 'b1': 0., 'a2': 0., 'b2': 0.}
-        sq_sums = {'a1': 0., 'b1': 0., 'a2': 0., 'b2': 0.}
-        counts = {'a1': 0, 'b1': 0, 'a2': 0, 'b2': 0}
-        
-        for item in tqdm(self.samples):
-            # 加载原始参数
-            p_path = Path(item['param_path'])
-            a1 = torch.load(p_path / f"{item['data_id']}_a1.pth", map_location='cpu')
-            b1 = torch.load(p_path / f"{item['data_id']}_b1.pth", map_location='cpu')
-            a2 = torch.load(p_path / f"{item['data_id']}_a2.pth", map_location='cpu')
-            b2 = torch.load(p_path / f"{item['data_id']}_b2.pth", map_location='cpu')
-
-            # 1. 规范化对齐
-            a1, b1 = self._canonicalize_lora(a1, b1)
-            a2, b2 = self._canonicalize_lora(a2, b2)
-
-            # 2. 转置 B 矩阵 [Dim_out, Rank] -> [Rank, Dim_out]
-            # 这样 flatten 后才是按 Rank 分组的
-            b1 = b1.T
-            b2 = b2.T
-
-            # 统计
-            for name, val in [('a1', a1), ('b1', b1), ('a2', a2), ('b2', b2)]:
-                sums[name] += val.sum().item()
-                sq_sums[name] += (val ** 2).sum().item()
-                counts[name] += val.numel()
-        
-        stats = {}
-        for k in sums.keys():
-            mean = sums[k] / counts[k]
-            var = (sq_sums[k] / counts[k]) - (mean ** 2)
-            std = np.sqrt(max(var, 1e-8))
-            stats[k] = {'mean': float(mean), 'std': float(std)}
-            
-        torch.save(stats, self.stats_path)
-        logger.info(f"Saved stats to {self.stats_path}")
-        return stats
     
     def __len__(self):
         return len(self.samples)
@@ -139,41 +36,14 @@ class LoRADataset(Dataset):
     def __getitem__(self, idx):
         item = self.samples[idx]
         
-        # --- 1. Load Condition ---
-        cond_data = torch.load(item['cond_path'], map_location='cpu')
-        cond = cond_data['cond'] # (cond_len, cond_dim)
-        cond_mask = cond_data['mask'] # (cond_len,) 0=pad
-        
-        # --- 2. Load Params & Co-sorting ---
-        params = {}
-        
-        # 分别加载两组 LoRA
-        a1 = torch.load(Path(item['param_path']) / f"{item['data_id']}_a1.pth", map_location='cpu')
-        b1 = torch.load(Path(item['param_path']) / f"{item['data_id']}_b1.pth", map_location='cpu')
-        a2 = torch.load(Path(item['param_path']) / f"{item['data_id']}_a2.pth", map_location='cpu')
-        b2 = torch.load(Path(item['param_path']) / f"{item['data_id']}_b2.pth", map_location='cpu')
+        cond = item['cond'] # (cond_len, cond_dim)
+        cond_mask = item['cond_mask'] # (cond_len,) 0=pad
+        a1 = item['a1']
+        b1 = item['b1'].T
+        a2 = item['a2']
+        b2 = item['b2'].T
 
-        # 1. 规范化 (Canonicalize)
-        a1, b1 = self._canonicalize_lora(a1, b1)
-        a2, b2 = self._canonicalize_lora(a2, b2)
-
-        # 2. 转置 B 矩阵 (Transpose B)
-        # B: [2048, 2] -> [2, 2048]
-        b1 = b1.T 
-        b2 = b2.T
-
-        # 3. Z-Score 归一化 (使用基于转置数据的统计量)
-        def process_group_data(a, b, suffix):
-            stats_a = self.stats[f'a{suffix}']
-            stats_b = self.stats[f'b{suffix}']
-            a = zscore(a, stats_a['mean'], stats_a['std'])
-            b = zscore(b, stats_b['mean'], stats_b['std'])
-            return a, b
-
-        params['a1'], params['b1'] = process_group_data(a1, b1, '1')
-        params['a2'], params['b2'] = process_group_data(a2, b2, '2')
-        
-        # --- 3. Flatten & Tokenize ---
+        # Flatten & Tokenize
         # 顺序: A1, B1, A2, B2
         tokens_list = []    # (token_len, token_size)
         layer_ids   = []    # 0 for A1, B1, 1 for A2, B2
@@ -190,10 +60,10 @@ class LoRADataset(Dataset):
             layer_ids.append(torch.full((chunks.size(0),), layer_id))
             matrix_ids.append(torch.full((chunks.size(0),), matrix_id))
         
-        process_mat(params['a1'], 0, 0)
-        process_mat(params['b1'], 0, 1)
-        process_mat(params['a2'], 1, 0)
-        process_mat(params['b2'], 1, 1)
+        process_mat(a1, 0, 0)
+        process_mat(b1, 0, 1)
+        process_mat(a2, 1, 0)
+        process_mat(b2, 1, 1)
         
         target_tokens     = torch.cat(tokens_list, dim=0)       # (token_len, token_size)
         target_layer_ids  = torch.cat(layer_ids, dim=0).long()  # (token_len,)
@@ -208,16 +78,12 @@ class LoRADataset(Dataset):
         }
 
 def get_dataloaders(cfg):
-    tot_datasets = LoRADataset(cfg.data.data_dir, cfg.data.stats_path, cfg.data.token_size)
+    tot_datasets = LoRADataset(cfg.data.data_dir, cfg.data.token_size)
     
     tot_size = len(tot_datasets)
     train_size = int(cfg.data.train_ratio * tot_size)
     val_size  = tot_size - train_size
     
-    # 按顺序划分
-    # train_datasets = torch.utils.data.Subset(tot_datasets, range(train_size))
-    # val_datasets = torch.utils.data.Subset(tot_datasets, range(train_size, len(tot_datasets)))
-    
     train_datasets, val_datasets = random_split(
         tot_datasets, 
         [train_size, val_size],
@@ -227,4 +93,40 @@ def get_dataloaders(cfg):
     train_loader = DataLoader(train_datasets, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.data.num_workers)
     val_loader = DataLoader(val_datasets, batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.data.num_workers)
     
-    return train_loader, val_loader
\ No newline at end of file
+    return train_loader, val_loader
+
+# # 测试代码
+# if __name__ == '__main__':
+#     from omegaconf import OmegaConf
+#     cfg = OmegaConf.load('../configs/config.yaml')
+#     train_loader, val_loader = get_dataloaders(cfg)
+    
+#     # 测试训练集
+#     for batch in tqdm(train_loader, desc='Training Batch'):
+#         cond = batch['cond']
+#         cond_mask = batch['cond_mask']
+#         tokens = batch['tokens']
+#         layer_ids = batch['layer_ids']
+#         matrix_ids = batch['matrix_ids']
+        
+#         print(f"cond shape: {cond.shape}")
+#         print(f"cond_mask shape: {cond_mask.shape}")
+#         print(f"tokens shape: {tokens.shape}")
+#         print(f"layer_ids shape: {layer_ids.shape}")
+#         print(f"matrix_ids shape: {matrix_ids.shape}")
+#         break
+    
+#     # 测试验证集
+#     for batch in tqdm(val_loader, desc='Validation Batch'):
+#         cond = batch['cond']
+#         cond_mask = batch['cond_mask']
+#         tokens = batch['tokens']
+#         layer_ids = batch['layer_ids']
+#         matrix_ids = batch['matrix_ids']
+        
+#         print(f"cond shape: {cond.shape}")
+#         print(f"cond_mask shape: {cond_mask.shape}")
+#         print(f"tokens shape: {tokens.shape}")
+#         print(f"layer_ids shape: {layer_ids.shape}")
+#         print(f"matrix_ids shape: {matrix_ids.shape}")
+#         break
\ No newline at end of file
```
\n### 4. DDPM模型 (models/ddpm.py)
```python
diff --git a/models/ddpm.py b/models/ddpm.py
index dfab6ad..3889ade 100644
--- a/models/ddpm.py
+++ b/models/ddpm.py
@@ -367,8 +367,8 @@ class GaussianDiffusion(nn.Module):
         loss_small = (loss_per_token * is_small.float()).sum(dim=1) / (is_small.float().sum(dim=1) + 1e-8)
         loss_large = (loss_per_token * is_large.float()).sum(dim=1) / (is_large.float().sum(dim=1) + 1e-8)
         
-        loss_batch = F.mse_loss(denoiser_output, target, reduction='none').mean(dim=[1, 2]) # (B,)
-        # loss_batch = self.small_weight * loss_small + (1 - self.small_weight) * loss_large
+        # loss_batch = F.mse_loss(denoiser_output, target, reduction='none').mean(dim=[1, 2]) # (B,)
+        loss_batch = self.small_weight * loss_small + (1 - self.small_weight) * loss_large
         
         # Min-SNR 加权
         if self.snr_gamma is not None:
@@ -386,12 +386,14 @@ class GaussianDiffusion(nn.Module):
             loss = (loss_batch * loss_weight).mean()
         else:
             loss = loss_batch.mean()
-
+        
+        
         # Metrics
         with torch.no_grad():
             flat_pred = denoiser_output.reshape(x_0.shape[0], -1)
             flat_target = target.reshape(x_0.shape[0], -1)
-            cos = F.cosine_similarity(flat_pred, flat_target, dim=1).mean()
+            cos_sim = F.cosine_similarity(flat_pred, flat_target, dim=1).mean()
+            norm_sim = (torch.norm(flat_pred, dim=1) / (torch.norm(flat_target, dim=1) + 1e-8)).mean()
             
             # 计算每个 Token 的 Cosine: (B, T)
             cos_per_token = F.cosine_similarity(denoiser_output, target, dim=2)
@@ -399,12 +401,31 @@ class GaussianDiffusion(nn.Module):
             # sum() 是对所有 batch 和 token 求和，再除以总数
             cos_small = (cos_per_token * is_small.float()).sum() / (is_small.float().sum() + 1e-8)
             cos_large = (cos_per_token * is_large.float()).sum() / (is_large.float().sum() + 1e-8)
+            
+            # 计算 Norm Sim Small/Large
+            # 1. 计算每个 Token 的平方和 (B, T)
+            pred_sq = denoiser_output.pow(2).sum(dim=2)
+            target_sq = target.pow(2).sum(dim=2)
+            
+            # 2. 根据 Mask 聚合 (B,)
+            pred_norm_small = (pred_sq * is_small.float()).sum(dim=1).sqrt()
+            target_norm_small = (target_sq * is_small.float()).sum(dim=1).sqrt()
+            
+            pred_norm_large = (pred_sq * is_large.float()).sum(dim=1).sqrt()
+            target_norm_large = (target_sq * is_large.float()).sum(dim=1).sqrt()
+            
+            # 3. 计算 Ratio 并平均 (避免除以零)
+            norm_sim_small = (pred_norm_small / (target_norm_small + 1e-8)).mean()
+            norm_sim_large = (pred_norm_large / (target_norm_large + 1e-8)).mean()
         
         loss_dict = {
             'loss': loss,
-            'cos': cos, 
+            'cos_sim': cos_sim,
+            'norm_sim': norm_sim, 
             'cos_small': cos_small,
-            'cos_large': cos_large
+            'cos_large': cos_large,
+            'norm_small': norm_sim_small,
+            'norm_large': norm_sim_large
         }
         if return_pred:
             loss_dict['pred'] = denoiser_output
```
\n### 5. 配置文件 (configs/config.yaml)
```yaml
diff --git a/configs/config.yaml b/configs/config.yaml
index c66d2b6..a7fb2f5 100644
--- a/configs/config.yaml
+++ b/configs/config.yaml
@@ -1,70 +1,78 @@
 data:
-  data_dir: /nfs5/zxd/Huawei/datasets/lora/  # 指向你的数据集根目录
+  data_dir: /nfs5/zxd/Huawei/datasets/lora/data_zscore
   stats_path: ./data/stats.pth
   num_workers: 8
   train_ratio: 0.9
-  device: cuda:0
-  token_size: 128     # 切分LoRA矩阵的块大小
-  cond_shape: [224, 512]    # 条件序列长度
-  original_shapes: 
-    a1: [2, 64]
-    b1: [2048, 2]
-    a2: [2, 2048]
-    b2: [64, 2]
+  device: cuda:1
+  token_size: 128
+  cond_shape:
+  - 224
+  - 512
+  original_shapes:
+    a1:
+    - 2
+    - 64
+    b1:
+    - 2048
+    - 2
+    a2:
+    - 2
+    - 2048
+    b2:
+    - 64
+    - 2
   max_len: 66
-
 resampler:
-  latent_cond_len: 64   # Resampler 压缩后的序列长度
-  hidden_dim: 768      # 内部特征维度 (DiT 和 Resampler 保持一致)
-  num_heads: 4
-  depth: 4  # Resampler 层数
-  dropout: 0.1
-
+  latent_cond_len: 128
+  hidden_dim: 1024
+  num_heads: 8
+  depth: 4
+  dropout: 0.2
 diffusion:
   timesteps: 1000
-  prediction_type: eps # eps, x, v
-  snr_gamma: 5.0      # Min-SNR 权重
+  prediction_type: eps
+  snr_gamma: 5.0
   betas:
     scheduler_type: linear
     beta_start: 0.0001
     beta_end: 0.02
     s: 0.008
-
+  small_weight: 0.2
 dit:
   num_heads: 8
-  depth: 12       # DiT 层数
-  mlp_ratio: 4.0      # DiT MLP 比例
-  dropout: 0.1
-
+  depth: 8
+  mlp_ratio: 4.0
+  dropout: 0.2
 train:
   seed: 3407
-  epochs: 100
-  batch_size: 180
-  weight_decay: 1e-4
-  grad_accum_steps: 1
+  epochs: 2000
+  batch_size: 256
+  weight_decay: 0.05
   save_interval: 100
   val_interval: 10
   grad_clip: 1.0
   cfg_drop_rate: 0.1
-  ema_rate: 0.999
+  ema_rate: 0.9999
   cond_noise_factor: 0.01
-
 lr_scheduler:
-  type: cosine_warmup
+  type: cosine_restart
+  restart_points: [0.1, 0.3, 0.8]
+  decay_ratio: 0.7
   max_lr: 5e-4
   warmup_ratio: 0.1
-  start_lr: 1.0e-5
-  eta_min: 1.0e-5
-
+  start_lr: 1.0e-05
+  eta_min: 1.0e-05
 inference:
+  seed: 3407
   use_ema: true
-  config: exps/eps12/logs/config.yaml
-  checkpoint_path: exps/eps6/ckpts/best.pth
-  cond_path: /home/zxd/zxd/Huawei/datasets/lora/conditions/seed5/01105.pth
-  cfg_scale: 7.5
-  output_dir: exps/eps6/inference
-
+  config: exps/eps25/logs/config.yaml
+  checkpoint_path: exps/eps25/ckpts/best.pth
+  cond_path: /home/zxd/zxd/Huawei/datasets/lora/test/conditions
+  cfg_scale: 1.0
+  use_ddim: true
+  ddim_steps: 50
+  eta: 1.0
+  output_dir: exps/eps25/inference
 mode: train
-exp_dir: exps/eps01
-
-msg: "train eps01 first time"
+exp_dir: exps/eps25
+msg: 1
```
\n### 6. 调度器 (utils/scheduler.py)
```python
diff --git a/utils/scheduler.py b/utils/scheduler.py
index 10d164a..7c34de1 100644
--- a/utils/scheduler.py
+++ b/utils/scheduler.py
@@ -5,6 +5,69 @@ from torch.optim import Optimizer
 from torch.optim import lr_scheduler
 
 
+import math
+import torch
+from torch.optim.lr_scheduler import _LRScheduler
+import bisect
+
+class CosineAnnealingRestartAtPoints(_LRScheduler):
+    def __init__(self, optimizer, restart_points, max_steps, decay_ratio=1.0, eta_min=0, last_epoch=-1):
+        """
+        Args:
+            optimizer (Optimizer): PyTorch 优化器
+            restart_points (list[float]): 重启点位的比例列表，例如 [0.1, 0.3, 0.8]
+            max_steps (int): 总训练步数 (tot_step)
+            decay_ratio (float): 每次重启后，LR 上限的衰减比例 (0 < ratio <= 1)
+            eta_min (float): 最小学习率 (默认为 0)
+            last_epoch (int): 上一个 epoch/step 的索引
+        """
+        self.max_steps = max_steps
+        self.decay_ratio = decay_ratio
+        self.eta_min = eta_min
+        
+        # 处理重启点：排序并转换为绝对步数
+        # 自动添加 0 和 max_steps 以形成完整的区间段
+        # 例如输入 [0.1], max=100 -> [0, 10, 100]
+        points = [0.0] + sorted(restart_points) + [1.0]
+        # 去重并转换为整数步数
+        self.milestones = sorted(list(set([int(p * max_steps) for p in points])))
+        
+        super().__init__(optimizer, last_epoch)
+
+    def get_lr(self):
+        # 1. 确定当前处于哪个区间 (使用 bisect 快速查找)
+        # bisect_right 返回插入点，减 1 得到当前区间的左边界索引
+        current_step = self.last_epoch
+        
+        # 防止 step 超出范围导致的索引越界
+        if current_step >= self.milestones[-1]:
+             idx = len(self.milestones) - 2
+        else:
+            idx = bisect.bisect_right(self.milestones, current_step) - 1
+            
+        # 2. 获取当前区间的起止点
+        start_step = self.milestones[idx]
+        end_step = self.milestones[idx + 1]
+        
+        # 3. 计算当前区间的进度 (0.0 到 1.0)
+        segment_len = end_step - start_step
+        if segment_len == 0: # 避免除以0
+            progress = 1.0
+        else:
+            progress = (current_step - start_step) / segment_len
+            
+        # 4. 计算当前的衰减系数 (Decay Ratio)
+        # 第0段(idx=0)不衰减，第1段(idx=1)衰减一次，以此类推
+        current_decay = self.decay_ratio ** idx
+
+        # 5. 应用 Cosine 公式
+        return [
+            self.eta_min + (base_lr * current_decay - self.eta_min) *
+            (1 + math.cos(math.pi * progress)) / 2
+            for base_lr in self.base_lrs
+        ]
+
+
 def get_lr_scheduler(
     optimizer: Optimizer,
     **kwargs
@@ -13,63 +76,16 @@ def get_lr_scheduler(
     根据 kwargs['scheduler_type'] 返回对应的学习率调度器 (lr scheduler)。
 
     支持的 scheduler_type：
-    - "step"                : StepLR，按固定间隔阶梯式衰减
-    - "multi_step"          : MultiStepLR，在指定 epoch 列表处阶梯式衰减
-    - "exponential"         : ExponentialLR，按固定比例每个 step/epoch 指数衰减
     - "cosine"              : CosineAnnealingLR，余弦退火
-    - "cosine_warm_restart" : CosineAnnealingWarmRestarts，带周期性重启的余弦退火
-    - "reduce_on_plateau"   : ReduceLROnPlateau，指标长期不提升时降低 lr
+    - "cosine_restart" : CosineAnnealingWarmRestarts，带周期性重启的余弦退火
     - "cosine_warmup"       : 先线性预热，再余弦退火（自定义 LambdaLR）
-    - "custom_multi_step"   : 多步自定义，指定到某些 epoch 时 lr 变为某个绝对值
     - "const"               : 保持不变
     """
 
     scheduler_type = kwargs['scheduler_type'].lower()
 
-    # 1. StepLR：固定间隔阶梯衰减
-    if scheduler_type == "step":
-        """
-        调度方式：
-        - 每隔 step_size 个 step/epoch，将学习率乘以 gamma：
-          lr_t = lr_0 * (gamma ** floor(t / step_size))
-
-        参数：
-        - step_size (int)  ：衰减间隔
-        - gamma (float)    ：每次衰减倍率，默认 0.5
-        """
-        step_size: int = kwargs.get("step_size", 30)
-        gamma: float = kwargs.get("gamma", 0.5)
-        return lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
-
-    # 2. MultiStepLR：在多处阶梯衰减
-    elif scheduler_type == "multi_step":
-        """
-        调度方式：
-        - 在 milestones 指定的 epoch/step 上，将 lr 乘以 gamma（可以多次）
-
-        参数：
-        - milestones (List[int])：衰减的 epoch/step 列表（必须提供）
-        - gamma (float)         ：每次衰减倍率，默认 0.5
-        """
-        milestones: List[int] = kwargs["milestones"]
-        gamma: float = kwargs.get("gamma", 0.5)
-        return lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
-
-    # 3. ExponentialLR：指数衰减
-    elif scheduler_type == "exponential":
-        """
-        调度方式：
-        - 每次调用 scheduler.step() 时，将 lr 乘以 gamma：
-          lr_t = lr_0 * (gamma ** t)
-
-        参数：
-        - gamma (float)：衰减因子，0<gamma<1 时衰减
-        """
-        gamma: float = kwargs.get("gamma", 0.95)
-        return lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
-
-    # 4. CosineAnnealingLR：余弦退火
-    elif scheduler_type == "cosine":
+    # CosineAnnealingLR：余弦退火
+    if scheduler_type == "cosine":
         """
         调度方式：
         - 在 [0, T_max] 内按照余弦函数从初始 lr 平滑下降到 eta_min：
@@ -83,61 +99,166 @@ def get_lr_scheduler(
         eta_min: float = kwargs.get("eta_min", 0.0)
         return lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
 
-    # 5. CosineAnnealingWarmRestarts：带重启的余弦退火
-    elif scheduler_type == "cosine_warm_restart":
+
+    # CosineAnnealingWarmRestarts：带重启的余弦退火
+    elif scheduler_type == "cosine_restart":
         """
         调度方式：
-        - 使用余弦退火，但会周期性地重启；
-        - 周期长度从 T_0 开始，每次乘以 T_mult；
-        - 每个周期内从 lr_0 退火到 eta_min，然后重置回 lr_0 再退火。
-
+        - 使用余弦退火，在指定比例位置重启
+        - 每个周期内从 lr 退火到 eta_min，然后重置并衰减
+        
         参数：
-        - T_0 (int)     ：第一个周期长度
-        - T_mult (int)  ：每次重启后周期长度放大倍数，默认 2
-        - lr_ratio (float)：每个周期 lr 与初始 lr 的比例，默认 1.0
-        - eta_min (float)：每个周期的最低 lr，默认 0.0
+        - restart_points (list): 重启点位置，用总步数的比例表示，例如 [0.1, 0.3, 0.8]
+        - lr_ratio (float): 每次重启时峰值学习率的衰减因子，默认 1.0（不衰减）
+        - eta_min (float): 每个周期的最低 lr，默认 0.0
+        - total_steps (int): 总训练步数
         """
-        T_0: int = kwargs.get("T_0", 10)
-        T_mult: int = kwargs.get("T_mult", 2)
+        restart_points: list = kwargs.get("restart_points", [0.1, 0.3, 0.8])
         lr_ratio: float = kwargs.get("lr_ratio", 1.0)
         eta_min: float = kwargs.get("eta_min", 0.0)
-        return lr_scheduler.CosineAnnealingWarmRestarts(
-            optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min
-        ) * lr_ratio
-
-
-
-    # 6. ReduceLROnPlateau：指标停滞时降低学习率
-    elif scheduler_type == "reduce_on_plateau":
-        """
-        调度方式：
-        - 根据监控指标（如 val_loss）变化情况来调整 lr；
-        - 当指标在 patience 个 epoch 内没有明显改善时，将 lr 乘以 factor；
-        - 使用方式：每个 epoch 结束时调用 scheduler.step(metric_value)。
-
-        参数（常用）：
-        - mode (str)      ："min" 或 "max"，默认 "min"
-        - factor (float)  ：每次降低 lr 的倍率，如 0.5
-        - patience (int)  ：容忍多少个 epoch 不提升
-        - threshold (float)：认为“有提升”的最小变化
-        - min_lr (float)  ：lr 下界
-        """
-        mode: str = kwargs.get("mode", "min")
-        factor: float = kwargs.get("factor", 0.5)
-        patience: int = kwargs.get("patience", 10)
-        threshold: float = kwargs.get("threshold", 1e-4)
-        min_lr: float = kwargs.get("min_lr", 0.0)
-
-        return lr_scheduler.ReduceLROnPlateau(
-            optimizer,
-            mode=mode,
-            factor=factor,
-            patience=patience,
-            threshold=threshold,
-            min_lr=min_lr,
+        total_steps: int = kwargs.get("total_steps", 100)
+        
+        class CosineAnnealingRestartAtPoints(lr_scheduler._LRScheduler):
+            def __init__(self, optimizer, restart_points, total_steps, lr_ratio=1.0, eta_min=0.0, last_epoch=-1):
+                """
+                Args:
+                    optimizer (Optimizer): 优化器
+                    restart_points (list): 重启点比例列表，如 [0.1, 0.3, 0.8]
+                    total_steps (int): 总训练步数
+                    lr_ratio (float): 每次重启时峰值学习率的衰减因子
+                    eta_min (float): 每个周期的最小学习率
+                """
+                # 验证输入
+                if not restart_points:
+                    raise ValueError("restart_points 不能为空")
+                if not all(0 < p < 1 for p in restart_points):
+                    raise ValueError("restart_points 必须在 (0, 1) 范围内")
+                if not sorted(restart_points) == restart_points:
+                    raise ValueError("restart_points 必须按升序排列")
+                
+                self.restart_points = restart_points
+                self.total_steps = total_steps
+                self.lr_ratio = lr_ratio
+                self.eta_min = eta_min
+                
+                # 计算实际的重启步数
+                self.restart_steps = [int(p * total_steps) for p in restart_points]
+                # 确保最后一个重启点在总步数之前
+                if self.restart_steps[-1] >= total_steps:
+                    raise ValueError("最后一个重启点必须在总步数之前")
+                
+                # 计算每个周期的长度
+                self.period_lengths = []
+                prev_step = 0
+                for step in self.restart_steps:
+                    self.period_lengths.append(step - prev_step)
+                    prev_step = step
+                # 最后一个周期到训练结束
+                self.period_lengths.append(total_steps - prev_step)
+                
+                self.num_periods = len(self.period_lengths)
+                self.current_period = 0  # 当前周期索引
+                self.steps_in_period = 0  # 当前周期内的步数
+                
+                # 保存初始学习率作为基准
+                self.base_lrs = [group['lr'] for group in optimizer.param_groups]
+                self.current_peak_lrs = self.base_lrs.copy()  # 当前周期的峰值学习率
+                
+                super().__init__(optimizer, last_epoch)
+                
+                # 初始化状态
+                if last_epoch != -1:
+                    self._initialize_state(last_epoch)
+            
+            def _initialize_state(self, last_epoch):
+                """根据给定的epoch初始化状态"""
+                self.current_period = 0
+                cumulative_steps = 0
+                
+                # 找到当前epoch所在的周期
+                for i, length in enumerate(self.period_lengths):
+                    if last_epoch < cumulative_steps + length:
+                        self.current_period = i
+                        self.steps_in_period = last_epoch - cumulative_steps
+                        break
+                    cumulative_steps += length
+                
+                # 更新当前周期的峰值学习率（应用衰减）
+                self.current_peak_lrs = [base_lr * (self.lr_ratio ** self.current_period) 
+                                    for base_lr in self.base_lrs]
+            
+            def get_lr(self):
+                """计算当前步数的学习率"""
+                current_length = self.period_lengths[self.current_period]
+                
+                # 如果周期长度为0，避免除零错误
+                if current_length <= 0:
+                    return [self.eta_min for _ in self.base_lrs]
+                
+                # 计算余弦退火
+                # 使用 (1 + cos(π * progress)) / 2 的公式
+                # progress 从 0 到 1
+                progress = self.steps_in_period / current_length
+                
+                return [
+                    self.eta_min + (peak_lr - self.eta_min) * 
+                    (1 + math.cos(math.pi * progress)) / 2
+                    for peak_lr in self.current_peak_lrs
+                ]
+            
+            def step(self, epoch=None):
+                """更新调度器状态"""
+                if epoch is None:
+                    epoch = self.last_epoch + 1
+                
+                self.last_epoch = epoch
+                
+                # 计算累积步数以确定当前周期
+                cumulative_steps = 0
+                for i, length in enumerate(self.period_lengths):
+                    if epoch < cumulative_steps + length:
+                        new_period = i
+                        steps_in_new_period = epoch - cumulative_steps
+                        break
+                    cumulative_steps += length
+                
+                # 检查是否切换到新周期
+                if new_period != self.current_period:
+                    self.current_period = new_period
+                    self.steps_in_period = steps_in_new_period
+                    # 更新峰值学习率（应用衰减）
+                    self.current_peak_lrs = [base_lr * (self.lr_ratio ** self.current_period) 
+                                        for base_lr in self.base_lrs]
+                else:
+                    self.steps_in_period = steps_in_new_period
+                
+                # 更新优化器的学习率
+                for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
+                    param_group['lr'] = lr
+            
+            def get_current_period(self):
+                """获取当前周期信息"""
+                return {
+                    'period': self.current_period,
+                    'total_periods': self.num_periods,
+                    'steps_in_period': self.steps_in_period,
+                    'period_length': self.period_lengths[self.current_period],
+                    'current_peak_lr': self.current_peak_lrs[0] if self.current_peak_lrs else None,
+                    'next_restart_at': None if self.current_period >= len(self.restart_steps) 
+                        else self.restart_steps[self.current_period]
+                }
+        
+        import math
+        return CosineAnnealingRestartAtPoints(
+            optimizer, 
+            restart_points=restart_points,
+            total_steps=total_steps,
+            lr_ratio=lr_ratio, 
+            eta_min=eta_min
         )
 
-    # 7. 自定义：预热 + 余弦退火 (cosine_warmup)
+
+    # 自定义：预热 + 余弦退火 (cosine_warmup)
     elif scheduler_type == "cosine_warmup":
         """
         调度方式：
@@ -193,51 +314,8 @@ def get_lr_scheduler(
 
         return lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
 
-    # 8. 自定义：多步自定义绝对 lr (custom_multi_step)
-    elif scheduler_type == "custom_multi_step":
-        """
-        调度方式：
-        - 用户指定一个字典：lr_milestones = {epoch: lr_value, ...}
-        - 当 current_epoch >= 某个 epoch 时，lr 变为对应的 lr_value；
-        - 如果有多个满足条件的 epoch，取“最大且不超过当前 epoch”的那个；
-        - 例如 lr_milestones = {10: 0.01, 30: 0.001}：
-            - 0 <= epoch < 10：lr = base_lr（optimizer 初始 lr）
-            - 10 <= epoch < 30：lr = 0.01
-            - epoch >= 30    ：lr = 0.001
-
-        参数：
-        - lr_milestones (Dict[int, float])：
-            key   : epoch 编号（从 0 开始的整数）
-            value : 该 epoch 及之后使用的绝对 lr 值
-
-        注意：
-        - 这是“绝对值调度”，不是按比例乘法；
-        - 通过 LambdaLR 实现，需要用 base_lr 把绝对 lr 转成倍率。
-        """
-        lr_milestones: Dict[int, float] = kwargs["lr_milestones"]
-        if not lr_milestones:
-            raise ValueError("`custom_multi_step` requires non-empty `lr_milestones` dict.")
-
-        # 初始 lr 作为 base_lr
-        base_lr = optimizer.param_groups[0]["lr"]
-
-        # 先把 milestone 的 epoch 排序，方便查找
-        sorted_epochs = sorted(lr_milestones.keys())
-
-        def lr_lambda(current_epoch: int):
-            # 找到最后一个 <= current_epoch 的 milestone
-            target_lr = base_lr  # 默认用初始 lr
-            for e in sorted_epochs:
-                if current_epoch >= e:
-                    target_lr = lr_milestones[e]
-                else:
-                    break
-            # 转成倍率
-            return target_lr / base_lr
 
-        return lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
-    
-    # 9. 自定义：保持不变 (const)
+    # 自定义：保持不变 (const)
     elif scheduler_type == "const":
         """
         调度方式：
```
