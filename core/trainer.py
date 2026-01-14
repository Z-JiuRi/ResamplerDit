import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
from omegaconf import OmegaConf
import torch.nn as nn

from models.ddpm import GaussianDiffusion
from models.dit import DiT
from models.resampler import PerceiverResampler
from utils.scheduler import get_lr_scheduler
from utils.tools import seed_everything, setup_logger, create_exp_dirs
from utils.visualize import setup_global_fonts, plot_gaussian, plot_heatmap, plot_histogram
from data.dataloader import get_dataloaders

import logging
logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, cfg):
# ==========================================================================================
        self.cfg = cfg
        self.device = torch.device(cfg.data.device)
        print(f"using device: {self.device}")
        # 创建实验目录
        self.exp_dir = create_exp_dirs(cfg.exp_dir)
        # 保存配置文件
        OmegaConf.save(cfg, self.exp_dir / "logs" / "config.yaml")
        logger.info(f"configs:\n{OmegaConf.to_yaml(cfg)}")
# ==========================================================================================
        setup_global_fonts()
        seed_everything(cfg.train.seed)
        setup_logger(self.exp_dir / "logs")
        self.writer = SummaryWriter(self.exp_dir / "logs")
# ==========================================================================================
        self.train_loader, self.test_loader = get_dataloaders(cfg)
        logger.info(f"[训练]: {len(self.train_loader.dataset)}")
        logger.info(f"[测试]: {len(self.test_loader.dataset)}")        
# ==========================================================================================
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
        
        self.null_cond = nn.Parameter(torch.zeros(1, cfg.resampler.latent_cond_len, cfg.resampler.hidden_dim, device=self.device), requires_grad=True)
        
        self.optimizer = optim.AdamW(
            list(self.resampler.parameters()) + list(self.dit.parameters()) + [self.null_cond],
            lr=cfg.lr_scheduler.max_lr,
            weight_decay=cfg.train.weight_decay
        )
        
        # 创建学习率调度器
        total_steps = len(self.train_loader) * cfg.train.epochs
        if cfg.lr_scheduler.type == 'cosine_warmup':
            kwargs = {
                'scheduler_type': cfg.lr_scheduler.type,
                'warmup_steps': cfg.lr_scheduler.warmup_ratio * total_steps,
                'max_steps': total_steps,
                'start_lr': cfg.lr_scheduler.start_lr,
                'eta_min': cfg.lr_scheduler.eta_min
            }
        self.scheduler = get_lr_scheduler(self.optimizer, **kwargs)
        
        self.start_epoch = 1 
        self.step = 0
        
        # 如果配置中有 resume 路径，可以在这里自动加载
        # if cfg.train.resume_path:
        #     self.load_checkpoint(cfg.train.resume_path)

    def train(self):
        logger.info("Starting training...")
        best_loss = float('inf')
        for epoch in range(self.start_epoch, self.cfg.train.epochs + 1):
            self.diffusion.train()
            self.resampler.train()
            self.dit.train()
            total_loss = 0
            total_cos = 0
            total_euclidean = 0
            
            pbar = tqdm(self.train_loader, desc=f"[Train] Epoch {epoch}")
            for batch in pbar:
                cond = batch['cond'].to(self.device)
                cond_mask = batch['cond_mask'].to(self.device)
                tokens = batch['tokens'].to(self.device)
                layer_ids = batch['layer_ids'].to(self.device)
                matrix_ids = batch['matrix_ids'].to(self.device)
                
                cond_feats = self.resampler(cond, cond_mask)
                if self.cfg.train.cfg_drop_rate > 0 and torch.rand(1).item() < self.cfg.train.cfg_drop_rate:
                    # 替换为 Null Condition (Broadcast 到 batch size)
                    current_cond = self.null_cond.expand(cond_feats.shape[0], -1, -1)
                else:
                    current_cond = cond_feats
                
                if epoch % self.cfg.train.test_interval == 0:
                    loss_dict = self.diffusion(tokens, current_cond, layer_ids=layer_ids, matrix_ids=matrix_ids, return_pred=True)
                    pred = loss_dict['pred']
                    target = loss_dict['target']
                    plot_heatmap(pred, target, self.exp_dir / "results" / "heatmap" / f"[Train]_{epoch}.png")
                    plot_histogram(pred, target, self.exp_dir / "results" / "hist" / f"[Train]_{epoch}.png")
                    plot_gaussian(pred - target, self.exp_dir / "results" / "diff" / f"[Train]_{epoch}.png")
                else:
                    loss_dict = self.diffusion(tokens, current_cond, layer_ids=layer_ids, matrix_ids=matrix_ids)
                
                loss = loss_dict['loss']
                cos = loss_dict['cos']
                euclidean = loss_dict['euclidean']
                
                # 2. Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.resampler.parameters()) \
                    + list(self.dit.parameters()) \
                    + [self.null_cond], 
                    self.cfg.train.grad_clip
                )
                self.optimizer.step()
                self.scheduler.step()
                
                total_loss += loss.item()
                total_cos += cos.item()
                total_euclidean += euclidean.item()
                
                # 4. Logging
                self.step += 1
                self.writer.add_scalar('Train/Loss', loss.item(), self.step)
                self.writer.add_scalar('Train/LR', self.scheduler.get_last_lr()[0], self.step)
                self.writer.add_scalar('Train/Cos', cos.item(), self.step)
                self.writer.add_scalar('Train/Euclidean', euclidean.item(), self.step)
                
                pbar.set_postfix({
                    'loss': f"{loss.item():.4e}",
                    'lr': f"{self.scheduler.get_last_lr()[0]:.4e}",
                    'cos': f"{cos.item():.4e}",
                    'euclidean': f"{euclidean.item():.4e}"
                })
            
            avg_loss = total_loss / len(self.train_loader)
            avg_cos = total_cos / len(self.train_loader)
            avg_euclidean = total_euclidean / len(self.train_loader)
            
            # Logging
            logger.info(f"[Train] Epoch {epoch}: Loss={avg_loss:.4e}, Cos={avg_cos:.4e}, Euclidean={avg_euclidean:.4e}")
            
            if epoch % self.cfg.train.test_interval == 0:
                test_loss = self.test(epoch)
                if test_loss < best_loss:
                    best_loss = test_loss
                    self.save_checkpoint('best')
            
            if epoch % self.cfg.train.save_interval == 0:
                self.save_checkpoint(epoch)
            

    @torch.no_grad()
    def test(self, epoch):
        self.diffusion.eval()
        self.resampler.eval()
        self.dit.eval()
        total_loss = 0
        total_cos = 0
        total_euclidean = 0
        
        for batch in self.test_loader:
            cond = batch['cond'].to(self.device)
            cond_mask = batch['cond_mask'].to(self.device)
            tokens = batch['tokens'].to(self.device)
            layer_ids = batch['layer_ids'].to(self.device)
            matrix_ids = batch['matrix_ids'].to(self.device)
            
            cond_feats = self.resampler(cond, cond_mask)
            loss_dict = self.diffusion(tokens, cond_feats, layer_ids=layer_ids, matrix_ids=matrix_ids, return_pred=True)
            pred = loss_dict['pred']
            target = loss_dict['target']
            plot_heatmap(pred, target, self.exp_dir / "results" / "heatmap" / f"[Test]_{epoch}.png")
            plot_histogram(pred, target, self.exp_dir / "results" / "hist" / f"[Test]_{epoch}.png")
            plot_gaussian(pred - target, self.exp_dir / "results" / "diff" / f"[Test]_{epoch}.png")
            total_loss += loss_dict['loss'].item()
            total_cos += loss_dict['cos'].item()
            total_euclidean += loss_dict['euclidean'].item()
            
        avg_loss = total_loss / len(self.test_loader)
        avg_cos = total_cos / len(self.test_loader)
        avg_euclidean = total_euclidean / len(self.test_loader)
        
        logger.info(f"[Test] Epoch {epoch}: Loss={avg_loss:.4e}, Cos={avg_cos:.4e}, Euclidean={avg_euclidean:.4e}")
        self.writer.add_scalar('Test/Loss', avg_loss, epoch)
        self.writer.add_scalar('Test/Cos', avg_cos, epoch)
        self.writer.add_scalar('Test/Euclidean', avg_euclidean, epoch)
        return avg_loss

    def save_checkpoint(self, epoch):
        path = self.exp_dir / "ckpts" / f"{epoch}.pth"
        torch.save({
            'resampler': self.resampler.state_dict(),
            'dit': self.dit.state_dict(),
            'null_cond': self.null_cond,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epoch': epoch
        }, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device) # 确保 map_location
        self.resampler.load_state_dict(checkpoint['resampler'])
        self.dit.load_state_dict(checkpoint['dit'])
        
        # 兼容性处理：防止旧模型没有 null_cond
        if 'null_cond' in checkpoint:
            self.null_cond.data = checkpoint['null_cond'].data
            
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        # [建议新增] 加载 LR 调度器
        if 'scheduler' in checkpoint and hasattr(self, 'scheduler'):
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            
        self.start_epoch = checkpoint['epoch'] + 1
        if self.training:
            logger.info(f"Loaded checkpoint from {path}, resuming from epoch {self.start_epoch}")
        else:
            logger.info(f"Loaded checkpoint from {path}")
