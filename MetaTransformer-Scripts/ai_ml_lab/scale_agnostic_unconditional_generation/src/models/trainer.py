"""
Scale-Agnostic Training Framework

Core Mathematical Architecture:
1. Multi-Scale Learning Objective:
   L(θ) = Σ_s w(s)L_s(θ)
   where L_s(θ) = E_x,t[||ε_θ(x_t, t, s) - ε||²]

2. Scale-Aware Gradient Updates:
   ∇θL(θ) = Σ_s w(s)∇θL_s(θ)
   with adaptive weight scheduling w(s,t)

3. Progressive Curriculum:
   s(t) = s_min + (s_max - s_min)σ(f(t))
   where σ is a smoothing function
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Dict, Optional, Tuple, List, Callable, Union
from dataclasses import dataclass
import wandb
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
import time
from collections import defaultdict

@dataclass
class TrainingConfig:
    """Configuration schema for scale-agnostic training."""
    # Core Parameters
    num_epochs: int = 1000
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-6

    # Scale-Aware Parameters
    min_scale: float = 0.1
    max_scale: float = 10.0
    scale_schedule: str = "cosine"
    scale_warmup_steps: int = 1000

    # Optimization
    optimizer_type: str = "adamw"
    scheduler_type: str = "cosine"
    warmup_steps: int = 1000
    grad_clip_norm: float = 1.0

    # Advanced Features
    use_amp: bool = True
    use_ema: bool = True
    ema_decay: float = 0.999

    # System Resources
    num_workers: int = 4
    pin_memory: bool = True

    # Logging & Checkpoints
    log_interval: int = 100
    eval_interval: int = 1000
    save_interval: int = 5000
    checkpoint_dir: str = "checkpoints"

class AdaptiveMemoryManager:
    """
    Manages dynamic memory allocation and optimization.

    Core Functions:
    1. Memory monitoring
    2. Adaptive batch sizing
    3. Resource optimization
    """
    def __init__(self, initial_batch_size: int, target_memory_usage: float = 0.8):
        self.current_batch_size = initial_batch_size
        self.target_memory_usage = target_memory_usage
        self.memory_buffer = defaultdict(list)

    def optimize_batch_size(self, current_memory_usage: float) -> int:
        """
        Dynamically adjust batch size based on memory utilization.

        Strategy:
        1. Monitor current usage
        2. Project optimal batch size
        3. Apply safety margins
        """
        memory_ratio = current_memory_usage / self.target_memory_usage
        if memory_ratio > 1.1:  # Memory pressure
            self.current_batch_size = max(1, int(self.current_batch_size * 0.8))
        elif memory_ratio < 0.8:  # Memory underutilization
            self.current_batch_size = int(self.current_batch_size * 1.2)

        return self.current_batch_size

    def update_memory_stats(self, usage: float):
        """Track memory statistics for optimization."""
        self.memory_buffer['usage'].append(usage)
        if len(self.memory_buffer['usage']) > 100:
            self.memory_buffer['usage'].pop(0)

class ScaleAgnosticTrainer:
    """
    Implements scale-aware training methodology for diffusion models.

    Core Components:
    1. Multi-resolution Learning Framework
    2. Adaptive Optimization Strategy
    3. Progressive Scale Curriculum
    """
    def __init__(
        self,
        model: nn.Module,
        noise_scheduler: nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
        config: TrainingConfig,
    ):
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config

        # Initialize training infrastructure
        self._setup_training_infrastructure()

    def _setup_training_infrastructure(self):
        """Initialize training components with scale awareness."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model distribution
        if torch.cuda.device_count() > 1:
            self.model = DDP(self.model)
        self.model.to(self.device)

        # Initialize components
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.scaler = GradScaler() if self.config.use_amp else None
        self.ema_model = self._initialize_ema() if self.config.use_ema else None

        # Memory management
        self.memory_manager = AdaptiveMemoryManager(self.config.batch_size)

        # Metrics tracking
        self.metrics = defaultdict(list)

        # Setup logging
        self._setup_logging()

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create scale-aware optimizer."""
        if self.config.optimizer_type == "adamw":
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=(0.9, 0.999)
            )
        elif self.config.optimizer_type == "adam":
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer_type}")

        return optimizer

    def _create_scheduler(self):
        """Create learning rate scheduler with warmup."""
        if self.config.scheduler_type == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs,
                eta_min=1e-6
            )
        elif self.config.scheduler_type == "linear":
            return optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=self.config.num_epochs
            )
        else:
            raise ValueError(f"Unsupported scheduler: {self.config.scheduler_type}")

    def _initialize_ema(self) -> nn.Module:
        """Initialize EMA model copy."""
        ema_model = type(self.model)(
            *self.model.args,
            **self.model.kwargs
        ).to(self.device)
        ema_model.load_state_dict(self.model.state_dict())
        return ema_model

    def _update_ema(self):
        """Update EMA model parameters."""
        with torch.no_grad():
            for ema_param, model_param in zip(
                self.ema_model.parameters(),
                self.model.parameters()
            ):
                ema_param.data.mul_(self.config.ema_decay)
                ema_param.data.add_(
                    model_param.data * (1 - self.config.ema_decay)
                )

    def save_checkpoint(
        self,
        path: Path,
        epoch: int,
        val_loss: Optional[float] = None
    ):
        """Save training checkpoint with complete state."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'val_loss': val_loss
        }

        if self.config.use_ema:
            checkpoint['ema_state_dict'] = self.ema_model.state_dict()

        if self.config.use_amp:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: Path):
        """Load training checkpoint and restore state."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if self.config.use_ema and 'ema_state_dict' in checkpoint:
            self.ema_model.load_state_dict(checkpoint['ema_state_dict'])

        if self.config.use_amp and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        return checkpoint['epoch']

    def train_epoch(self, epoch: int):
        """Execute single training epoch with scale awareness."""
        self.model.train()
        epoch_metrics = defaultdict(float)
        num_batches = len(self.train_dataloader)

        with tqdm(self.train_dataloader, desc=f"Epoch {epoch}") as pbar:
            for batch_idx, batch in enumerate(pbar):
                # Calculate current scale factor
                scale_factor = self._get_scale_factor(
                    epoch * num_batches + batch_idx,
                    self.config.num_epochs * num_batches
                )

                # Adaptive batch size adjustment
                if torch.cuda.is_available():
                    current_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
                    batch_size = self.memory_manager.optimize_batch_size(current_memory)

                # Training step
                step_metrics = self._training_step(batch, scale_factor)

                # Update metrics
                for key, value in step_metrics.items():
                    epoch_metrics[key] += value

                # Update progress bar
                pbar.set_postfix({
                    'loss': epoch_metrics['total_loss'] / (batch_idx + 1),
                    'scale': scale_factor,
                    'batch_size': batch_size if torch.cuda.is_available() else self.config.batch_size
                })

                # Log metrics
                if batch_idx % self.config.log_interval == 0:
                    self._log_metrics(epoch, batch_idx, step_metrics, scale_factor)

        return {k: v / num_batches for k, v in epoch_metrics.items()}

    def validate(self, epoch: int) -> float:
        """Perform validation with current model state."""
        self.model.eval()
        total_loss = 0
        num_batches = len(self.val_dataloader)

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_dataloader):
                scale_factor = self._get_scale_factor(
                    epoch * num_batches + batch_idx,
                    self.config.num_epochs * num_batches
                )

                loss_dict = self._compute_loss(batch, scale_factor)
                total_loss += loss_dict['total_loss'].item()

        return total_loss / num_batches

    def train(self):
        """Execute complete training procedure."""
        best_val_loss = float('inf')
        train_start_time = time.time()

        for epoch in range(self.config.num_epochs):
            epoch_start_time = time.time()

            # Training epoch
            train_metrics = self.train_epoch(epoch)

            # Validation
            if epoch % self.config.eval_interval == 0:
                val_loss = self.validate(epoch)

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(
                        Path(self.config.checkpoint_dir) / "best_model.pt",
                        epoch,
                        val_loss
                    )

                self.logger.info(f"Validation Loss: {val_loss:.4f}")

            # Learning rate scheduling
            self.scheduler.step()

            # Regular checkpointing
            if epoch % self.config.save_interval == 0:
                self.save_checkpoint(
                    Path(self.config.checkpoint_dir) / f"checkpoint_{epoch}.pt",
                    epoch
                )

            # Log epoch statistics
            epoch_time = time.time() - epoch_start_time
            total_time = time.time() - train_start_time

            self.logger.info(
                f"Epoch {epoch} completed in {epoch_time:.2f}s | "
                f"Total training time: {total_time:.2f}s | "
                f"Training Loss: {train_metrics['total_loss']:.4f}"
            )

    def _compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
        scale_factor: float
    ) -> Dict[str, torch.Tensor]:
        """
        Compute scale-aware loss for current batch.

        Mathematical formulation:
        L_s(θ) = E_x,t[||ε_θ(x_t, t, s) - ε||²]
        """
        # Extract features and move to device
        node_features = batch['node_features'].to(self.device)
        edge_features = batch['edge_features'].to(self.device)
        edge_index = batch['edge_index'].to(self.device)

        # Sample timesteps
        batch_size = node_features.shape[0]
        t = torch.randint(
            0,
            self.noise_scheduler.num_diffusion_steps,
            (batch_size,),
            device=self.device
        )

        # Add noise to features
        noise_scale = scale_factor * torch.ones(batch_size, device=self.device)
        noised_nodes, node_noise = self.noise_scheduler.add_noise(
            node_features, t, scale_factor=noise_scale
        )
        noised_edges, edge_noise = self.noise_scheduler.add_noise(
            edge_features, t, scale_factor=noise_scale
        )

        # Predict noise
        with autocast(enabled=self.config.use_amp):
            node_pred, edge_pred = self.model(
                noised_nodes,
                edge_index,
noised_edges,
                t,
                scale_factor=noise_scale
            )

            # Compute losses with scale awareness
            node_loss = torch.nn.functional.mse_loss(node_pred, node_noise)
            edge_loss = torch.nn.functional.mse_loss(edge_pred, edge_noise)

            # Scale-weighted total loss
            total_loss = node_loss + edge_loss

        return {
            'total_loss': total_loss,
            'node_loss': node_loss,
            'edge_loss': edge_loss
        }

    def _training_step(
        self,
        batch: Dict[str, torch.Tensor],
        scale_factor: float
    ) -> Dict[str, float]:
        """Execute single training step with scale awareness."""
        self.optimizer.zero_grad()

        if self.config.use_amp:
            with autocast():
                loss_dict = self._compute_loss(batch, scale_factor)

            # Scale loss and backward pass
            self.scaler.scale(loss_dict['total_loss']).backward()

            if self.config.grad_clip_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip_norm
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss_dict = self._compute_loss(batch, scale_factor)
            loss_dict['total_loss'].backward()

            if self.config.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip_norm
                )

            self.optimizer.step()

        # Update EMA model if enabled
        if self.config.use_ema:
            self._update_ema()

        return {k: v.item() for k, v in loss_dict.items()}

    def _get_scale_factor(self, step: int, total_steps: int) -> float:
        """
        Calculate current scale factor based on training progress.

        Mathematical formulation:
        s(t) = s_min + (s_max - s_min)σ(f(t))
        where σ is a smoothing function
        """
        progress = step / total_steps

        if self.config.scale_schedule == "linear":
            factor = progress
        elif self.config.scale_schedule == "cosine":
            factor = 0.5 * (1 + np.cos(np.pi * progress))
        elif self.config.scale_schedule == "exponential":
            factor = np.exp(-5 * (1 - progress))
        else:
            raise ValueError(f"Unsupported scale schedule: {self.config.scale_schedule}")

        return self.config.min_scale + (
            self.config.max_scale - self.config.min_scale
        ) * factor

    def _log_metrics(
        self,
        epoch: int,
        batch_idx: int,
        metrics: Dict[str, float],
        scale_factor: float
    ):
        """Log training metrics with structured organization."""
        log_dict = {
            'training/epoch': epoch,
            'training/batch': batch_idx,
            'training/scale_factor': scale_factor,
            'training/learning_rate': self.optimizer.param_groups[0]['lr']
        }

        # Add loss components
        for key, value in metrics.items():
            log_dict[f'training/{key}'] = value

        # Log resource utilization
        if torch.cuda.is_available():
            log_dict['system/gpu_memory_used'] = torch.cuda.max_memory_allocated() / 1e9

        # Log to wandb if enabled
        if wandb.run is not None:
            wandb.log(log_dict)

        # Console logging
        self.logger.info(
            f"Epoch {epoch} | Batch {batch_idx} | "
            f"Loss: {metrics['total_loss']:.4f} | "
            f"Scale: {scale_factor:.4f}"
        )

    def _setup_logging(self):
        """Initialize hierarchical logging system."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)s | %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(
                    Path(self.config.checkpoint_dir) / 'training.log'
                )
            ]
        )
        self.logger = logging.getLogger(__name__)

        # Initialize wandb if available
        try:
            wandb.init(
                project="scale-agnostic-generation",
                config=vars(self.config),
                resume=True
            )
        except:
            self.logger.warning("Failed to initialize wandb. Continuing without it.")
