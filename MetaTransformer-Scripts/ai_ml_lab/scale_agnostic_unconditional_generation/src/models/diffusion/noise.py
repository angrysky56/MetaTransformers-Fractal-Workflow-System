"""
Noise scheduling and application for scale-agnostic diffusion models.
"""
import torch
import numpy as np
from typing import Tuple, Optional

class NoiseScheduler:
    """
    Manages noise scheduling for the diffusion process with scale awareness.
    """
    def __init__(
        self,
        num_diffusion_steps: int,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        schedule_type: str = "cosine"
    ):
        """
        Initialize noise scheduler.
        
        Args:
            num_diffusion_steps: Total number of diffusion steps
            beta_start: Starting value for noise schedule
            beta_end: Ending value for noise schedule
            schedule_type: Type of scheduling ("linear", "cosine", or "quadratic")
        """
        self.num_diffusion_steps = num_diffusion_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.schedule_type = schedule_type
        
        # Initialize noise schedule
        self.betas = self._get_noise_schedule()
        
        # Calculate derived quantities
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)
        
    def _get_noise_schedule(self) -> torch.Tensor:
        """
        Create noise schedule based on specified type.
        """
        if self.schedule_type == "linear":
            return torch.linspace(self.beta_start, self.beta_end, self.num_diffusion_steps)
        
        elif self.schedule_type == "cosine":
            steps = torch.linspace(0, self.num_diffusion_steps, self.num_diffusion_steps + 1)
            alpha_bars = torch.cos(((steps / self.num_diffusion_steps) + 0.008) / 1.008 * torch.pi * 0.5) ** 2
            alpha_bars = alpha_bars / alpha_bars[0]
            betas = 1 - (alpha_bars[1:] / alpha_bars[:-1])
            return torch.clip(betas, 0.0001, 0.9999)
        
        elif self.schedule_type == "quadratic":
            steps = torch.linspace(0, self.num_diffusion_steps, self.num_diffusion_steps)
            return torch.square(torch.linspace(self.beta_start**0.5, self.beta_end**0.5, self.num_diffusion_steps))
        
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")
    
    def add_noise(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        scale_factor: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add noise to input tensor with scale awareness.
        
        Args:
            x: Input tensor
            t: Timesteps
            noise: Optional pre-generated noise
            scale_factor: Factor to adjust noise scale
            
        Returns:
            tuple: (noised_input, noise)
        """
        if noise is None:
            noise = torch.randn_like(x)
            
        # Apply scale-aware noise
        sqrt_alpha_bars = self.sqrt_alpha_bars[t].view(-1, 1, 1)
        sqrt_one_minus_alpha_bars = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1)
        
        # Scale noise based on input scale
        scaled_noise = noise * scale_factor
        
        # Return noised input and noise
        return (
            sqrt_alpha_bars * x + sqrt_one_minus_alpha_bars * scaled_noise,
            scaled_noise
        )
    
    def get_variance(self, t: torch.Tensor) -> torch.Tensor:
        """
        Get variance at specified timesteps.
        """
        return self.betas[t]
    
    def predict_start_from_noise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict original input from noised version and noise.
        """
        sqrt_recip_alpha_bars = torch.rsqrt(self.alpha_bars[t]).view(-1, 1, 1)
        sqrt_recipm1_alpha_bars = torch.sqrt(1.0 / self.alpha_bars[t] - 1).view(-1, 1, 1)
        
        return sqrt_recip_alpha_bars * x_t - sqrt_recipm1_alpha_bars * noise
    
    def q_posterior_mean_variance(
        self,
        x_start: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate posterior mean and variance for the reverse process.
        """
        posterior_variance = (
            self.betas[t] * (1.0 - self.alpha_bars[t-1]) / (1.0 - self.alpha_bars[t])
        ).view(-1, 1, 1)
        
        # Calculate posterior mean
        coef1 = (
            torch.sqrt(self.alpha_bars[t-1]) * self.betas[t] / (1.0 - self.alpha_bars[t])
        ).view(-1, 1, 1)
        coef2 = (
            torch.sqrt(self.alphas[t]) * (1.0 - self.alpha_bars[t-1]) /
            (1.0 - self.alpha_bars[t])
        ).view(-1, 1, 1)
        
        posterior_mean = coef1 * x_start + coef2 * x_t
        
        return posterior_mean, posterior_variance

class ScaleAwareNoise:
    """
    Implements scale-aware noise application for different structural levels.
    """
    def __init__(
        self,
        base_scheduler: NoiseScheduler,
        min_scale: float = 0.1,
        max_scale: float = 10.0
    ):
        self.scheduler = base_scheduler
        self.min_scale = min_scale
        self.max_scale = max_scale
        
    def get_scale_factor(self, structure_size: torch.Tensor) -> float:
        """
        Calculate appropriate noise scale factor based on structure size.
        """
        # Normalize structure size to [0, 1] range
        normalized_size = (
            (structure_size - self.min_scale) / (self.max_scale - self.min_scale)
        ).clamp(0, 1)
        
        # Apply smooth scaling function
        scale_factor = torch.sin(normalized_size * np.pi / 2)
        
        return scale_factor
    
    def apply_scaled_noise(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        structure_size: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply noise with scale-dependent intensity.
        """
        scale_factor = self.get_scale_factor(structure_size)
        return self.scheduler.add_noise(x, t, scale_factor=scale_factor)
