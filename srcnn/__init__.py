"""SRCNN (Super-Resolution Convolutional Neural Network) package.

Supports:
- From-scratch pure-NumPy implementation with hand-written forward/backward passes.
- Maintainable PyTorch implementation with CPU/CUDA acceleration and full checkpointing.
- Image quality metrics: PSNR, SSIM, Y-channel luminance conversion, and border shaving.
"""
from .core import SRCNN, Adam
from .data import load_split, make_lr, validate_image_path
from .metrics import psnr, ssim, compute_metrics, rgb_to_y, shave_borders
from .trainer import train, evaluate
from .models_pytorch import SRCNN_PyTorch
from .trainer_pytorch import train_torch, evaluate_torch

__all__ = [
    "SRCNN",
    "Adam",
    "SRCNN_PyTorch",
    "train",
    "evaluate",
    "train_torch",
    "evaluate_torch",
    "load_split",
    "make_lr",
    "validate_image_path",
    "psnr",
    "ssim",
    "compute_metrics",
    "rgb_to_y",
    "shave_borders",
]
