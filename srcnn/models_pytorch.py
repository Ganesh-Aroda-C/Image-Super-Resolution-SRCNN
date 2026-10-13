"""PyTorch implementation of SRCNN (Dong et al., 2015).

Supports:
- 3-stage convolutional pipeline: patch extraction, non-linear mapping, reconstruction
- Configurable layer filters, kernel sizes, activations, batch normalization, and residual learning
- Bidirectional weight loading/exporting with the pure-NumPy checkpoints (.npz)
- Seamless CPU and CUDA execution
- Full PyTorch checkpointing (.pt) with training history and configuration
"""
import os
from typing import Tuple, Dict, Any, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SRCNN_PyTorch(nn.Module):
    """Super-Resolution Convolutional Neural Network (PyTorch implementation)."""
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        f: Tuple[int, int] = (64, 32),
        k: Tuple[int, int, int] = (9, 1, 5),
        use_bn: bool = False,
        use_tanh: bool = False,
        residual: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.f = f
        self.k = k
        self.use_bn = use_bn
        self.use_tanh = use_tanh
        self.residual = residual
        
        pad = tuple((s - 1) // 2 for s in k)
        self.pad = pad
        
        # Layer 1: Patch extraction
        self.conv1 = nn.Conv2d(in_channels, f[0], kernel_size=k[0], padding=pad[0])
        self.bn1 = nn.BatchNorm2d(f[0]) if use_bn else nn.Identity()
        self.relu1 = nn.ReLU(inplace=True)
        
        # Layer 2: Non-linear mapping
        self.conv2 = nn.Conv2d(f[0], f[1], kernel_size=k[1], padding=pad[1])
        self.bn2 = nn.BatchNorm2d(f[1]) if use_bn else nn.Identity()
        self.relu2 = nn.ReLU(inplace=True)
        
        # Layer 3: Reconstruction
        self.conv3 = nn.Conv2d(f[1], out_channels, kernel_size=k[2], padding=pad[2])
        self.tanh = nn.Tanh() if use_tanh else None
        
        self._initialize_weights()

    def _initialize_weights(self):
        # He (Kaiming) initialization for feature extraction and mapping
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.conv1.bias)
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.conv2.bias)
        # Small standard deviation initialization for reconstruction layer
        nn.init.normal_(self.conv3.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.conv3.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor, shape (B, C, H, W) with pixel values in [0, 1].
        Returns:
            Reconstructed high-resolution tensor, shape (B, C, H, W).
        """
        identity = x
        
        h = self.relu1(self.bn1(self.conv1(x)))
        h = self.relu2(self.bn2(self.conv2(h)))
        out = self.conv3(h)
        
        if self.residual:
            out = out + identity
            
        if self.use_tanh:
            out = (self.tanh(out) + 1.0) * 0.5
            
        return out

    def predict(
        self,
        x: Union[np.ndarray, torch.Tensor],
        device: Optional[torch.device] = None
    ) -> np.ndarray:
        """Perform inference on arbitrary image arrays (H, W, C) or (B, H, W, C).
        
        Returns:
            NumPy array in range [0, 1], layout matching input (NHWC or HWC).
        """
        self.eval()
        if device is None:
            device = next(self.parameters()).device
            
        is_single = False
        if isinstance(x, np.ndarray):
            if x.ndim == 3:  # (H, W, C)
                is_single = True
                x = x[np.newaxis, ...]
            # Convert NHWC -> NCHW
            x_t = torch.from_numpy(x.transpose(0, 3, 1, 2)).float().to(device)
        else:
            if x.dim() == 3:
                is_single = True
                x = x.unsqueeze(0)
            x_t = x.to(device)

        with torch.no_grad():
            out_t = self.forward(x_t)
            out_t = torch.clamp(out_t, 0.0, 1.0)
            
        # NCHW -> NHWC
        out_np = out_t.cpu().numpy().transpose(0, 2, 3, 1)
        if is_single:
            return out_np[0]
        return out_np

    def get_config(self) -> Dict[str, Any]:
        """Return architecture configuration dictionary."""
        return {
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "f": self.f,
            "k": self.k,
            "use_bn": self.use_bn,
            "use_tanh": self.use_tanh,
            "residual": self.residual
        }

    def save_checkpoint(
        self,
        filepath: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: int = 0,
        history: Optional[Dict[str, Any]] = None,
        best_val_psnr: float = 0.0
    ) -> None:
        """Save a complete training checkpoint."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        checkpoint = {
            "config": self.get_config(),
            "model_state_dict": self.state_dict(),
            "epoch": epoch,
            "best_val_psnr": best_val_psnr,
            "history": history or {}
        }
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        torch.save(checkpoint, filepath)

    @classmethod
    def load_checkpoint(
        cls,
        filepath: str,
        device: torch.device = torch.device("cpu"),
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> Tuple["SRCNN_PyTorch", Dict[str, Any]]:
        """Load a PyTorch checkpoint (.pt) and reconstruct model."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"PyTorch checkpoint not found at: {filepath}")
        ckpt = torch.load(filepath, map_location=device, weights_only=False)
        cfg = ckpt.get("config", {})
        model = cls(**cfg).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        return model, ckpt

    def load_from_numpy(self, npz_path: str) -> None:
        """Load weights from a pure-NumPy SRCNN .npz checkpoint."""
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"NumPy checkpoint not found: {npz_path}")
        z = np.load(npz_path)
        
        # NumPy conv weights: (Cin, k, k, Cout) -> PyTorch: (Cout, Cin, k, k)
        w0 = torch.from_numpy(z["W0"].transpose(3, 0, 1, 2)).float()
        b0 = torch.from_numpy(z["b0"]).float()
        w1 = torch.from_numpy(z["W1"].transpose(3, 0, 1, 2)).float()
        b1 = torch.from_numpy(z["b1"]).float()
        w2 = torch.from_numpy(z["W2"].transpose(3, 0, 1, 2)).float()
        b2 = torch.from_numpy(z["b2"]).float()
        
        self.conv1.weight.data.copy_(w0)
        self.conv1.bias.data.copy_(b0)
        self.conv2.weight.data.copy_(w1)
        self.conv2.bias.data.copy_(b1)
        self.conv3.weight.data.copy_(w2)
        self.conv3.bias.data.copy_(b2)
        
        if self.use_bn and "gamma0" in z.files:
            self.bn1.weight.data.copy_(torch.from_numpy(z["gamma0"]).float())
            self.bn1.bias.data.copy_(torch.from_numpy(z["beta0"]).float())
            self.bn1.running_mean.data.copy_(torch.from_numpy(z["rm0"]).float())
            self.bn1.running_var.data.copy_(torch.from_numpy(z["rv0"]).float())
            self.bn2.weight.data.copy_(torch.from_numpy(z["gamma1"]).float())
            self.bn2.bias.data.copy_(torch.from_numpy(z["beta1"]).float())
            self.bn2.running_mean.data.copy_(torch.from_numpy(z["rm1"]).float())
            self.bn2.running_var.data.copy_(torch.from_numpy(z["rv1"]).float())

    def export_to_numpy(self, npz_path: str) -> None:
        """Export PyTorch weights into a pure-NumPy compatible .npz checkpoint."""
        os.makedirs(os.path.dirname(os.path.abspath(npz_path)), exist_ok=True)
        # PyTorch: (Cout, Cin, k, k) -> NumPy: (Cin, k, k, Cout)
        w0 = self.conv1.weight.detach().cpu().numpy().transpose(1, 2, 3, 0)
        b0 = self.conv1.bias.detach().cpu().numpy()
        w1 = self.conv2.weight.detach().cpu().numpy().transpose(1, 2, 3, 0)
        b1 = self.conv2.bias.detach().cpu().numpy()
        w2 = self.conv3.weight.detach().cpu().numpy().transpose(1, 2, 3, 0)
        b2 = self.conv3.bias.detach().cpu().numpy()
        
        s = {
            "W0": w0, "b0": b0,
            "W1": w1, "b1": b1,
            "W2": w2, "b2": b2,
            "use_bn": int(self.use_bn),
            "use_tanh": int(self.use_tanh),
            "residual": int(self.residual),
            "f": np.array(self.f, dtype=np.int32),
            "k": np.array(self.k, dtype=np.int32)
        }
        if self.use_bn:
            s.update({
                "gamma0": self.bn1.weight.detach().cpu().numpy(),
                "beta0": self.bn1.bias.detach().cpu().numpy(),
                "rm0": self.bn1.running_mean.detach().cpu().numpy(),
                "rv0": self.bn1.running_var.detach().cpu().numpy(),
                "gamma1": self.bn2.weight.detach().cpu().numpy(),
                "beta1": self.bn2.bias.detach().cpu().numpy(),
                "rm1": self.bn2.running_mean.detach().cpu().numpy(),
                "rv1": self.bn2.running_var.detach().cpu().numpy(),
            })
        np.savez(npz_path, **s)
