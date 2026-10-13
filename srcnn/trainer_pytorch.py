"""PyTorch training and evaluation pipeline for SRCNN."""
import time
from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from .models_pytorch import SRCNN_PyTorch
from .metrics import psnr, compute_metrics


class PatchDataset(Dataset):
    """PyTorch Dataset that extracts random patches from full-size images."""
    
    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        patch_size: int = 48,
        length: int = 320,
        augment: bool = True,
        seed: Optional[int] = None
    ):
        """
        Args:
            X: Input images (N, H, W, 3) in [0, 1].
            Y: Target images (N, H, W, 3) in [0, 1].
            patch_size: Square patch spatial dimension.
            length: Virtual dataset length per epoch.
            augment: Whether to apply random horizontal/vertical flips.
        """
        self.X = X
        self.Y = Y
        self.patch_size = patch_size
        self.length = length
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        self.h_max = X.shape[1] - patch_size + 1
        self.w_max = X.shape[2] - patch_size + 1

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_idx = self.rng.integers(0, len(self.X))
        r = int(self.rng.integers(0, self.h_max))
        c = int(self.rng.integers(0, self.w_max))
        
        x_patch = self.X[img_idx, r:r + self.patch_size, c:c + self.patch_size].copy()
        y_patch = self.Y[img_idx, r:r + self.patch_size, c:c + self.patch_size].copy()
        
        if self.augment:
            if self.rng.random() < 0.5:
                x_patch = x_patch[:, ::-1, :]
                y_patch = y_patch[:, ::-1, :]
            if self.rng.random() < 0.5:
                x_patch = x_patch[::-1, :, :]
                y_patch = y_patch[::-1, :, :]
                
        # Ensure contiguous array layout (eliminates negative strides from flips)
        x_patch = np.ascontiguousarray(x_patch)
        y_patch = np.ascontiguousarray(y_patch)
                
        # HWC -> CHW float tensor
        x_t = torch.from_numpy(x_patch.transpose(2, 0, 1)).float()
        y_t = torch.from_numpy(y_patch.transpose(2, 0, 1)).float()
        return x_t, y_t


def evaluate_torch(
    model: SRCNN_PyTorch,
    X: np.ndarray,
    Y: np.ndarray,
    device: torch.device,
    batch_size: int = 4
) -> Tuple[float, float, np.ndarray]:
    """Evaluate PyTorch model on full test images."""
    model.eval()
    preds = []
    
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = X[i:i + batch_size]
            xb_t = torch.from_numpy(xb.transpose(0, 3, 1, 2)).float().to(device)
            out_t = model(xb_t)
            out_t = torch.clamp(out_t, 0.0, 1.0)
            out_np = out_t.cpu().numpy().transpose(0, 2, 3, 1)
            preds.append(out_np)
            
    pred_all = np.concatenate(preds, axis=0)
    mse = float(((pred_all - Y) ** 2).mean())
    p = psnr(pred_all, Y)
    return mse, float(np.mean(p)), pred_all


def train_torch(
    model: SRCNN_PyTorch,
    Xtr: np.ndarray,
    Ytr: np.ndarray,
    Xva: np.ndarray,
    Yva: np.ndarray,
    lr: float = 1e-3,
    epochs: int = 30,
    batch_size: int = 16,
    patch_size: int = 48,
    patches_per_epoch: int = 160,
    weight_decay: float = 0.0,
    device: Optional[torch.device] = None,
    save_checkpoint_path: Optional[str] = None,
    resume_checkpoint_path: Optional[str] = None,
    seed: int = 0,
    log: Any = print,
    tag: str = "torch_run"
) -> Tuple[SRCNN_PyTorch, Dict[str, Any]]:
    """Train SRCNN model using PyTorch."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    start_epoch = 1
    best_val_psnr = -1.0
    best_state_dict = None
    hist: Dict[str, Any] = {
        "epoch": [],
        "train_loss": [],
        "train_psnr": [],
        "val_loss": [],
        "val_psnr": [],
        "best_epoch": 0,
        "best_val_psnr": -1.0
    }
    
    # Support resuming from checkpoint
    if resume_checkpoint_path and os.path.exists(resume_checkpoint_path):
        ckpt = torch.load(resume_checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_psnr = ckpt.get("best_val_psnr", -1.0)
        hist = ckpt.get("history", hist)
        log(f"[{tag}] Resumed from checkpoint {resume_checkpoint_path} at epoch {start_epoch}")

    dataset = PatchDataset(Xtr, Ytr, patch_size=patch_size, length=patches_per_epoch, seed=seed)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    t0 = time.time()
    for ep in range(start_epoch, epochs + 1):
        model.train()
        losses = []
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
        train_loss = float(np.mean(losses)) if losses else 0.0
        train_psnr = 10.0 * np.log10(1.0 / max(train_loss, 1e-12))
        
        # Evaluate validation set
        val_loss, val_psnr, _ = evaluate_torch(model, Xva, Yva, device=device)
        
        hist["epoch"].append(ep)
        hist["train_loss"].append(train_loss)
        hist["train_psnr"].append(train_psnr)
        hist["val_loss"].append(val_loss)
        hist["val_psnr"].append(val_psnr)
        
        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            hist["best_epoch"] = ep
            hist["best_val_psnr"] = best_val_psnr
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
            if save_checkpoint_path:
                model.save_checkpoint(
                    save_checkpoint_path,
                    optimizer=optimizer,
                    epoch=ep,
                    history=hist,
                    best_val_psnr=best_val_psnr
                )
                
        log(f"[{tag}] epoch {ep:3d}/{epochs}  train_loss {train_loss:.5f}  val_loss {val_loss:.5f}  val_PSNR {val_psnr:6.3f} dB  ({time.time()-t0:.0f}s)")

    # Restore best validation weights
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        
    return model, hist
