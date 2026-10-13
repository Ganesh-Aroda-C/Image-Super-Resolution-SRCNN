"""Evaluation metrics for Image Super-Resolution: PSNR, SSIM, and Color Conversion.

Supports:
- Per-image or batched inputs (NumPy arrays or PyTorch tensors)
- Evaluation on RGB or Y-channel (luminance, standard in SR literature)
- Configurable border shaving (to exclude boundary artifacts)
- Robust handling of edge cases (identical images, small images, zero variance)
"""
import numpy as np
from typing import Union, Tuple, Optional


def rgb_to_y(img: np.ndarray) -> np.ndarray:
    """Convert RGB image array (in [0, 1]) to luminance Y channel.
    
    Uses standard ITU-R BT.601 conversion:
        Y = 0.299 * R + 0.587 * G + 0.114 * B
    
    Args:
        img: Array of shape (..., 3) with values in [0, 1].
    Returns:
        Array of shape (..., 1) or (...) containing luminance values in [0, 1].
    """
    if img.shape[-1] != 3:
        raise ValueError(f"Expected 3 channels in last dimension, got shape {img.shape}")
    r = img[..., 0]
    g = img[..., 1]
    b = img[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return y[..., np.newaxis]


def shave_borders(img: np.ndarray, shave: int = 0) -> np.ndarray:
    """Crop boundary pixels from an image array.
    
    Args:
        img: Array of shape (H, W, C) or (N, H, W, C).
        shave: Number of border pixels to remove on each side.
    Returns:
        Cropped array.
    """
    if shave <= 0:
        return img
    if img.ndim == 3:  # (H, W, C)
        h, w = img.shape[:2]
        if 2 * shave >= h or 2 * shave >= w:
            return img
        return img[shave:-shave, shave:-shave, :]
    elif img.ndim == 4:  # (N, H, W, C)
        h, w = img.shape[1:3]
        if 2 * shave >= h or 2 * shave >= w:
            return img
        return img[:, shave:-shave, shave:-shave, :]
    elif img.ndim == 2:  # (H, W)
        h, w = img.shape
        if 2 * shave >= h or 2 * shave >= w:
            return img
        return img[shave:-shave, shave:-shave]
    return img


def psnr(a: np.ndarray, b: np.ndarray, max_val: float = 1.0) -> np.ndarray:
    """Compute Peak Signal-to-Noise Ratio (PSNR) in decibels (dB).
    
    Args:
        a: Predicted image(s), float array in range [0, max_val].
        b: Ground truth image(s), float array in range [0, max_val].
        max_val: Maximum possible pixel value (typically 1.0 or 255.0).
        
    Returns:
        1D NumPy array of PSNR values for each image in the batch,
        or scalar float if input was a single 2D/3D image.
    """
    is_single = False
    if a.ndim <= 3:
        is_single = True
        a = np.expand_dims(a, 0)
        b = np.expand_dims(b, 0)

    a_f64 = a.astype(np.float64)
    b_f64 = b.astype(np.float64)
    
    # Per-sample MSE
    mse = ((a_f64 - b_f64) ** 2).reshape(len(a), -1).mean(axis=1)
    
    # Handle perfect match (MSE == 0) safely: return 100.0 dB
    psnr_vals = np.zeros_like(mse)
    zero_mask = mse <= 1e-12
    psnr_vals[zero_mask] = 100.0
    psnr_vals[~zero_mask] = 10.0 * np.log10((max_val ** 2) / mse[~zero_mask])
    
    if is_single:
        return float(psnr_vals[0])
    return psnr_vals


def ssim(a: np.ndarray, b: np.ndarray, max_val: float = 1.0) -> np.ndarray:
    """Compute Structural Similarity Index Measure (SSIM).
    
    Uses scikit-image's structural_similarity with automatic window size adjustment
    for small image patches.
    
    Args:
        a: Predicted image(s) array, values in [0, max_val].
        b: Ground truth image(s) array, values in [0, max_val].
        max_val: Maximum pixel value (1.0 for normalized, 255 for uint8).
        
    Returns:
        1D NumPy array of SSIM values (one per image), or scalar float if single image.
    """
    try:
        from skimage.metrics import structural_similarity as _ssim_fn
    except ImportError:
        raise ImportError("scikit-image is required for SSIM calculation: pip install scikit-image")

    is_single = False
    if a.ndim <= 3:
        is_single = True
        a = np.expand_dims(a, 0)
        b = np.expand_dims(b, 0)

    ssim_vals = []
    for i in range(len(a)):
        img1 = a[i]
        img2 = b[i]
        
        # Determine minimum spatial dimension
        min_dim = min(img1.shape[0], img1.shape[1])
        win_size = min(7, min_dim)
        if win_size % 2 == 0:
            win_size -= 1
        if win_size < 3:
            # Fallback for ultra-small image patches where SSIM window doesn't fit
            score = 1.0 if np.allclose(img1, img2, atol=1e-3) else float(1.0 - np.mean(np.abs(img1 - img2)))
            ssim_vals.append(score)
            continue
            
        is_multichannel = (img1.ndim == 3 and img1.shape[-1] in (1, 3))
        channel_axis = -1 if is_multichannel else None
        
        score = _ssim_fn(
            img1, img2,
            data_range=max_val,
            win_size=win_size,
            channel_axis=channel_axis
        )
        ssim_vals.append(float(score))
        
    res = np.array(ssim_vals, dtype=np.float64)
    if is_single:
        return float(res[0])
    return res


def compute_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    shave: int = 0,
    use_y_channel: bool = False
) -> Tuple[float, float]:
    """Compute both PSNR and SSIM with optional border shaving and Y-channel extraction.
    
    Args:
        pred: Predicted image, (H, W, C) in [0, 1].
        target: Target image, (H, W, C) in [0, 1].
        shave: Border pixels to shave off before evaluation.
        use_y_channel: If True, convert RGB to Y channel before evaluation.
        
    Returns:
        (psnr_val, ssim_val) tuple of floats.
    """
    p = np.clip(pred.astype(np.float64), 0.0, 1.0)
    t = np.clip(target.astype(np.float64), 0.0, 1.0)
    
    if shave > 0:
        p = shave_borders(p, shave)
        t = shave_borders(t, shave)
        
    if use_y_channel and p.ndim == 3 and p.shape[-1] == 3:
        p = rgb_to_y(p)
        t = rgb_to_y(t)
        
    p_val = float(psnr(p, t, max_val=1.0))
    s_val = float(ssim(p, t, max_val=1.0))
    return p_val, s_val
