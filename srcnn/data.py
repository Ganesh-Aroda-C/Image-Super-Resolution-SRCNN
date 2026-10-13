"""Data loading, low-resolution generation, validation, and metrics."""
import glob
import os
from typing import List, Tuple, Union, Optional
import numpy as np
from PIL import Image

# Re-export metrics for unified access
from .metrics import psnr, ssim, compute_metrics, rgb_to_y, shave_borders

SIZE = 224


def validate_image_path(image_path: str) -> None:
    """Validate that an image path exists and can be opened as an image."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    try:
        with Image.open(image_path) as img:
            img.verify()
    except Exception as e:
        raise ValueError(f"Corrupted or invalid image file at {image_path}: {e}")


def make_lr(
    hr_img: Union[Image.Image, np.ndarray],
    size: Optional[Union[int, Tuple[int, int]]] = SIZE,
    scale: float = 2.0,
    down: str = "bilinear",
    up: str = "bicubic"
) -> Image.Image:
    """Degrade a high-resolution image: downsample by scale factor, then bicubic upsample back.
    
    Args:
        hr_img: High-resolution PIL Image or NumPy array (H, W, 3).
        size: Target (width, height) or single int dimension for square crop. If None, uses hr_img.size.
        scale: Degradation scale factor (default: 2.0).
        down: Downsampling filter ("nearest", "bilinear", "bicubic", "lanczos").
        up: Upsampling filter ("nearest", "bilinear", "bicubic", "lanczos").
        
    Returns:
        Bicubic-upsampled PIL Image matching target dimensions.
    """
    modes = {
        "nearest": Image.NEAREST,
        "bilinear": Image.BILINEAR,
        "bicubic": Image.BICUBIC,
        "lanczos": Image.LANCZOS,
    }
    if isinstance(hr_img, np.ndarray):
        if hr_img.dtype != np.uint8:
            hr_img = np.clip(hr_img * 255.0, 0, 255).astype(np.uint8)
        hr_img = Image.fromarray(hr_img)
    else:
        hr_img = hr_img.copy()

    if size is not None:
        if isinstance(size, int):
            w_target, h_target = size, size
        else:
            w_target, h_target = size
    else:
        w_target, h_target = hr_img.size

    # Ensure dimensions are multiples of scale for clean down/up sampling
    w_small = max(1, int(round(w_target / scale)))
    h_small = max(1, int(round(h_target / scale)))

    down_mode = modes.get(down.lower(), Image.BILINEAR)
    up_mode = modes.get(up.lower(), Image.BICUBIC)

    small = hr_img.resize((w_small, h_small), down_mode)
    return small.resize((w_target, h_target), up_mode)


def load_split(
    data_dir: str,
    split: str,
    size: int = SIZE,
    scale: float = 2.0,
    down: str = "bilinear",
    up: str = "bicubic"
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Loads a dataset split (train, val, test) and creates aligned (X, Y) pairs.
    
    Args:
        data_dir: Base directory containing train/, val/, test/ folders.
        split: Subdirectory name ('train', 'val', or 'test').
        size: Dimension to resize HR images to (default 224x224).
        scale: Super-resolution scale factor (default 2.0).
        down: Downsampling interpolation mode.
        up: Upsampling interpolation mode.
        
    Returns:
        (X, Y, files):
            X: float32 array in [0, 1] of degraded/upsampled inputs, shape (N, size, size, 3).
            Y: float32 array in [0, 1] of ground-truth targets, shape (N, size, size, 3).
            files: List of file paths loaded.
    """
    split_dir = os.path.join(data_dir, split)
    if not os.path.exists(split_dir):
        raise FileNotFoundError(
            f"Dataset split directory '{split_dir}' does not exist.\n"
            "Run 'python scripts/make_dataset.py' to generate the bundled 100-image dataset."
        )

    files = sorted(glob.glob(os.path.join(split_dir, "*.png")) +
                   glob.glob(os.path.join(split_dir, "*.jpg")) +
                   glob.glob(os.path.join(split_dir, "*.jpeg")))
    if not files:
        raise FileNotFoundError(
            f"No images found in {split_dir}.\n"
            "Run 'python scripts/make_dataset.py' (bundled sample photos) or "
            "'python scripts/prepare_div2k.py --src <DIV2K_train_HR folder>' first."
        )

    X, Y = [], []
    for f in files:
        hr = Image.open(f).convert("RGB")
        if hr.size != (size, size):
            hr = hr.resize((size, size), Image.BICUBIC)
        Y.append(np.asarray(hr, np.float32) / 255.0)
        X.append(np.asarray(make_lr(hr, size=size, scale=scale, down=down, up=up), np.float32) / 255.0)

    return np.stack(X), np.stack(Y), files
