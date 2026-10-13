"""Data loading, low-resolution generation and metrics."""
import glob, os
import numpy as np
from PIL import Image

SIZE = 224


def make_lr(hr_u8, size=SIZE, down="bilinear", up="bicubic"):
    """Degrade a HR PIL image: downsize by 0.5 (bilinear) then upsample back (bicubic by default)."""
    modes = {"nearest": Image.NEAREST, "bilinear": Image.BILINEAR, "bicubic": Image.BICUBIC}
    small = hr_u8.resize((size // 2, size // 2), modes[down])
    return small.resize((size, size), modes[up])


def load_split(data_dir, split, up="bicubic"):
    """Returns (X, Y) float32 arrays in [0,1], shape (N,224,224,3). X = degraded input, Y = target."""
    files = sorted(glob.glob(os.path.join(data_dir, split, "*.png")))
    if not files:
        raise FileNotFoundError(
            f"No PNG images found in {os.path.join(data_dir, split)}. "
            "Run `python scripts/make_dataset.py` (bundled sample photos) or "
            "`python scripts/prepare_div2k.py --src <DIV2K_train_HR folder>` first.")
    X, Y = [], []
    for f in files:
        hr = Image.open(f).convert("RGB")
        if hr.size != (SIZE, SIZE):
            hr = hr.resize((SIZE, SIZE), Image.BICUBIC)
        Y.append(np.asarray(hr, np.float32) / 255.0)
        X.append(np.asarray(make_lr(hr, up=up), np.float32) / 255.0)
    return np.stack(X), np.stack(Y), files


def psnr(a, b, max_val=1.0):
    """Per-image PSNR (dB) for arrays shaped (N,H,W,C); returns array of N values."""
    mse = ((a.astype(np.float64) - b.astype(np.float64)) ** 2).reshape(len(a), -1).mean(1)
    return 10 * np.log10(max_val ** 2 / np.maximum(mse, 1e-12))
