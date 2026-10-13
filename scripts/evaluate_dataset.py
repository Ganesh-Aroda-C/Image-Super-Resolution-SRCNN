"""Benchmark evaluation script for SRCNN on any image dataset (e.g. Set5, Set14, or data/test).

Evaluates:
- Bicubic vs SRCNN
- RGB PSNR and SSIM
- Luminance Y-channel PSNR and SSIM (standard SR literature convention with border shaving)
- Per-image breakdown and average metric gains

Usage:
  python scripts/evaluate_dataset.py --dataset data/test
  python scripts/evaluate_dataset.py --dataset path/to/Set5 --model results/residual_lr_0.001.npz
"""
import argparse
import glob
import json
import os
import sys
import time
import numpy as np
from PIL import Image

import _path
from _path import ROOT
from srcnn import SRCNN, SRCNN_PyTorch, psnr, ssim, compute_metrics, make_lr


def parse_args():
    ap = argparse.ArgumentParser(description="Evaluate SRCNN on a dataset directory")
    ap.add_argument("--dataset", default=os.path.join(ROOT, "data", "test"),
                    help="Directory containing test images (*.png, *.jpg, *.bmp)")
    ap.add_argument("--model", default=os.path.join(ROOT, "results", "residual_lr_0.001.npz"),
                    help="Model checkpoint (.npz or .pt)")
    ap.add_argument("--scale", type=float, default=2.0, help="Downsampling scale factor")
    ap.add_argument("--shave", type=int, default=2, help="Border shave width in pixels for Y-channel metrics")
    ap.add_argument("--out", default=None, help="Optional JSON output path for metrics summary")
    return ap.parse_args()


def load_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    if path.endswith(".pt"):
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, _ = SRCNN_PyTorch.load_checkpoint(path, device=device)
        return model, "pytorch"
    else:
        model = SRCNN.load(path)
        return model, "numpy"


def main():
    a = parse_args()
    if not os.path.exists(a.dataset):
        print(f"[ERROR] Dataset directory not found: {a.dataset}", file=sys.stderr)
        sys.exit(1)

    model, backend = load_model(a.model)
    
    # Collect images
    extensions = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(a.dataset, ext)))
        files.extend(glob.glob(os.path.join(a.dataset, "**", ext), recursive=True))
    files = sorted(list(set(files)))
    
    if not files:
        print(f"[ERROR] No images found in {a.dataset}", file=sys.stderr)
        sys.exit(1)

    print("=" * 80)
    print(f"BENCHMARK EVALUATION: {os.path.basename(a.dataset)} ({len(files)} images)")
    print(f"Model Checkpoint   : {os.path.basename(a.model)} ({backend.upper()})")
    print(f"Scale Factor       : {a.scale}x  |  Border Shave: {a.shave}px")
    print("=" * 80)
    print(f"{'Image':<26} | {'Bicubic PSNR':<12} | {'SRCNN PSNR':<12} | {'Gain (dB)':<10} | {'SRCNN SSIM':<10}")
    print("-" * 80)

    results = []
    bic_psnr_list, src_psnr_list = [], []
    bic_ssim_list, src_ssim_list = [], []
    bic_psnr_y_list, src_psnr_y_list = [], []
    bic_ssim_y_list, src_ssim_y_list = [], []

    for f in files:
        fname = os.path.basename(f)
        img = Image.open(f).convert("RGB")
        w = int(img.width // a.scale * a.scale)
        h = int(img.height // a.scale * a.scale)
        if w < 16 or h < 16:
            continue
            
        hr_pil = img.crop((0, 0, w, h))
        lr_w, lr_h = int(w // a.scale), int(h // a.scale)
        lr_pil = hr_pil.resize((lr_w, lr_h), Image.BILINEAR)
        bic_pil = lr_pil.resize((w, h), Image.BICUBIC)

        x_np = np.asarray(bic_pil, np.float32)[np.newaxis, ...] / 255.0
        hr_np = np.asarray(hr_pil, np.float32)[np.newaxis, ...] / 255.0

        sr_np = model.predict(x_np)

        # RGB metrics
        b_p = float(psnr(x_np, hr_np)[0])
        s_p = float(psnr(sr_np, hr_np)[0])
        b_s = float(ssim(x_np, hr_np)[0])
        s_s = float(ssim(sr_np, hr_np)[0])

        # Y-channel metrics
        b_py, b_sy = compute_metrics(x_np[0], hr_np[0], shave=a.shave, use_y_channel=True)
        s_py, s_sy = compute_metrics(sr_np[0], hr_np[0], shave=a.shave, use_y_channel=True)

        gain_rgb = s_p - b_p
        print(f"{fname[:25]:<26} | {b_p:9.3f} dB  | {s_p:9.3f} dB  | {gain_rgb:+8.3f} dB | {s_s:8.4f}")

        bic_psnr_list.append(b_p); src_psnr_list.append(s_p)
        bic_ssim_list.append(b_s); src_ssim_list.append(s_s)
        bic_psnr_y_list.append(b_py); src_psnr_y_list.append(s_py)
        bic_ssim_y_list.append(b_sy); src_ssim_y_list.append(s_sy)

        results.append({
            "file": fname,
            "bicubic_psnr_rgb": b_p, "srcnn_psnr_rgb": s_p,
            "bicubic_ssim_rgb": b_s, "srcnn_ssim_rgb": s_s,
            "bicubic_psnr_y": b_py, "srcnn_psnr_y": s_py,
            "bicubic_ssim_y": b_sy, "srcnn_ssim_y": s_sy,
        })

    n = len(results)
    avg_bic_p, avg_src_p = np.mean(bic_psnr_list), np.mean(src_psnr_list)
    avg_bic_s, avg_src_s = np.mean(bic_ssim_list), np.mean(src_ssim_list)
    avg_bic_py, avg_src_py = np.mean(bic_psnr_y_list), np.mean(src_psnr_y_list)
    avg_bic_sy, avg_src_sy = np.mean(bic_ssim_y_list), np.mean(src_ssim_y_list)

    print("=" * 80)
    print("AVERAGE SUMMARY:")
    print(f"  RGB PSNR     : Bicubic = {avg_bic_p:.3f} dB | SRCNN = {avg_src_p:.3f} dB | Gain = {avg_src_p - avg_bic_p:+.3f} dB")
    print(f"  RGB SSIM     : Bicubic = {avg_bic_s:.4f}    | SRCNN = {avg_src_s:.4f}    | Gain = {avg_src_s - avg_bic_s:+.4f}")
    print(f"  Y-Channel PSNR (shave={a.shave}): Bicubic = {avg_bic_py:.3f} dB | SRCNN = {avg_src_py:.3f} dB | Gain = {avg_src_py - avg_bic_py:+.3f} dB")
    print(f"  Y-Channel SSIM (shave={a.shave}): Bicubic = {avg_bic_sy:.4f}    | SRCNN = {avg_src_sy:.4f}    | Gain = {avg_src_sy - avg_bic_sy:+.4f}")
    print("=" * 80)

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        summary = {
            "dataset": a.dataset,
            "num_images": n,
            "scale": a.scale,
            "model": a.model,
            "averages": {
                "bicubic_psnr_rgb": float(avg_bic_p), "srcnn_psnr_rgb": float(avg_src_p),
                "bicubic_ssim_rgb": float(avg_bic_s), "srcnn_ssim_rgb": float(avg_src_s),
                "bicubic_psnr_y": float(avg_bic_py), "srcnn_psnr_y": float(avg_src_py),
                "bicubic_ssim_y": float(avg_bic_sy), "srcnn_ssim_y": float(avg_src_sy),
            },
            "per_image": results
        }
        with open(a.out, "w") as fp:
            json.dump(summary, fp, indent=2)
        print(f"Saved evaluation summary to: {a.out}")


if __name__ == "__main__":
    main()
