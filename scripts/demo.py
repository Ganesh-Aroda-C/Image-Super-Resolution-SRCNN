"""Super-resolve any image with a trained SRCNN model (NumPy or PyTorch).

Usage examples:
  # Standard evaluation (simulates degradation -> bicubic baseline -> SRCNN reconstruction)
  python scripts/demo.py --input data/test/test_000_retina.png

  # Real-world image enlargement (no reference ground-truth)
  python scripts/demo.py --input photo.jpg --no_degrade

  # Generate detail zoom crops and select a specific checkpoint
  python scripts/demo.py --input data/test/test_000_retina.png --crop --model results/residual_lr_0.001.npz
"""
import argparse
import os
import sys
import time
import numpy as np
from PIL import Image

import _path
from _path import ROOT
from srcnn import SRCNN, SRCNN_PyTorch, psnr, ssim, compute_metrics, make_lr, validate_image_path


def parse_args():
    ap = argparse.ArgumentParser(description="SRCNN Image Super-Resolution Demo")
    ap.add_argument("--input", required=True, help="Path to input image file (PNG/JPG)")
    ap.add_argument("--model", default=os.path.join(ROOT, "results", "lr_0.001_no_bn.npz"),
                    help="Path to trained checkpoint (.npz or .pt)")
    ap.add_argument("--out", default=os.path.join(ROOT, "results"), help="Directory to save outputs")
    ap.add_argument("--scale", type=float, default=2.0, help="Super-resolution scale factor")
    ap.add_argument("--no_degrade", action="store_true",
                    help="Treat input as already low-resolution (no ground truth available)")
    ap.add_argument("--crop", action="store_true", help="Generate an extra magnified crop comparison")
    ap.add_argument("--max_side", type=int, default=600,
                    help="Downscale excessively large inputs to prevent slow CPU convolution")
    return ap.parse_args()


def load_model(model_path: str):
    """Load model from either .npz (NumPy) or .pt (PyTorch) checkpoint with clear errors."""
    if not os.path.exists(model_path):
        print(f"\n[ERROR] Checkpoint not found: '{model_path}'", file=sys.stderr)
        print("Available trained models in results/:", file=sys.stderr)
        if os.path.exists(os.path.join(ROOT, "results")):
            for f in sorted(os.listdir(os.path.join(ROOT, "results"))):
                if f.endswith((".npz", ".pt")):
                    print(f"  - results/{f}", file=sys.stderr)
        print("\nTo train a new model, run: python scripts/train.py --epochs 30 --lr 0.001", file=sys.stderr)
        sys.exit(1)

    if model_path.endswith(".pt"):
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, _ = SRCNN_PyTorch.load_checkpoint(model_path, device=device)
        print(f"Loaded PyTorch model from: {model_path} (device: {device})")
        return model, "pytorch"
    else:
        model = SRCNN.load(model_path)
        print(f"Loaded NumPy model from: {model_path}")
        return model, "numpy"


def make_crop_panel(img: np.ndarray, box: tuple, zoom: int = 3) -> np.ndarray:
    """Extract a patch from an image and magnify it using nearest neighbor to visualize pixels."""
    x0, y0, x1, y1 = box
    patch = img[y0:y1, x0:x1]
    pil_patch = Image.fromarray((patch * 255.0).round().astype(np.uint8))
    zoomed = pil_patch.resize((patch.shape[1] * zoom, patch.shape[0] * zoom), Image.NEAREST)
    return np.asarray(zoomed, np.float32) / 255.0


def main():
    a = parse_args()
    validate_image_path(a.input)
    os.makedirs(a.out, exist_ok=True)
    
    model, backend = load_model(a.model)
    img_raw = Image.open(a.input).convert("RGB")
    
    # Restrict maximum dimension for fast CPU inference if requested
    if max(img_raw.size) > a.max_side:
        factor = a.max_side / max(img_raw.size)
        new_w = max(1, int(round(img_raw.width * factor)))
        new_h = max(1, int(round(img_raw.height * factor)))
        img_raw = img_raw.resize((new_w, new_h), Image.LANCZOS)
        print(f"Resized large input to {img_raw.size} for fast inference.")

    base_name = os.path.splitext(os.path.basename(a.input))[0]

    if a.no_degrade:
        # User supplied a low-resolution image to 2x super-resolve
        w, h = img_raw.size
        target_w = int(round(w * a.scale))
        target_h = int(round(h * a.scale))
        bicubic_pil = img_raw.resize((target_w, target_h), Image.BICUBIC)
        
        x = np.asarray(bicubic_pil, np.float32)[np.newaxis, ...] / 255.0
        t0 = time.time()
        y = model.predict(x)[0]
        infer_time = time.time() - t0
        
        print(f"\nSuper-resolved image from {img_raw.size} to {(target_w, target_h)} in {infer_time:.2f}s")
        print("Note: No reference ground-truth image available; PSNR/SSIM cannot be computed.")
        
        # Save reconstructed output and side-by-side comparison
        out_srcnn_path = os.path.join(a.out, f"{base_name}_srcnn.png")
        out_comp_path = os.path.join(a.out, f"{base_name}_comparison.png")
        
        Image.fromarray((np.clip(y, 0, 1) * 255).round().astype(np.uint8)).save(out_srcnn_path)
        
        comp_array = np.concatenate([np.asarray(bicubic_pil) / 255.0, y], axis=1)
        Image.fromarray((np.clip(comp_array, 0, 1) * 255).round().astype(np.uint8)).save(out_comp_path)
        print(f"Saved reconstructed image : {out_srcnn_path}")
        print(f"Saved side-by-side [Bicubic | SRCNN]: {out_comp_path}")
    else:
        # Standard super-resolution evaluation with ground truth
        # Adjust dimensions to be cleanly divisible by scale
        w = int(img_raw.width // a.scale * a.scale)
        h = int(img_raw.height // a.scale * a.scale)
        hr_pil = img_raw.crop((0, 0, w, h))
        
        lr_w = int(w // a.scale)
        lr_h = int(h // a.scale)
        lr_pil = hr_pil.resize((lr_w, lr_h), Image.BILINEAR)
        bicubic_pil = lr_pil.resize((w, h), Image.BICUBIC)
        
        x = np.asarray(bicubic_pil, np.float32)[np.newaxis, ...] / 255.0
        hr_np = np.asarray(hr_pil, np.float32)[np.newaxis, ...] / 255.0
        
        t0 = time.time()
        pred = model.predict(x)
        infer_time = time.time() - t0
        
        # Calculate metrics
        bicubic_psnr = float(psnr(x, hr_np)[0])
        srcnn_psnr = float(psnr(pred, hr_np)[0])
        
        bicubic_ssim = float(ssim(x, hr_np)[0])
        srcnn_ssim = float(ssim(pred, hr_np)[0])
        
        shave = int(round(a.scale))
        b_py, b_sy = compute_metrics(x[0], hr_np[0], shave=shave, use_y_channel=True)
        s_py, s_sy = compute_metrics(pred[0], hr_np[0], shave=shave, use_y_channel=True)
        
        print("\n" + "=" * 55)
        print(f"SUPER-RESOLUTION RESULTS: {base_name}")
        print(f"Image Resolution : {w}x{h} (downsampled to {lr_w}x{lr_h}, scale={a.scale}x)")
        print(f"Inference Time   : {infer_time:.3f}s ({backend.upper()})")
        print("-" * 55)
        print(f"RGB Metrics:")
        print(f"  Bicubic PSNR   : {bicubic_psnr:6.3f} dB   | SSIM: {bicubic_ssim:.4f}")
        print(f"  SRCNN PSNR     : {srcnn_psnr:6.3f} dB   | SSIM: {srcnn_ssim:.4f}")
        gain_p = srcnn_psnr - bicubic_psnr
        gain_s = srcnn_ssim - bicubic_ssim
        print(f"  Gain           : {gain_p:+6.3f} dB   | SSIM: {gain_s:+.4f}")
        print("-" * 55)
        print(f"Y-Channel (Luminance) Metrics (shave={shave}px):")
        print(f"  Bicubic PSNR-Y : {b_py:6.3f} dB   | SSIM-Y: {b_sy:.4f}")
        print(f"  SRCNN PSNR-Y   : {s_py:6.3f} dB   | SSIM-Y: {s_sy:.4f}")
        print(f"  Gain-Y         : {s_py - b_py:+6.3f} dB   | SSIM-Y: {s_sy - b_sy:+.4f}")
        print("=" * 55)

        # Save images
        out_srcnn_path = os.path.join(a.out, f"{base_name}_srcnn.png")
        out_comp_path = os.path.join(a.out, f"{base_name}_comparison.png")
        
        y = np.clip(pred[0], 0.0, 1.0)
        Image.fromarray((y * 255.0).round().astype(np.uint8)).save(out_srcnn_path)
        
        # 3-panel comparison: [Bicubic | SRCNN | Ground Truth]
        panels = [x[0], y, hr_np[0]]
        comp_array = np.concatenate(panels, axis=1)
        Image.fromarray((np.clip(comp_array, 0, 1) * 255.0).round().astype(np.uint8)).save(out_comp_path)
        
        print(f"Saved reconstructed image : {out_srcnn_path}")
        print(f"Saved side-by-side [Bicubic | SRCNN | Ground Truth]: {out_comp_path}")
        
        if a.crop:
            # Crop a central 48x48 region and magnify 3x for fine detail comparison
            cw, ch = min(48, w // 2), min(48, h // 2)
            cx, cy = (w - cw) // 2, (h - ch) // 2
            box = (cx, cy, cx + cw, cy + ch)
            
            crop_bic = make_crop_panel(x[0], box, zoom=3)
            crop_sr = make_crop_panel(y, box, zoom=3)
            crop_gt = make_crop_panel(hr_np[0], box, zoom=3)
            
            crop_comp = np.concatenate([crop_bic, crop_sr, crop_gt], axis=1)
            out_crop_path = os.path.join(a.out, f"{base_name}_crop_comparison.png")
            Image.fromarray((crop_comp * 255.0).round().astype(np.uint8)).save(out_crop_path)
            print(f"Saved magnified detail crop: {out_crop_path}")


if __name__ == "__main__":
    main()
