"""Super-resolve any image with a trained model (live-demo script).

  python scripts/demo.py --input some_photo.jpg            # simulates degradation (half-res -> bicubic), then restores
  python scripts/demo.py --input low_res.png --no_degrade  # treats the input as an already low-res image and 2x-enlarges it
Writes <name>_comparison.png with  [bicubic | SRCNN | original].
"""
import argparse, os
import numpy as np
from PIL import Image
import _path
from _path import ROOT
from srcnn import SRCNN, psnr

ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True)
ap.add_argument("--model", default=os.path.join(ROOT, "results", "lr_0.001_no_bn.npz"))
ap.add_argument("--out", default=os.path.join(ROOT, "results"))
ap.add_argument("--no_degrade", action="store_true")
ap.add_argument("--max_side", type=int, default=448, help="downscale huge inputs to keep CPU time small")
a = ap.parse_args()

model = SRCNN.load(a.model)
img = Image.open(a.input).convert("RGB")
if max(img.size) > a.max_side:
    s = a.max_side / max(img.size); img = img.resize((int(img.width * s), int(img.height * s)), Image.LANCZOS)
name = os.path.splitext(os.path.basename(a.input))[0]
os.makedirs(a.out, exist_ok=True)
if a.no_degrade:
    big = img.resize((img.width * 2, img.height * 2), Image.BICUBIC)
    x = np.asarray(big, np.float32)[None] / 255
    y = model.predict(x)[0]
    panels = [np.asarray(big) / 255., y]
    print("2x super-resolved (no reference image, so no PSNR).")
else:
    w, h = img.width // 2 * 2, img.height // 2 * 2
    hr = img.crop((0, 0, w, h))
    lr = hr.resize((w // 2, h // 2), Image.BILINEAR).resize((w, h), Image.BICUBIC)
    x = np.asarray(lr, np.float32)[None] / 255; t = np.asarray(hr, np.float32)[None] / 255
    y = model.predict(x)
    print(f"PSNR  bicubic: {psnr(x, t)[0]:.3f} dB   SRCNN: {psnr(y, t)[0]:.3f} dB")
    panels = [x[0], y[0], t[0]]; y = y[0]
Image.fromarray((np.concatenate(panels, 1) * 255).round().astype(np.uint8)).save(os.path.join(a.out, f"{name}_comparison.png"))
Image.fromarray((np.clip(y, 0, 1) * 255).round().astype(np.uint8)).save(os.path.join(a.out, f"{name}_srcnn.png"))
print("saved to", a.out)
