"""Optional: build the same 100-image / 60-20-20 dataset from the REAL DIV2K images.

1. Download DIV2K_train_HR.zip (3.5 GB) from https://data.vision.ee.ethz.ch/cvl/DIV2K/ and unzip it.
2. python scripts/prepare_div2k.py --src path/to/DIV2K_train_HR
   (centre-crops each image to 800x800, resizes to 224x224, samples 100 images, splits 60/20/20,
    and overwrites ./data/{train,val,test}).
"""
import argparse, glob, os, shutil
import numpy as np
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True, help="folder containing DIV2K *.png images")
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--seed", type=int, default=229)
a = ap.parse_args()

files = sorted(glob.glob(os.path.join(a.src, "*.png")))
if len(files) < a.n:
    raise SystemExit(f"Found only {len(files)} PNGs in {a.src}")
rng = np.random.default_rng(a.seed)
files = [files[i] for i in rng.permutation(len(files))[:a.n]]
cuts = {"train": files[:int(.6 * a.n)], "val": files[int(.6 * a.n):int(.8 * a.n)], "test": files[int(.8 * a.n):]}
for split, fl in cuts.items():
    d = os.path.join(a.out, split)
    shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    for i, f in enumerate(fl):
        im = Image.open(f).convert("RGB"); w, h = im.size
        s = min(800, w, h); x0, y0 = (w - s) // 2, (h - s) // 2
        im.crop((x0, y0, x0 + s, y0 + s)).resize((224, 224), Image.LANCZOS).save(os.path.join(d, f"{split}_{i:03d}_{os.path.basename(f)}"))
    print(split, len(fl))
