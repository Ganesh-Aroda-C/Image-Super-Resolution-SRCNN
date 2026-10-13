"""Build the bundled 100-image dataset (60 train / 20 val / 20 test) of 224x224 high-resolution PNGs.

No internet is needed: the source photos ship with scikit-image (`skimage.data`).  Random crops of each
source photo are centre-resized to 224x224 (downscale only, so targets stay sharp).  The split is done
BY SOURCE PHOTO, so val/test images never share content with training images.

To use the real DIV2K dataset instead, see scripts/prepare_div2k.py.
"""
import argparse, os, sys
import numpy as np
from PIL import Image
import skimage.data as sd

SPLITS = {   # source photo -> split
    "train": ["astronaut", "coffee", "rocket", "hubble_deep_field", "brick", "gravel", "grass", "moon"],
    "val":   ["chelsea", "immunohistochemistry"],
    "test":  ["retina", "camera", "cell"],
}
COUNTS = {"train": 60, "val": 20, "test": 20}
SIZE = 224


def load(name):
    im = getattr(sd, name)()
    if im.ndim == 2:
        im = np.stack([im] * 3, -1)
    return Image.fromarray(im[..., :3].astype(np.uint8))


def random_crop(img, rng):
    w, h = img.size
    m = min(w, h)
    side = int(rng.uniform(max(SIZE, 0.45 * m), m))             # never smaller than 224 -> no upscaling
    x0 = int(rng.integers(0, w - side + 1)); y0 = int(rng.integers(0, h - side + 1))
    c = img.crop((x0, y0, x0 + side, y0 + side))
    if rng.random() < 0.5:
        c = c.transpose(Image.FLIP_LEFT_RIGHT)
    return c.resize((SIZE, SIZE), Image.LANCZOS)


def main(out, seed):
    rng = np.random.default_rng(seed)
    for split, names in SPLITS.items():
        d = os.path.join(out, split); os.makedirs(d, exist_ok=True)
        for f in os.listdir(d):
            if f.endswith(".png"): os.remove(os.path.join(d, f))
        n, k = COUNTS[split], len(names)
        per = [n // k + (1 if i < n % k else 0) for i in range(k)]
        idx = 0
        for name, cnt in zip(names, per):
            src = load(name)
            for _ in range(cnt):
                random_crop(src, rng).save(os.path.join(d, f"{split}_{idx:03d}_{name}.png"))
                idx += 1
        print(f"{split}: {idx} images from {names}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
    ap.add_argument("--seed", type=int, default=229)
    a = ap.parse_args()
    main(a.out, a.seed)
