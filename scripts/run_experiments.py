"""Run every experiment of the study (baselines, lr grid, tanh, batch-norm, regularisation).
Resumable: finished runs (results/<name>.json) are skipped.
    python scripts/run_experiments.py --epochs 40        (~40 min on one CPU core)
    python scripts/run_experiments.py --quick            (~3 min smoke version)
"""
import argparse, json, os, time
import numpy as np
import _path
from _path import ROOT
from srcnn import load_split, psnr
from srcnn.trainer import train, evaluate

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=os.path.join(ROOT, "data"))
ap.add_argument("--out", default=os.path.join(ROOT, "results"))
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--patches_per_epoch", type=int, default=320)
ap.add_argument("--eval_every", type=int, default=2)
ap.add_argument("--quick", action="store_true")
a = ap.parse_args()
if a.quick:
    a.epochs, a.patches_per_epoch, a.eval_every, a.out = 4, 64, 1, os.path.join(ROOT, "results_quick")
os.makedirs(a.out, exist_ok=True)

# ---- baselines (non-neural interpolation of the 112x112 image to 224x224) ----------------------
base_path = os.path.join(a.out, "baselines.json")
if not os.path.exists(base_path):
    res = {}
    for mode in ("nearest", "bilinear", "bicubic"):
        r = {}
        for split in ("val", "test"):
            X, Y, _ = load_split(a.data, split, up=mode)
            r[split + "_loss"] = float(((X - Y) ** 2).mean()); r[split + "_psnr"] = float(psnr(X, Y).mean())
        res[mode] = r
        print(f"baseline {mode:9s} val PSNR {r['val_psnr']:.3f}  test PSNR {r['test_psnr']:.3f}")
    json.dump(res, open(base_path, "w"), indent=1)

Xtr, Ytr, _ = load_split(a.data, "train"); Xva, Yva, _ = load_split(a.data, "val"); Xte, Yte, _ = load_split(a.data, "test")

RUNS = [  # name, kwargs
    ("lr_0.001_no_bn", dict(lr=1e-3)),                       # <- "Basic SRCNN"
    ("lr_0.01_no_bn", dict(lr=1e-2)),
    ("lr_0.0001_no_bn", dict(lr=1e-4)),
    ("tanh_lr_0.001", dict(lr=1e-3, use_tanh=True)),
    ("bn_lr_0.001", dict(lr=1e-3, use_bn=True)),
    ("bn_lr_0.01", dict(lr=1e-2, use_bn=True)),
    ("residual_lr_0.001", dict(lr=1e-3, residual=True)),      # refinement beyond the paper
    ("lr_0.001_wd_0.01", dict(lr=1e-3, wd=1e-2)),
    ("lr_0.001_wd_0.001", dict(lr=1e-3, wd=1e-3)),
    ("lr_0.001_wd_0.0001", dict(lr=1e-3, wd=1e-4)),
    ("lr_0.001_wd_0.00001", dict(lr=1e-3, wd=1e-5)),
]
for name, kw in RUNS:
    jp = os.path.join(a.out, name + ".json")
    if os.path.exists(jp):
        print("skip", name); continue
    t0 = time.time()
    model, hist = train(Xtr, Ytr, Xva, Yva, epochs=a.epochs, patches_per_epoch=a.patches_per_epoch,
                        eval_every=a.eval_every, tag=name, **kw)
    tl, tp, _ = evaluate(model, Xte, Yte)
    model.save(os.path.join(a.out, name + ".npz"))
    json.dump({"history": hist, "test_loss": tl, "test_psnr": tp, "kwargs": kw, "seconds": time.time() - t0},
              open(jp, "w"), indent=1)
    print(f"==> {name}: best val {hist['best_val_psnr']:.3f}  TEST loss {tl:.5f} PSNR {tp:.3f}  ({time.time()-t0:.0f}s)", flush=True)
print("ALL DONE")
