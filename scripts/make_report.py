"""Turn results/*.json into tables (Markdown + CSV) and figures (PNG).   python scripts/make_report.py"""
import argparse, json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import _path
from _path import ROOT
from srcnn import SRCNN, load_split, psnr

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=os.path.join(ROOT, "data"))
ap.add_argument("--res", default=os.path.join(ROOT, "results"))
a = ap.parse_args()
R = lambda n: json.load(open(os.path.join(a.res, n + ".json")))
base = R("baselines")

# ---------------- table -------------------------------------------------------------------------
rows = []
def add(label, run):
    r = R(run); rows.append((label, r["history"]["best_val_psnr"], r["test_loss"], r["test_psnr"]))
add("Basic SRCNN (lr 0.001)", "lr_0.001_no_bn")
add("+ Regularization (wd 1e-4)", "lr_0.001_wd_0.0001")
add("+ Tanh", "tanh_lr_0.001")
add("+ Batch Norm", "bn_lr_0.001")
add("Residual SRCNN (refinement)", "residual_lr_0.001")
for m in ("bicubic", "bilinear", "nearest"):
    rows.append((m.capitalize(), base[m]["val_psnr"], base[m]["test_loss"], base[m]["test_psnr"]))
with open(os.path.join(a.res, "results_table.md"), "w") as f:
    f.write("| Experiment | Max Val PSNR (dB) | Test Loss (MSE) | Test PSNR (dB) |\n|---|---|---|---|\n")
    for l, v, tl, tp in rows: f.write(f"| {l} | {v:.3f} | {tl:.6f} | {tp:.3f} |\n")
with open(os.path.join(a.res, "results_table.csv"), "w") as f:
    f.write("experiment,max_val_psnr,test_loss,test_psnr\n")
    for l, v, tl, tp in rows: f.write(f'"{l}",{v:.4f},{tl:.7f},{tp:.4f}\n')
print(open(os.path.join(a.res, "results_table.md")).read())

# ---------------- curves ------------------------------------------------------------------------
def curves(names, fname, title):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for n in names:
        h = R(n)["history"]
        ax[0].plot(h["epoch"], h["val_loss"], label=n); ax[1].plot(h["epoch"], h["val_psnr"], label=n)
    for m, c in (("nearest", "r"), ("bilinear", "m"), ("bicubic", "k")):
        ax[0].axhline(base[m]["val_loss"], color=c, ls="--", lw=.8, label=m)
        ax[1].axhline(base[m]["val_psnr"], color=c, ls="--", lw=.8, label=m)
    ax[0].set_ylabel("Average validation loss (MSE)"); ax[1].set_ylabel("Average validation PSNR (dB)")
    ax[0].set_ylim(0, min(0.02, ax[0].get_ylim()[1])); ax[1].set_ylim(max(15, ax[1].get_ylim()[0]), None)
    for x in ax: x.set_xlabel("Epoch"); x.grid(alpha=.3)
    ax[1].legend(fontsize=7, loc="lower right"); fig.suptitle(title); fig.tight_layout()
    fig.savefig(os.path.join(a.res, fname), dpi=130); plt.close(fig)
curves(["lr_0.01_no_bn", "lr_0.001_no_bn", "lr_0.0001_no_bn"], "fig_learning_rate.png", "Learning-rate comparison")
curves(["lr_0.001_no_bn", "residual_lr_0.001"], "fig_residual.png", "Residual refinement vs basic SRCNN")
curves(["lr_0.001_no_bn", "tanh_lr_0.001"], "fig_tanh.png", "Effect of tanh output activation")
curves(["lr_0.001_no_bn", "bn_lr_0.001", "bn_lr_0.01"], "fig_batchnorm.png", "Effect of batch normalisation")
curves(["lr_0.001_no_bn"] + [f"lr_0.001_wd_{w}" for w in ("0.01", "0.001", "0.0001", "0.00001")], "fig_regularization.png", "Effect of L2 regularisation")

# ---------------- qualitative samples -------------------------------------------------------------
Xte, Yte, files = load_split(a.data, "test")
model = SRCNN.load(os.path.join(a.res, "lr_0.001_no_bn.npz")); rmodel = SRCNN.load(os.path.join(a.res, "residual_lr_0.001.npz"))
pred, rpred = model.predict(Xte), rmodel.predict(Xte)
pb, ps, pr = psnr(Xte, Yte), psnr(pred, Yte), psnr(rpred, Yte)
gain = ps - pb
pick = [0, 6, 12, 18][:len(Xte)]
fig, ax = plt.subplots(len(pick), 4, figsize=(12, 3 * len(pick)))
for r, i in enumerate(pick):
    for c, (im, t) in enumerate(((Xte[i], f"Input (bicubic) {pb[i]:.2f} dB"), (pred[i], f"SRCNN {ps[i]:.2f} dB"),
                                 (rpred[i], f"Residual SRCNN {pr[i]:.2f} dB"), (Yte[i], "Target (HR)"))):
        ax[r, c].imshow(im); ax[r, c].set_title(t, fontsize=9); ax[r, c].axis("off")
fig.tight_layout(); fig.savefig(os.path.join(a.res, "fig_samples.png"), dpi=110); plt.close(fig)
print(f"SRCNN improved PSNR on {(ps > pb).sum()}/{len(ps)} test images; mean gain {gain.mean():+.3f} dB")
print(f"Residual SRCNN improved PSNR on {(pr > pb).sum()}/{len(pr)} test images; mean gain {(pr-pb).mean():+.3f} dB")
json.dump({"mean_gain_db": float(gain.mean()), "n_improved": int((ps > pb).sum()), "n_test": len(ps),
           "res_mean_gain_db": float((pr - pb).mean()), "res_n_improved": int((pr > pb).sum())},
          open(os.path.join(a.res, "summary.json"), "w"))
