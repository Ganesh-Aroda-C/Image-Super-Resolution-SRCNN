"""Train a single SRCNN and report test PSNR.   python scripts/train.py --epochs 30 --lr 0.001"""
import argparse, json, os
import _path
from _path import ROOT
from srcnn import load_split
from srcnn.trainer import train, evaluate

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=os.path.join(ROOT, "data"))
ap.add_argument("--out", default=os.path.join(ROOT, "results"))
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--patch", type=int, default=48)
ap.add_argument("--patches_per_epoch", type=int, default=160)
ap.add_argument("--wd", type=float, default=0.0, help="L2 regularisation factor")
ap.add_argument("--bn", action="store_true", help="add batch-norm after conv1/conv2")
ap.add_argument("--tanh", action="store_true", help="tanh on the output layer")
ap.add_argument("--residual", action="store_true", help="refinement: predict a residual added to the bicubic input")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--name", default="srcnn")
a = ap.parse_args()

Xtr, Ytr, _ = load_split(a.data, "train"); Xva, Yva, _ = load_split(a.data, "val"); Xte, Yte, _ = load_split(a.data, "test")
print(f"train {Xtr.shape}  val {Xva.shape}  test {Xte.shape}")
model, hist = train(Xtr, Ytr, Xva, Yva, lr=a.lr, epochs=a.epochs, batch=a.batch, patch=a.patch,
                    patches_per_epoch=a.patches_per_epoch, use_bn=a.bn, use_tanh=a.tanh, residual=a.residual, wd=a.wd, seed=a.seed, tag=a.name)
tl, tp, _ = evaluate(model, Xte, Yte)
print(f"\nTEST  loss {tl:.6f}   PSNR {tp:.3f} dB")
os.makedirs(a.out, exist_ok=True)
model.save(os.path.join(a.out, f"{a.name}.npz"))
json.dump({"history": hist, "test_loss": tl, "test_psnr": tp, "args": vars(a)}, open(os.path.join(a.out, f"{a.name}.json"), "w"), indent=1)
print("saved", os.path.join(a.out, f"{a.name}.npz"))
