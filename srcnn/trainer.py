"""Training / evaluation loop."""
import time
import numpy as np
from .core import SRCNN, Adam
from .data import psnr


def random_patches(X, Y, n, size, rng):
    idx = rng.integers(0, len(X), n)
    xs = np.empty((n, size, size, 3), np.float32); ys = np.empty_like(xs)
    for j, i in enumerate(idx):
        r = int(rng.integers(0, X.shape[1] - size + 1)); c = int(rng.integers(0, X.shape[2] - size + 1))
        a, b = X[i, r:r + size, c:c + size], Y[i, r:r + size, c:c + size]
        if rng.random() < .5: a, b = a[:, ::-1], b[:, ::-1]
        if rng.random() < .5: a, b = a[::-1], b[::-1]
        xs[j], ys[j] = a, b
    return xs, ys


def evaluate(model, X, Y):
    pred = model.predict(X)
    p = psnr(pred, Y)
    mse = float(((pred - Y) ** 2).mean())
    return mse, float(p.mean()), pred


def train(Xtr, Ytr, Xva, Yva, lr=1e-3, epochs=30, batch=16, patch=48, patches_per_epoch=160,
          use_bn=False, use_tanh=False, residual=False, wd=0.0, seed=0, eval_every=1, log=print, tag="run"):
    """Train SRCNN on random patches (fully-convolutional, so evaluation uses full 224x224 images).
    Returns (best_model, history).  `best` = epoch with highest validation PSNR."""
    rng = np.random.default_rng(seed)
    model = SRCNN(use_bn=use_bn, use_tanh=use_tanh, residual=residual, seed=seed)
    opt = Adam(model.params(), lr=lr, wd=wd, decay_mask=model.decay_mask())
    hist = {"epoch": [], "train_loss": [], "train_psnr": [], "val_loss": [], "val_psnr": []}
    best, best_state, t0 = -1.0, None, time.time()
    for ep in range(1, epochs + 1):
        losses = []
        xs, ys = random_patches(Xtr, Ytr, patches_per_epoch, patch, rng)
        for s in range(0, patches_per_epoch, batch):
            xb, yb = xs[s:s + batch], ys[s:s + batch]
            out = model.forward(xb, train=True)
            diff = out - yb
            losses.append(float((diff ** 2).mean()))
            opt.step(model.backward((2.0 / diff.size) * diff))
        if ep % eval_every == 0 or ep == epochs:
            vl, vp, _ = evaluate(model, Xva, Yva)
            tl = float(np.mean(losses))
            hist["epoch"].append(ep); hist["train_loss"].append(tl)
            hist["train_psnr"].append(10 * np.log10(1 / tl)); hist["val_loss"].append(vl); hist["val_psnr"].append(vp)
            if vp > best:
                best = vp
                best_state = ([w.copy() for w in model.W], [b.copy() for b in model.b],
                              [(bn.gamma.copy(), bn.beta.copy(), bn.rm.copy(), bn.rv.copy()) for bn in model.bn])
            log(f"[{tag}] epoch {ep:3d}/{epochs}  train_loss {tl:.5f}  val_loss {vl:.5f}  val_PSNR {vp:6.3f} dB  ({time.time()-t0:.0f}s)")
    W, b, bns = best_state                      # restore best-validation weights
    model.W, model.b = W, b
    for bn, (g, be, rm, rv) in zip(model.bn, bns):
        bn.gamma, bn.beta, bn.rm, bn.rv = g, be, rm, rv
    hist["best_val_psnr"] = best
    return model, hist
