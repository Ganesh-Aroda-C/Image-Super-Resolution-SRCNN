"""Training and evaluation loop for pure-NumPy SRCNN."""
import time
from typing import Callable, Dict, Tuple, Any
import numpy as np
from .core import SRCNN, Adam
from .metrics import psnr, compute_metrics


def random_patches(
    X: np.ndarray,
    Y: np.ndarray,
    n: int,
    size: int,
    rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample n random aligned (input, target) patches with random horizontal/vertical flips."""
    idx = rng.integers(0, len(X), n)
    xs = np.empty((n, size, size, 3), np.float32)
    ys = np.empty_like(xs)
    h_max = X.shape[1] - size + 1
    w_max = X.shape[2] - size + 1
    for j, i in enumerate(idx):
        r = int(rng.integers(0, h_max))
        c = int(rng.integers(0, w_max))
        a, b = X[i, r:r + size, c:c + size], Y[i, r:r + size, c:c + size]
        if rng.random() < 0.5:
            a, b = a[:, ::-1], b[:, ::-1]  # Horizontal flip
        if rng.random() < 0.5:
            a, b = a[::-1], b[::-1]        # Vertical flip
        xs[j], ys[j] = a, b
    return xs, ys


def evaluate(
    model: SRCNN,
    X: np.ndarray,
    Y: np.ndarray
) -> Tuple[float, float, np.ndarray]:
    """Evaluate model on a dataset split, returning (MSE_loss, mean_PSNR_dB, predictions)."""
    pred = model.predict(X)
    p = psnr(pred, Y)
    mse = float(((pred - Y) ** 2).mean())
    return mse, float(np.mean(p)), pred


def train(
    Xtr: np.ndarray,
    Ytr: np.ndarray,
    Xva: np.ndarray,
    Yva: np.ndarray,
    lr: float = 1e-3,
    epochs: int = 30,
    batch: int = 16,
    patch: int = 48,
    patches_per_epoch: int = 160,
    use_bn: bool = False,
    use_tanh: bool = False,
    residual: bool = False,
    wd: float = 0.0,
    seed: int = 0,
    eval_every: int = 1,
    log: Callable[[str], None] = print,
    tag: str = "run"
) -> Tuple[SRCNN, Dict[str, Any]]:
    """Train SRCNN on random patches (fully-convolutional, so evaluation uses full images).
    
    Returns (best_model, history), where best_model has weights from the epoch with highest validation PSNR.
    """
    rng = np.random.default_rng(seed)
    model = SRCNN(use_bn=use_bn, use_tanh=use_tanh, residual=residual, seed=seed)
    opt = Adam(model.params(), lr=lr, wd=wd, decay_mask=model.decay_mask())
    hist = {
        "epoch": [],
        "train_loss": [],
        "train_psnr": [],
        "val_loss": [],
        "val_psnr": [],
        "best_epoch": 0,
        "best_val_psnr": -1.0
    }
    
    # Initialize best_state with initial weights to guarantee safe restoration
    best_state = (
        [w.copy() for w in model.W],
        [b.copy() for b in model.b],
        [(bn.gamma.copy(), bn.beta.copy(), bn.rm.copy(), bn.rv.copy()) for bn in model.bn]
    )
    best_psnr = -1.0
    t0 = time.time()

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
            tl = float(np.mean(losses)) if losses else 0.0
            t_psnr = 10.0 * np.log10(1.0 / max(tl, 1e-12))
            
            hist["epoch"].append(ep)
            hist["train_loss"].append(tl)
            hist["train_psnr"].append(t_psnr)
            hist["val_loss"].append(vl)
            hist["val_psnr"].append(vp)
            
            if vp > best_psnr:
                best_psnr = vp
                hist["best_epoch"] = ep
                best_state = (
                    [w.copy() for w in model.W],
                    [b.copy() for b in model.b],
                    [(bn.gamma.copy(), bn.beta.copy(), bn.rm.copy(), bn.rv.copy()) for bn in model.bn]
                )
            log(f"[{tag}] epoch {ep:3d}/{epochs}  train_loss {tl:.5f}  val_loss {vl:.5f}  val_PSNR {vp:6.3f} dB  ({time.time()-t0:.0f}s)")

    # Restore best validation checkpoint weights
    W, b, bns = best_state
    model.W = W
    model.b = b
    for bn, (g, be, rm, rv) in zip(model.bn, bns):
        bn.gamma, bn.beta, bn.rm, bn.rv = g, be, rm, rv
        
    hist["best_val_psnr"] = best_psnr
    return model, hist
