"""Numerical gradient check of the hand-written backprop.

Can be run directly:
    python tests/test_gradients.py
Or via pytest:
    pytest tests/test_gradients.py
"""
import os
import sys
import numpy as np

# Ensure repository root is on path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import srcnn.core as core
core.F32 = np.float64  # float64 so finite differences are noise-free
from srcnn.core import SRCNN


def check(use_bn, use_tanh, residual=False):
    """Compute worst relative error between analytic backward pass and finite differences."""
    rng = np.random.default_rng(1)
    m = SRCNN(use_bn=use_bn, use_tanh=use_tanh, residual=residual, f=(6, 4), seed=3)
    for w in m.W:  # bigger weights -> larger, easier-to-check signal
        w *= 3
    x = rng.random((2, 12, 12, 3)).astype(np.float64)
    y = rng.random((2, 12, 12, 3)).astype(np.float64)

    def loss():
        return float(((m.forward(x, train=True) - y) ** 2).mean(dtype=np.float64))

    out = m.forward(x, train=True)
    grads = m.backward(2 * (out - y) / out.size)
    worst = 0.0
    for p, g in zip(m.params(), grads):
        for _ in range(5):
            idx = tuple(int(rng.integers(0, s)) for s in p.shape)
            old = p[idx]
            eps = 1e-5
            p[idx] = old + eps
            lp = loss()
            p[idx] = old - eps
            lm = loss()
            p[idx] = old
            num = (lp - lm) / (2 * eps)
            if abs(num) < 1e-9 and abs(g[idx]) < 1e-9:
                continue
            worst = max(worst, abs(num - g[idx]) / (abs(num) + abs(g[idx]) + 1e-8))
    return worst


# Pytest test cases
def test_gradients_vanilla():
    err = check(use_bn=False, use_tanh=False, residual=False)
    assert err < 1e-4, f"Vanilla SRCNN gradient error too high: {err}"


def test_gradients_tanh():
    err = check(use_bn=False, use_tanh=True, residual=False)
    assert err < 1e-4, f"Tanh SRCNN gradient error too high: {err}"


def test_gradients_bn():
    err = check(use_bn=True, use_tanh=False, residual=False)
    assert err < 1e-4, f"BatchNorm SRCNN gradient error too high: {err}"


def test_gradients_residual():
    err = check(use_bn=False, use_tanh=False, residual=True)
    assert err < 1e-4, f"Residual SRCNN gradient error too high: {err}"


if __name__ == "__main__":
    ok = True
    for bn, th, rs in [(False, False, False), (False, True, False), (True, False, False), (False, False, True)]:
        w = check(bn, th, rs)
        flag = w < 1e-4
        ok &= flag
        print(f"bn={bn!s:5} tanh={th!s:5} residual={rs!s:5} worst relative error = {w:.2e}  {'PASS' if flag else 'FAIL'}")
    sys.exit(0 if ok else 1)
