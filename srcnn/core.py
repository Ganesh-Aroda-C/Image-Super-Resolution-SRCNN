"""Pure-NumPy implementation of SRCNN with hand-written forward/backward passes.

Architecture (Dong et al., 2015), "same" padding, NHWC layout:
    conv1: 9x9, 3  -> 64  (+ optional BatchNorm) + ReLU   patch extraction
    conv2: 1x1, 64 -> 32  (+ optional BatchNorm) + ReLU   non-linear mapping
    conv3: 5x5, 32 -> 3   (+ optional tanh output)         reconstruction
Input is the bicubic-upsampled low-resolution image, values in [0, 1].
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

F32 = np.float32


def _cols(x, k, pad):
    """im2col for NHWC input -> (B*H*W, C*k*k)."""
    if pad:
        x = np.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)))
    w = sliding_window_view(x, (k, k), axis=(1, 2))      # B,H,W,C,k,k
    B, H, W = w.shape[:3]
    return np.ascontiguousarray(w).reshape(B * H * W, -1), (B, H, W)


def conv_forward(x, w, b, pad):
    """x: (B,H,W,Cin); w: (Cin,k,k,Cout) ; returns (B,H,W,Cout), cols."""
    cin, k, _, cout = w.shape
    if k == 1:
        B, H, W, _ = x.shape
        cols = x.reshape(-1, cin)
    else:
        cols, (B, H, W) = _cols(x, k, pad)
    out = cols @ w.reshape(-1, cout) + b
    return out.reshape(B, H, W, cout), cols


def conv_backward(dout, cols, w, pad, need_dx=True):
    cin, k, _, cout = w.shape
    B, H, W, _ = dout.shape
    d2 = dout.reshape(-1, cout)
    dw = (cols.T @ d2).reshape(w.shape)
    db = d2.sum(0)
    dx = None
    if need_dx:
        if k == 1:
            dx = (d2 @ w.reshape(cin, cout).T).reshape(B, H, W, cin)
        else:
            # input-gradient = "same" correlation of dout with the flipped, transposed kernel
            wf = np.ascontiguousarray(w[:, ::-1, ::-1, :].transpose(3, 1, 2, 0))  # (Cout,k,k,Cin)
            dx, _ = conv_forward(dout, wf, np.zeros(cin, F32), k - 1 - pad)
    return dx, dw, db


class BatchNorm:
    def __init__(self, c, momentum=0.1, eps=1e-5):
        self.gamma, self.beta = np.ones(c, F32), np.zeros(c, F32)
        self.rm, self.rv = np.zeros(c, F32), np.ones(c, F32)
        self.m, self.eps = momentum, eps

    def forward(self, x, train):
        if train:
            mu = x.mean((0, 1, 2)); var = x.var((0, 1, 2))
            self.rm = (1 - self.m) * self.rm + self.m * mu
            self.rv = (1 - self.m) * self.rv + self.m * var
        else:
            mu, var = self.rm, self.rv
        inv = 1.0 / np.sqrt(var + self.eps)
        xh = (x - mu) * inv
        self.cache = (xh, inv)
        return self.gamma * xh + self.beta

    def backward(self, d):
        xh, inv = self.cache
        n = d.shape[0] * d.shape[1] * d.shape[2]
        self.dgamma = (d * xh).sum((0, 1, 2)); self.dbeta = d.sum((0, 1, 2))
        dxh = d * self.gamma
        return (inv / n) * (n * dxh - dxh.sum((0, 1, 2)) - xh * (dxh * xh).sum((0, 1, 2)))


class SRCNN:
    def __init__(self, use_bn=False, use_tanh=False, residual=False, f=(64, 32), k=(9, 1, 5), seed=0):
        rng = np.random.default_rng(seed)
        self.use_bn, self.use_tanh, self.residual, self.k = use_bn, use_tanh, residual, k
        self.pad = tuple((s - 1) // 2 for s in k)
        chans = [3, f[0], f[1], 3]
        self.W, self.b = [], []
        for i in range(3):
            fan_in = chans[i] * k[i] * k[i]
            std = np.sqrt(2.0 / fan_in) if i < 2 else 0.01      # He init / small output init
            self.W.append((rng.standard_normal((chans[i], k[i], k[i], chans[i + 1])) * std).astype(F32))
            self.b.append(np.zeros(chans[i + 1], F32))
        self.bn = [BatchNorm(f[0]), BatchNorm(f[1])] if use_bn else []

    def params(self):
        p = self.W + self.b
        for bn in self.bn:
            p = p + [bn.gamma, bn.beta]
        return p

    def decay_mask(self):
        """L2 regularisation applies to conv weights only (not biases / BN params)."""
        return [True] * 3 + [False] * (3 + 2 * len(self.bn))

    def forward(self, x, train=False):
        x = x.astype(F32, copy=False)
        self._c = []
        h = x
        for i in range(3):
            h, cols = conv_forward(h, self.W[i], self.b[i], self.pad[i])
            self._c.append(cols)
            if i < 2:
                if self.use_bn:
                    h = self.bn[i].forward(h, train)
                mask = h > 0
                self._c.append(mask)
                h = h * mask
        if self.residual:                                        # optional refinement: learn a correction to the input
            h = h + x
        if self.use_tanh:
            t = np.tanh(h)
            self._c.append(t)
            h = (t + 1.0) * 0.5                                  # map (-1,1) -> (0,1)
        return h

    def predict(self, x, bs=2):
        out = [self.forward(x[i:i + bs], train=False) for i in range(0, len(x), bs)]
        return np.clip(np.concatenate(out), 0.0, 1.0)

    def backward(self, dout):
        """dout = dLoss/dOutput. Returns grads aligned with self.params()."""
        c = self._c
        # cache layout: [cols0, mask0, cols1, mask1, cols2, (tanh)]
        if self.use_tanh:
            t = c[5]
            dout = dout * 0.5 * (1 - t * t)
        dW, db = [None] * 3, [None] * 3
        d, dW[2], db[2] = conv_backward(dout, c[4], self.W[2], self.pad[2])
        d = d * c[3]
        if self.use_bn: d = self.bn[1].backward(d)
        d, dW[1], db[1] = conv_backward(d, c[2], self.W[1], self.pad[1])
        d = d * c[1]
        if self.use_bn: d = self.bn[0].backward(d)
        _, dW[0], db[0] = conv_backward(d, c[0], self.W[0], self.pad[0], need_dx=False)
        g = dW + db
        for bn in self.bn:
            g += [bn.dgamma, bn.dbeta]
        return g

    def save(self, path):
        s = {f"W{i}": self.W[i] for i in range(3)}
        s.update({f"b{i}": self.b[i] for i in range(3)})
        for i, bn in enumerate(self.bn):
            s.update({f"gamma{i}": bn.gamma, f"beta{i}": bn.beta, f"rm{i}": bn.rm, f"rv{i}": bn.rv})
        s["use_bn"], s["use_tanh"], s["residual"] = int(self.use_bn), int(self.use_tanh), int(self.residual)
        np.savez(path, **s)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        m = cls(use_bn=bool(z["use_bn"]), use_tanh=bool(z["use_tanh"]),
                residual=bool(z["residual"]) if "residual" in z.files else False)
        m.W = [z[f"W{i}"].copy() for i in range(3)]
        m.b = [z[f"b{i}"].copy() for i in range(3)]
        for i, bn in enumerate(m.bn):
            bn.gamma, bn.beta, bn.rm, bn.rv = (z[f"{n}{i}"].copy() for n in ("gamma", "beta", "rm", "rv"))
        return m


class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, wd=0.0, decay_mask=None):
        self.p, self.lr, self.b1, self.b2, self.eps, self.wd = params, lr, b1, b2, eps, wd
        self.mask = decay_mask
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        for i, (p, g) in enumerate(zip(self.p, grads)):
            if self.wd and (self.mask is None or self.mask[i]):
                g = g + self.wd * p                              # L2 regularisation
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            mh = self.m[i] / (1 - self.b1 ** self.t)
            vh = self.v[i] / (1 - self.b2 ** self.t)
            p -= (self.lr * mh / (np.sqrt(vh) + self.eps)).astype(F32)
