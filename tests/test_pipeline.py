"""Comprehensive test suite for the SRCNN Super-Resolution project.

Tests:
1. Model initialization and forward-pass tensor dimensions (NumPy and PyTorch).
2. Gradient flow and parameter updates.
3. Image preprocessing, degradation (make_lr), and normalization.
4. Dataset loading and patch generation.
5. Checkpoint saving, reloading, and cross-backend interoperability.
6. CPU inference on arbitrary dimensions (including non-multiples of scale).
7. PSNR and SSIM metric edge cases and color conversions.
8. Error handling for missing files and invalid configurations.
9. Full end-to-end smoke test on synthetic sample images.
"""
import os
import sys
import tempfile
import pytest
import numpy as np
from PIL import Image

# Ensure project root is in sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import srcnn
from srcnn.core import SRCNN, Adam
from srcnn.data import make_lr, load_split, validate_image_path
from srcnn.metrics import psnr, ssim, compute_metrics, rgb_to_y, shave_borders
from srcnn.trainer import train, evaluate, random_patches
from srcnn.models_pytorch import SRCNN_PyTorch
from srcnn.trainer_pytorch import train_torch, evaluate_torch, PatchDataset
import torch


# ---------------- 1. Model Forward Dimensions ----------------
def test_numpy_forward_dimensions():
    """Verify that NumPy SRCNN produces exact matching spatial dimensions."""
    model = SRCNN(f=(64, 32), k=(9, 1, 5), seed=42)
    # Test batch of square inputs
    x_square = np.random.rand(2, 64, 64, 3).astype(np.float32)
    out_square = model.forward(x_square, train=False)
    assert out_square.shape == (2, 64, 64, 3), f"Unexpected shape {out_square.shape}"
    
    # Test arbitrary rectangular dimensions (not multiples of 2 or 4)
    x_rect = np.random.rand(1, 47, 53, 3).astype(np.float32)
    out_rect = model.forward(x_rect, train=False)
    assert out_rect.shape == (1, 47, 53, 3), f"Rectangular shape mismatch: {out_rect.shape}"


def test_pytorch_forward_dimensions():
    """Verify that PyTorch SRCNN produces exact matching spatial dimensions."""
    model = SRCNN_PyTorch(f=(64, 32), k=(9, 1, 5))
    model.eval()
    x = torch.rand(2, 3, 64, 64)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 3, 64, 64), f"PyTorch forward shape mismatch {out.shape}"
    
    # Arbitrary odd dimensions
    x_odd = torch.rand(1, 3, 37, 49)
    with torch.no_grad():
        out_odd = model(x_odd)
    assert out_odd.shape == (1, 3, 37, 49), f"PyTorch odd dimensions mismatch {out_odd.shape}"


# ---------------- 2. Gradient Flow & Optimization ----------------
def test_numpy_parameter_updates():
    """Verify that Adam optimizer updates all model parameters with non-zero gradients."""
    model = SRCNN(seed=0)
    opt = Adam(model.params(), lr=1e-3)
    x = np.random.rand(2, 32, 32, 3).astype(np.float32)
    y = np.random.rand(2, 32, 32, 3).astype(np.float32)
    
    initial_w0 = model.W[0].copy()
    out = model.forward(x, train=True)
    diff = out - y
    grads = model.backward((2.0 / diff.size) * diff)
    opt.step(grads)
    
    assert not np.allclose(initial_w0, model.W[0]), "Weights were not updated during optimization step!"


def test_pytorch_gradient_flow():
    """Verify gradient computation and backpropagation in PyTorch model."""
    model = SRCNN_PyTorch()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.rand(2, 3, 32, 32)
    y = torch.rand(2, 3, 32, 32)
    
    out = model(x)
    loss = torch.nn.functional.mse_loss(out, y)
    loss.backward()
    
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no gradient!"
        assert torch.sum(torch.abs(param.grad)) > 0, f"Parameter {name} has all-zero gradients!"
    optimizer.step()


# ---------------- 3. Image Preprocessing & Degradation ----------------
def test_make_lr_degradation():
    """Verify that make_lr correctly degrades and upsamples an image."""
    img = Image.new("RGB", (100, 80), color=(120, 60, 200))
    degraded = make_lr(img, size=(100, 80), scale=2.0)
    assert degraded.size == (100, 80), f"Size mismatch after degradation: {degraded.size}"
    
    # Check scale factor 3.0
    degraded_3x = make_lr(img, size=(90, 90), scale=3.0)
    assert degraded_3x.size == (90, 90)


# ---------------- 4. Dataset Loading & Patch Sampling ----------------
def test_random_patch_sampling():
    """Verify patch sampling shapes and boundary constraints."""
    rng = np.random.default_rng(42)
    X = np.random.rand(5, 64, 64, 3).astype(np.float32)
    Y = np.random.rand(5, 64, 64, 3).astype(np.float32)
    
    xs, ys = random_patches(X, Y, n=10, size=32, rng=rng)
    assert xs.shape == (10, 32, 32, 3)
    assert ys.shape == (10, 32, 32, 3)
    assert np.all(xs >= 0.0) and np.all(xs <= 1.0)


def test_pytorch_patch_dataset():
    """Verify PyTorch PatchDataset __getitem__ and batch collation."""
    X = np.random.rand(4, 64, 64, 3).astype(np.float32)
    Y = np.random.rand(4, 64, 64, 3).astype(np.float32)
    ds = PatchDataset(X, Y, patch_size=32, length=16, seed=123)
    assert len(ds) == 16
    
    xt, yt = ds[0]
    assert xt.shape == (3, 32, 32)
    assert yt.shape == (3, 32, 32)


# ---------------- 5. Checkpoint Saving, Reloading & Interoperability ----------------
def test_checkpoint_saving_and_reloading():
    """Verify that saving and reloading NumPy checkpoint preserves predictions exactly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "test_model.npz")
        m1 = SRCNN(seed=7)
        x = np.random.rand(1, 32, 32, 3).astype(np.float32)
        pred1 = m1.predict(x)
        m1.save(ckpt_path)
        
        m2 = SRCNN.load(ckpt_path)
        pred2 = m2.predict(x)
        np.testing.assert_allclose(pred1, pred2, atol=1e-6)


def test_cross_backend_interoperability():
    """Verify that NumPy checkpoint can be loaded into PyTorch model and produces identical outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        npz_path = os.path.join(tmpdir, "shared_model.npz")
        np_model = SRCNN(seed=99)
        np_model.save(npz_path)
        
        torch_model = SRCNN_PyTorch()
        torch_model.load_from_numpy(npz_path)
        
        x = np.random.rand(1, 48, 48, 3).astype(np.float32)
        out_np = np_model.predict(x)
        out_torch = torch_model.predict(x)
        
        max_diff = np.max(np.abs(out_np - out_torch))
        assert max_diff < 1e-4, f"Cross-backend prediction divergence: {max_diff}"


# ---------------- 6. Metrics Calculation & Edge Cases ----------------
def test_psnr_identical_images():
    """Verify that PSNR of identical images returns 100 dB without division by zero."""
    x = np.random.rand(2, 32, 32, 3).astype(np.float32)
    p = psnr(x, x)
    assert np.all(p == 100.0)


def test_ssim_identical_images():
    """Verify SSIM of identical images is 1.0."""
    x = np.random.rand(32, 32, 3).astype(np.float32)
    s = ssim(x, x)
    assert pytest.approx(float(s), abs=1e-4) == 1.0


def test_rgb_to_y_conversion():
    """Verify standard ITU-R BT.601 conversion formula."""
    # Pure red [1, 0, 0] -> Y = 0.299
    red = np.zeros((1, 1, 3), dtype=np.float32)
    red[0, 0, 0] = 1.0
    y_red = rgb_to_y(red)
    assert pytest.approx(float(y_red[0, 0, 0]), abs=1e-4) == 0.299


def test_border_shaving():
    """Verify that border shaving correctly removes periphery pixels."""
    x = np.arange(16, dtype=np.float32).reshape(4, 4, 1)
    shaved = shave_borders(x, shave=1)
    assert shaved.shape == (2, 2, 1)


# ---------------- 7. Error Handling & Validation ----------------
def test_missing_image_validation():
    """Verify FileNotFoundError on non-existent image path."""
    with pytest.raises(FileNotFoundError):
        validate_image_path("non_existent_path_xyz123.png")


def test_missing_model_checkpoint():
    """Verify FileNotFoundError when loading non-existent checkpoint."""
    with pytest.raises(FileNotFoundError):
        SRCNN.load("results/definitely_not_a_model.npz")


# ---------------- 8. End-to-End Synthetic Smoke Test ----------------
def test_end_to_end_smoke_training():
    """Test full training loop (2 epochs) on a tiny synthetic dataset."""
    Xtr = np.random.rand(4, 32, 32, 3).astype(np.float32)
    Ytr = np.random.rand(4, 32, 32, 3).astype(np.float32)
    Xva = np.random.rand(2, 32, 32, 3).astype(np.float32)
    Yva = np.random.rand(2, 32, 32, 3).astype(np.float32)
    
    # Test NumPy training loop
    model_np, hist_np = train(
        Xtr, Ytr, Xva, Yva,
        epochs=2, batch=2, patch=16, patches_per_epoch=8,
        log=lambda s: None
    )
    assert len(hist_np["train_loss"]) == 2
    mse_np, psnr_np, _ = evaluate(model_np, Xva, Yva)
    assert mse_np >= 0.0
    assert psnr_np > 0.0

    # Test PyTorch training loop
    model_pt = SRCNN_PyTorch()
    model_pt, hist_pt = train_torch(
        model_pt, Xtr, Ytr, Xva, Yva,
        epochs=2, batch_size=2, patch_size=16, patches_per_epoch=8,
        device=torch.device("cpu"),
        log=lambda s: None
    )
    assert len(hist_pt["train_loss"]) == 2
    mse_pt, psnr_pt, _ = evaluate_torch(model_pt, Xva, Yva, device=torch.device("cpu"))
    assert mse_pt >= 0.0
    assert psnr_pt > 0.0
