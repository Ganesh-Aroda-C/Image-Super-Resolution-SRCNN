"""Train a single SRCNN model and report test PSNR/SSIM.

Usage:
  python scripts/train.py --epochs 30 --lr 0.001                    # Pure-NumPy backend (default)
  python scripts/train.py --backend pytorch --epochs 30 --lr 0.001  # PyTorch backend
"""
import argparse
import json
import os
import time

import _path
from _path import ROOT
import srcnn
from srcnn import load_split, compute_metrics


def parse_args():
    ap = argparse.ArgumentParser(description="Train an SRCNN model")
    ap.add_argument("--backend", choices=["numpy", "pytorch"], default="numpy",
                    help="Engine backend: 'numpy' (from scratch) or 'pytorch' (CPU/CUDA)")
    ap.add_argument("--data", default=os.path.join(ROOT, "data"), help="Path to data directory")
    ap.add_argument("--out", default=os.path.join(ROOT, "results"), help="Directory for saved model and metrics")
    ap.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    ap.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    ap.add_argument("--batch", type=int, default=16, help="Batch size")
    ap.add_argument("--patch", type=int, default=48, help="Patch size for training")
    ap.add_argument("--patches_per_epoch", type=int, default=160, help="Total patches sampled per epoch")
    ap.add_argument("--wd", type=float, default=0.0, help="L2 regularization / weight decay")
    ap.add_argument("--bn", action="store_true", help="Add batch-norm after conv1 and conv2")
    ap.add_argument("--tanh", action="store_true", help="Use tanh on the output layer")
    ap.add_argument("--residual", action="store_true", help="Predict residual added to bicubic input")
    ap.add_argument("--scale", type=float, default=2.0, help="Super-resolution degradation scale")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="PyTorch compute device")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    ap.add_argument("--name", default="srcnn", help="Experiment / model checkpoint name")
    return ap.parse_args()


def main():
    a = parse_args()
    os.makedirs(a.out, exist_ok=True)
    
    print(f"=== Training SRCNN [{a.backend.upper()} Backend] ===")
    print(f"Data directory: {a.data}")
    print(f"Configuration : epochs={a.epochs}, lr={a.lr}, batch={a.batch}, patch={a.patch}, seed={a.seed}")
    print(f"Architecture  : bn={a.bn}, tanh={a.tanh}, residual={a.residual}")

    Xtr, Ytr, _ = load_split(a.data, "train", scale=a.scale)
    Xva, Yva, _ = load_split(a.data, "val", scale=a.scale)
    Xte, Yte, _ = load_split(a.data, "test", scale=a.scale)
    print(f"Dataset loaded: train {Xtr.shape}, val {Xva.shape}, test {Xte.shape}")

    t0 = time.time()
    if a.backend == "numpy":
        from srcnn.trainer import train, evaluate
        model, hist = train(
            Xtr, Ytr, Xva, Yva,
            lr=a.lr, epochs=a.epochs, batch=a.batch, patch=a.patch,
            patches_per_epoch=a.patches_per_epoch,
            use_bn=a.bn, use_tanh=a.tanh, residual=a.residual,
            wd=a.wd, seed=a.seed, tag=a.name
        )
        tl, tp, pred_te = evaluate(model, Xte, Yte)
        
        # Save NumPy weights
        npz_out = os.path.join(a.out, f"{a.name}.npz")
        model.save(npz_out)
        print(f"Saved NumPy checkpoint: {npz_out}")
    else:
        import torch
        from srcnn.models_pytorch import SRCNN_PyTorch
        from srcnn.trainer_pytorch import train_torch, evaluate_torch
        
        if a.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(a.device)
        print(f"Using PyTorch device: {device}")
        
        torch_model = SRCNN_PyTorch(
            use_bn=a.bn,
            use_tanh=a.tanh,
            residual=a.residual
        )
        pt_path = os.path.join(a.out, f"{a.name}.pt")
        model, hist = train_torch(
            torch_model, Xtr, Ytr, Xva, Yva,
            lr=a.lr, epochs=a.epochs, batch_size=a.batch, patch_size=a.patch,
            patches_per_epoch=a.patches_per_epoch, weight_decay=a.wd,
            device=device, save_checkpoint_path=pt_path,
            seed=a.seed, tag=a.name
        )
        tl, tp, pred_te = evaluate_torch(model, Xte, Yte, device=device)
        
        # Also export to .npz for cross-backend compatibility
        npz_out = os.path.join(a.out, f"{a.name}.npz")
        model.export_to_numpy(npz_out)
        print(f"Saved PyTorch checkpoint: {pt_path} and exported {npz_out}")

    # Compute test set SSIM and Y-channel PSNR
    test_ssim_rgb = float(srcnn.ssim(pred_te, Yte).mean())
    psnr_y, ssim_y = [], []
    for i in range(len(pred_te)):
        py, sy = compute_metrics(pred_te[i], Yte[i], shave=int(a.scale), use_y_channel=True)
        psnr_y.append(py)
        ssim_y.append(sy)
    test_psnr_y = float(sum(psnr_y) / len(psnr_y))
    test_ssim_y = float(sum(ssim_y) / len(ssim_y))

    elapsed = time.time() - t0
    print("\n" + "=" * 50)
    print(f"TEST RESULTS [{a.name}]:")
    print(f"  Test Loss (MSE)      : {tl:.6f}")
    print(f"  Test PSNR (RGB)      : {tp:.3f} dB")
    print(f"  Test SSIM (RGB)      : {test_ssim_rgb:.4f}")
    print(f"  Test PSNR (Y-channel): {test_psnr_y:.3f} dB (shave={int(a.scale)})")
    print(f"  Test SSIM (Y-channel): {test_ssim_y:.4f} (shave={int(a.scale)})")
    print(f"  Total Training Time  : {elapsed:.1f}s")
    print("=" * 50)

    # Save summary JSON
    summary_path = os.path.join(a.out, f"{a.name}.json")
    json_data = {
        "history": hist,
        "test_loss": tl,
        "test_psnr": tp,
        "test_ssim": test_ssim_rgb,
        "test_psnr_y": test_psnr_y,
        "test_ssim_y": test_ssim_y,
        "training_time_sec": elapsed,
        "args": vars(a)
    }
    with open(summary_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"Saved experiment metadata: {summary_path}")


if __name__ == "__main__":
    main()
