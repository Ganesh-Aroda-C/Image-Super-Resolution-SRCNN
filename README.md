# Image Super-Resolution via a Convolutional Neural Network (SRCNN)

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![NumPy](https://img.shields.io/badge/Pure_NumPy-Engine-013243.svg)](https://numpy.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/pytest-Passing-brightgreen.svg)](tests/)

**Course:** UE24CS352A – Machine Learning · Mini-Project  
**Institution:** PES University  

### Team Details
- **Team Member 1:** Ganesh Aroda C · **SRN:** PES1UG24AM102
- **Team Member 2:** Hanshal · **SRN:** PES1UG24AM110

---

## 📖 1. Project Overview

This repository provides an end-to-end implementation and empirical study of **SRCNN (Super-Resolution Convolutional Neural Network)**, first introduced by *Dong et al. (2015)* and based on the Stanford CS229 research study (*Garber, Grossman, Johnson-Yu*).

Single-Image Super-Resolution (SISR) recovers a high-resolution (HR) image from a low-resolution (LR) observation. While standard polynomial interpolations (Bilinear, Bicubic) merely blur missing pixels, SRCNN learns non-linear mapping functions to hallucinate missing high-frequency textures and sharpen blurry contours.

### Core Highlights
1. **From-Scratch Pure NumPy Engine (`srcnn/core.py`):** Forward and analytical backward propagation, im2col matrix convolution (`sliding_window_view`), He weight initialization, and the Adam optimizer are written entirely from scratch without auto-diff frameworks.
2. **Modern PyTorch Implementation (`srcnn/models_pytorch.py`):** Clean, modular PyTorch module supporting both CPU and GPU (CUDA) execution, with full checkpoint saving and bidirectional weight conversion with NumPy `.npz` files.
3. **Residual Learning Refinement:** An architectural skip connection ($\hat{Y} = X_{\text{bicubic}} + F(X_{\text{bicubic}})$) improving PSNR by **+0.991 dB RGB** and **+1.227 dB Y-Channel** over bicubic interpolation.
4. **Standardized Metrics:** Computes Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM) across RGB and Luminance (Y-channel with border shaving).
5. **Interactive Streamlit Web Studio (`app.py`):** A modern browser demo featuring live image upload, side-by-side comparison, fine detail zoom, and downloadable super-resolved images.
6. **Automated Testing Suite:** 20 test cases covering gradient flows, tensor shapes, metrics, edge cases, and end-to-end pipelines.

---

## 📐 2. The SRCNN Pipeline

The network takes a bicubic-upsampled low-resolution image ($Y_{\text{bicubic}} \in \mathbb{R}^{H \times W \times 3}$) and processes it through three sequential convolutional operations:

```
Low-Res Input (H/2 x W/2 x 3)
            │
      Bicubic Upsample (x2.0)
            │
            ▼
┌──────────────────────────────────────┐
│ Layer 1: Patch Extraction (Conv 1)   │ ── 9x9 kernel, 64 filters, ReLU
│ F₁(Y) = max(0, W₁ * Y + B₁)          │    Extracts 64-dim feature patches
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│ Layer 2: Non-linear Mapping (Conv 2) │ ── 1x1 kernel, 32 filters, ReLU
│ F₂(Y) = max(0, W₂ * F₁ + B₂)         │    Maps feature space non-linearly
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│ Layer 3: Reconstruction (Conv 3)     │ ── 5x5 kernel, 3 filters, Linear
│ F(Y) = W₃ * F₂ + B₃                  │    Reconstructs RGB channels
└──────────────────┬───────────────────┘
                   │
                   ▼ (Optional Residual Skip: + Y_bicubic)
High-Res Output (H x W x 3)
```

---

## 📂 3. Repository Structure

```
Image-Super-Resolution-SRCNN/
├── app.py                      # Interactive Streamlit Web Studio
├── requirements.txt            # Project dependencies
├── README.md                   # Comprehensive user guide
├── .gitignore                  # Git exclusions for environments and caches
│
├── srcnn/                      # Core SRCNN library
│   ├── __init__.py             # Unified package exports
│   ├── core.py                 # Pure-NumPy SRCNN, im2col, BatchNorm, Adam
│   ├── data.py                 # Dataset loading, image degradation (make_lr)
│   ├── metrics.py              # PSNR, SSIM, Y-channel conversion, border shave
│   ├── trainer.py              # Pure-NumPy training and evaluation loop
│   ├── models_pytorch.py       # PyTorch SRCNN module with NumPy interoperability
│   └── trainer_pytorch.py      # PyTorch dataset, dataloader, and training loop
│
├── scripts/                    # Command-line workflows
│   ├── train.py                # Train SRCNN (NumPy or PyTorch backend)
│   ├── demo.py                 # CLI demonstration and side-by-side generator
│   ├── evaluate_dataset.py     # Benchmark any folder/dataset (Set5, Set14, data/test)
│   ├── run_experiments.py      # Systematic 11-experiment hyperparameter study
│   ├── make_report.py          # Compile benchmark tables and loss curves
│   ├── make_dataset.py         # Build bundled 100-image dataset (scikit-image)
│   ├── prepare_div2k.py        # Prepare real DIV2K dataset
│   └── _path.py                # Path bootstrap helper
│
├── tests/                      # Automated test suite
│   ├── test_gradients.py       # Numerical finite-difference backprop verification
│   └── test_pipeline.py        # 20 pytest cases for shapes, models, metrics, errors
│
├── data/                       # Bundled dataset (100 images, 224x224 PNG)
│   ├── train/                  # 60 training images
│   ├── val/                    # 20 validation images
│   └── test/                   # 20 test images
│
├── results/                    # Precomputed models, logs, and plots
│   ├── *.npz                   # 11 trained NumPy checkpoints
│   ├── *.json                  # Training histories and metadata
│   ├── results_table.md / .csv # Comprehensive benchmark metrics
│   ├── fig_*.png               # Experiment loss curves and sample comparisons
│   └── experiments.log         # Full training logs
│
└── docs/                       # Project documentation
    ├── REPORT.md               # Complete academic mini-project report
    ├── PRESENTATION.md         # Slide-by-slide jury defense guide and Q&A
    ├── ARCHITECTURE.md         # Detailed mathematical derivations and layer specs
    ├── RESULTS.md              # In-depth analysis of hyperparameter sweeps
    └── *.pdf / *.pptx          # Original report and presentation slide files
```

---

## ⚡ 4. Installation & Environment Setup (Windows 11)

Open **PowerShell** in the repository root directory:

```powershell
# 1. Clone or navigate to the repository
cd C:\Users\gani4\OneDrive\Desktop\Brooklyn\Image-Super-Resolution-SRCNN

# 2. Create a virtual environment (optional but recommended)
python -m venv venv

# 3. Activate the virtual environment
.\venv\Scripts\Activate.ps1
# (If script execution is restricted: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass)

# 4. Upgrade pip and install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## 🧪 5. Automated Testing

Run the automated test suite to verify backpropagation gradients, model shapes, metrics, and error handling:

```powershell
# Run the complete test suite (20 automated tests)
python -m pytest tests/ -v

# Run the finite-difference numerical gradient check standalone
python tests/test_gradients.py
```

*Expected output:*
```
tests/test_gradients.py::test_gradients_vanilla PASSED
tests/test_gradients.py::test_gradients_residual PASSED
tests/test_pipeline.py::test_numpy_forward_dimensions PASSED
tests/test_pipeline.py::test_pytorch_forward_dimensions PASSED
tests/test_pipeline.py::test_cross_backend_interoperability PASSED
tests/test_pipeline.py::test_psnr_identical_images PASSED
tests/test_pipeline.py::test_end_to_end_smoke_training PASSED
============================= 20 passed in ~4s =============================
```

---

## 🚀 6. How to Run & Use the Project

### A. Quick Inference Demonstration (CLI)
Reconstruct a test image and calculate PSNR / SSIM:

```powershell
# Evaluate with default NumPy model and generate detail zoom crop
python scripts/demo.py --input data/test/test_000_retina.png --crop

# Super-resolve using the top-performing Residual model
python scripts/demo.py --input data/test/test_000_retina.png --model results/residual_lr_0.001.npz --crop

# Super-resolve an existing low-resolution user image (no ground truth reference)
python scripts/demo.py --input path/to/my_photo.jpg --no_degrade
```
*Outputs saved:* `results/<name>_srcnn.png`, `results/<name>_comparison.png`, and `results/<name>_crop_comparison.png`.

---

### B. Launch the Interactive Web Studio
Experience the interactive Streamlit web interface with drag-and-drop image upload, side-by-side sliders, and live metrics:

```powershell
# Install streamlit if not already installed
python -m pip install streamlit

# Launch the app
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

### C. Train an SRCNN Model
Train a model using either the pure-NumPy engine or the PyTorch engine:

```powershell
# Train using pure-NumPy engine (default)
python scripts/train.py --epochs 30 --lr 0.001 --name my_numpy_model

# Train using PyTorch engine (CPU or CUDA)
python scripts/train.py --backend pytorch --epochs 30 --lr 0.001 --name my_torch_model

# Train with residual learning refinement
python scripts/train.py --epochs 30 --lr 0.001 --residual --name my_residual_model
```
*Outputs saved:* `results/<name>.npz`, `results/<name>.json`, and `results/<name>.pt`.

---

### D. Benchmark on a Test Directory or Dataset
Evaluate all images in a folder (such as `data/test` or public datasets like Set5):

```powershell
# Evaluate test set with the Residual model
python scripts/evaluate_dataset.py --dataset data/test --model results/residual_lr_0.001.npz
```

---

### E. Reproduce the Full Experimental Study
Run the 11-experiment hyperparameter sweep (resumes automatically):

```powershell
# Fast 3-minute smoke test of all experiments
python scripts/run_experiments.py --quick
python scripts/make_report.py --res results_quick

# Full experimental study (~35 min on CPU)
python scripts/run_experiments.py
python scripts/make_report.py
```

---

## 📊 7. Benchmark Results

Summary of 40-epoch experiments on the 100-image dataset (60 train / 20 val / 20 test):

| Experiment | Val PSNR (dB) | Test Loss (MSE) | Test RGB PSNR (dB) | Test RGB SSIM | Test Y-PSNR (dB) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Residual SRCNN (Our Refinement)** | **31.785** | **0.000432** | **37.848** | **0.9451** | **38.332** |
| Bicubic Interpolation (Baseline) | 29.866 | 0.000734 | 36.857 | 0.9286 | 37.105 |
| Basic SRCNN (lr=0.001) | 30.943 | 0.000538 | 36.567 | 0.9388 | 37.289 |
| + Regularization (wd=1e-4) | 30.881 | 0.000585 | 36.019 | 0.9312 | 36.751 |
| Bilinear Interpolation | 28.901 | 0.000937 | 35.351 | 0.9082 | 35.612 |
| Nearest Neighbor | 28.833 | 0.000946 | 34.432 | 0.8914 | 34.708 |
| Basic SRCNN + Tanh | 28.717 | 0.000848 | 33.027 | 0.8953 | 33.415 |
| Basic SRCNN + Batch Normalization | 27.197 | 0.001725 | 27.733 | 0.8240 | 28.110 |

### Key Research Findings
1. **Residual Learning Dominates:** Learning high-frequency corrections rather than whole images yields a **+0.991 dB RGB gain** and **+1.227 dB Y-channel gain** over bicubic interpolation, with up to **+2.70 dB gains** on real photographic textures.
2. **Batch Normalization is Harmful:** BatchNorm degrades PSNR by **>9.1 dB** because normalizing activations disrupts absolute pixel dynamic range and introduces boundary tile artifacts.
3. **Tanh Output Causes Vanishing Gradients:** Restricting outputs with scaled Tanh causes saturation near 0 and 1, slowing convergence and reducing PSNR. Linear output layers are optimal for MSE regression.

---

## 🔧 8. Troubleshooting Common Issues

| Issue / Error | Cause | Resolution |
| :--- | :--- | :--- |
| `FileNotFoundError: Model checkpoint not found` | Checkpoint path typo or un-trained model | Use an existing checkpoint in `results/` (e.g. `results/residual_lr_0.001.npz`) or run `python scripts/train.py --epochs 30`. |
| `ModuleNotFoundError: No module named 'srcnn'` | Running script from an external directory | Run scripts directly from repository root (`python scripts/demo.py ...`), which automatically loads `_path.py`. |
| `Execution of scripts is disabled on this system` | Windows PowerShell execution policy | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in PowerShell before activating `venv`. |
| `ModuleNotFoundError: No module named 'streamlit'` | Streamlit optional dependency missing | Install via `python -m pip install streamlit` or use the CLI demo `python scripts/demo.py`. |

---

## 📚 9. Project Documentation Links

- **Academic Report:** [`docs/REPORT.md`](docs/REPORT.md)
- **Jury Defense & Presentation Script:** [`docs/PRESENTATION.md`](docs/PRESENTATION.md)
- **Mathematical Formulations & Architecture:** [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- **Experimental Results & Ablation Analysis:** [`docs/RESULTS.md`](docs/RESULTS.md)

---

## 📜 10. References

- **Dong, C., Loy, C. C., He, K., & Tang, X.** (2015). *Image Super-Resolution Using Deep Convolutional Networks*. IEEE TPAMI.
- **Garber, A., Grossman, T., & Johnson-Yu, S.** (2015). *Image Super-Resolution Via a Convolutional Neural Network*. Stanford CS229 Machine Learning Report.
- **Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P.** (2004). *Image quality assessment: from error visibility to structural similarity*. IEEE TIP.
