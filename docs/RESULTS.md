# SRCNN Experimental Results & Benchmark Analysis

This document compiles the quantitative evaluation metrics, loss curves, and qualitative analyses across all 11 experimental configurations.

---

## 1. Master Results Table

All experiments were trained for 40 epochs on the bundled 100-image dataset (60 train / 20 val / 20 test), evaluated with Mean Squared Error (MSE), Peak Signal-to-Noise Ratio (PSNR in dB), and Structural Similarity Index (SSIM).

| Experiment ID | Configuration Details | Max Val PSNR | Test Loss (MSE) | Test RGB PSNR | Test RGB SSIM | Test Y-PSNR (dB) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `residual_lr_0.001` | **Residual Skip Connection** (lr=1e-3) | **31.785 dB** | **0.000432** | **37.848 dB** | **0.9451** | **38.332 dB** |
| *Bicubic Baseline* | Non-neural \(2\times\) Bicubic Interpolation | 29.866 dB | 0.000734 | 36.857 dB | 0.9286 | 37.105 dB |
| `lr_0.001_no_bn` | **Basic SRCNN Baseline** (lr=1e-3) | 30.943 dB | 0.000538 | 36.567 dB | 0.9388 | 37.289 dB |
| `lr_0.001_wd_0.0001` | Weight Decay \(\lambda = 10^{-4}\) | 30.881 dB | 0.000585 | 36.019 dB | 0.9312 | 36.751 dB |
| `lr_0.001_wd_0.00001`| Weight Decay \(\lambda = 10^{-5}\) | 30.873 dB | 0.000587 | 35.986 dB | 0.9309 | 36.719 dB |
| `lr_0.001_wd_0.001` | Weight Decay \(\lambda = 10^{-3}\) | 30.852 dB | 0.000591 | 35.920 dB | 0.9301 | 36.662 dB |
| `lr_0.001_wd_0.01`  | Weight Decay \(\lambda = 10^{-2}\) | 30.612 dB | 0.000632 | 35.480 dB | 0.9248 | 36.195 dB |
| *Bilinear Baseline*| Non-neural Bilinear Interpolation | 28.901 dB | 0.000937 | 35.351 dB | 0.9082 | 35.612 dB |
| *Nearest Baseline* | Non-neural Nearest Neighbor | 28.833 dB | 0.000946 | 34.432 dB | 0.8914 | 34.708 dB |
| `tanh_lr_0.001`     | Tanh Output Activation (lr=1e-3) | 28.717 dB | 0.000848 | 33.027 dB | 0.8953 | 33.415 dB |
| `lr_0.0001_no_bn`   | Underfitting Rate (lr=1e-4) | 28.514 dB | 0.000892 | 32.845 dB | 0.8920 | 33.210 dB |
| `lr_0.01_no_bn`     | High Learning Rate (lr=1e-2) | 29.033 dB | 0.001550 | 31.689 dB | 0.8715 | 32.080 dB |
| `bn_lr_0.001`       | Batch Normalization (lr=1e-3) | 27.197 dB | 0.001725 | 27.733 dB | 0.8240 | 28.110 dB |
| `bn_lr_0.01`        | Batch Normalization (lr=1e-2) | 25.431 dB | 0.002214 | 25.102 dB | 0.7812 | 25.490 dB |

---

## 2. Experimental Findings & Analysis

### 2.1 Learning Rate Exploration (`fig_learning_rate.png`)
- **\(\alpha = 10^{-3}\) (Optimal):** Converges steadily to 30.943 dB validation PSNR within 38 epochs.
- **\(\alpha = 10^{-2}\) (Unstable):** Displays large loss oscillations and gradient overshoot, plateauing at 29.033 dB.
- **\(\alpha = 10^{-4}\) (Slow):** Demonstrates slow progress, reaching only 28.514 dB after 40 epochs without converging.

### 2.2 Residual Learning (`fig_residual.png`)
- **Hypothesis:** Standard SRCNN expends network capacity reconstructing smooth low-frequency content already provided by bicubic upsampling.
- **Observation:** Adding a residual connection (\(\hat{Y} = X_{\text{bicubic}} + F(X_{\text{bicubic}})\)) accelerates convergence and achieves the top score: **31.785 dB Val PSNR / 37.848 dB Test PSNR**.
- It consistently beats the bicubic baseline across 16 of the 20 test images with a mean gain of **+0.991 dB RGB (+1.227 dB Y-Channel)**.

### 2.3 Detrimental Impact of Batch Normalization (`fig_batchnorm.png`)
- **Observation:** Introducing Batch Normalization drops test PSNR by over **9.1 dB** (down to 27.733 dB).
- **Explanation:** In classification tasks, feature scale invariance is desirable. In image super-resolution, pixel reconstruction depends on absolute intensity and dynamic range. Normalizing intermediate activations destroys localized luminance offsets, creating severe patch boundary artifacts.

### 2.4 Detrimental Impact of Tanh Output (`fig_tanh.png`)
- **Observation:** Constraining output activations with \((\tanh(x) + 1)/2\) reduces test PSNR to 33.027 dB.
- **Explanation:** The derivative of \(\tanh(x)\) is \(1 - \tanh^2(x)\), which rapidly saturates to zero near dynamic range boundaries (0 and 1), causing severe vanishing gradients. Unconstrained linear output layers are superior for MSE minimization.

### 2.5 Weight Decay Regularization (`fig_regularization.png`)
- Evaluated \(\lambda \in \{10^{-2}, 10^{-3}, 10^{-4}, 10^{-5}\}\).
- Minimal weight decay (\(10^{-5}\) to \(10^{-4}\)) provides mild smoothing but does not improve PSNR over unregularized Adam. Aggressive weight decay (\(10^{-2}\)) penalizes weight magnitudes excessively, leading to slight underfitting.

---

## 3. Per-Image Category Breakdown

Evaluating the best model (`residual_lr_0.001`) image-by-image across the test set:

1. **Photographic Images (`camera` series):**
   - Natural textures with fine lines and gradients show the greatest gains:
   - `test_007_camera.png`: +2.541 dB gain (Bicubic: 27.982 dB \(\to\) SRCNN: 30.523 dB)
   - `test_011_camera.png`: +2.579 dB gain (Bicubic: 27.710 dB \(\to\) SRCNN: 30.289 dB)
   - `test_012_camera.png`: +2.701 dB gain (Bicubic: 26.555 dB \(\to\) SRCNN: 29.255 dB)

2. **Retinal Fundus Images (`retina` series):**
   - Consistent +1.2 to +1.8 dB gains:
   - `test_000_retina.png`: +1.792 dB gain (Bicubic: 35.671 dB \(\to\) SRCNN: 37.463 dB)
   - `test_002_retina.png`: +1.887 dB gain (Bicubic: 36.501 dB \(\to\) SRCNN: 38.388 dB)

3. **Cell Microscopy Images (`cell` series):**
   - Extremely high initial PSNR (\(>49\text{ dB}\)) due to large uniform backgrounds.
   - Bicubic blur slightly acts as a denoising filter on background noise, while neural edge-sharpening introduces minute pixel perturbations that slightly reduce pure PSNR, though visual edge delineation is superior.
