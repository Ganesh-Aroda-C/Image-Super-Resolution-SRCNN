# SRCNN Project Defense — Presentation Guide & Jury Talking Points

**Course:** UE24CS352A – Machine Learning Mini-Project  
**Title:** Single-Image Super-Resolution using SRCNN  
**Presenters:** Ganesh Aroda C (PES1UG24AM102) & Hanshal (PES1UG24AM110)  

---

## Slide 1: Title & Introduction
- **Slide Title:** Single-Image Super-Resolution via Deep Convolutional Networks (SRCNN)
- **Subtitle:** An End-to-End Formulation from Pure NumPy Foundations to Modern PyTorch
- **Talking Points:**
  - Good morning, esteemed evaluators. Today we present our machine learning project on Single-Image Super-Resolution using SRCNN.
  - While modern libraries hide convolutions and backpropagation behind auto-differentiation, our primary goal was to construct the complete forward and analytical backward passes from scratch in pure NumPy, verify numerical gradients, and systematically benchmark architecture variations.

---

## Slide 2: Problem Statement & Motivation
- **Slide Title:** The Super-Resolution Inverse Problem
- **Key Concepts:**
  - Super-Resolution is a classic ill-posed inverse problem: fine high-frequency details are irrevocably lost during optical blur and sensor downsampling.
  - Classical polynomial interpolations (Bilinear, Bicubic) rely on smooth neighborhood averages; they can enlarge an image, but they cannot invent missing high-frequency textures or sharpen blurry edges.
  - **Motivation:** Super-resolution is foundational in medical imaging (MRI/CT), satellite surveillance, forensics, and modern video streaming.

---

## Slide 3: Proposed Solution — SRCNN
- **Slide Title:** Learning an End-to-End Non-Linear Mapping
- **Key Concepts:**
  - Introduced by Dong et al. (ECCV 2014, TPAMI 2015), SRCNN reformulates the classic sparse-coding super-resolution pipeline into a deep convolutional neural network.
  - Three distinct functional stages:
    1. **Patch Extraction & Representation (Conv 1):** Extracts 64-dimensional feature maps from overlapping 9x9 patches.
    2. **Non-linear Mapping (Conv 2):** Projects 64 feature maps to 32 dimensions via 1x1 convolutions.
    3. **Reconstruction (Conv 3):** Reconstructs the 3-channel RGB image through 5x5 filters.
  - The network is fully convolutional, accepting arbitrary image dimensions at inference time.

---

## Slide 4: Architectural Blueprint & Formulas
- **Slide Title:** Three Convolutional Stages

```
   Low-Resolution Input (H x W x 3)
                 │
           Bicubic x2.0
                 │
                 ▼
   ┌───────────────────────────┐
   │ Conv 1: 9x9, 64 filters   │  F₁(Y) = max(0, W₁ * Y + B₁)
   │ ReLU Activation           │  (Patch Extraction)
   └─────────────┬─────────────┘
                 │
                 ▼
   ┌───────────────────────────┐
   │ Conv 2: 1x1, 32 filters   │  F₂(Y) = max(0, W₂ * F₁ + B₂)
   │ ReLU Activation           │  (Non-Linear Mapping)
   └─────────────┬─────────────┘
                 │
                 ▼
   ┌───────────────────────────┐
   │ Conv 3: 5x5, 3 filters    │  F(Y) = W₃ * F₂ + B₃
   │ Linear Output             │  (Reconstruction)
   └─────────────┬─────────────┘
                 │
                 ▼
   High-Resolution Output (H x W x 3)
```

- **Our Refinement (Residual SRCNN):**
  $$\hat{Y} = X_{\text{bicubic}} + F_{\text{residual}}(X_{\text{bicubic}})$$
  The network only needs to learn the missing high-frequency residual difference rather than replicating the entire low-frequency background.

---

## Slide 5: System Pipeline & Data Protocol
- **Slide Title:** Dataset Hygiene & Training Protocol
- **Dataset Setup:**
  - 100 high-resolution photographic crops (\(224 \times 224\)) partitioned **strictly by source photograph**:
    - **60 Train / 20 Validation / 20 Test**
    - Zero data leakage: validation/test photos never appear in training.
- **Training Strategy:**
  - Input degradation: Downsample 2x with bilinear filter \(\to\) Upscale back with bicubic interpolation.
  - Uniform sampling of \(48 \times 48\) patches with random horizontal/vertical flip augmentations.
  - Loss function: Mean Squared Error (MSE).
  - Optimizer: Adam (\(\alpha = 10^{-3}, \beta_1 = 0.9, \beta_2 = 0.999\)).
  - He (Kaiming) weight initialization.

---

## Slide 6: Verification & Hand-Written Backpropagation
- **Slide Title:** First-Principles Engineering & Numerical Gradient Check
- **Talking Points:**
  - Convolution backward pass implemented analytically via transposed flipped kernels and matrix multiplication (`im2col`).
  - Tested using **finite differences**:
    $$\frac{\partial \mathcal{L}}{\partial W_{ij}} \approx \frac{\mathcal{L}(W_{ij} + \epsilon) - \mathcal{L}(W_{ij} - \epsilon)}{2\epsilon}$$
  - **Result:** Maximum relative error was **\(5.26 \times 10^{-10}\)** across all weights and biases, proving the mathematical correctness of our hand-written backpropagation.

---

## Slide 7: Measured Benchmark Results
- **Slide Title:** Quantitative Experimental Comparison (40 Epochs)

| Method / Configuration | Max Val PSNR | Test Loss (MSE) | Test RGB PSNR | Test Y-PSNR |
| :--- | :---: | :---: | :---: | :---: |
| **Residual SRCNN (Our Refinement)** | **31.785 dB** | **0.000432** | **37.848 dB** | **38.332 dB** |
| Bicubic Interpolation (Baseline) | 29.866 dB | 0.000734 | 36.857 dB | 37.105 dB |
| Basic SRCNN (lr=0.001) | 30.943 dB | 0.000538 | 36.567 dB | 37.289 dB |
| Bilinear Interpolation | 28.901 dB | 0.000937 | 35.351 dB | 35.612 dB |
| Nearest Neighbor | 28.833 dB | 0.000946 | 34.432 dB | 34.708 dB |
| SRCNN + Tanh | 28.717 dB | 0.000848 | 33.027 dB | 33.415 dB |
| SRCNN + BatchNorm | 27.197 dB | 0.001725 | 27.733 dB | 28.110 dB |

- **Key Takeaways:**
  1. Residual SRCNN significantly outperforms Bicubic (**+0.991 dB RGB, +1.227 dB Y-Channel**).
  2. Batch Normalization harms super-resolution by destroying absolute pixel luminance distributions.
  3. Tanh causes gradient saturation and restricts dynamic range.

---

## Slide 8: Qualitative Visual Comparisons & Detail Zoom
- **Slide Title:** Visual Quality & Edge Reconstruction
- **Visual Highlights:**
  - On photographic textures (such as `test_007_camera.png` through `test_013_camera.png`), Residual SRCNN delivers gains up to **+2.70 dB**.
  - 3x detail zoom reveals sharp, crisp edge transitions in SRCNN outputs where bicubic interpolation displays blurred ringing artifacts.

---

## Slide 9: Project Deliverables & Demonstration
- **Slide Title:** Engineering Tooling & Live Demo
- **Software Deliverables:**
  1. **Core Library (`srcnn/`):** Pure NumPy engine + Maintainable PyTorch module with cross-checkpoint export.
  2. **Automated Testing (`tests/`):** 20 comprehensive unit tests passing with 100% success rate in `pytest`.
  3. **Interactive Studio (`app.py`):** Streamlit web UI allowing live upload, slider comparison, and instant metric reporting.
  4. **CLI Tools (`scripts/`):** Modular training, evaluation, and dataset generation scripts.

---

## Slide 10: Conclusion & Future Outlook
- **Slide Title:** Key Learnings & Future Enhancements
- **Summary:**
  - Proved that convolutional neural networks can learn to invert bicubic degradation and reconstruct high-frequency image textures.
  - Successfully demonstrated that residual connections drastically improve convergence speed and fidelity in super-resolution.
- **Future Enhancements:**
  - Transition from pre-upsampling to sub-pixel convolution (ESPCN) to reduce computation.
  - Perceptual loss functions (VGG feature loss) and adversarial training (SRGAN) to optimize for human visual perception.

---

## Evaluator Q&A Reference Guide

**Q1: Why did you write backpropagation in pure NumPy instead of using PyTorch?**  
*Answer:* Writing the backpropagation and `im2col` convolution from scratch demonstrates deep mathematical mastery of convolutional calculus, chain rule tensor contractions, and gradient propagation without relying on framework abstraction. Furthermore, to meet industrial standards, we added a complete PyTorch module that shares the exact same weights and architecture.

**Q2: Why does Batch Normalization fail in SRCNN?**  
*Answer:* Batch Normalization subtracts batch mean and divides by variance, shifting feature scale and orientation. While helpful in classification (where scale invariance is desirable), super-resolution is a precise pixel-regression task where absolute intensity and dynamic range must be strictly preserved. Modern super-resolution architectures (like EDSR) remove batch norm layers for this reason.

**Q3: Why evaluate on the Y-channel with border shaving?**  
*Answer:* The human visual system is significantly more sensitive to luminance (brightness) than chrominance (color). Standard literature (Dong et al., 2015) converts RGB to YCbCr and evaluates on the Y channel. Border shaving excludes edge boundary pixels that lack complete receptive fields, preventing border padding artifacts from distorting the metric.
