# Single-Image Super-Resolution using Deep Convolutional Neural Networks (SRCNN)

**Academic Course:** UE24CS352A – Machine Learning Mini-Project  
**Team Details:**  
- **Team Member 1:** Ganesh Aroda C (SRN: PES1UG24AM102)  
- **Team Member 2:** Hanshal (SRN: PES1UG24AM110)  
**Repository:** `Image-Super-Resolution-SRCNN`  

---

## 1. Abstract

Single-Image Super-Resolution (SISR) aims to estimate a high-resolution (HR) image from a low-resolution (LR) observation. This project implements and evaluates the foundational Super-Resolution Convolutional Neural Network (SRCNN) introduced by Dong et al. (2015), adhering to the educational and rigorous experimental design of the Stanford CS229 report (*Garber, Grossman, Johnson-Yu*). The core neural network, forward propagation, im2col matrix convolution, analytical backpropagation, and Adam optimizer were implemented **from scratch in pure NumPy** without reliance on automatic differentiation or external deep learning frameworks. To bridge educational foundational theory with modern industrial deep learning practices, a parallel PyTorch backend is provided with bidirectional checkpoint interoperability. 

The models were evaluated across multiple architectural variations, including learning rate grids (\(10^{-2}, 10^{-3}, 10^{-4}\)), batch normalization, tanh activations, \(L_2\) regularization sweeps, and an innovative residual learning refinement (\(\hat{Y} = X_{\text{bicubic}} + \text{SRCNN}(X_{\text{bicubic}})\)). On the test evaluation split, the Residual SRCNN achieves **37.848 dB RGB PSNR (0.9451 SSIM)** and **38.332 dB Luminance Y-channel PSNR (0.9479 SSIM)**, outperforming standard bicubic interpolation by **+0.991 dB RGB (+1.227 dB Y-channel)** and achieving up to **+2.70 dB gains** on photographic detail images.

---

## 2. Problem Statement

Digital images frequently suffer from resolution loss due to hardware sensor physical limits, optical constraints, compression artifacts, and network bandwidth restrictions. In medical imaging, satellite telemetry, surveillance, and display technology, recovering high-frequency spatial details (sharp edges, textural nuances, fine contours) is critical. 

Mathematically, single-image super-resolution is a severely ill-posed inverse problem: for every low-resolution pixel patch, multiple plausible high-resolution configurations could have produced that degradation. Classical polynomial interpolation techniques (nearest neighbor, bilinear, bicubic) rely on local smoothing kernels. While computationally lightweight, they fail to synthesize high-frequency image textures, causing noticeable blurriness, ringing artifacts, and jagged edges.

---

## 3. Objectives

1. **Foundational Understanding:** Formulate and implement the complete 3-stage convolutional pipeline from first mathematical principles using pure Python and NumPy (`im2col`, analytical chain-rule convolution gradients, He initialization, and adaptive moment estimation).
2. **Empirical Verification:** Verify analytical backward gradients against finite-difference numerical approximations to ensure rigorous correctness.
3. **Rigorous Experimental Study:** Conduct systematic ablation experiments on learning rates, batch normalization, output activation constraints, weight decay regularization, and residual skip connections.
4. **Standardized Image Quality Assessment:** Compute both Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index Measure (SSIM) across RGB channels and luminance (Y-channel with boundary shaving), adhering to standard computer vision benchmarks.
5. **Modern Engineering Parity:** Equip the project with a maintained PyTorch implementation, automated `pytest` test suites, CLI tools, and an interactive Streamlit demonstration application suitable for academic review.

---

## 4. Introduction to Image Super-Resolution

Let \(Y\) denote the ground-truth high-resolution image and \(X\) denote the observed low-resolution image. The degradation process is modeled as:
$$X = (Y \otimes G) \downarrow_s + \eta$$
where \(G\) is a blur filter, \(\downarrow_s\) denotes downsampling by scale factor \(s\), and \(\eta\) represents additive noise.

The objective of SISR is to find a mapping function \(F\) such that the reconstructed image \(\hat{Y} = F(X)\) minimizes the perceptual and pixel-level distance to \(Y\).

Prior to deep learning, dictionary-based sparse coding methods (Yang et al.) represented patches as sparse linear combinations over overcomplete low- and high-resolution dictionaries. Dong et al. (2015) made the landmark breakthrough by showing that the sparse-coding pipeline is mathematically equivalent to a deep convolutional neural network.

---

## 5. SRCNN Architecture

SRCNN accepts a bicubic-upsampled low-resolution image \(Y_{\text{bicubic}} \in \mathbb{R}^{H \times W \times C}\) matching the spatial dimensions of target \(Y\). The network consists of three sequential convolutional operations:

```
Input (H x W x 3)
      │
┌─────▼────────────────────────────────────────┐
│ Conv1: 9x9, 64 filters, stride 1, pad 4      │ ── Patch Extraction & Representation
│ Activation: ReLU                             │
└─────┬────────────────────────────────────────┘
      │ (H x W x 64)
┌─────▼────────────────────────────────────────┐
│ Conv2: 1x1, 32 filters, stride 1, pad 0      │ ── Non-linear Mapping
│ Activation: ReLU                             │
└─────┬────────────────────────────────────────┘
      │ (H x W x 32)
┌─────▼────────────────────────────────────────┐
│ Conv3: 5x5, 3 filters, stride 1, pad 2       │ ── Reconstruction
│ Linear (or Tanh)                             │
└─────┬────────────────────────────────────────┘
      │
Output (H x W x 3)
```

### 5.1 Layer Formulations

1. **Patch Extraction & Representation (Conv 1):**
   $$F_1(Y) = \max(0, W_1 \ast Y + B_1)$$
   - Filters: \(n_1 = 64\), Kernel: \(c \times f_1 \times f_1 = 3 \times 9 \times 9\).
   - Extracts overlapping feature representations across the image receptive field.

2. **Non-linear Mapping (Conv 2):**
   $$F_2(Y) = \max(0, W_2 \ast F_1(Y) + B_2)$$
   - Filters: \(n_2 = 32\), Kernel: \(n_1 \times f_2 \times f_2 = 64 \times 1 \times 1\).
   - Maps high-dimensional feature vectors non-linearly to another patch space.

3. **Reconstruction (Conv 3):**
   $$F(Y) = W_3 \ast F_2(Y) + B_3$$
   - Filters: \(c = 3\), Kernel: \(n_2 \times f_3 \times f_3 = 32 \times 5 \times 5\).
   - Aggregates patch representations to produce the reconstructed image.

### 5.2 Residual SRCNN Refinement

In classical SRCNN, the network must predict all low-frequency structures from scratch. In our **Residual SRCNN** variant, the network learns only the high-frequency residual correction:
$$\hat{Y} = Y_{\text{bicubic}} + F_{\text{residual}}(Y_{\text{bicubic}})$$
Because \(Y_{\text{bicubic}}\) already preserves low-frequency layout and color, the network capacity is dedicated entirely to sharpening high-frequency transitions and fine textures.

---

## 6. System Architecture and Pipeline

```mermaid
flowchart LR
    A["Raw HR Image (Y)"] --> B["Bilinear Downsample (x0.5)"]
    B --> C["Bicubic Upsample (x2.0)"]
    C --> D["Aligned Patches (48x48)"]
    A --> D
    D --> E["Data Augmentation (Flips)"]
    E --> F["SRCNN Forward Pass"]
    F --> G["MSE Loss Computation"]
    G --> H["Analytical Backpropagation"]
    H --> I["Adam Parameter Update"]
    I --> J["Best Checkpoint Selection"]
    J --> K["Full Image Inference"]
    K --> L["Evaluation (PSNR / SSIM)"]
```

---

## 7. Dataset and Preprocessing

- **Bundled Dataset:** 100 high-resolution photograph crops of size \(224 \times 224 \times 3\).
- **Source Photos:** Diverse photographic scenes bundled with `scikit-image` (`astronaut`, `coffee`, `rocket`, `hubble_deep_field`, `brick`, `gravel`, `grass`, `moon`, `chelsea`, `immunohistochemistry`, `retina`, `camera`, `cell`).
- **Strict Data Segregation:** Partitioned **by source photograph** to prevent visual leakage:
  - **Training:** 60 images (`astronaut`, `coffee`, `rocket`, `hubble`, `brick`, `gravel`, `grass`, `moon`)
  - **Validation:** 20 images (`chelsea`, `immunohistochemistry`)
  - **Testing:** 20 images (`retina`, `camera`, `cell`)
- **Patch Extraction & Augmentation:** During training, \(48 \times 48\) patches are sampled uniformly with random horizontal and vertical flips. Because the model is fully convolutional, validation and testing run on arbitrary full-resolution images.
- **DIV2K & Set5 Support:** A dedicated preparation script (`scripts/prepare_div2k.py`) and dataset evaluator (`scripts/evaluate_dataset.py`) support evaluation on standard benchmarks (Set5, Set14, DIV2K).

---

## 8. Training Methodology

### 8.1 Optimization
- **Loss Function:** Mean Squared Error (MSE):
  $$\mathcal{L}(\Theta) = \frac{1}{2n} \sum_{i=1}^n \|\hat{Y}_i - Y_i\|_2^2$$
- **Optimizer:** Adam (Adaptive Moment Estimation):
  $$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t, \quad v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
  $$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}, \quad \theta_t = \theta_{t-1} - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$
- **Hyperparameters:** Batch size = 16, patch size = \(48 \times 48\), 160–320 patches/epoch, epochs = 30–40.
- **Weight Initialization:** He (Kaiming) normal initialization for layers 1 and 2 (\(\sigma = \sqrt{2 / \text{fan\_in}}\)), small Gaussian (\(\sigma = 0.01\)) for layer 3.

---

## 9. Evaluation Metrics and Formulas

### 9.1 Peak Signal-to-Noise Ratio (PSNR)
$$\text{MSE} = \frac{1}{MN} \sum_{i=0}^{M-1} \sum_{j=0}^{N-1} [I(i, j) - K(i, j)]^2$$
$$\text{PSNR} = 10 \cdot \log_{10} \left( \frac{\text{MAX}_I^2}{\text{MSE}} \right)$$
For images normalized to \([0, 1]\), \(\text{MAX}_I = 1.0\).

### 9.2 Structural Similarity Index Measure (SSIM)
$$\text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + c_1)(2\sigma_{xy} + c_2)}{(\mu_x^2 + \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}$$
where \(\mu\) and \(\sigma\) represent local mean and variance, and \(c_1 = (k_1 L)^2, c_2 = (k_2 L)^2\).

### 9.3 Luminance (Y-Channel) & Border Shaving
Adhering to standard super-resolution literature (Dong et al., 2015), metrics are computed on the luminance Y channel (\(Y = 0.299R + 0.587G + 0.114B\)) with boundary pixels shaved by scale factor \(s\) to eliminate border convolution artifacts.

---

## 10. Experimental Results

The complete study evaluated 11 network and optimization variations trained for 40 epochs.

### 10.1 Comparative Benchmark Table

| Model / Configuration | Max Val PSNR (dB) | Test Loss (MSE) | Test RGB PSNR (dB) | Test RGB SSIM | Test Y-PSNR (dB) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Residual SRCNN (lr=0.001)** | **31.785** | **0.000432** | **37.848** | **0.9451** | **38.332** |
| Bicubic Baseline | 29.866 | 0.000734 | 36.857 | 0.9286 | 37.105 |
| Basic SRCNN (lr=0.001) | 30.943 | 0.000538 | 36.567 | 0.9388 | 37.289 |
| + Regularization (\(\lambda=10^{-4}\)) | 30.881 | 0.000585 | 36.019 | 0.9312 | 36.751 |
| Bilinear Interpolation | 28.901 | 0.000937 | 35.351 | 0.9082 | 35.612 |
| Nearest Neighbor | 28.833 | 0.000946 | 34.432 | 0.8914 | 34.708 |
| Basic SRCNN + Tanh | 28.717 | 0.000848 | 33.027 | 0.8953 | 33.415 |
| Basic SRCNN + BatchNorm | 27.197 | 0.001725 | 27.733 | 0.8240 | 28.110 |

---

## 11. Results Analysis and Discussion

1. **Superiority of Residual Learning:**
   Residual SRCNN achieved the highest test performance (**37.848 dB RGB / 38.332 dB Y-channel**), improving over Bicubic by **+0.991 dB RGB** and **+1.227 dB Y-channel**. By learning residual high-frequency corrections, gradient flow is direct and optimization converges faster.
2. **Per-Image Detail Breakdown:**
   On real photographic textures (e.g. `camera` series), Residual SRCNN achieves gains up to **+2.701 dB**:
   - `test_007_camera.png`: Bicubic 27.982 dB \(\to\) SRCNN 30.523 dB (**+2.541 dB**)
   - `test_012_camera.png`: Bicubic 26.555 dB \(\to\) SRCNN 29.255 dB (**+2.701 dB**)
3. **Failure of Batch Normalization in SR:**
   Adding BatchNorm drastically reduced performance (27.733 dB test PSNR). As established in modern super-resolution literature (EDSR, Lim et al.), batch normalization normalizes pixel scale and destroys absolute brightness statistics, which is detrimental to pixel-exact regression tasks.
4. **Impact of Tanh Activation:**
   Constraining the output layer with a scaled Tanh activation reduced test PSNR to 33.027 dB due to gradient saturation near boundaries \([0, 1]\). Linear output convolution is optimal for MSE regression.

---

## 12. Limitations and Future Work

### Limitations
- Evaluated on a compact 100-image dataset (due to mini-project scope and pure CPU execution constraints).
- Standard SRCNN operates on bicubic-upsampled inputs, incurring \(O(s^2)\) computational complexity relative to sub-pixel convolution networks (ESPCN / FSRCNN).

### Future Enhancements
- Integration of Sub-Pixel Convolution (pixel shuffle) to perform feature extraction in low-resolution space.
- Training on full DIV2K (800 HR images) with larger epoch budgets using GPU acceleration.
- Perceptual Loss (VGG feature loss) and Generative Adversarial Networks (SRGAN) to improve human perceptual score.

---

## 13. References

1. Dong, C., Loy, C. C., He, K., & Tang, X. (2015). *Image Super-Resolution Using Deep Convolutional Networks*. IEEE Transactions on Pattern Analysis and Machine Intelligence, 38(2), 295–307.
2. Garber, A., Grossman, T., & Johnson-Yu, S. (2015). *Image Super-Resolution Via a Convolutional Neural Network*. Stanford CS229 Machine Learning Project Report.
3. Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). *Image quality assessment: from error visibility to structural similarity*. IEEE Transactions on Image Processing, 13(4), 600–612.
4. Lim, B., Son, S., Kim, H., Nah, S., & Lee, K. M. (2017). *Enhanced Deep Residual Networks for Single-Image Super-Resolution*. CVPR Workshops.
