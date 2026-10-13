# SRCNN Architectural & Mathematical Deep-Dive

This document details the mathematical derivations, algorithmic implementations, and architectural specifications of the **SRCNN (Super-Resolution Convolutional Neural Network)** project.

---

## 1. Network Topology

```
                  Input: Y_bicubic ∈ ℝ^(H × W × 3)
                                 │
                   ┌─────────────▼─────────────┐
                   │  Conv 1: 9×9, 64 filters   │
                   │  Pad = 4, Stride = 1      │
                   │  Activation: ReLU         │
                   └─────────────┬─────────────┘
                                 │
                      h₁ ∈ ℝ^(H × W × 64)
                                 │
                   ┌─────────────▼─────────────┐
                   │  Conv 2: 1×1, 32 filters   │
                   │  Pad = 0, Stride = 1      │
                   │  Activation: ReLU         │
                   └─────────────┬─────────────┘
                                 │
                      h₂ ∈ ℝ^(H × W × 32)
                                 │
                   ┌─────────────▼─────────────┐
                   │  Conv 3: 5×5, 3 filters   │
                   │  Pad = 2, Stride = 1      │
                   │  Activation: Identity     │
                   └─────────────┬─────────────┘
                                 │
                      h₃ ∈ ℝ^(H × W × 3)
                                 │
                   ┌─────────────▼─────────────┐
                   │ Residual Addition (opt)   │ ── h₃ + Y_bicubic
                   └─────────────┬─────────────┘
                                 │
                 Output: Ŷ ∈ [0, 1]^(H × W × 3)
```

| Layer | Kernel Size \(k \times k\) | Input Channels | Output Channels | Padding | Output Activation | Functional Role |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Conv 1** | \(9 \times 9\) | 3 | 64 | 4 ("same") | ReLU | Patch Extraction & Feature Representation |
| **Conv 2** | \(1 \times 1\) | 64 | 32 | 0 ("same") | ReLU | Non-linear Representation Mapping |
| **Conv 3** | \(5 \times 5\) | 32 | 3 | 2 ("same") | Identity | Spatial Reconstruction & Pixel Aggregation |

---

## 2. Forward Propagation via `im2col`

For spatial convolution without loops, each image is unfolded using `im2col` via NumPy stride tricks (`sliding_window_view`).

Given input tensor \(X \in \mathbb{R}^{B \times H \times W \times C_{\text{in}}}\) and kernel \(W \in \mathbb{R}^{C_{\text{in}} \times k \times k \times C_{\text{out}}}\):
1. **Zero Padding:** Pad spatial dimensions by \(p = (k - 1) / 2\).
2. **Column Extraction:** Unfold into matrix \(\text{Cols} \in \mathbb{R}^{(B \cdot H \cdot W) \times (C_{\text{in}} \cdot k \cdot k)}\).
3. **Matrix Multiplication:**
   $$\text{Out}_{\text{flat}} = \text{Cols} \cdot W_{\text{flat}} + b^T$$
   where \(W_{\text{flat}} \in \mathbb{R}^{(C_{\text{in}} \cdot k \cdot k) \times C_{\text{out}}}\) and \(b \in \mathbb{R}^{C_{\text{out}}}\).
4. **Reshape:** Reshape back to \(\mathbb{R}^{B \times H \times W \times C_{\text{out}}}\).

---

## 3. Analytical Backpropagation Derivations

Let \(\mathcal{L}\) denote the scalar Mean Squared Error loss. Given downstream gradient \(\delta_{\text{out}} = \frac{\partial \mathcal{L}}{\partial \text{Out}} \in \mathbb{R}^{B \times H \times W \times C_{\text{out}}}\):

### 3.1 Bias Gradient
$$\frac{\partial \mathcal{L}}{\partial b} = \sum_{b=1}^B \sum_{h=1}^H \sum_{w=1}^W \delta_{\text{out}}(b, h, w, :)$$

### 3.2 Weight Gradient
$$\frac{\partial \mathcal{L}}{\partial W_{\text{flat}}} = \text{Cols}^T \cdot \delta_{\text{flat}}$$
Reshaping back yields \(\frac{\partial \mathcal{L}}{\partial W} \in \mathbb{R}^{C_{\text{in}} \times k \times k \times C_{\text{out}}}\).

### 3.3 Input Gradient
To propagate gradients to earlier layers, the input gradient \(\delta_{\text{in}} = \frac{\partial \mathcal{L}}{\partial X}\) is computed by full correlation of \(\delta_{\text{out}}\) with the spatially flipped, channel-transposed kernel:
$$W_{\text{flip}} = W[:, ::-1, ::-1, :].\text{transpose}(3, 1, 2, 0) \in \mathbb{R}^{C_{\text{out}} \times k \times k \times C_{\text{in}}}$$
$$\delta_{\text{in}} = \text{conv\_forward}(\delta_{\text{out}}, W_{\text{flip}}, \text{pad} = k - 1 - p)$$

---

## 4. Parameter Optimization: Adam Formulation

Parameters are updated using decoupled adaptive moments:
1. First moment vector (momentum):
   $$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
2. Second moment vector (uncentered variance):
   $$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
3. Bias-corrected moment estimates:
   $$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
4. Parameter update with weight decay \(\lambda\):
   $$g_t \leftarrow g_t + \lambda \theta_{t-1} \quad (\text{conv weights only})$$
   $$\theta_t = \theta_{t-1} - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

Default hyperparameters: \(\alpha = 10^{-3}, \beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-8}\).

---

## 5. Weight Initialization

To prevent vanishing or exploding activations across deep layers with ReLU non-linearities, He (Kaiming) normal initialization is applied:
$$\text{fan\_in}_i = C_{\text{in}, i} \cdot k_i \cdot k_i$$
$$\sigma_i = \sqrt{\frac{2}{\text{fan\_in}_i}}$$
For the final reconstruction layer (Conv 3), a small standard deviation \(\sigma_3 = 0.01\) is chosen to initialize near-zero reconstruction offsets.

---

## 6. PyTorch & NumPy Checkpoint Interoperability

The repository provides seamless bidirectional weight translation between pure NumPy and PyTorch:

| Tensor Format | Dimension Ordering | Conv1 Weights Shape |
| :--- | :--- | :--- |
| **NumPy (`core.py`)** | \((C_{\text{in}}, k, k, C_{\text{out}})\) | `(3, 9, 9, 64)` |
| **PyTorch (`nn.Conv2d`)** | \((C_{\text{out}}, C_{\text{in}}, k, k)\) | `(64, 3, 9, 9)` |

**Mapping Formulas:**
- NumPy to PyTorch: `W_torch = W_numpy.transpose(3, 0, 1, 2)`
- PyTorch to NumPy: `W_numpy = W_torch.transpose(1, 2, 3, 0)`

Biases share identical 1D vector shapes `(C_out,)` in both formats.
