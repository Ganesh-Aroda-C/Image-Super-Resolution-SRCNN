"""Streamlit Interactive Demonstration for SRCNN Image Super-Resolution.

Run with:
    streamlit run app.py
"""
import io
import os
import sys
import time
import glob
from typing import Optional, Tuple
import numpy as np
from PIL import Image

try:
    import streamlit as st
except ImportError:
    print("Streamlit is not installed in the current environment.")
    print("To launch the web interface, install it using: pip install streamlit")
    print("Or run the CLI demo instead: python scripts/demo.py --input data/test/test_000_retina.png")
    sys.exit(1)

# Ensure project root is available in path
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import srcnn
from srcnn.core import SRCNN
from srcnn.models_pytorch import SRCNN_PyTorch
from srcnn.metrics import psnr, ssim, compute_metrics, rgb_to_y, shave_borders
from srcnn.data import make_lr


st.set_page_config(
    page_title="SRCNN Image Super-Resolution Studio",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_cached_model(model_path: str, backend: str = "numpy"):
    if not os.path.exists(model_path):
        return None, f"Checkpoint not found at: {model_path}"
    try:
        if backend == "pytorch" or model_path.endswith(".pt"):
            import torch
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if model_path.endswith(".pt"):
                model, _ = SRCNN_PyTorch.load_checkpoint(model_path, device=device)
            else:
                model = SRCNN_PyTorch()
                model.load_from_numpy(model_path)
                model = model.to(device)
            return model, None
        else:
            model = SRCNN.load(model_path)
            return model, None
    except Exception as e:
        return None, f"Failed to load checkpoint: {e}"


def get_available_checkpoints():
    res_dir = os.path.join(ROOT, "results")
    if not os.path.exists(res_dir):
        return []
    ckpts = sorted(glob.glob(os.path.join(res_dir, "*.npz")) + glob.glob(os.path.join(res_dir, "*.pt")))
    return [os.path.relpath(p, ROOT) for p in ckpts]


def main():
    st.title("🔍 SRCNN Image Super-Resolution Studio")
    st.markdown(
        "**UE24CS352A Machine Learning Mini-Project** · "
        "From-scratch re-implementation of *Dong et al. (2015)* & Stanford CS229."
    )

    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    available_ckpts = get_available_checkpoints()
    if not available_ckpts:
        st.sidebar.error("No model checkpoints found in `results/`!")
        st.sidebar.info("Train a model first:\n`python scripts/train.py --epochs 30`")
        selected_ckpt = None
    else:
        # Default to residual_lr_0.001.npz or lr_0.001_no_bn.npz if present
        default_idx = 0
        for i, c in enumerate(available_ckpts):
            if "residual_lr_0.001.npz" in c:
                default_idx = i
                break
            elif "lr_0.001_no_bn.npz" in c:
                default_idx = i

        selected_ckpt = st.sidebar.selectbox(
            "Model Checkpoint",
            available_ckpts,
            index=default_idx,
            help="Select a trained weights checkpoint (.npz from pure NumPy or .pt from PyTorch)"
        )

    backend = st.sidebar.radio(
        "Execution Engine",
        ["NumPy (Pure CPU)", "PyTorch (Accelerated)"],
        index=0,
        help="NumPy uses hand-written convolutions and backprop; PyTorch uses native tensor ops."
    )
    backend_key = "pytorch" if "PyTorch" in backend else "numpy"

    mode = st.sidebar.radio(
        "Evaluation Mode",
        ["Ground Truth Evaluation (Degrade & Restore)", "Direct Super-Resolution (Real Low-Res)"],
        index=0,
        help="Ground Truth mode measures PSNR/SSIM by simulating degradation; Direct mode enlarges without reference."
    )

    scale_factor = st.sidebar.slider("Super-Resolution Scale", min_value=2.0, max_value=4.0, value=2.0, step=1.0)
    eval_channel = st.sidebar.selectbox("Metric Channel", ["Y-Channel (Luminance, Shaved)", "RGB (Full Image)"], index=0)

    # Main content tabs
    tab_demo, tab_architecture, tab_benchmark = st.tabs(["🖼️ Interactive Demo", "📐 Model Architecture", "📊 Benchmark Results"])

    with tab_demo:
        col_input, col_sample = st.columns([2, 1])
        with col_input:
            uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])
        with col_sample:
            st.markdown("**Or pick a bundled test photo:**")
            sample_options = ["None", "data/test/test_000_retina.png", "data/test/test_006_camera.png", "data/test/test_012_cell.png"]
            chosen_sample = st.selectbox("Sample image", sample_options, index=1 if os.path.exists(os.path.join(ROOT, "data/test/test_000_retina.png")) else 0)

        img_to_process = None
        if uploaded_file is not None:
            img_to_process = Image.open(uploaded_file).convert("RGB")
        elif chosen_sample != "None" and os.path.exists(os.path.join(ROOT, chosen_sample)):
            img_to_process = Image.open(os.path.join(ROOT, chosen_sample)).convert("RGB")

        if img_to_process is None:
            st.info("👆 Please upload an image or select a bundled test photo to begin super-resolution.")
            return

        # Model validation
        if not selected_ckpt:
            st.error("Cannot run inference: No model checkpoint selected.")
            return

        model_path = os.path.join(ROOT, selected_ckpt)
        model, err = load_cached_model(model_path, backend=backend_key)
        if err:
            st.error(err)
            st.info("Run `python scripts/train.py --epochs 30 --lr 0.001` in PowerShell to train the baseline model.")
            return

        st.success(f"Loaded `{os.path.basename(selected_ckpt)}` using `{backend_key.upper()}` engine.")

        # Processing logic
        w_raw, h_raw = img_to_process.size
        # Limit max side to prevent huge CPU inference delays in web app
        if max(w_raw, h_raw) > 512:
            scale_down = 512 / max(w_raw, h_raw)
            img_to_process = img_to_process.resize((int(w_raw * scale_down), int(h_raw * scale_down)), Image.LANCZOS)
            st.caption(f"Resized large input to {img_to_process.size} for web responsiveness.")

        w, h = img_to_process.size
        w_adj = int(w // scale_factor * scale_factor)
        h_adj = int(h // scale_factor * scale_factor)
        img_hr = img_to_process.crop((0, 0, w_adj, h_adj))

        if mode.startswith("Ground Truth"):
            # Create low-resolution image
            w_lr = int(w_adj // scale_factor)
            h_lr = int(h_adj // scale_factor)
            img_lr = img_hr.resize((w_lr, h_lr), Image.BILINEAR)
            img_bicubic = img_lr.resize((w_adj, h_adj), Image.BICUBIC)

            x_np = np.asarray(img_bicubic, np.float32)[np.newaxis, ...] / 255.0
            hr_np = np.asarray(img_hr, np.float32)[np.newaxis, ...] / 255.0

            # Run inference
            t0 = time.time()
            sr_np = model.predict(x_np)
            inference_time = (time.time() - t0) * 1000.0  # ms
            sr_img = Image.fromarray((np.clip(sr_np[0], 0, 1) * 255.0).round().astype(np.uint8))

            # Metrics
            shave = int(scale_factor) if "Shaved" in eval_channel else 0
            use_y = "Y-Channel" in eval_channel

            b_psnr, b_ssim = compute_metrics(x_np[0], hr_np[0], shave=shave, use_y_channel=use_y)
            s_psnr, s_ssim = compute_metrics(sr_np[0], hr_np[0], shave=shave, use_y_channel=use_y)
            delta_psnr = s_psnr - b_psnr
            delta_ssim = s_ssim - b_ssim

            # Metrics Cards
            st.subheader("📊 Objective Quality Metrics")
            m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
            m_col1.metric("Bicubic Baseline PSNR", f"{b_psnr:.2f} dB")
            m_col2.metric("SRCNN Reconstructed PSNR", f"{s_psnr:.2f} dB", delta=f"{delta_psnr:+.2f} dB")
            m_col3.metric("Bicubic SSIM", f"{b_ssim:.4f}")
            m_col4.metric("SRCNN SSIM", f"{s_ssim:.4f}", delta=f"{delta_ssim:+.4f}")
            m_col5.metric("Inference Latency", f"{inference_time:.1f} ms")

            # Visual Comparison
            st.subheader("🖼️ Visual Comparison")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.image(img_bicubic, caption=f"Bicubic Interpolation ({b_psnr:.2f} dB)", use_container_width=True)
            with c2:
                st.image(sr_img, caption=f"SRCNN Reconstruction ({s_psnr:.2f} dB)", use_container_width=True)
            with c3:
                st.image(img_hr, caption="Ground Truth (Original HR)", use_container_width=True)

            # Zoom crop comparison
            st.subheader("🔍 Magnified Fine Detail Inspection (3x Center Zoom)")
            cw, ch = min(48, w_adj // 2), min(48, h_adj // 2)
            cx, cy = (w_adj - cw) // 2, (h_adj - ch) // 2
            box = (cx, cy, cx + cw, cy + ch)

            crop_bic = img_bicubic.crop(box).resize((cw * 3, ch * 3), Image.NEAREST)
            crop_sr = sr_img.crop(box).resize((cw * 3, ch * 3), Image.NEAREST)
            crop_gt = img_hr.crop(box).resize((cw * 3, ch * 3), Image.NEAREST)

            z1, z2, z3 = st.columns(3)
            with z1:
                st.image(crop_bic, caption="Bicubic Detail Crop (Blurred)", use_container_width=True)
            with z2:
                st.image(crop_sr, caption="SRCNN Detail Crop (Sharpened)", use_container_width=True)
            with z3:
                st.image(crop_gt, caption="Ground Truth Detail Crop", use_container_width=True)

        else:
            # Direct 2x Super-Resolution mode
            target_w = int(round(w * scale_factor))
            target_h = int(round(h * scale_factor))
            img_bicubic = img_to_process.resize((target_w, target_h), Image.BICUBIC)

            x_np = np.asarray(img_bicubic, np.float32)[np.newaxis, ...] / 255.0
            t0 = time.time()
            sr_np = model.predict(x_np)
            inference_time = (time.time() - t0) * 1000.0

            sr_img = Image.fromarray((np.clip(sr_np[0], 0, 1) * 255.0).round().astype(np.uint8))

            st.subheader(f"🖼️ Direct Super-Resolution ({scale_factor}x Upscale)")
            st.caption(f"Original input {w}x{h} enlarged to {target_w}x{target_h} in {inference_time:.1f} ms.")

            c1, c2 = st.columns(2)
            with c1:
                st.image(img_bicubic, caption="Bicubic Interpolated Baseline", use_container_width=True)
            with c2:
                st.image(sr_img, caption="SRCNN Restored Output", use_container_width=True)

        # Download button
        buf = io.BytesIO()
        sr_img.save(buf, format="PNG")
        byte_im = buf.getvalue()
        st.download_button(
            label="💾 Download Super-Resolved Image",
            data=byte_im,
            file_name="srcnn_super_resolved.png",
            mime="image/png"
        )

    with tab_architecture:
        st.subheader("SRCNN Network Architecture")
        st.markdown("""
        **SRCNN (Dong et al., 2015)** models single-image super-resolution as a three-stage deep convolutional process:

        1. **Patch Extraction & Representation (Conv 1):**
           - Kernel: `9x9`, Input Channels: `3`, Output Filters: `64`, Stride: `1`, Padding: `4` (`same`).
           - Non-linearity: `ReLU(x) = max(0, x)`.
           - Extracts overlapping high-dimensional feature representations from the bicubic input.

        2. **Non-linear Mapping (Conv 2):**
           - Kernel: `1x1`, Input Filters: `64`, Output Filters: `32`, Stride: `1`, Padding: `0`.
           - Non-linearity: `ReLU(x)`.
           - Maps 64-dimensional feature representations non-linearly to 32-dimensional high-resolution patch spaces.

        3. **Reconstruction (Conv 3):**
           - Kernel: `5x5`, Input Filters: `32`, Output Channels: `3`, Stride: `1`, Padding: `2` (`same`).
           - Aggregates patch-wise representations into the final continuous-valued super-resolved image.

        4. **Residual Refinement Extension:**
           $$\\hat{Y} = X_{\\text{bicubic}} + \\text{SRCNN}(X_{\\text{bicubic}})$$
           - Rather than predicting the full image from scratch, the residual network learns the high-frequency correction residual.
        """)

    with tab_benchmark:
        st.subheader("Experimental Results (40 Epochs, Full Grid)")
        table_path = os.path.join(ROOT, "results", "results_table.md")
        if os.path.exists(table_path):
            with open(table_path, "r") as f:
                st.markdown(f.read())
        else:
            st.info("Results table not found. Run `python scripts/make_report.py` to compile.")


if __name__ == "__main__":
    main()
