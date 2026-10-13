# Image Super-Resolution via a Convolutional Neural Network (SRCNN)

UE24CS352A – Machine Learning · Mini-Project

A from-scratch re-implementation of **SRCNN** (Dong et al., 2015) following the CS229 report
*"Image Super-Resolution Via a Convolutional Neural Network"* (Garber, Grossman, Johnson-Yu).
The network, back-propagation and Adam optimiser are written **in pure NumPy** (no PyTorch/TensorFlow, no GPU),
so the project runs on any laptop with Python 3.9+.

```
Low-res image (112x112) -> bicubic x2 -> [9x9 conv, 64] ReLU -> [1x1 conv, 32] ReLU -> [5x5 conv, 3] -> High-res (224x224)
                                          patch extraction        non-linear mapping      reconstruction
```
**Team Details**

- **Team Member 1:** Ganesh Aroda C
- **SRN:** PES1UG24AM102
- **Team Member 2:** Hanshal
- **SRN:** PES1UG24AM110


## 1. Setup (2 minutes)

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Dataset  (already included in `data/`)

`data/{train,val,test}/*.png` – 100 high-resolution 224x224 images, split **60 / 20 / 20**, exactly like the paper's
protocol.  Low-resolution inputs are generated on the fly: image -> 112x112 (bilinear) -> 224x224 (bicubic).

* The included images are crops of the real photographs bundled with scikit-image. The split is done **per source
  photo**, so validation/test images never overlap with training content. (Rebuild with `python scripts/make_dataset.py`.)
* **To use the real DIV2K data** used in the paper: download `DIV2K_train_HR.zip` from
  <https://data.vision.ee.ethz.ch/cvl/DIV2K/>, unzip, then run
  `python scripts/prepare_div2k.py --src path/to/DIV2K_train_HR` – it overwrites `data/` with 100 DIV2K images
  (centre-crop 800x800 -> 224x224, 60/20/20). Every other command below stays the same.

## 3. Run

| Goal | Command | Time (1 CPU core) |
|---|---|---|
| Check back-propagation is correct (numerical gradient test) | `python tests/test_gradients.py` | seconds |
| Train one model and print test PSNR | `python scripts/train.py --epochs 30 --lr 0.001` | ~3 min |
| Quick end-to-end smoke test of everything | `python scripts/run_experiments.py --quick && python scripts/make_report.py --res results_quick` | ~3 min |
| Full study (baselines + lr grid + tanh + batch-norm + regularisation) | `python scripts/run_experiments.py` | ~35 min |
| Tables + figures from the results | `python scripts/make_report.py` | seconds |
| **Live demo** on any image | `python scripts/demo.py --input path/to/photo.jpg` | seconds |

`results/` already contains the outputs of the full study that produced the write-up (trained weights `*.npz`,
per-run histories `*.json`, `results_table.md/csv`, and the `fig_*.png` figures).

### Demo examples
```bash
python scripts/demo.py --input data/test/test_000_retina.png        # prints bicubic vs SRCNN PSNR, writes results/*_comparison.png
python scripts/demo.py --input my_small_photo.png --no_degrade      # 2x enlarge a real low-res image
```

## 4. Results obtained (full study, 40 epochs, 1 CPU core)

| Experiment | Max val PSNR (dB) | Test PSNR (dB) |
|---|---|---|
| Basic SRCNN (lr 0.001) | 30.943 | 36.567 |
| + Regularization (1e-4) | 30.881 | 36.019 |
| + Tanh | 28.717 | 33.027 |
| + Batch Norm | 27.197 | 27.733 |
| **Residual SRCNN (our refinement)** | **31.785** | **37.848** |
| Bicubic | 29.866 | 36.857 |
| Bilinear | 28.901 | 35.351 |
| Nearest | 28.833 | 34.432 |

Honest notes: the plain SRCNN beats bicubic on validation (+1.1 dB) but not on the (very smooth) test photos
(-0.3 dB); the residual variant, added afterwards, fixes this. Single seed, 20 test images - differences of a few
tenths of a dB are not statistically established. Numbers are not comparable with the paper's DIV2K numbers.

## 5. Project layout

```
srcnn/core.py            SRCNN layers, forward/backward passes, BatchNorm, Adam (NumPy)
srcnn/data.py            image loading, LR generation, PSNR
srcnn/trainer.py         patch sampler, training loop, evaluation
scripts/make_dataset.py  builds data/ ;  prepare_div2k.py  builds data/ from real DIV2K
scripts/train.py         train one configuration
scripts/run_experiments.py  all experiments of the study
scripts/make_report.py   tables & figures
scripts/demo.py          live demonstration
tests/test_gradients.py  finite-difference check of the hand-written backprop
docs/                    one-page write-up (PDF) and presentation slides
```

## 6. Implementation notes
* Loss = MSE, metric = PSNR (`10·log10(1/MSE)` for images in [0,1]); optimiser = Adam; He initialisation.
* Training uses random 48x48 patches (flips augmentation); the network is fully convolutional so validation/test run
  on whole 224x224 images. The weights of the best validation epoch are kept.
* Variants studied: learning rate {1e-2, 1e-3, 1e-4}, tanh output activation, batch-norm after conv1/conv2,
  L2 regularisation {1e-2 … 1e-5}, and an optional `--residual` refinement (network output + bicubic input).
* Everything is seeded, so results are reproducible.
