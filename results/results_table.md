| Experiment | Max Val PSNR (dB) | Test Loss (MSE) | Test PSNR (dB) |
|---|---|---|---|
| Basic SRCNN (lr 0.001) | 30.943 | 0.000538 | 36.567 |
| + Regularization (wd 1e-4) | 30.881 | 0.000585 | 36.019 |
| + Tanh | 28.717 | 0.000848 | 33.027 |
| + Batch Norm | 27.197 | 0.001725 | 27.733 |
| Residual SRCNN (refinement) | 31.785 | 0.000432 | 37.848 |
| Bicubic | 29.866 | 0.000734 | 36.857 |
| Bilinear | 28.901 | 0.000937 | 35.351 |
| Nearest | 28.833 | 0.000946 | 34.432 |
