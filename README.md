# SelfReg-UNet: Self-Regularized UNet for Medical Image Segmentation (MICCAI 2024) [![paper](https://img.shields.io/badge/arXiv-Paper-brightgreen)](https://arxiv.org/abs/2405.03140)

Paper link (preprint): [https://arxiv.org/abs/2405.03140]

## News :fire:
- **May 9, 2024:** Congratulations ! Paper has been accepted by MICCAI 2024 !

<img align="right" width="50%" height="100%" src="https://github.com/xiwenc1/TimeMIL/blob/main/Figs/intro2_1.jpg">

> **Abstract.** Deep neural networks, including transformers and convolutional neural networks, have significantly improved multivariate time series classification (MTSC). However, these methods often rely on supervised learning, which does not fully account for the sparsity and locality of patterns in time series data (e.g., diseases-related anomalous points in ECG). To address this challenge, we formally reformulate MTSC as a weakly supervised problem, introducing a novel multiple-instance learning (MIL) framework for better localization of patterns of interest and modeling time dependencies within time series. Our novel approach, TimeMIL, formulates the temporal correlation and ordering within a time-aware MIL pooling, leveraging a tokenized transformer with a specialized learnable wavelet positional token. The proposed method surpassed 26 recent state-of-the-art methods, underscoring the effectiveness of the weakly supervised TimeMIL in MTSC. 

> **Method.**  Demostrating the operation based on feature for (a) semantic consistency regularization and  (b) internal feature distillation..
<img src="https://github.com/xiwenc1/TimeMIL/blob/main/Figs/network_v2.jpg">


## Thanks for the code provided by:
- Todynet:https://github.com/liuxz1011/TodyNet

