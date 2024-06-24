# SelfReg-UNet: Self-Regularized UNet for Medical Image Segmentation (MICCAI 2024) [![paper](https://img.shields.io/badge/arXiv-Paper-brightgreen)](https://arxiv.org/abs/2406.14896)

Paper link (preprint): [https://arxiv.org/abs/2406.14896]

## News :fire:
- **June 17, 2024:** Congratulations ! Paper has been accepted by MICCAI 2024 !

<img align="right" width="50%" height="100%" src="https://github.com/ChongQingNoSubway/SelfReg-UNet/blob/main/MICCAI2024/findings.jpg">

> **Abstract.**  Since its introduction, UNet has been leading a variety of medical image segmentation tasks. Although numerous follow-up studies have also been dedicated to improving the performance of standard UNet, few have conducted in-depth analyses of the underlying interest pattern of UNet in medical image segmentation. In this paper, we explore the patterns learned in a UNet and observe two important factors that potentially affect its performance: (i) irrelative feature learned caused by asymmetric supervision; (ii) feature redundancy in the feature map. To this end, we propose to balance the supervision between encoder and decoder and reduce the redundant information in the UNet. Specifically, we use the feature map that contains the most semantic information (i.e., the last layer of the decoder) to provide additional supervision to other blocks to provide additional supervision and reduce feature redundancy by leveraging feature distillation. The proposed method can be easily integrated into existing UNet architecture in a plug-and-play fashion with negligible computational cost. The experimental results suggest that the proposed method consistently improves the performance of standard UNets on four medical image segmentation datasets.

> **Method.**  Demostrating the operation based on feature for (a) semantic consistency regularization and  (b) internal feature distillation.
<img src="https://github.com/ChongQingNoSubway/SelfReg-UNet/blob/main/MICCAI2024/featDist.jpg">


## Thanks for the code provided by:
- HiFormer: https://github.com/amirhossein-kz/hiformer
- CASCADE: https://github.com/SLDGroup/CASCADE
- UCTransNet: https://github.com/mcgregorwwww/uctransnet
