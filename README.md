# SelfReg-UNet: Self-Regularized UNet for Medical Image Segmentation (MICCAI 2024) [![paper](https://img.shields.io/badge/arXiv-Paper-brightgreen)](https://arxiv.org/abs/2406.14896)

Paper link (preprint): [https://arxiv.org/abs/2406.14896]

## News :fire:
- **June 17, 2024:** Congratulations ! Paper has been accepted by MICCAI 2024 !

<img align="right" width="50%" height="100%" src="https://github.com/ChongQingNoSubway/SelfReg-UNet/blob/main/MICCAI2024/findings.jpg">

> **Abstract.**  Since its introduction, UNet has been leading a variety of medical image segmentation tasks. Although numerous follow-up studies have also been dedicated to improving the performance of standard UNet, few have conducted in-depth analyses of the underlying interest pattern of UNet in medical image segmentation. In this paper, we explore the patterns learned in a UNet and observe two important factors that potentially affect its performance: (i) irrelative feature learned caused by asymmetric supervision; (ii) feature redundancy in the feature map. To this end, we propose to balance the supervision between encoder and decoder and reduce the redundant information in the UNet. Specifically, we use the feature map that contains the most semantic information (i.e., the last layer of the decoder) to provide additional supervision to other blocks to provide additional supervision and reduce feature redundancy by leveraging feature distillation. The proposed method can be easily integrated into existing UNet architecture in a plug-and-play fashion with negligible computational cost. The experimental results suggest that the proposed method consistently improves the performance of standard UNets on four medical image segmentation datasets.

> **Method.**  Demostrating the operation based on feature for (a) semantic consistency regularization and  (b) internal feature distillation.
<img src="https://github.com/ChongQingNoSubway/SelfReg-UNet/blob/main/MICCAI2024/featDist.jpg">


## Key Code
```

class KDloss(nn.Module):

    def __init__(self,lambda_x):
        super(KDloss,self).__init__()
        self.lambda_x = lambda_x

    def inter_fd(self,f_s, f_t):
        s_C, t_C, s_H, t_H = f_s.shape[1], f_t.shape[1], f_s.shape[2], f_t.shape[2]
        if s_H > t_H:
            f_s = F.adaptive_avg_pool2d(f_s, (t_H, t_H))
        elif s_H < t_H:
            f_t = F.adaptive_avg_pool2d(f_t, (s_H, s_H))
        else:
            pass
        
        idx_s = random.sample(range(s_C),min(s_C,t_C))
        idx_t = random.sample(range(t_C),min(s_C,t_C))

        # inter_fd_loss = F.mse_loss(f_s[:, 0:min(s_C,t_C), :, :], f_t[:, 0:min(s_C,t_C), :, :].detach())

        inter_fd_loss = F.mse_loss(f_s[:, idx_s, :, :], f_t[:, idx_t, :, :].detach())
        return inter_fd_loss 
    
    def intra_fd(self,f_s):
        sorted_s, indices_s = torch.sort(F.normalize(f_s, p=2, dim=(2,3)).mean([0, 2, 3]), dim=0, descending=True)
        f_s = torch.index_select(f_s, 1, indices_s)
        intra_fd_loss = F.mse_loss(f_s[:, 0:f_s.shape[1]//2, :, :], f_s[:, f_s.shape[1]//2: f_s.shape[1], :, :])
        return intra_fd_loss
    
    def forward(self,feature,feature_decoder,final_up,epoch):
        f1 = feature[0][-1] # 
        f2 = feature[1][-1]
        f3 = feature[2][-1]
        f4 = feature[3][-1] # lower feature 

        f1_0 = feature[0][0] # 
        f2_0 = feature[1][0]
        f3_0 = feature[2][0]
        f4_0 = feature[3][0] # lower feature 

        f1_d = feature_decoder[0][-1] # 14 x 14
        f2_d = feature_decoder[1][-1] # 28 x 28
        f3_d = feature_decoder[2][-1] # 56 x 56

        f1_d_0 = feature_decoder[0][0] # 14 x 14
        f2_d_0 = feature_decoder[1][0] # 28 x 28
        f3_d_0 = feature_decoder[2][0] # 56 x 56

        #print(f3_d.shape)

        final_layer = final_up


        loss =  (self.intra_fd(f1)+self.intra_fd(f2)+self.intra_fd(f3)+self.intra_fd(f4))/4
        loss += (self.intra_fd(f1_0)+self.intra_fd(f2_0)+self.intra_fd(f3_0)+self.intra_fd(f4_0))/4
        loss += (self.intra_fd(f1_d_0)+self.intra_fd(f2_d_0)+self.intra_fd(f3_d_0))/3
        loss += (self.intra_fd(f1_d)+self.intra_fd(f2_d)+self.intra_fd(f3_d))/3


        loss += (self.inter_fd(f1_d,final_layer)+self.inter_fd(f2_d,final_layer)+self.inter_fd(f3_d,final_layer)
                  +self.inter_fd(f1,final_layer)+self.inter_fd(f2,final_layer)+self.inter_fd(f3,final_layer)+self.inter_fd(f4,final_layer))/7
        
        loss += (self.inter_fd(f1_d_0,final_layer)+self.inter_fd(f2_d_0,final_layer)+self.inter_fd(f3_d_0,final_layer)
                   +self.inter_fd(f1_0,final_layer)+self.inter_fd(f2_0,final_layer)+self.inter_fd(f3_0,final_layer)+self.inter_fd(f4_0,final_layer))/7
        

        loss = loss * self.lambda_x
        return loss 
```


## How to run

First download the pre-trained imagenet for SwinUnet according to ```https://github.com/HuCaoFighting/Swin-Unet```.

In ``` ./src/train_synapse```:

**Train**
```python train.py --dataset Synapse --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --max_epochs 150 --output_dir 11_1  --gpu_id 0 --img_size 224 --base_lr 0.05 --batch_size 32 --lambda_x 0.010 ```

**test**
```python test.py --dataset Synapse --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --is_saveni --output_dir 11_1 --max_epoch 150 --base_lr 0.05 --img_size 224 --batch_size 24```

**Check weights.**  https://drive.google.com/drive/folders/1V9y3fOgKExOFS8namk46UwJqH3yFoPu_?usp=sharing


## Thanks for the code provided by:
- SwinUnet: https://github.com/HuCaoFighting/Swin-Unet
- HiFormer: https://github.com/amirhossein-kz/hiformer
- CASCADE: https://github.com/SLDGroup/CASCADE
- UCTransNet: https://github.com/mcgregorwwww/uctransnet
- ...
