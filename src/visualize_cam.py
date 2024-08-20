import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from scipy.ndimage import zoom
from pytorch_grad_cam import GradCAM, ScoreCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


from torch.utils.data import DataLoader
from datasets.dataset_synapse import Synapse_dataset

join = os.path.join

class SemanticSegmentationTarget:
    def __init__(self, category, mask):
        self.category = category
        self.mask = torch.from_numpy(mask)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()
    
    def __call__(self, model_output):
        return  (model_output[self.category, :, : ] * self.mask).sum()

def reshape_transform(tensor, height=7, width=7):
    # print(tensor.size(1))
    factor = (tensor.size(1) // (height * width)) ** 0.5
    
    result = tensor.reshape(tensor.size(0),
                            int(height*factor), int(width*factor), tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def maybe_mkdir_p(directory: str) -> None:
    os.makedirs(directory, exist_ok=True)

def compute_cam_single_class(net, testloader, args):
    for layer_name, target_layer in args.target_layers.items():
        for i_batch, sampled_batch in tqdm(enumerate(testloader)):
            h, w = sampled_batch["image"].size()[2:]
            image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
            image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()

            case_dir = os.path.join(save_dir, case_name)
            maybe_mkdir_p(os.path.join(case_dir, "cam_v3", f"all", layer_name))
            maybe_mkdir_p(os.path.join(case_dir, "mask"))
            maybe_mkdir_p(os.path.join(case_dir, "prediction"))

            for ind in range(image.shape[0]):
                slice = image[ind, :, :]
                label_slice = label[ind, :, :]
                x, y = slice.shape[0], slice.shape[1]
                if x != args.patch_size[0] or y != args.patch_size[1]:
                    slice = zoom(slice, (args.patch_size[0] / x, args.patch_size[1] / y), order=3)  # previous using 0
                    label_slice = zoom(label_slice, (args.patch_size[0] / x, args.patch_size[1] / y), order=0)  # previous using 0

                rgb_img = (slice - slice.min()) / (slice.max() - slice.min())
                rgb_img = np.tile(rgb_img[..., np.newaxis], (1, 1, 3))
                    
                input_tensor = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float()
                input_tensor = input_tensor.cuda()
                unique_label = np.unique(label_slice)[1:]

                if len(unique_label) > 1:
                    with torch.no_grad():
                        logits = net(input_tensor)
                        prediction = torch.argmax(torch.softmax(logits, dim=1), dim=1).squeeze(0)
                        prediction = prediction.cpu().detach()

                    cam_all_classes = 0
                    for class_id in range(1, 9):
                        targets = [SemanticSegmentationTarget(class_id, np.float32(label_slice == class_id))]
                        cam = GradCAM(model=net, target_layers=target_layer, use_cuda=True, reshape_transform=reshape_transform if args.model=='swin_unet' else None)
                        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :, :]
                        # grayscale_cam = grayscale_cam / grayscale_cam.max()
                        cam_all_classes += grayscale_cam
                        # cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
                    cam_all_classes = cam_all_classes / cam_all_classes.max()
                    cam_image = show_cam_on_image(rgb_img, cam_all_classes, use_rgb=True)
                    # print(case_dir)
                    Image.fromarray(cam_image).save(os.path.join(case_dir, "cam_v3", f"all", layer_name, f"slice{ind}.png"))
                    Image.fromarray(np.uint8(label_slice * 30)).save(os.path.join(case_dir, "mask", f"slice{ind}.png"))
                    Image.fromarray(np.uint8(prediction * 30)).save(os.path.join(case_dir, "prediction", f"slice{ind}.png"))
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='swin_unet')
    # parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--class_id", type=int, default=1)
    args = parser.parse_args()

    save_dir = join("attn_maps", args.model)
    maybe_mkdir_p(save_dir)

    ## loading dataset
    dataset_config = {
            'Synapse': {
                'Dataset': Synapse_dataset,
                'volume_path': '/home/peijie.qiu/Study/datasets/Synapse/test_vol_h5',
                'list_dir': './lists/lists_Synapse',
                'num_classes': 9,
                'z_spacing': 1,
            },
        }

    args.patch_size=[224, 224]

    db_test = dataset_config['Synapse']['Dataset'](base_dir=dataset_config['Synapse']['volume_path'], split="test_vol", list_dir=dataset_config['Synapse']['list_dir'])
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    print("{} test iterations per epoch".format(len(testloader)))

    ## load model
    if args.model == 'unet':
        # from networks.unet import unet_2D
        # net = unet_2D(feature_scale=4, in_channels=1, n_classes=9).cuda()
        # ckpt = torch.load("UNet_epoch_149.pth")
        # net.load_state_dict(ckpt)
        # # target_layers = [net.up_concat4]
        # target_layers_dict = {
        #     'encoder_1': [net.conv1],
        #     'encoder_2': [net.conv2],
        #     'encoder_3': [net.conv3],
        #     'encoder_4': [net.conv4],
        #     'bottleneck': [net.center],
        #     'decoder_4': [net.up_concat4],
        #     'decoder_3': [net.up_concat3],
        #     'decoder_2': [net.up_concat2],
        #     'decoder_1': [net.up_concat1]
        # }
        pass
    elif args.model == 'swin_unet':
        from config import _C
        from networks.vision_transformer import SwinUnet as ViT_seg
        
        config = _C.clone()
        config.MODEL.SWIN.DEPTHS = [2, 2, 2, 2]
        config.MODEL.SWIN.DECODER_DEPTHS = [2, 2, 2, 1]
        config.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]

        net = net = ViT_seg(config, img_size=224, num_classes=9).cuda()
        ckpt = torch.load("trained_ckpt/epoch_149.pth")
        net.load_state_dict(ckpt)

        # print(net.swin_unet.layers_up[0])

        # target_layers = [net.swin_unet.layers_up[3].blocks[-1].norm2]
        target_layers_dict = {
            'encoder_1': [net.swin_unet.layers[0].blocks[0].norm2],
            'encoder_2': [net.swin_unet.layers[1].blocks[0].norm2],
            'encoder_3': [net.swin_unet.layers[2].blocks[0].norm2],
            'encoder_4': [net.swin_unet.layers[3].blocks[0].norm2],
            'bottleneck': [net.swin_unet.layers_up[0].norm],
            'decoder_4': [net.swin_unet.layers_up[1].blocks[0].norm2],
            'decoder_3': [net.swin_unet.layers_up[2].blocks[0].norm2],
            'decoder_2': [net.swin_unet.layers_up[3].blocks[0].norm2],
            'decoder_1': [net.swin_unet.up]
        }

    net.eval()
    args.target_layers = target_layers_dict
    compute_cam_single_class(net, testloader, args)