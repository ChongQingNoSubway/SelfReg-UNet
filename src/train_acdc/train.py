import os
import sys
import logging
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import random
import time

import numpy as np
from tqdm import tqdm
from medpy.metric import dc,hd95
from scipy.ndimage import zoom

from utils.utils import powerset
from utils.utils import DiceLoss, calculate_dice_percase, val_single_volume
from utils.dataset_ACDC import ACDCdataset, RandomGenerator,ACDCdataset_train,RandomGenerator_DINO,RandomGenerator_DINO_Deform
from test_ACDC import inference

from network.vision_transformer import SwinUnet as ViT_seg
from config import get_config
from torchvision.transforms import functional as VF

# from lib.networks import PVT_GCASCADE, MERIT_GCASCADE




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

        #inter_fd_loss = F.mse_loss(f_s[:, 0:min(s_C,t_C), :, :], f_t[:, 0:min(s_C,t_C), :, :].detach())

        inter_fd_loss = F.mse_loss(f_s[:, idx_s, :, :], f_t[:, idx_t, :, :].detach())
        return inter_fd_loss 
    
    def intra_fd(self,f_s):
        sorted_s, indices_s = torch.sort(F.normalize(f_s, p=2, dim=(2,3)).mean([0, 2, 3]), dim=0, descending=True)
        f_s = torch.index_select(f_s, 1, indices_s)
        intra_fd_loss = F.mse_loss(f_s[:, 0:f_s.shape[1]//2, :, :], f_s[:, f_s.shape[1]//2: f_s.shape[1], :, :])
        return intra_fd_loss
    
    def forward(self,feature,feature_decoder,final_up):
        f1 = feature[0][-1] # 
        f2 = feature[1][-1]
        f3 = feature[2][-1]
        f4 = feature[3][-1] # lower feature 

        f1_0 = feature[0][0] # 
        f2_0 = feature[1][0]
        f3_0 = feature[2][0]
        f4_0 = feature[3][0] # lower feature 

        f1_d = feature_decoder[0][-1] # 
        f2_d = feature_decoder[1][-1] # 
        f3_d = feature_decoder[2][-1] # 

        f1_d_0 = feature_decoder[0][0] # 
        f2_d_0 = feature_decoder[1][0] # 
        f3_d_0 = feature_decoder[2][0] # 

        #print(f3_d.shape)

        final_layer = final_up
        #print(final_layer.shape)


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

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule








parser = argparse.ArgumentParser()
parser.add_argument('--encoder', default='PVT', help='Name of encoder: PVT or MERIT')
parser.add_argument('--skip_aggregation', default='additive', help='Type of skip-aggregation: additive or concatenation')                        
parser.add_argument("--batch_size", default=12, help="batch size")
parser.add_argument("--lr", default=0.0001, help="learning rate")
parser.add_argument("--max_epochs", default=400)
parser.add_argument("--img_size", default=224)
parser.add_argument("--save_path", default="./model_pth/ACDC")
parser.add_argument("--n_gpu", default=1)
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--list_dir", default="./data/ACDC/lists_ACDC")
parser.add_argument("--root_dir", default="./data/ACDC/")
parser.add_argument("--volume_path", default="./data/ACDC/test")
parser.add_argument("--z_spacing", default=10)
parser.add_argument("--num_classes", default=4)
parser.add_argument('--test_save_dir', default='./predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--seed', type=int,
                    default=2222, help='random seed')        
parser.add_argument('--gpu_id',default=1,type=int)     
parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )      
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                            'full: cache all data, '
                            'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--kd', type=float,
                    default=0.1, help='dino weight')

parser.add_argument('--model', type=str,
                    default='Our', help='dino weight')


args = parser.parse_args()

config = get_config(args)

torch.cuda.set_device(args.gpu_id)

if not args.deterministic:
    cudnn.benchmark = True
    cudnn.deterministic = False
else:
    cudnn.benchmark = False
    cudnn.deterministic = True
    
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


args.is_pretrain = True
args.exp = args.model + str(args.img_size)
snapshot_path = "{}/{}/{}".format(args.save_path, args.exp, 'PVT_GCASCADE_MUTATION_w3_7_Run1')
snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
snapshot_path = snapshot_path + '_epo' +str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
snapshot_path = snapshot_path + '_lr' + str(args.lr) if args.lr != 0.01 else snapshot_path
snapshot_path = snapshot_path + '_'+str(args.img_size)
snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path

#current_time = time.strftime("%H%M%S")
#print("The current time is", current_time)
#snapshot_path = snapshot_path +'_t'+current_time
    
if not os.path.exists(snapshot_path):
    os.makedirs(snapshot_path)

args.test_save_dir = os.path.join(snapshot_path, args.test_save_dir)
test_save_path = os.path.join(args.test_save_dir, args.exp)
if not os.path.exists(test_save_path):
    os.makedirs(test_save_path, exist_ok=True)

net = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes).cuda()
net.load_from(config)

train_dataset = ACDCdataset_train(args.root_dir, args.list_dir, split="train", transform=
                                   transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])])
                                   )

print("The length of train set is: {}".format(len(train_dataset)))
Train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)


db_val=ACDCdataset(base_dir=args.root_dir, list_dir=args.list_dir, split="valid")
valloader=DataLoader(db_val, batch_size=1, shuffle=False)
db_test =ACDCdataset(base_dir=args.volume_path,list_dir=args.list_dir, split="test")
testloader = DataLoader(db_test, batch_size=1, shuffle=False)

if args.n_gpu > 1:
    net = nn.DataParallel(net)

net = net.cuda()
net.train()
ce_loss = CrossEntropyLoss()
dice_loss = DiceLoss(args.num_classes)
kd_loss = KDloss(lambda_x=args.kd)

save_interval = 100

iterator = tqdm(range(0, args.max_epochs), ncols=70)
iter_num = 0

Loss = []
Test_Accuracy = []

Best_dcs = 0.80
Best_test_dcs = 0.80
logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')

max_iterations = args.max_epochs * len(Train_loader)
base_lr = args.lr
optimizer = optim.AdamW(net.parameters(), lr=base_lr, weight_decay=0.0001)

#optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)





def val():
    logging.info("Validation ===>")
    dc_sum=0
    metric_list = 0.0
    net.eval()
    for i, val_sampled_batch in enumerate(valloader):
        val_image_batch, val_label_batch = val_sampled_batch["image"], val_sampled_batch["label"]
        val_image_batch, val_label_batch = val_image_batch.squeeze(0).cpu().detach().numpy(), val_label_batch.squeeze(0).cpu().detach().numpy()
        x, y = val_image_batch.shape[0], val_image_batch.shape[1]
        if x != args.img_size or y != args.img_size:
            val_image_batch = zoom(val_image_batch, (args.img_size / x, args.img_size / y), order=3)
        val_image_batch = torch.from_numpy(val_image_batch).unsqueeze(0).unsqueeze(0).float().cuda()
        
        val_outputs = net(val_image_batch)

        # val_outputs = 0.0
        # for idx in range(len(P)):
        #     val_outputs += P[idx]
        
        val_outputs = torch.softmax(val_outputs, dim=1)
        val_outputs = torch.argmax(val_outputs, dim=1).squeeze(0)
        val_outputs = val_outputs.cpu().detach().numpy()

        if x != args.img_size or y != args.img_size:
            val_outputs = zoom(val_outputs, (x / args.img_size, y / args.img_size), order=0)
        else:
            val_outputs = val_outputs
        dc_sum+=dc(val_outputs,val_label_batch[:])
    performance = dc_sum / len(valloader)
    logging.info('Testing performance in val model: mean_dice : %f, best_dice : %f' % (performance, Best_dcs))

    print('Testing performance in val model: mean_dice : %f, best_dice : %f' % (performance, Best_dcs))
    return performance

l = [0, 1, 2, 3]
ss = [x for x in powerset(l)]
#ss = [[0],[1],[2],[3]]
# print(ss)
    
for epoch in iterator:
    net.train()
    train_loss = 0
    for i_batch, sampled_batch in enumerate(Train_loader):
        image_batch, label_batch = sampled_batch["image"], sampled_batch["label"]
        image_batch, label_batch = image_batch.type(torch.FloatTensor), label_batch.type(torch.FloatTensor)
        image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
       

        outputs, kd_encorder,kd_decorder, final_up = net(image_batch)

        lc1, lc2 = 0.3, 0.7

        loss_ce = ce_loss(outputs, label_batch[:].long())

        loss_dice = dice_loss(outputs, label_batch, softmax=True)

        loss_kd = kd_loss(kd_encorder,kd_decorder,final_up)

        loss = lc1 * loss_ce + lc2 * loss_dice  + loss_kd  

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9 # We did not use this
        lr_ = base_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_

        iter_num = iter_num + 1
        if iter_num%50 == 0:
            logging.info('iteration %d : loss : %f  loss kd : %f lr_: %f' % (iter_num, loss.item(),loss_kd.item(), lr_))
            print('iteration %d : loss : %f lr_: %f' % (iter_num, loss.item(), lr_))
        train_loss += loss.item()

      
    Loss.append(train_loss/len(train_dataset))
    logging.info('iteration %d : loss : %f lr_: %f' % (iter_num, loss.item(), lr_))
    print('iteration %d : loss : %f lr_: %f' % (iter_num, loss.item(), lr_))
    
    save_model_path = os.path.join(snapshot_path, 'last.pth')
    torch.save(net.state_dict(), save_model_path)
    #logging.info("save model to {}".format(save_model_path))
    
    avg_dcs = val()
  
    if avg_dcs >= Best_dcs:
        save_model_path = os.path.join(snapshot_path, 'best.pth')
        torch.save(net.state_dict(), save_model_path)
        logging.info("save model to {} at epoch {}".format(save_model_path,epoch))
        print("save model to {}".format(save_model_path))
        Best_dcs = avg_dcs
        
        avg_test_dcs, avg_hd, avg_jacard, avg_asd = inference(args, net, testloader, args.test_save_dir)
        print("test avg_dsc: %f" % (avg_test_dcs))
        logging.info("test avg_dsc: %f" % (avg_test_dcs))
        Test_Accuracy.append(avg_test_dcs)
        if(Best_test_dcs <= avg_test_dcs):
            Best_test_dcs = avg_test_dcs
            save_model_path = os.path.join(snapshot_path, 'test_best.pth')
            torch.save(net.state_dict(), save_model_path)
            logging.info("save model to {}".format(save_model_path))
            print("save model to {}".format(save_model_path))
        
    if epoch >= args.max_epochs - 1:
        save_model_path = os.path.join(snapshot_path,  'epoch={}_lr={}_avg_dcs={}.pth'.format(epoch, lr_, avg_dcs))
        torch.save(net.state_dict(), save_model_path)
        logging.info("save model to {}".format(save_model_path))
        print("save model to {}".format(save_model_path))
        iterator.close()
        break
