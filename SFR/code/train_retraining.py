import os
import pdb
import sys
import random
import shutil
import argparse
import logging
import numpy as np
import nibabel as nib
from tqdm import tqdm
from scipy import ndimage
from importlib import import_module
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/mnt/e3dbc9b9-6856-470d-84b1-ff55921cd906/SFR_modeling/data', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='name', help='model_name')
parser.add_argument('--dataset', type=str, default='atlas', help='dataset to use')
parser.add_argument('--label_num', type=int, default=24, help='number of labeled data')

parser.add_argument('--model_type', type=str, default='vnet', help='model_type')
parser.add_argument('--rank', type=int, default=4, help='Rank for LoRA adaptation')
parser.add_argument('--module', type=str, default='sam_lora_image_encoder')

parser.add_argument('--patch_size', type=int, default=256, help='shape of data')
parser.add_argument('--input_size', type=int, default=1024, help='shape of data')
parser.add_argument('--num_classes', type=int, default=1, help='number of class')
parser.add_argument('--save_img', type=int, default=250, help='img saving iterations')

# load
parser.add_argument('--pre_exp', type=str, default='sam_ft', help='model_name')
parser.add_argument('--pre_iter', type=int, default=6000, help='maximum epoch number to train')
parser.add_argument('--load', action="store_true", help='load reg & seg net')
parser.add_argument('--load_iter', type=int, default=0, help='load iter')
parser.add_argument('--change_lr', type=int,  default=2500, help='iter for changing lr')

parser.add_argument('--pre_seg_iter', type=int, default=0, help='seg number to train')
parser.add_argument('--save_iter', type=int, default=1000, help='maximum epoch number to train')
parser.add_argument('--max_iterations', type=int, default=12000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=2, help='labeled_batch_size per gpu')

parser.add_argument('--lr_seg', type=float, default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')

parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency', type=float,  default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,  default=40.0, help='consistency_rampup')
args = parser.parse_args()

root = "../"

train_data_path = args.root_path
snapshot_path = root + "model_" + args.dataset + "/" + args.exp + "/"

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size
labeled_bs = args.labeled_bs
n_gpu = len(args.gpu.split(','))
print(batch_size)
max_iterations, base_lr, input_size, patch_size = args.max_iterations, args.lr_seg, args.input_size, args.patch_size
num_classes = args.num_classes

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn.modules.loss import CrossEntropyLoss

# from sam_lora_image_encoder import LoRA_Sam
# from segment_anything_lora import sam_model_registry
# from segment_anything_lora.utils.transforms import ResizeLongestSide
from networks.hierarchical_vnet import VNet
# from networks.hierarchical_unet_3d import UNet_3D
from dataloaders.dataset import *
from utils import ramps, losses
from utils.util import *

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def get_current_consistency_weight(iter_num):
    epoch = iter_num // 150
    consistency_weight = args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)
    return consistency_weight

from detectron2.config import get_cfg, CfgNode
from detectron2.utils.events import _CURRENT_STORAGE_STACK, EventStorage

from efficientps.model import EffificientPS

from dataloaders.dataset import *
from utils import ramps, losses
from utils.util import *

def add_custom_param(cfg):
    """
    In order to add custom config parameter in the .yaml those parameter must
    be initialised
    """
    # Model
    cfg.MODEL_CUSTOM = CfgNode()
    cfg.MODEL_CUSTOM.BACKBONE = CfgNode()
    cfg.MODEL_CUSTOM.BACKBONE.EFFICIENTNET_ID = 7
    cfg.MODEL_CUSTOM.BACKBONE.LOAD_PRETRAIN = True
    # DATASET
    cfg.NUM_CLASS = 2
    cfg.DATASET_PATH = "/mnt/e3dbc9b9-6856-470d-84b1-ff55921cd906/EPS_Medical_rgb/Liver_dataset"
    cfg.TRAIN_JSON = "/mnt/e3dbc9b9-6856-470d-84b1-ff55921cd906/EPS_Medical_rgb/Liver_dataset/output/cityscapes_panoptic_train.json"
    cfg.VALID_JSON = "/mnt/e3dbc9b9-6856-470d-84b1-ff55921cd906/EPS_Medical_rgb/Liver_dataset/output/cityscapes_panoptic_val.json"
    cfg.PRED_DIR = "/mnt/e3dbc9b9-6856-470d-84b1-ff55921cd906/EPS_Medical_rgb/Liver_dataset/preds"
    cfg.PRED_JSON = "/mnt/e3dbc9b9-6856-470d-84b1-ff55921cd906/EPS_Medical_rgb/Liver_dataset/preds/cityscapes_panoptic_preds.json"
    # Transfom
    cfg.TRANSFORM = CfgNode()
    cfg.TRANSFORM.NORMALIZE = CfgNode()
    cfg.TRANSFORM.NORMALIZE.MEAN = (106.433, 116.617, 119.559)
    cfg.TRANSFORM.NORMALIZE.STD = (65.496, 67.6, 74.123)
    cfg.TRANSFORM.RESIZE = CfgNode()
    cfg.TRANSFORM.RESIZE.HEIGHT = 1024
    cfg.TRANSFORM.RESIZE.WIDTH = 1024
    cfg.TRANSFORM.RANDOMCROP = CfgNode()
    cfg.TRANSFORM.RANDOMCROP.HEIGHT = 1024
    cfg.TRANSFORM.RANDOMCROP.WIDTH = 1024
    cfg.TRANSFORM.HFLIP = CfgNode()
    cfg.TRANSFORM.HFLIP.PROB = 0.5
    # Solver
    cfg.SOLVER.NAME = "Adam"
    cfg.SOLVER.ACCUMULATE_GRAD = 1
    cfg.SOLVER.MAX_EPOCHS = 40
    # Runner
    cfg.BATCH_SIZE = 2
    cfg.CHECKPOINT_PATH = ""
    cfg.PRECISION = 32
    # Callbacks
    cfg.CALLBACKS = CfgNode()
    cfg.CALLBACKS.CHECKPOINT_DIR = None
    # Inference
    cfg.INFERENCE = CfgNode()
    cfg.INFERENCE.AREA_TRESH = 64


cfg = get_cfg()
add_custom_param(cfg)
cfg.merge_from_file("/mnt/e3dbc9b9-6856-470d-84b1-ff55921cd906/SFR_modeling/SFR/code/config.yaml")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
        os.makedirs(snapshot_path + 'saveimg')

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    if args.dataset == 'la':
        db_train = LAHeart(base_dir=train_data_path,
                           transform=transforms.Compose([
                               RandomRotFlip(),
                               RandomCrop((112, 112, 80)),
                               ToTensor(),
                           ]))
        labeled_idxs = list(range(args.label_num))
        unlabeled_idxs = list(range(args.label_num, 80))

    elif args.dataset == 'duke':
        db_train = DukeLiverDataset(base_dir=train_data_path,
                            # list_num='0',  # or whichever list you want to use
                            split='train',
                            transform=transforms.Compose([
                                # RandomRotFlip(),
                                RandomCrop((patch_size, patch_size, 16)),
                                # CenterCrop3D((256, 256, 20)),
                                ToTensor(),
                            ]))
        labeled_idxs = list(range(args.label_num))
        unlabeled_idxs = list(range(args.label_num, 260))
    elif args.dataset == 'atlas':
        # db_train = Atlas23(base_dir=train_data_path,
        #                     # list_num='0',  # or whichever list you want to use
        #                     split='train',
        #                     transform=transforms.Compose([
        #                         # RandomRotFlip(),
        #                         RandomCrop((patch_size, patch_size, 16)),
        #                         # CenterCrop3D((256, 256, 20)),
        #                         ToTensor(),
        #                     ]))
        # labeled_idxs = list(range(args.label_num))
        # unlabeled_idxs = list(range(args.label_num, 48))

        db_train = AtlasIRCAD(base_dir=train_data_path,
                            # list_num='0',  # or whichever list you want to use
                            split='train',
                            transform=transforms.Compose([
                                # RandomRotFlip(),
                                RandomCrop((patch_size, patch_size, 16)),
                                # CenterCrop3D((256, 256, 20)),
                                ToTensor(),
                            ]))
        labeled_idxs = list(range(args.label_num))
        unlabeled_idxs = list(range(48, 58))

    elif args.dataset == 'kits19':
        db_train = Kits19(base_dir=train_data_path,
                            # list_num='0',  # or whichever list you want to use
                            # num=args.label_num,
                            split='train',
                            transform=transforms.Compose([
                                # RandomRotFlip(),
                                RandomCrop((patch_size, patch_size, 16)),
                                # CenterCrop3D((256, 256, 20)),
                                ToTensor(),
                            ]))
        labeled_idxs = list(range(args.label_num))
        unlabeled_idxs = list(range(args.label_num, 168))

    multimask_output = True if num_classes > 2 else False
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size-labeled_bs)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=False, worker_init_fn=worker_init_fn)
    
    

    # model, img_embedding_size = sam_model_registry["vit_b"](image_size=args.input_size, num_classes=num_classes, checkpoint='pre_weight/sam_vit_b_01ec64.pth', pixel_mean=[0, 0, 0], pixel_std=[1, 1, 1])
    # pkg = import_module(args.module)
    # model = pkg.LoRA_Sam(model, args.rank).cuda()
    # sam_checkpoint = root + "model_" + args.dataset + "/" + args.pre_exp + "/sam_iter_" + str(args.pre_iter) + ".pth"
    # model.load_lora_parameters(sam_checkpoint)
    # print("sam weight from {}".format(sam_checkpoint))
    model = EffificientPS(cfg).to(device)
    checkpoint = torch.load("/mnt/e3dbc9b9-6856-470d-84b1-ff55921cd906/SFR_modeling/SFR/model_atlas/4by4_WA_indet_12k/eps_iter_best_11412.pth")
    model.load_state_dict(checkpoint)

    def create_model(ema=False):
        if args.model_type == "vnet":
            net = VNet(n_channels=1, n_classes=num_classes + 1, normalization='batchnorm', has_dropout=True, pyramid_has_dropout=True)
            # save_mode_path = "/mnt/e3dbc9b9-6856-470d-84b1-ff55921cd906/SFR_modeling/SFR/model_kits19/3d_kits19_WA/iter_12000.pth"
            # net.load_state_dict(torch.load(save_mode_path))
        # elif args.model_type == "unet_3d":
        #     net = UNet_3D(in_channels=1, n_classes=num_classes + 1)
        model = net.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model_seg = create_model()
    model_seg_ema = create_model(ema=True)

    if args.load:
        seg_checkpoint = os.path.join(snapshot_path, 'iter_' + str(args.load_iter) + '.pth')
        model_seg.load_state_dict(torch.load(seg_checkpoint))
        print("init weight from {}".format(seg_checkpoint))

    model.eval()
    model_seg.train()
    model_seg_ema.train()

    # Set optimizer and losses
    optimizer_seg = optim.SGD(model_seg.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes+1)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))

    iter_num = args.load_iter
    max_epoch = (max_iterations - args.load_iter) // len(trainloader) + 1
    lr_ = base_lr
    kl_distance = torch.nn.KLDivLoss(reduction='none')

    for epoch_num in tqdm(range(max_epoch), ncols=70):
        
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']

            # label_batch = (label_batch==255).long()  # for Duke liver

            # import torch
            # import matplotlib.pyplot as plt

            # # Assume label_batch is your tensor of shape [4, 256, 256, 16]
            # # Convert to numpy for visualization
            # label_batch_np = label_batch.cpu().numpy()  # shape: [4, 256, 256, 16]
            
            # # raise Exception
            # # Setup figure: 4 rows (volumes), 16 columns (slices)
            # fig, axes = plt.subplots(nrows=4, ncols=16, figsize=(20, 5))
            # fig.suptitle("Binary Mask Slices per Volume", fontsize=16)

            # for vol_idx in range(4):  # 4 volumes
            #     for slice_idx in range(16):  # 16 slices per volume
            #         ax = axes[vol_idx, slice_idx]
            #         ax.imshow(label_batch_np[vol_idx, :, :, slice_idx], cmap='gray')
            #         ax.axis('off')  # Hide axis ticks

            # plt.tight_layout()
            # plt.subplots_adjust(top=0.9)  # Add space for the title
            # plt.show()
            # continue
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            noise = torch.clamp(torch.randn_like(volume_batch) * 0.1, -0.2, 0.2)
            volume_batch_ema = volume_batch + noise
            if iter_num % args.save_img == 0:
                nib.save(nib.Nifti1Image(volume_batch[labeled_bs,0].cpu().detach().numpy().astype(np.float32), np.eye(4)), snapshot_path + '/saveimg' + '/img_' + str(iter_num) + '.nii.gz')
                nib.save(nib.Nifti1Image(label_batch[labeled_bs].cpu().detach().numpy().astype(np.float32), np.eye(4)), snapshot_path + '/saveimg' + '/gt_' + str(iter_num) + '.nii.gz')

            ## Train Segmentation Module
            outputs, _, _, _ = model_seg(volume_batch)

            outputs_soft = F.softmax(outputs, dim=1)
            prediction_seg = torch.argmax(outputs_soft, dim=1)
            # import matplotlib.pyplot as plt

            # # Assume label_batch is your tensor of shape [4, 256, 256, 16]
            # # Convert to numpy for visualization
            # prediction_seg_np = prediction_seg.cpu().numpy()  # shape: [4, 256, 256, 16]

            # # Setup figure: 4 rows (volumes), 16 columns (slices)
            # fig, axes = plt.subplots(nrows=4, ncols=16, figsize=(20, 5))
            # fig.suptitle("Binary Mask Slices per Volume", fontsize=16)

            # for vol_idx in range(4):  # 4 volumes
            #     for slice_idx in range(16):  # 16 slices per volume
            #         ax = axes[vol_idx, slice_idx]
            #         ax.imshow(prediction_seg_np[vol_idx, :, :, slice_idx], cmap='gray')
            #         ax.axis('off')  # Hide axis ticks

            # plt.tight_layout()
            # plt.subplots_adjust(top=0.9)  # Add space for the title
            # plt.show()
            # raise Exception

            with torch.no_grad():
                outputs_ema, _, _, _ = model_seg_ema(volume_batch_ema)
                outputs_soft_ema = F.softmax(outputs_ema, dim=1)
            prediction_seg_ema = torch.argmax(outputs_soft_ema, dim=1)

            ## Generate pseudo labels
            image = volume_batch[labeled_bs:]
            label = label_batch[labeled_bs:]
            

            output_soft_single, prediction_sam = sam_test_tensor(model, image, label, input_size, batch_size, labeled_bs, num_classes, multimask_output)
            # w_loss_p = 1 - iter_num / max_iterations
            prediction_sam = prediction_sam.long().cuda()
            ## calculate the loss
            loss_ce_seg_l = ce_loss(outputs[:labeled_bs], label_batch[:labeled_bs])

            # print(outputs_soft[:labeled_bs].shape, label_batch[:labeled_bs].unsqueeze(1).shape)
            
            loss_dice_seg_l = dice_loss(outputs_soft[:labeled_bs], label_batch[:labeled_bs].unsqueeze(1))
            loss_seg_l = 0.5 * (loss_ce_seg_l + loss_dice_seg_l)
            consistency_weight = get_current_consistency_weight(iter_num)
            loss_ce_seg_u = ce_loss(outputs[labeled_bs:], prediction_sam)
            print(outputs[labeled_bs:].shape, prediction_sam.shape)
            print(outputs[labeled_bs:].unique(), prediction_sam.unique())
            print(outputs_soft[labeled_bs:].shape, outputs_soft[labeled_bs:].unique())
            
            import matplotlib.pyplot as plt
            import numpy as np
            import torch

            # Extract tensors
            os = outputs[labeled_bs:, 1].cpu().detach().numpy()     # (2, 256, 256, 16)
            gt = label.cpu().detach().numpy()          # (2, 256, 256, 16)
            ps = prediction_sam.cpu().detach().numpy()              # (2, 256, 256, 16)

            fig, axs = plt.subplots(3, 16, figsize=(24, 6))  # 3 rows now

            for i in range(16):
                # Row 1: outputs (logits)
                axs[0, i].imshow(os[0, :, :, i], cmap='viridis')
                axs[0, i].set_title(f"os_{i+1}")
                axs[0, i].axis('off')

                # Row 2: ground truth labels
                axs[1, i].imshow(gt[0, :, :, i], cmap='gray')
                axs[1, i].set_title(f"gt_{i+1}")
                axs[1, i].axis('off')

                # Row 3: predictions
                axs[2, i].imshow(ps[0, :, :, i], cmap='gray')
                axs[2, i].set_title(f"ps_{i+1}")
                axs[2, i].axis('off')

            plt.tight_layout()
            plt.show()

            
            loss_dice_seg_u = dice_loss(outputs_soft[labeled_bs:], prediction_sam.unsqueeze(1))
            loss_seg_u = 0.5 * (loss_ce_seg_u + loss_dice_seg_u)
            # loss_seg_u = w_loss_p * loss_seg_u

            consistency_dist = losses.softmax_mse_loss(outputs, outputs_ema)
            consistency_dist = torch.mean(consistency_dist)
            consistency_loss = consistency_weight * consistency_dist

            loss_seg = loss_seg_l + loss_seg_u + consistency_loss

            optimizer_seg.zero_grad()
            loss_seg.backward()
            optimizer_seg.step()
            update_ema_variables(model_seg, model_seg_ema, args.ema_decay, iter_num)
           
            if iter_num % args.save_img == 0:
                nib.save(nib.Nifti1Image(prediction_seg[labeled_bs].cpu().detach().numpy().astype(np.float32), np.eye(4)), snapshot_path + '/saveimg' + '/pred_seg_' + str(iter_num) + '_u.nii.gz')
                nib.save(nib.Nifti1Image(prediction_sam[0].cpu().detach().numpy().astype(np.float32), np.eye(4)), snapshot_path + '/saveimg' + '/pred_sam_' + str(iter_num) + '_u.nii.gz')
            iter_num = iter_num + 1
            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/loss_seg', loss_seg, iter_num)
            writer.add_scalar('loss/loss_seg_l', loss_seg_l, iter_num)
            writer.add_scalar('loss/loss_seg_u', loss_seg_u, iter_num)
            logging.info('iter %d : seg loss : %f, sup loss : %f, unsup loss : %f' % (iter_num, loss_seg, loss_seg_l, loss_seg_u.item()))

            ## change lr
            if args.dataset == "btcv":
                lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            else:
                if iter_num % args.change_lr == 0:
                    lr_ = base_lr * 0.1 ** (iter_num // args.change_lr)
            for param_group in optimizer_seg.param_groups:
                param_group['lr'] = lr_

            if iter_num % args.save_iter == 0:
                save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model_seg.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                    break
        if iter_num >= max_iterations:
            break
    save_mode_path = os.path.join(snapshot_path, 'iter_' + str(max_iterations) + '.pth')
    torch.save(model_seg.state_dict(), save_mode_path)
    logging.info("save model to {}".format(save_mode_path))
    writer.close()

