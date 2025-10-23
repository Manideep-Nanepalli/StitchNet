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
# from tensorboardX import SummaryWriter
parser = argparse.ArgumentParser() 
parser.add_argument('--root_path', type=str, default='/mnt/e3dbc9b9-6856-470d-84b1-ff55921cd906/SFR_modeling/data', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='name', help='model_name')
parser.add_argument('--dataset', type=str, default='kits19', help='dataset to use')
parser.add_argument('--label_num', type=int, default=126, help='number of labeled data')

parser.add_argument('--pretrain_model', type=str, default='vit_b', help='vit to select')
parser.add_argument('--patch_size', type=int, default=128, help='shape of data')
parser.add_argument('--input_size', type=int, default=1024, help='shape of data')
parser.add_argument('--num_classes', type=int, default=1, help='number of class')
parser.add_argument('--save_img', type=int, default=250, help='img saving iterations')
# load
parser.add_argument('--load', action="store_true", help='load net')
parser.add_argument('--load_iter', type=int, default=0, help='load iter')

parser.add_argument('--save_iter', type=int, default=1000, help='maximum epoch number to train')
parser.add_argument('--max_iterations', type=int, default=12000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=1, help='batch_size per gpu')

parser.add_argument('--nt', type=int, default=2, help='nonlinear_transformation')
parser.add_argument('--nonlinear_rate', type=float, default=0.5, help='nonlinear_rate')
parser.add_argument('--rdmrotflip', action="store_true", help='rdmrotflip')

# parser.add_argument("--lr_sam", type=float, default=0.001, help="sam learning rate")
parser.add_argument('--warmup', action='store_true', help='If activated, warp up the learning from a lower lr to the base_lr')
parser.add_argument('--warmup_period', type=int, default=250, help='Warp up iterations, only valid whrn warmup is activated')

parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--rank', type=int, default=4, help='Rank for LoRA adaptation')
parser.add_argument('--module', type=str, default='sam_lora_image_encoder')
args = parser.parse_args()

root = "../"

train_data_path = args.root_path
snapshot_path = root + "model_" + args.dataset + "/" + args.exp + "/"

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size
n_gpu = len(args.gpu.split(','))

max_iterations, input_size, patch_size = args.max_iterations, args.input_size, args.patch_size
num_classes = args.num_classes
# lr_sam = args.lr_sam

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn.modules.loss import CrossEntropyLoss

# from sam_lora_image_encoder import LoRA_Sam
# from segment_anything_lora import sam_model_registry
from efficientps.model import EffificientPS

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

# import yaml
# from types import SimpleNamespace

# def load_config(path_to_yaml):
#     with open(path_to_yaml, 'r') as f:
#         config_dict = yaml.safe_load(f)
#     return dict_to_namespace(config_dict)

# def dict_to_namespace(d):
#     # Recursively convert dict to SimpleNamespace
#     if isinstance(d, dict):
#         return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
#     elif isinstance(d, list):
#         return [dict_to_namespace(item) if isinstance(item, dict) else item for item in d]
#     else:
#         return d

# cfg = load_config("/mnt/e3dbc9b9-6856-470d-84b1-ff55921cd906/SFR_modeling/SFR/code/config.yaml")


from detectron2.config import get_cfg, CfgNode
from detectron2.utils.events import _CURRENT_STORAGE_STACK, EventStorage

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

logging.getLogger("pytorch_lightning").setLevel(logging.INFO)
logger = logging.getLogger("pytorch_lightning.core")
if not os.path.exists(cfg.CALLBACKS.CHECKPOINT_DIR):
    os.makedirs(cfg.CALLBACKS.CHECKPOINT_DIR)
logger.addHandler(logging.FileHandler(
    os.path.join(cfg.CALLBACKS.CHECKPOINT_DIR,"core.log")))
with open("/mnt/e3dbc9b9-6856-470d-84b1-ff55921cd906/SFR_modeling/SFR/code/config.yaml") as file:
    logger.info(file.read())
_CURRENT_STORAGE_STACK.append(EventStorage())


if __name__ == "__main__":
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
        os.makedirs(snapshot_path + 'saveimg')

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    if args.dataset == 'kits19':
        db_train = Kits19(base_dir=train_data_path,
                            # list_num='0',  # or whichever list you want to use
                            num=args.label_num,
                            split='train',
                            transform=transforms.Compose([
                                RandomRotFlip(),
                                RandomCrop((patch_size, patch_size, 4)),
                                # CenterCrop3D((256, 256, 20)),
                                ToTensor(),
                            ]))
        db_val = Kits19(
                        base_dir=train_data_path,
                        # list_num='0',
                        num=args.label_num,
                        split='val',
                        transform=transforms.Compose([
                            # RandomRotFlip(),
                            RandomCrop((patch_size, patch_size, 4)),
                            ToTensor(),
                        ])
                    )
        db_test = Kits19(
                        base_dir=train_data_path,
                        # list_num='0',
                        num=args.label_num,
                        split='test',
                        transform=transforms.Compose([
                            # RandomRotFlip(),
                            RandomCrop((patch_size, patch_size, 4)),
                            ToTensor(),
                        ])
                    )

    multimask_output = True if num_classes > 2 else False
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    testloader = DataLoader(db_test, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_dice=0.0
    batch_dice = 0.0
    def dice_score(pred, gt, smooth=1e-6):
            """
            Compute Dice coefficient between two binary volumes.
            """
            intersection = np.sum(pred * gt)
            volume_sum = np.sum(pred) + np.sum(gt)
            dice = (2. * intersection + smooth) / (volume_sum + smooth)
            return dice
    
    if os.path.exists(cfg.CHECKPOINT_PATH):
        print('""""""""""""""""""""""""""""""""""""""""""""""')
        print("Loading model from {}".format(cfg.CHECKPOINT_PATH))
        print('""""""""""""""""""""""""""""""""""""""""""""""')
        model = EffificientPS.load_from_checkpoint(cfg=cfg,
            checkpoint_path=cfg.CHECKPOINT_PATH).to(device)
    else:
        print('""""""""""""""""""""""""""""""""""""""""""""""')
        print("Creating a new model")
        print('""""""""""""""""""""""""""""""""""""""""""""""')
        model = EffificientPS(cfg).to(device)
        # model = EffificientPS(cfg)  # Initialize with the same config
        checkpoint = torch.load("/mnt/e3dbc9b9-6856-470d-84b1-ff55921cd906/SFR_modeling/SFR/2x2results/eps_iter_best_11592.pth")
        print("Yes")
        model.load_state_dict(checkpoint)
            
    optimizer_config = model.configure_optimizers()
    optimizer = optimizer_config['optimizer']
    scheduler = optimizer_config['lr_scheduler']

    iter_num = args.load_iter
    max_epoch = (max_iterations - args.load_iter) // len(trainloader) + 1
    kl_distance = torch.nn.KLDivLoss(reduction='none')
    

    model.eval()
    total_dice = 0.0
    num_samples = 0
    num_val_iters = 0
    with torch.no_grad():
        val_loader_tqdm = tqdm(valloader, desc=" Validating", leave=False)
        for i_batch, sampled_batch in enumerate(val_loader_tqdm):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']      # [2, 1, 128, 128, 64], [2, 128, 128, 64]

            ### Train EPS Module
            eps_volume_batch = volume_batch.cpu().detach().numpy()
            eps_label_batch = label_batch.cpu().detach().numpy()

            ## labeled data
            image = eps_volume_batch       # [B, 1, 128, 128, 64]
            label = eps_label_batch        # [B, 128, 128, 64]
            
            image_inshape, label_inshape, n_row, n_col, pw, ph, ps, pww, phh, s_l, s_w = Spread_bs_aug(image, label, input_size, args.nt, args.nonlinear_rate)  # (B, 1024, 1024)

            volume_batch_inshape, label_batch_inshape = ToTensor_sam_bs(image_inshape, label_inshape)  # [B, 3, 1024, 1024], [B, 1024, 1024]
                                
            volume_batch_inshape = volume_batch_inshape.to(device)
            label_batch_inshape = label_batch_inshape.to(device)
            batch = dict()
            batch['image'] = volume_batch_inshape
            semantic = label_batch_inshape

            batch['semantic'] = torch.where(semantic == 0, torch.tensor(0, dtype=semantic.dtype, device=semantic.device),
            torch.where(semantic == 255, torch.tensor(1, dtype=semantic.dtype, device=semantic.device), semantic))
            
            outputs, loss = model.shared_step(batch)
            output_masks = outputs['semantic']
            
            output_soft = F.softmax(output_masks, dim=1)                    # [1, 2, 1024, 1024]
                        
            num_val_iters += 1

            prediction_inshape = torch.argmax(output_soft, dim=1)  # (B, 1024, 1024)
            
            prediction_inshape = prediction_inshape.cpu().detach().numpy()  # (B, 1024, 1024)
            
            pred_single = np.zeros((batch_size, image.shape[-1] + ps * 2, image.shape[-2] + pw * 2, image.shape[-3] + ph * 2))  # [B, 80, 112, 112]  (1, 112, 114, 86)
            if pww < 0 or phh < 0:
                pww_r, phh_r = pww, phh
                prediction_inshape = np.pad(prediction_inshape, [(0, 0), (abs(pww), abs(pww)), (abs(phh), abs(phh))], mode='constant', constant_values=255)
                pww, phh = 0, 0

            # Inverse Stitching
            for row in range(n_row):
                for col in range(n_col):
                    if row * n_col + col < pred_single.shape[1]:
                        pred_single[:, row * n_col + col] = prediction_inshape[:, pww + row * s_l:pww + (row + 1) * s_l, phh + col * s_w:phh + (col + 1) * s_w]
            pred_single = pred_single[:, ps:pred_single.shape[-3] - ps, pw:pred_single.shape[-2] - pw, ph:pred_single.shape[-1] - ph]  # (B, 64, 128, 128)
            pred_single = np.swapaxes(pred_single, -3, -1)  # (B, 128, 128, 64)
            prediction_eps = pred_single

            prediction_binary = (prediction_eps > 0).astype(np.uint8)  # or whatever label is foreground
            # label_binary = (batch['semantic'].cpu().detach().numpy() == 1).astype(np.uint8)      # ground truth foreground
            
            eps_label_batch = eps_label_batch.squeeze(0)
                       
            label_binary = (eps_label_batch > 0).astype(np.uint8)  

            prediction_binary = prediction_binary.squeeze(0)

            # Compute Dice score
            assert prediction_binary.shape == label_binary.shape

            # print(prediction_binary.shape)
            print(np.unique(prediction_binary))
            raise Exception
            # import matplotlib.pyplot as plt

            # # Assuming label_binary and prediction_binary are NumPy arrays
            # # label_binary: (512, 512, 4), prediction_binary: (512, 512, 4)
            # if hasattr(label_binary, 'numpy'):
            #     label_binary = label_binary.cpu().numpy()
            # if hasattr(prediction_binary, 'numpy'):
            #     prediction_binary = prediction_binary.cpu().numpy()

            # fig, axes = plt.subplots(4, 2, figsize=(8, 16))

            # for i in range(4):
            #     # GT mask
            #     axes[i, 0].imshow(label_binary[:, :, i], cmap='gray', vmin=0, vmax=1)
            #     axes[i, 0].set_title(f"GT Mask {i}")
            #     axes[i, 0].axis('off')
                
            #     # Predicted mask
            #     axes[i, 1].imshow(prediction_binary[:, :, i], cmap='gray', vmin=0, vmax=1)
            #     axes[i, 1].set_title(f"Pred Mask {i}")
            #     axes[i, 1].axis('off')

            # plt.tight_layout()
            # plt.show()


            total_dice += dice_score(prediction_binary, label_binary)

        print(f"Mean Val Dice Score: {total_dice/len(valloader):.4f}")
        

    total_dice = 0.0
    num_samples = 0
    with torch.no_grad():
        test_loader_tqdm = tqdm(testloader, desc="Testing", leave=False)
        for i_batch, sampled_batch in enumerate(test_loader_tqdm):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']      # [2, 1, 128, 128, 64], [2, 128, 128, 64]

            ### Train EPS Module
            eps_volume_batch = volume_batch.cpu().detach().numpy()
            eps_label_batch = label_batch.cpu().detach().numpy()

            ## labeled data
            image = eps_volume_batch       # [B, 1, 128, 128, 64]
            label = eps_label_batch        # [B, 128, 128, 64]
            

            image_inshape, label_inshape, n_row, n_col, pw, ph, ps, pww, phh, s_l, s_w = Spread_bs_aug(image, label, input_size, args.nt, args.nonlinear_rate)  # (B, 1024, 1024)

            volume_batch_inshape, label_batch_inshape = ToTensor_sam_bs(image_inshape, label_inshape)  # [B, 3, 1024, 1024], [B, 1024, 1024]
                                
            volume_batch_inshape = volume_batch_inshape.to(device)
            label_batch_inshape = label_batch_inshape.to(device)
            batch = dict()
            batch['image'] = volume_batch_inshape
            semantic = label_batch_inshape
            
            batch['semantic'] = torch.where(semantic == 0, torch.tensor(0, dtype=semantic.dtype, device=semantic.device),
            torch.where(semantic == 255, torch.tensor(1, dtype=semantic.dtype, device=semantic.device), semantic))
            
            outputs, loss = model.shared_step(batch)
            output_masks = outputs['semantic']
            
            output_soft = F.softmax(output_masks, dim=1)                    # [1, 2, 1024, 1024]
                        
            prediction_inshape = torch.argmax(output_soft, dim=1)  # (B, 1024, 1024)
            
            prediction_inshape = prediction_inshape.cpu().detach().numpy()  # (B, 1024, 1024)
            
            pred_single = np.zeros((batch_size, image.shape[-1] + ps * 2, image.shape[-2] + pw * 2, image.shape[-3] + ph * 2))  # [B, 80, 112, 112]  (1, 112, 114, 86)
            if pww < 0 or phh < 0:
                pww_r, phh_r = pww, phh
                prediction_inshape = np.pad(prediction_inshape, [(0, 0), (abs(pww), abs(pww)), (abs(phh), abs(phh))], mode='constant', constant_values=255)
                pww, phh = 0, 0

            # Inverse Stitching
            for row in range(n_row):
                for col in range(n_col):
                    if row * n_col + col < pred_single.shape[1]:
                        pred_single[:, row * n_col + col] = prediction_inshape[:, pww + row * s_l:pww + (row + 1) * s_l, phh + col * s_w:phh + (col + 1) * s_w]
            pred_single = pred_single[:, ps:pred_single.shape[-3] - ps, pw:pred_single.shape[-2] - pw, ph:pred_single.shape[-1] - ph]  # (B, 64, 128, 128)
            pred_single = np.swapaxes(pred_single, -3, -1)  # (B, 128, 128, 64)
            prediction_eps = pred_single


            prediction_binary = (prediction_eps == 1).astype(np.uint8)  # or whatever label is foreground
            # label_binary = (batch['semantic'].cpu().detach().numpy() == 1).astype(np.uint8)      # ground truth foreground

            eps_label_batch = np.where(eps_label_batch == 0, 
                    0, 
                    np.where(eps_label_batch == 255, 
                            1, 
                            eps_label_batch))
            
            label_binary = (eps_label_batch == 1).astype(np.uint8)  
            
            # Compute Dice score
            assert prediction_binary.shape == eps_label_batch.shape
            batch_dice = 0.0
            for b in range(prediction_binary.shape[0]):
                batch_dice += dice_score(prediction_binary[b], label_binary[b])
            # dice = dice_score(prediction_binary, label_binary)
            total_dice += batch_dice
            num_samples += prediction_binary.shape[0]
            mean_test_dice = total_dice / num_samples

        print(f"Mean Test Dice Score: {mean_test_dice:.4f}")
