# python3 train_finetuning.py --gpu 0 --exp duke_50p_4by4_eca --label_num 154 --patch_size 256 --rdmrotflip
# python3 train_retraining.py --gpu 0 --exp 3d_seg_4by4_eca --label_num 154 --patch_size 256
# python3 test_singleclass.py --gpu 0 --model 3d_seg_4by4_eca --iteration 12000

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
parser.add_argument('--root_path', type=str, default='/mnt/ab146a07-47de-444d-8709-50981a4043c2/SFR_enet/data', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='name', help='model_name')
parser.add_argument('--dataset', type=str, default='atlas', help='dataset to use')
parser.add_argument('--label_num', type=int, default=14, help='number of labeled data')

parser.add_argument('--pretrain_model', type=str, default='vit_b', help='vit to select')
parser.add_argument('--patch_size', type=int, default=256, help='shape of data')
parser.add_argument('--input_size', type=int, default=1024, help='shape of data')
parser.add_argument('--num_classes', type=int, default=1, help='number of class')
parser.add_argument('--save_img', type=int, default=250, help='img saving iterations')
# load
parser.add_argument('--load', action="store_true", help='load net')
parser.add_argument('--load_iter', type=int, default=0, help='load iter')

parser.add_argument('--save_iter', type=int, default=1000, help='maximum epoch number to train')
parser.add_argument('--max_iterations', type=int, default=6000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=2, help='batch_size per gpu')

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
from torch.amp import autocast

# from torch.amp import GradScaler
# scaler = GradScaler()

# seed function

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

# import os
# import random
# import numpy as np
# import torch
# import torch.backends.cudnn as cudnn

# def set_reproducibility(seed: int, deterministic: bool = True):
#     if deterministic:
#         # cuDNN settings
#         cudnn.benchmark = False
#         cudnn.deterministic = True

#         # Set seeds for Python, NumPy, and PyTorch (CPU and GPU)
#         random.seed(seed)
#         np.random.seed(seed)
#         torch.manual_seed(seed)
#         torch.cuda.manual_seed(seed)

#         # Use deterministic algorithms (raises error if not possible)
#         torch.use_deterministic_algorithms(True)

#         # For deterministic cuBLAS operations
#         os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

#         print(f"[INFO] Deterministic mode enabled with seed {seed}")
#     else:
#         cudnn.benchmark = True
#         print(f"[INFO] Non-deterministic mode enabled (faster, less reproducible)")
        
# set_reproducibility(args.seed, args.deterministic)


def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)



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


# class CustomSliceSampler:
#     def __init__(self, target_depth=16, mask_ratio=0.8, pad_value_image=0, pad_value_label=255):
#         """
#         Initializes the custom slice sampler.

#         Args:
#             target_depth (int): The desired number of slices in the output volume.
#             mask_ratio (float): The desired ratio of masked (non-blank) slices
#                                  in the output volume (e.g., 0.8 for 80%).
#             pad_value_image (float): Value to pad image slices if not enough are available.
#             pad_value_label (int): Value to pad label slices if not enough are available.
#         """
#         self.target_depth = target_depth
#         self.num_mask_needed = round(target_depth * mask_ratio)
#         self.num_blank_needed = target_depth - self.num_mask_needed
#         self.pad_value_image = pad_value_image
#         self.pad_value_label = pad_value_label

#     def __call__(self, sample):
#         image, label = sample['image'], sample['label'] # Assuming shape (W, H, D) from Kits19

#         original_depth = image.shape[2] # Depth is the last dimension after Kits19 swapaxes
#         height, width = image.shape[0], image.shape[1]

#         # 1. Identify blank and non-blank slices
#         # Labels are binarized to 0 or 1 by Kits19. 255 is the padding value if added later.
#         mask_indices = []
#         blank_indices = []

#         for d in range(original_depth):
#             # Check if the slice contains any foreground pixels (value > 0)
#             if np.any(label[:, :, d] > 0):
#                 mask_indices.append(d)
#             else:
#                 blank_indices.append(d)
        
#         random.shuffle(mask_indices) # Shuffle to ensure randomness in selection
#         random.shuffle(blank_indices)

#         selected_slice_indices = []

#         # 2. Select masked slices
#         num_selected_mask = min(self.num_mask_needed, len(mask_indices))
#         selected_slice_indices.extend(mask_indices[:num_selected_mask])

#         # 3. Select blank slices
#         num_selected_blank = min(self.num_blank_needed, len(blank_indices))
#         selected_slice_indices.extend(blank_indices[:num_selected_blank])

#         # 4. Handle cases where total selected slices are less than target_depth
#         remaining_slots = self.target_depth - len(selected_slice_indices)

#         if remaining_slots > 0:
#             # First, try to fill from remaining masked slices
#             if len(mask_indices) > num_selected_mask:
#                 remaining_mask_indices = mask_indices[num_selected_mask:]
#                 num_to_add_from_mask = min(remaining_slots, len(remaining_mask_indices))
#                 selected_slice_indices.extend(random.sample(remaining_mask_indices, num_to_add_from_mask))
#                 remaining_slots -= num_to_add_from_mask
            
#             # Then, try to fill from remaining blank slices
#             if remaining_slots > 0 and len(blank_indices) > num_selected_blank:
#                 remaining_blank_indices = blank_indices[num_selected_blank:]
#                 num_to_add_from_blank = min(remaining_slots, len(remaining_blank_indices))
#                 selected_slice_indices.extend(random.sample(remaining_blank_indices, num_to_add_from_blank))
#                 remaining_slots -= num_to_add_from_blank
            
#             # 5. Pad if still not enough slices
#             if remaining_slots > 0:
#                 print(f"Warning: Not enough slices in volume (original depth {original_depth}) to meet target_depth ({self.target_depth}). Padding {remaining_slots} slices.")
#                 for _ in range(remaining_slots):
#                     # Add dummy index to signify padding later
#                     selected_slice_indices.append(-1) # Use a sentinel value for padding

#         # Ensure we have exactly target_depth indices
#         # If we selected too many due to complex sampling, trim. If too few, padding above handles it.
#         # This line ensures it's exactly target_depth for the slice extraction below
#         if len(selected_slice_indices) > self.target_depth:
#             selected_slice_indices = random.sample(selected_slice_indices, self.target_depth)
#         elif len(selected_slice_indices) < self.target_depth:
#             # This case should ideally be covered by the padding logic above
#             # But as a failsafe, if for some reason it's short, we add more dummy indices
#             selected_slice_indices.extend([-1] * (self.target_depth - len(selected_slice_indices)))

#         # Sort indices to maintain relative order if desired, or keep them random if that's an augmentation
#         # For stitching, maintaining relative order is usually preferred
#         selected_slice_indices.sort() 

#         # 6. Create the new cropped image and label volumes
#         cropped_image_slices = []
#         cropped_label_slices = []

#         for idx in selected_slice_indices:
#             if idx != -1: # It's a real slice
#                 cropped_image_slices.append(image[:, :, idx])
#                 cropped_label_slices.append(label[:, :, idx])
#             else: # It's a padded slice
#                 cropped_image_slices.append(np.full((height, width), self.pad_value_image, dtype=image.dtype))
#                 cropped_label_slices.append(np.full((height, width), self.pad_value_label, dtype=label.dtype))
        
#         # Stack the slices to form the new 3D volume
#         cropped_image = np.stack(cropped_image_slices, axis=2) # Shape (W, H, target_depth)
#         cropped_label = np.stack(cropped_label_slices, axis=2) # Shape (W, H, target_depth)

#         # Update the sample with the new cropped volumes
#         sample['image'] = cropped_image
#         sample['label'] = cropped_label.astype(np.uint8) # Ensure label dtype

#         return sample

cfg = get_cfg()
add_custom_param(cfg)
cfg.merge_from_file("/mnt/ab146a07-47de-444d-8709-50981a4043c2/SFR_enet/SFR/code/config.yaml")

logging.getLogger("pytorch_lightning").setLevel(logging.INFO)
logger = logging.getLogger("pytorch_lightning.core")
if not os.path.exists(cfg.CALLBACKS.CHECKPOINT_DIR):
    os.makedirs(cfg.CALLBACKS.CHECKPOINT_DIR)
logger.addHandler(logging.FileHandler(
    os.path.join(cfg.CALLBACKS.CHECKPOINT_DIR,"core.log")))
with open("/mnt/ab146a07-47de-444d-8709-50981a4043c2/SFR_enet/SFR/code/config.yaml") as file:
    logger.info(file.read())
_CURRENT_STORAGE_STACK.append(EventStorage())


if __name__ == "__main__":
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
        os.makedirs(snapshot_path + 'saveimg')
        os.makedirs(snapshot_path + 'savevalimg')

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    if args.dataset == 'duke':
        db_train = DukeLiverDataset(base_dir=train_data_path,
                            # list_num='0',  # or whichever list you want to use
                            num=args.label_num,
                            split='train',
                            transform=transforms.Compose([
                                RandomRotFlip(),
                                RandomCrop((patch_size, patch_size, 16)),
                                # CenterCrop3D((256, 256, 20)),
                                ToTensor(),
                            ]))
        # db_val = DukeLiverDataset(
        #                 base_dir=train_data_path,
        #                 # list_num='0',
        #                 num=args.label_num,
        #                 split='val',
        #                 transform=transforms.Compose([
        #                     # RandomRotFlip(),
        #                     RandomCrop((patch_size, patch_size, 16)),
        #                     ToTensor(),
        #                 ])
        #             )
        # db_test = DukeLiverDataset(
        #                 base_dir=train_data_path,
        #                 # list_num='0',
        #                 num=args.label_num,
        #                 split='test',
        #                 transform=transforms.Compose([
        #                     # RandomRotFlip(),
        #                     RandomCrop((patch_size, patch_size, 16)),
        #                     ToTensor(),
        #                 ])
        #             )
    elif args.dataset == 'kits19':
        db_train = Kits19(base_dir=train_data_path,
                            # list_num='0',  # or whichever list you want to use
                            num=args.label_num,
                            split='train',
                            transform=transforms.Compose([
                                RandomRotFlip(),
                                RandomCrop((patch_size, patch_size, 16)),
                                # CustomSliceSampler(target_depth=16, mask_ratio=0.8),
                                # CenterCrop3D((256, 256, 20)),
                                ToTensor(),
                            ]))
        # db_val = Kits19(
        #                 base_dir=train_data_path,
        #                 # list_num='0',
        #                 num=args.label_num,
        #                 split='val',
        #                 transform=transforms.Compose([
        #                     # RandomRotFlip(),
        #                     RandomCrop((patch_size, patch_size, 16)),
        #                     ToTensor(), 
        #                 ])
        #             )
        # db_test = Kits19(
        #                 base_dir=train_data_path,
        #                 # list_num='0',
        #                 num=args.label_num,
        #                 split='test',
        #                 transform=transforms.Compose([
        #                     # RandomRotFlip(),
        #                     RandomCrop((patch_size, patch_size, 16)),
        #                     ToTensor(), 
        #                 ])
        #             )
    elif args.dataset == 'atlas':
        # db_train = Atlas23(base_dir=train_data_path,
        #                     # list_num='0',  # or whichever list you want to use
        #                     num=args.label_num,
        #                     split='train',
        #                     transform=transforms.Compose([
        #                         RandomRotFlip(),
        #                         RandomCrop((patch_size, patch_size, 16)),
        #                         # CenterCrop3D((256, 256, 20)),
        #                         ToTensor(),
        #                     ]))
        
        db_train = AtlasIRCAD(base_dir=train_data_path,
                            # list_num='0',  # or whichever list you want to use
                            num=args.label_num,
                            split='train',
                            transform=transforms.Compose([
                                RandomRotFlip(),
                                RandomCrop((patch_size, patch_size, 16)),
                                # CenterCrop3D((256, 256, 20)),
                                ToTensor(),
                            ]))
        
        # db_val = Atlas23(
        #                 base_dir=train_data_path,
        #                 # list_num='0',
        #                 num=args.label_num,
        #                 split='val',
        #                 transform=transforms.Compose([
        #                     # RandomRotFlip(),
        #                     RandomCrop((patch_size, patch_size, 16)),
        #                     ToTensor(), 
        #                 ])
        #             )
        # db_test = Atlas23(
        #                 base_dir=train_data_path,
        #                 # list_num='0',
        #                 num=args.label_num,
        #                 split='test',
        #                 transform=transforms.Compose([
        #                     # RandomRotFlip(),
        #                     RandomCrop((patch_size, patch_size, 16)),
        #                     ToTensor(), 
        #                 ])
        #             )


    multimask_output = True if num_classes > 2 else False
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    # valloader = DataLoader(db_val, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    # testloader = DataLoader(db_test, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_dice = 0.0
    # def dice_score(pred, gt, smooth=1e-6):
    #         """
    #         Compute Dice coefficient between two binary volumes.
    #         """
    #         intersection = np.sum(pred * gt)
    #         volume_sum = np.sum(pred) + np.sum(gt)
    #         dice = (2. * intersection + smooth) / (volume_sum + smooth)
    #         return dice
    
    
    def dice_coeff(y_pred, y_true, smooth=1e-6):
        # import matplotlib.pyplot as plt

        # plt.figure(figsize=(16, 8))
        # for i in range(16):
        #     plt.subplot(4, 8, 2*i + 1)
        #     plt.imshow(y_true[:, :, i], cmap='gray')
        #     plt.title(f"GT {i}")
        #     plt.axis('off')

        #     plt.subplot(4, 8, 2*i + 2)
        #     plt.imshow(y_pred[:, :, i], cmap='gray')
        #     plt.title(f"Pred {i}")
        #     plt.axis('off')

        # plt.tight_layout()
        # plt.show()

        # raise Exception
        
        y_true_f = y_true.reshape(-1)
        y_pred_f = y_pred.reshape(-1)
        intersection = np.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)
    
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
        # checkpoint = torch.load("/mnt/e3dbc9b9-6856-470d-84b1-ff55921cd906/SFR_modeling/SFR/model_duke/eps_ft/eps_iter_6000.pth")
        # model.load_state_dict(checkpoint)
        

        # cfg.CHECKPOINT_PATH = None
    
    optimizer_config = model.configure_optimizers()
    optimizer = optimizer_config['optimizer']
    scheduler = optimizer_config['lr_scheduler']

    # # Set losses

    ce_loss = CrossEntropyLoss(ignore_index=255)
    dice_loss = losses.DiceLoss(num_classes+1)

    # writer = SummaryWriter(snapshot_path + '/log')
    # logging.info("{} itertations per epoch".format(len(trainloader)))

    iter_num = args.load_iter
    max_epoch = (max_iterations - args.load_iter) // len(trainloader) + 1
    kl_distance = torch.nn.KLDivLoss(reduction='none')
    # lr_ = base_lr_sam
    log_file = "dice_val_log_atlas.txt"

    log1_file = "dice_train_log_atlas.txt"

    
    # best_val_loss = float('inf') 

    for epoch_num in range(max_epoch):
        print(f"\n🧪 Epoch {epoch_num+1}/{max_epoch}")
        model.train()
        total_dice = 0.0
        num_samples = 0.0
        total_train_loss = 0.0
        
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']      # [2, 1, 128, 128, 64], [2, 128, 128, 64]

            ### Train EPS Module
            eps_volume_batch = volume_batch.cpu().detach().numpy()
            eps_label_batch = label_batch.cpu().detach().numpy()

            ## labeled data
            image = eps_volume_batch       # [B, 1, 128, 128, 64]
            label = eps_label_batch        # [B, 128, 128, 64]
            # print(image.shape, label.shape)
            # raise Exception            
            image_inshape, label_inshape, n_row, n_col, pw, ph, ps, pww, phh, s_l, s_w = Spread_bs_aug(image, label, input_size, args.nt, args.nonlinear_rate)  # (B, 1024, 1024)

            # import matplotlib.pyplot as plt

            # plt.figure(figsize=(10, 5))

            # for i in range(2):
            #     plt.subplot(1, 2, i + 1)
            #     plt.imshow(label_inshape[i], cmap='tab10', vmin=0, vmax=1)
            #     plt.title(f"Slice {i}")
            #     plt.axis('off')

            # plt.tight_layout()
            # plt.show()
            # raise Exception

            # continue
            

            if args.rdmrotflip:     # 2d RandomRotFlip
                k = np.array([np.random.randint(0, 4) for _ in range(image_inshape.shape[0])])
                axis = np.array([np.random.randint(0, 2) for _ in range(image_inshape.shape[0])])
                for i in range(image_inshape.shape[0]):
                    image_inshape[i] = RandomRotFlip_2d(image_inshape[i], k[i], axis[i])
                    label_inshape[i] = RandomRotFlip_2d(label_inshape[i], k[i], axis[i])

            volume_batch_inshape, label_batch_inshape = ToTensor_sam_bs(image_inshape, label_inshape)  # [B, 3, 1024, 1024], [B, 1024, 1024]
                         
            volume_batch_inshape = volume_batch_inshape.to(device)
            label_batch_inshape = label_batch_inshape.to(device)
            batch = dict()
            batch['image'] = volume_batch_inshape
            semantic = label_batch_inshape

            batch['semantic'] = torch.where(semantic == 0, torch.tensor(0, dtype=semantic.dtype, device=semantic.device),
                        torch.where(semantic == 255, torch.tensor(1, dtype=semantic.dtype, device=semantic.device), semantic))

            # with autocast(device_type='cuda', dtype=torch.float16):
            outputs, loss = model.shared_step(batch)
            output_masks = outputs['semantic']
            
            
            output_soft = F.softmax(output_masks, dim=1)                    # [1, 2, 1024, 1024]
            label_batch_inshape = torch.where(label_batch_inshape == 255,
                                  torch.tensor(1, dtype=label_batch_inshape.dtype, device=label_batch_inshape.device),
                                  label_batch_inshape)
            
            # print(output_soft.shape) #[2, 2, 1024, 1024]
            # print(label_batch_inshape.shape) # [2, 1024, 1024]
            # print(label_batch_inshape.unsqueeze(1).shape) # [2, 1, 1024, 1024]
            # print(output_soft[0][0,0,0], output_soft[0][1,0,0])
            #raise Exception

            ce_loss = loss['semantic_loss']
            dice_loss = loss['dice_loss']
            # loss_dice = dice_loss(output_soft, label_batch_inshape.unsqueeze(1))
            
            # loss += loss_dice

            # loss_eps = 0.5 * (loss + loss_dice)
            # loss_eps = 0.5 * ce_loss + 1 * dice_loss
            loss_eps = 5.0 * ce_loss + 10.0 * dice_loss


            optimizer.zero_grad()


            loss_eps.backward()
            optimizer.step()
            
            # scaler.scale(loss_eps).backward()
            # scaler.step(optimizer)
            # scaler.update()

            # loss_ce = ce_loss(output_masks, label_batch_inshape)

            prediction_inshape = torch.argmax(output_soft, dim=1)  # (B, 1024, 1024)
            prediction_inshape = prediction_inshape.cpu().detach().numpy()  # (B, 1024, 1024)

            if iter_num % args.save_img == 0:
                nib.save(nib.Nifti1Image(image_inshape.astype(np.float32), np.eye(4)), snapshot_path + '/saveimg' + '/inshape_img_' + str(iter_num) + '.nii.gz')
                nib.save(nib.Nifti1Image(label_inshape.astype(np.float32), np.eye(4)), snapshot_path + '/saveimg' + '/inshape_gt_' + str(iter_num) + '.nii.gz')
                nib.save(nib.Nifti1Image(prediction_inshape.astype(np.float32), np.eye(4)), snapshot_path + '/saveimg' + '/inshape_pred_' + str(iter_num) + '_u.nii.gz')

            if args.rdmrotflip:
                for i in range(image_inshape.shape[0]):
                    prediction_inshape[i] = RandomFlipRot_2d(prediction_inshape[i], k[i]*(-1), axis[i]*(-1))

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
            
            eps_label_batch = np.where(eps_label_batch == 0, 
                    0, 
                    np.where(eps_label_batch == 255, 
                            1, 
                            eps_label_batch))

            label_binary = (eps_label_batch == 1).astype(np.uint8)  
            # print(prediction_binary.shape, eps_label_batch.shape)
            
            assert prediction_binary.shape == eps_label_batch.shape
            batch_dice = 0.0
            for b in range(prediction_binary.shape[0]):
                batch_dice += dice_coeff(prediction_binary[b], label_binary[b])
            total_dice += batch_dice
            num_samples += prediction_binary.shape[0]
            mean_train_dice = total_dice / num_samples
            
            if iter_num % args.save_img == 0:
                nib.save(nib.Nifti1Image(image[0,0].astype(np.float32), np.eye(4)), snapshot_path + '/saveimg' + '/img_' + str(iter_num) + '.nii.gz')
                nib.save(nib.Nifti1Image(label[0].astype(np.float32), np.eye(4)), snapshot_path + '/saveimg' + '/gt_' + str(iter_num) + '.nii.gz')
                nib.save(nib.Nifti1Image(prediction_eps[0].astype(np.float32), np.eye(4)), snapshot_path + '/saveimg' + '/pred_eps_' + str(iter_num) + '_u.nii.gz')
            
            logging.info('iter %d : eps loss : %f, ce loss : %f, dice loss : %f' % (iter_num, loss_eps, ce_loss, dice_loss.item()))
            total_train_loss += loss_eps
            iter_num = iter_num + 1
            

            if iter_num >= max_iterations:
                break

        print("epoch Number :", epoch_num)
        print(f"Mean Train Dice Score: {mean_train_dice:.4f}")

        avg_train_loss = total_train_loss / iter_num
                
        # Save to file
        with open(log1_file, "a") as f:
            f.write(f"Epoch {epoch_num+1}, Mean Dice: {mean_train_dice:.4f}, Train Loss: {avg_train_loss:.4f}\n")
        
    
        scheduler.step(avg_train_loss.item())
        if iter_num >= max_iterations:
            break  

        

        # total_val_loss = 0.0 


        # num_val_iters = 0
        # with torch.no_grad():
        #     val_loader_tqdm = tqdm(valloader, desc=" Validating", leave=False)
        #     for i_batch, sampled_batch in enumerate(val_loader_tqdm):

        #         volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']      # [2, 1, 128, 128, 64], [2, 128, 128, 64]

        #         ### Train EPS Module
        #         eps_volume_batch = volume_batch.cpu().detach().numpy()
        #         eps_label_batch = label_batch.cpu().detach().numpy()

        #         ## labeled data
        #         image = eps_volume_batch       # [B, 1, 128, 128, 64]
        #         label = eps_label_batch        # [B, 128, 128, 64]
               

        #         image_inshape, label_inshape, n_row, n_col, pw, ph, ps, pww, phh, s_l, s_w = Spread_bs_aug(image, label, input_size, args.nt, args.nonlinear_rate)  # (B, 1024, 1024)

        #         # if args.rdmrotflip:     # 2d RandomRotFlip
        #         #     k = np.array([np.random.randint(0, 4) for _ in range(image_inshape.shape[0])])
        #         #     axis = np.array([np.random.randint(0, 2) for _ in range(image_inshape.shape[0])])
        #         #     for i in range(image_inshape.shape[0]):
        #         #         image_inshape[i] = RandomRotFlip_2d(image_inshape[i], k[i], axis[i])
        #         #         label_inshape[i] = RandomRotFlip_2d(label_inshape[i], k[i], axis[i])

        #         volume_batch_inshape, label_batch_inshape = ToTensor_sam_bs(image_inshape, label_inshape)  # [B, 3, 1024, 1024], [B, 1024, 1024]
                                   
        #         volume_batch_inshape = volume_batch_inshape.to(device)
        #         label_batch_inshape = label_batch_inshape.to(device)
        #         batch = dict()
        #         batch['image'] = volume_batch_inshape
        #         semantic = label_batch_inshape
                
        #         batch['semantic'] = torch.where(semantic == 0, torch.tensor(0, dtype=semantic.dtype, device=semantic.device),
        #         torch.where(semantic == 255, torch.tensor(1, dtype=semantic.dtype, device=semantic.device), semantic))
                
        #         outputs, loss = model.shared_step(batch)
        #         output_masks = outputs['semantic']
                
        #         output_soft = F.softmax(output_masks, dim=1)                    # [1, 2, 1024, 1024]
                            
        #         ce_loss = loss['semantic_loss']
        #         dice_loss = loss['dice_loss']
        #         # loss_dice = dice_loss(output_soft, label_batch_inshape.unsqueeze(1))
                
        #         # loss += loss_dice

        #         # loss_eps = 0.5 * (loss + loss_dice)
        #         loss_eps = 0.5 * ce_loss + 1 * dice_loss
        #         total_val_loss += loss_eps.item()
        #         num_val_iters += 1
        #         prediction_inshape = torch.argmax(output_soft, dim=1)  # (B, 1024, 1024)
                
        #         prediction_inshape = prediction_inshape.cpu().detach().numpy()  # (B, 1024, 1024)
                
        #         nib.save(nib.Nifti1Image(label_inshape.astype(np.float32), np.eye(4)), snapshot_path + '/savevalimg' + '/inshape_gt_' + str(iter_num) + '.nii.gz')
        #         nib.save(nib.Nifti1Image(prediction_inshape.astype(np.float32), np.eye(4)), snapshot_path + '/savevalimg' + '/inshape_pred_' + str(iter_num) + '_u.nii.gz')

        #         # if args.rdmrotflip:
        #         #     for i in range(image_inshape.shape[0]):
        #         #         prediction_inshape[i] = RandomFlipRot_2d(prediction_inshape[i], k[i]*(-1), axis[i]*(-1))

        #         pred_single = np.zeros((batch_size, image.shape[-1] + ps * 2, image.shape[-2] + pw * 2, image.shape[-3] + ph * 2))  # [B, 80, 112, 112]  (1, 112, 114, 86)
        #         if pww < 0 or phh < 0:
        #             pww_r, phh_r = pww, phh
        #             prediction_inshape = np.pad(prediction_inshape, [(0, 0), (abs(pww), abs(pww)), (abs(phh), abs(phh))], mode='constant', constant_values=255)
        #             pww, phh = 0, 0

        #         # Inverse Stitching
        #         for row in range(n_row):
        #             for col in range(n_col):
        #                 if row * n_col + col < pred_single.shape[1]:
        #                     pred_single[:, row * n_col + col] = prediction_inshape[:, pww + row * s_l:pww + (row + 1) * s_l, phh + col * s_w:phh + (col + 1) * s_w]
        #         pred_single = pred_single[:, ps:pred_single.shape[-3] - ps, pw:pred_single.shape[-2] - pw, ph:pred_single.shape[-1] - ph]  # (B, 64, 128, 128)
        #         pred_single = np.swapaxes(pred_single, -3, -1)  # (B, 128, 128, 64)
        #         prediction_eps = pred_single

        #         nib.save(nib.Nifti1Image(label[0].astype(np.float32), np.eye(4)), snapshot_path + '/savevalimg' + '/gt_' + str(iter_num) + '.nii.gz')
        #         nib.save(nib.Nifti1Image(prediction_eps[0].astype(np.float32), np.eye(4)), snapshot_path + '/savevalimg' + '/pred_eps_' + str(iter_num) + '_u.nii.gz')

        #         prediction_binary = (prediction_eps == 1).astype(np.uint8)  # or whatever label is foreground
        #         # label_binary = (batch['semantic'].cpu().detach().numpy() == 1).astype(np.uint8)      # ground truth foreground
                
        #         # eps_label_batch = np.where(eps_label_batch == 0, 
        #         #      0, 
        #         #      np.where(eps_label_batch == 255, 
        #         #               1, 
        #         #               eps_label_batch))
                
        #         label_binary = (eps_label_batch == 1).astype(np.uint8)  
                
        #         # Compute Dice score
        #         assert prediction_binary.shape == eps_label_batch.shape
        #         batch_dice = 0.0
        #         for b in range(prediction_binary.shape[0]):
        #             batch_dice += dice_coeff(prediction_binary[b], label_binary[b])
        #         total_dice += batch_dice
        #         num_samples += prediction_binary.shape[0]
        #         mean_val_dice = total_dice / num_samples

        #     print(f"Mean Val Dice Score: {mean_val_dice:.4f}")
            
        # avg_val_loss = total_val_loss / num_val_iters
        # print(f"Epoch {epoch_num+1}: Avg Val Loss = {avg_val_loss:.4f}")

        save_mode_path_eps = os.path.join(snapshot_path, 'eps_iter_best_' + str(iter_num) + '.pth')
        torch.save(model.state_dict(), save_mode_path_eps)
        # print(f"Saved Model at Epoch {epoch_num+1} with Val Loss = {avg_val_loss:.4f}")

        # # # --- SAVE BEST MODEL ---
        # # if avg_val_loss < best_val_loss:
        # #     best_val_loss = avg_val_loss
        # #     # model.save_lora_parameters(save_mode_path_sam)
        
        # total_dice = 0.0
        # num_samples = 0
        # with torch.no_grad():
        #     test_loader_tqdm = tqdm(testloader, desc="Testing", leave=False)
        #     for i_batch, sampled_batch in enumerate(test_loader_tqdm):

        #         volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']      # [2, 1, 128, 128, 64], [2, 128, 128, 64]

        #         ### Train EPS Module
        #         eps_volume_batch = volume_batch.cpu().detach().numpy()
        #         eps_label_batch = label_batch.cpu().detach().numpy()

        #         ## labeled data
        #         image = eps_volume_batch       # [B, 1, 128, 128, 64]
        #         label = eps_label_batch        # [B, 128, 128, 64]
               
        #         image_inshape, label_inshape, n_row, n_col, pw, ph, ps, pww, phh, s_l, s_w = Spread_bs_aug(image, label, input_size, args.nt, args.nonlinear_rate)  # (B, 1024, 1024)

        #         volume_batch_inshape, label_batch_inshape = ToTensor_sam_bs(image_inshape, label_inshape)  # [B, 3, 1024, 1024], [B, 1024, 1024]
                                   
        #         volume_batch_inshape = volume_batch_inshape.to(device)
        #         label_batch_inshape = label_batch_inshape.to(device)
        #         batch = dict()
        #         batch['image'] = volume_batch_inshape 
        #         semantic = label_batch_inshape
                
        #         batch['semantic'] = torch.where(semantic == 0, torch.tensor(0, dtype=semantic.dtype, device=semantic.device),
        #         torch.where(semantic == 255, torch.tensor(1, dtype=semantic.dtype, device=semantic.device), semantic))
                
        #         outputs, loss = model.shared_step(batch)
        #         output_masks = outputs['semantic']
                
        #         output_soft = F.softmax(output_masks, dim=1)                    # [1, 2, 1024, 1024]
                            
        #         prediction_inshape = torch.argmax(output_soft, dim=1)  # (B, 1024, 1024)
                
        #         prediction_inshape = prediction_inshape.cpu().detach().numpy()  # (B, 1024, 1024)
                
        #         nib.save(nib.Nifti1Image(label_inshape.astype(np.float32), np.eye(4)), snapshot_path + '/savevalimg' + '/inshape_gt_' + str(iter_num) + '.nii.gz')
        #         nib.save(nib.Nifti1Image(prediction_inshape.astype(np.float32), np.eye(4)), snapshot_path + '/savevalimg' + '/inshape_pred_' + str(iter_num) + '_u.nii.gz')

        #         pred_single = np.zeros((batch_size, image.shape[-1] + ps * 2, image.shape[-2] + pw * 2, image.shape[-3] + ph * 2))  # [B, 80, 112, 112]  (1, 112, 114, 86)
        #         if pww < 0 or phh < 0:
        #             pww_r, phh_r = pww, phh
        #             prediction_inshape = np.pad(prediction_inshape, [(0, 0), (abs(pww), abs(pww)), (abs(phh), abs(phh))], mode='constant', constant_values=255)
        #             pww, phh = 0, 0

        #         # Inverse Stitching
        #         for row in range(n_row):
        #             for col in range(n_col):
        #                 if row * n_col + col < pred_single.shape[1]:
        #                     pred_single[:, row * n_col + col] = prediction_inshape[:, pww + row * s_l:pww + (row + 1) * s_l, phh + col * s_w:phh + (col + 1) * s_w]
        #         pred_single = pred_single[:, ps:pred_single.shape[-3] - ps, pw:pred_single.shape[-2] - pw, ph:pred_single.shape[-1] - ph]  # (B, 64, 128, 128)
        #         pred_single = np.swapaxes(pred_single, -3, -1)  # (B, 128, 128, 64)
        #         prediction_eps = pred_single

        #         nib.save(nib.Nifti1Image(label[0].astype(np.float32), np.eye(4)), snapshot_path + '/savevalimg' + '/gt_' + str(iter_num) + '.nii.gz')
        #         nib.save(nib.Nifti1Image(prediction_eps[0].astype(np.float32), np.eye(4)), snapshot_path + '/savevalimg' + '/pred_eps_' + str(iter_num) + '_u.nii.gz')

        #         prediction_binary = (prediction_eps == 1).astype(np.uint8)  # or whatever label is foreground
        #         # label_binary = (batch['semantic'].cpu().detach().numpy() == 1).astype(np.uint8)      # ground truth foreground

        #         # eps_label_batch = np.where(eps_label_batch == 0, 
        #         #      0, 
        #         #      np.where(eps_label_batch == 255, 
        #         #               1, 
        #         #               eps_label_batch))
                
        #         label_binary = (eps_label_batch == 1).astype(np.uint8)  
                
        #         # Compute Dice score
        #         assert prediction_binary.shape == eps_label_batch.shape
        #         batch_dice = 0.0
        #         for b in range(prediction_binary.shape[0]):
        #             batch_dice += dice_coeff(prediction_binary[b], label_binary[b])
        #         # dice = dice_score(prediction_binary, label_binary)
        #         total_dice += batch_dice
        #         num_samples += prediction_binary.shape[0]
        #         mean_test_dice = total_dice / num_samples

        #     print(f"Mean Test Dice Score: {mean_test_dice:.4f}")
            
        #     # Save to file
        #     with open(log_file, "a") as f:
        #         f.write(f"Epoch {epoch_num+1}, Avg. Val Loss:{avg_val_loss:.4f}, Mean Val Dice: {mean_val_dice:.4f}, Mean Test Dice: {mean_test_dice:.4f}\n")


        # # save_mode_path_eps = os.path.join(snapshot_path, 'eps_iter_' + str(iter_num) + '.pth')

        # # # model_sam.save_lora_parameters(save_mode_path_sam)
        # # torch.save(model.state_dict(), save_mode_path_eps)
        # # logging.info("save model to {}".format(save_mode_path_eps))