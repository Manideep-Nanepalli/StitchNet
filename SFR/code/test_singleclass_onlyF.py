import os
import argparse
import torch
import numpy as np
from networks.hierarchical_vnet import VNet
# from networks.hierarchical_unet_3d import UNet_3D
from test_util_singleclass import test_all_case
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/mnt/ab146a07-47de-444d-8709-50981a4043c2/SFR_enet/data', help='Name of Experiment')
parser.add_argument('--model', type=str,  default='sparse_mt_294', help='model_name')
parser.add_argument('--dataset', type=str,  default='atlas', help='dataset to use')
parser.add_argument('--data_version', type=str,  default='v2', help='dataset version to use')
parser.add_argument('--set_version', type=str,  default='0', help='dataset version to use')
parser.add_argument('--semantic_class', type=str, default='kidney', choices=['kidney', 'tumor'])
parser.add_argument('--list_num', type=str,  default='', help='data list to use')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--iteration', type=int,  default=12000, help='GPU to use')
parser.add_argument('--patch_size', type=int, default=256, help='patch size')
parser.add_argument('--model_type', type=str, default='eps', help='model_type')
args = parser.parse_args()

root = "../"

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
snapshot_path = root + "model_" + args.dataset + "/" + args.model + "/"
test_save_path = root + "model_" + args.dataset + "/prediction/" + args.model + "_post/"
if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)

num_classes = 2

if args.dataset == 'la':
    with open(args.root_path + '/../test' + args.list_num + '.list', 'r') as f:
        image_list = f.readlines()
    image_list = [args.root_path + item.replace('\n', '') + "/mri_norm2.h5" for item in image_list]
elif args.dataset == 'btcv':
    num_classes = 14
    with open(args.root_path + '/../test' + args.list_num + '.list', 'r') as f:
        image_list = f.readlines()
    image_list = [args.root_path + '/' + item.replace('\n', '') + ".h5" for item in image_list]
elif args.dataset == 'mact':
    num_classes = 9
    with open(args.root_path + '/../test' + args.list_num + '.list', 'r') as f:
        image_list = f.readlines()
    image_list = [args.root_path + '/' + item.replace('\n', '') + ".h5" for item in image_list]
elif args.dataset == 'brats':
    with open(args.root_path + '/../test_follow.list', 'r') as f:
        image_list = f.readlines()
    image_list = [args.root_path + '/' + item.replace('\n', '') + ".h5" for item in image_list]

elif args.dataset == 'duke':
    
    with open(args.root_path + '/../test.list', 'r') as f:
    # with open("/mnt/e3dbc9b9-6856-470d-84b1-ff55921cd906/SFR_modeling/liver_data/train.list", 'r') as f:
        image_list = f.readlines()
    image_list = [item.replace('\n', '') for item in image_list]
    # image_list = image_list[:260]
elif args.dataset == 'atlas':
    
    with open(args.root_path + '/test_atlas.list', 'r') as f:
    # with open(args.root_path + '/train_atlas.list', 'r') as f:
        image_list = f.readlines()
    image_list = [item.replace('\n', '') for item in image_list]
    
elif args.dataset == 'kits19':
    
    with open('/mnt/e3dbc9b9-6856-470d-84b1-ff55921cd906/SFR_modeling/test_kits.list', 'r') as f:
    # with open(args.root_path + '/train_atlas.list', 'r') as f:
        image_list = f.readlines()
    image_list = [item.replace('\n', '') for item in image_list]

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
cfg.merge_from_file("/mnt/ab146a07-47de-444d-8709-50981a4043c2/SFR_enet/SFR/code/config.yaml")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_calculate_metric(epoch_num):
    # if args.model_type == "vnet":
    # net = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=False, pyramid_has_dropout=False).cuda()
    # save_mode_path = os.path.join(snapshot_path, 'iter_' + str(epoch_num) + '.pth')
    # net.load_state_dict(torch.load(save_mode_path))
    
    # elif args.model_type == "unet_3d":
    #     net = UNet_3D(in_channels=1, n_classes=num_classes).cuda()
    net = EffificientPS(cfg).to(device)
    # save_model_path = f"/mnt/ab146a07-47de-444d-8709-50981a4043c2/SFR_enet/SFR/model_atlas/ircad_unet++/eps_iter_best_{epoch_num}.pth"
    # save_model_path = f"/mnt/ab146a07-47de-444d-8709-50981a4043c2/SFR_enet/Best model/IRCAD/ircad_unet++_5705.pth"
    save_model_path = f"/mnt/ab146a07-47de-444d-8709-50981a4043c2/SFR_enet/Best model/ATLAS/atlas_unet++_11784.pth"

    checkpoint = torch.load(save_model_path)
    net.load_state_dict(checkpoint)
    print("init weight from {}".format(save_model_path))
    net.eval()

    if args.dataset == 'la':
        if args.patch_size == 112:
            ps = (112, 112, 80)
        elif args.patch_size == 128:
            ps = (128, 128, 64)
        avg_metric = test_all_case(net, args.dataset, args.semantic_class, image_list, num_classes=num_classes, patch_size=ps,
                                   save_result=True, stride_xy=18, stride_z=4, test_save_path=test_save_path)

    elif args.dataset == 'btcv' or args.dataset == 'mact':
        patch_size = args.patch_size
        avg_metric = test_all_case(net, args.dataset, args.semantic_class, image_list, num_classes=num_classes, patch_size=(patch_size, patch_size, patch_size),
                                   save_result=True, stride_xy=12, stride_z=12, test_save_path=test_save_path)

    elif args.dataset == 'brats':
        patch_size = args.patch_size
        avg_metric = test_all_case(net, args.dataset, args.semantic_class, image_list, num_classes=num_classes, patch_size=(patch_size, patch_size, patch_size),
                                   save_result=True, stride_xy=64, stride_z=64, test_save_path=test_save_path)
    
    elif args.dataset == 'duke':
        patch_size = args.patch_size
        avg_metric = test_all_case(net, args.dataset, args.semantic_class, image_list, num_classes=num_classes, patch_size=(patch_size, patch_size, 16),
                                   save_result=True, stride_xy=18, stride_z=4, test_save_path=test_save_path)
        
    elif args.dataset == 'atlas' or args.dataset == 'kits19':
        patch_size = args.patch_size
        avg_metric = test_all_case(net, args.dataset, args.semantic_class, image_list, num_classes=num_classes, patch_size=(patch_size, patch_size, 16),
                                   save_result=True, stride_xy=18, stride_z=4, test_save_path=test_save_path)
    return avg_metric


if __name__ == '__main__':
    metric, std = test_calculate_metric(5705)

    # best_average = 0.0
    # best_epoch = 0
    # for epoch in range(800*7, 858*7, 7):
    #     metric, std = test_calculate_metric(epoch)
    #     if best_average < metric[0]:
    #         best_average = metric[0]
    #         best_epoch = epoch
    # print(best_average, epoch)

    print(metric)
    with open(root + "model_" + args.dataset + "/prediction_v2.txt", "a") as f:
        f.write(args.model + " - " + str(args.iteration) + ": " + ", ".join(str(i) for i in metric) + "\n")
        f.write(args.model + " - " + str(args.iteration) + ": " + ", ".join(str(i) for i in std) + "\n")
