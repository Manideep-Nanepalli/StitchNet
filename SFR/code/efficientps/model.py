# import os
# import torch
# from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
# from .fpn import TwoWayFpn
# from .fpn.two_way_fpn import OneWayFpn
# import pytorch_lightning as pl
# from .backbone import generate_backbone_EfficientPS, output_feature_size
# from .semantic_head import SemanticHead
# from .instance_head import InstanceHead
# from .panoptic_segmentation_module import panoptic_segmentation
# from .panoptic_metrics import generate_pred_panoptic
# from panopticapi.evaluation import pq_compute
# from torch.amp import autocast

# class EffificientPS(pl.LightningModule):
#     """
#     EfficientPS model see http://panoptic.cs.uni-freiburg.de/
#     Here pytorch lightningis used https://pytorch-lightning.readthedocs.io/en/latest/
#     """
    
#     def __init__(self, cfg):
#         """
#         Args:
#         - cfg (Config) : Config object from detectron2
#         """
#         super().__init__()
#         self.cfg = cfg
#         self.backbone = generate_backbone_EfficientPS(cfg)
#         self.fpn = TwoWayFpn(
#             output_feature_size[cfg.MODEL_CUSTOM.BACKBONE.EFFICIENTNET_ID])
#         self.semantic_head = SemanticHead(cfg.NUM_CLASS)
#         # self.instance_head = InstanceHead(cfg)

#     def forward(self, x):
#         # in lightning, forward defines the prediction/inference actions
#         predictions, _ = self.shared_step(x)
#         return predictions

#     def training_step(self, batch, batch_idx):
#         # training_step defined the train loop.
#         # It is independent of forward
#         _, loss = self.shared_step(batch)
#         # Add losses to logs
#         [self.log(k, v) for k,v in loss.items()]
#         self.log('train_loss', sum(loss.values()))
#         return {'loss': sum(loss.values())}

#     def shared_step(self, inputs):
#         loss = dict()
#         predictions = dict()
#         # Feature extraction
#         # with autocast(device_type='cuda', dtype=torch.float16):
#         features = self.backbone.extract_endpoints(inputs['image'])
#         # for i in features.keys():
#         #     print(f"{i} Shape : ", features[i].shape)
#         # raise Exception

#         pyramid_features = self.fpn(features)

#         # Heads Predictions
#         semantic_logits, semantic_loss = self.semantic_head(pyramid_features, inputs)
#         # pred_instance, instance_losses = self.instance_head(pyramid_features, inputs)
        
#         # Output set up
#         loss.update(semantic_loss)
#         # loss.update(instance_losses)
#         predictions.update({'semantic': semantic_logits})
#         # predictions.update({'instance': pred_instance})
#         return predictions, loss

#     def validation_step(self, batch, batch_idx):
#         predictions, loss = self.shared_step(batch)
#         panoptic_result = panoptic_segmentation(self.cfg,
#             predictions,
#             self.device)
#         return {
#             'val_loss': sum(loss.values()),
#             'panoptic': panoptic_result,
#             'image_id': batch['image_id']
#         }

#     def validation_epoch_end(self, outputs):
#         # Create and save all predictions files
#         generate_pred_panoptic(self.cfg, outputs)

#         # Compute PQ metric with panpticapi
#         pq_res = pq_compute(
#             gt_json_file= os.path.join(self.cfg.DATASET_PATH,
#                                        self.cfg.VALID_JSON),
#             pred_json_file= os.path.join(self.cfg.DATASET_PATH,
#                                          self.cfg.PRED_JSON),
#             gt_folder= os.path.join(self.cfg.DATASET_PATH,
#                                     "gtFine/cityscapes_panoptic_val/"),
#             pred_folder=os.path.join(self.cfg.DATASET_PATH, self.cfg.PRED_DIR)
#         )
#         self.log("PQ", 100 * pq_res["All"]["pq"])
#         self.log("SQ", 100 * pq_res["All"]["sq"])
#         self.log("RQ", 100 * pq_res["All"]["rq"])
#         self.log("PQ_th", 100 * pq_res["Things"]["pq"])
#         self.log("SQ_th", 100 * pq_res["Things"]["sq"])
#         self.log("RQ_th", 100 * pq_res["Things"]["rq"])
#         self.log("PQ_st", 100 * pq_res["Stuff"]["pq"])
#         self.log("SQ_st", 100 * pq_res["Stuff"]["sq"])
#         self.log("RQ_st", 100 * pq_res["Stuff"]["rq"])

#     def configure_optimizers(self):
#         if self.cfg.SOLVER.NAME == "Adam":
#             self.optimizer = torch.optim.Adam(self.parameters(),
#                                          lr=self.cfg.SOLVER.BASE_LR,
#                                          weight_decay=self.cfg.SOLVER.WEIGHT_DECAY)
#         elif self.cfg.SOLVER.NAME == "AdamW":
#             self.optimizer = torch.optim.AdamW(self.parameters(),
#                                          lr=self.cfg.SOLVER.BASE_LR,
#                                          weight_decay=self.cfg.SOLVER.WEIGHT_DECAY)
#         elif self.cfg.SOLVER.NAME == "SGD":
#             self.optimizer = torch.optim.SGD(self.parameters(),
#                                         lr=self.cfg.SOLVER.BASE_LR,
#                                         momentum=0.9,
#                                         weight_decay=self.cfg.SOLVER.WEIGHT_DECAY)
#         else:
#             raise ValueError("Solver name is not supported, \
#                 Adam or SGD : {}".format(self.cfg.SOLVER.NAME))
#         return {
#             'optimizer': self.optimizer,
#             'lr_scheduler': ReduceLROnPlateau(self.optimizer,
#                                               mode='max',
#                                               patience=3,
#                                               factor=0.1,
#                                               min_lr=1e-4,
#                                               verbose=True),
#             'monitor': 'PQ'
#         }

#     def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_idx, closure, on_tpu=False, using_native_amp=False, using_lbfgs=False):
#         # warm up lr
#         if self.trainer.global_step < self.cfg.SOLVER.WARMUP_ITERS:
#             lr_scale = min(1., float(self.trainer.global_step + 1) /
#                                     float(self.cfg.SOLVER.WARMUP_ITERS))
#             for pg in optimizer.param_groups:
#                 pg['lr'] = lr_scale * self.cfg.SOLVER.BASE_LR

#         # update params
#         optimizer.step(closure=closure)

#Unet ++

import os
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
# from .fpn import TwoWayFpn
# from .fpn.two_way_fpn import OneWayFpn
import pytorch_lightning as pl
# from .backbone import generate_backbone_EfficientPS, output_feature_size
# from .semantic_head import SemanticHead
# from .instance_head import InstanceHead
from .panoptic_segmentation_module import panoptic_segmentation
from .panoptic_metrics import generate_pred_panoptic
from panopticapi.evaluation import pq_compute
# from torch.amp import autocast

from efficientps.unet_plus_plus.model import NestedUNet
import torch.nn as nn
from torch.nn import functional as F
from math import ceil

class DiceLossBin(nn.Module):
    def __init__(self, n_classes, softmax=True):
        super(DiceLossBin, self).__init__()
        self.n_classes = n_classes
        self.softmax = softmax

    def forward(self, inputs, targets):
        """
        inputs: [B, C, H, W] — logits or probabilities
        targets: [B, 1, H, W] — integer labels (e.g. 0 or 1)
        """
        if self.softmax:
            inputs = F.softmax(inputs, dim=1)

        # Convert target to one-hot: [B, C, H, W]
        # targets = targets.squeeze(1)  # [B, H, W]
        
        targets_one_hot = F.one_hot(targets, num_classes=self.n_classes)  # [B, H, W, C]
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()      # [B, C, H, W]
        
        # Dice loss computation
        smooth = 1e-6
        intersection = torch.sum(inputs * targets_one_hot, dim=(2, 3))
        inputs_sum = torch.sum(inputs * inputs, dim=(2, 3))
        targets_sum = torch.sum(targets_one_hot * targets_one_hot, dim=(2, 3))

        dice = (2. * intersection + smooth) / (inputs_sum + targets_sum + smooth)
        loss = 1 - dice
        return loss.mean()
    
class EffificientPS(pl.LightningModule):
    """
    EfficientPS model see http://panoptic.cs.uni-freiburg.de/
    Here pytorch lightningis used https://pytorch-lightning.readthedocs.io/en/latest/
    """
    
    def __init__(self, cfg):
        """
        Args:
        - cfg (Config) : Config object from detectron2
        """
        super().__init__()
        self.cfg = cfg
        self.softmax = nn.Softmax(dim=1)
        self.net = NestedUNet(num_classes=cfg.NUM_CLASS, input_channels=3, deep_supervision=False)
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')
        self.dice_loss = DiceLossBin(n_classes=cfg.NUM_CLASS, softmax=True)
        # self.instance_head = InstanceHead(cfg)

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        predictions, _ = self.shared_step(x)
        return predictions

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        _, loss = self.shared_step(batch)
        # Add losses to logs
        [self.log(k, v) for k,v in loss.items()]
        self.log('train_loss', sum(loss.values()))
        return {'loss': sum(loss.values())}

    def shared_step(self, inputs):
        loss = dict()
        predictions = dict()
        
        semantic_logits = self.net(inputs['image'])

        if 'semantic' in inputs.keys():
            loss.update(self.loss(semantic_logits, inputs['semantic']))
        
        predictions.update({'semantic': self.softmax(semantic_logits)})


        return predictions, loss
    
    def loss(self, inputs, targets):
        """
        Weighted pixel loss, described in the paper as :
        if loss \in worst 25% of per pixel loss then w = 4/(H*W)
        else w = 0
        We keep 25% of each image appy the weigth and then compute the mean.
        """
        # First apply cross entropy on the image.
        # print(inputs.unique())
        # print(targets.unique())
        # raise Exception
        loss = self.cross_entropy_loss(inputs, targets)
        # import torch.nn.functional as F

        
        # sort the loss and take 25 % worst pixel
        # [B, 1, H, W] -> [B, H * W]ut
        loss = loss.view(loss.shape[0], -1)
        size = loss.shape[1]
        max_id = int(ceil(size * 0.25))
        sorted_loss = torch.sort(loss, descending=True).values
        kept_loss = sorted_loss[:, : max_id]
        kept_loss = kept_loss * 4.0 / size
        ce_loss_value = torch.sum(kept_loss) / loss.shape[0]

        # Compute Dice loss (you can define your own Dice or use torchmetrics)

        # Dice loss calculation for binary classification

        # probs = torch.softmax(inputs, dim=1)[:, 1, :, :]  # take class 1 (liver)

        # target_bin = (targets == 1).float()
        # intersection = (probs * target_bin).sum(dim=(1, 2))
        # union = probs.sum(dim=(1, 2)) + target_bin.sum(dim=(1, 2))

        # dice = 1 - ((2 * intersection + 1e-6) / (union + 1e-6))



        # dice_loss = dice.mean()

        dice_loss = self.dice_loss(inputs, targets)
        

        # dice_loss = dice.mean()
        
        # total_loss = 5.0 * ce_loss_value + 10.0 * dice_loss
        # total_loss = ce_loss_value

        return{
            'semantic_loss': ce_loss_value,
            'dice_loss': dice_loss
        }

    def configure_optimizers(self):
        if self.cfg.SOLVER.NAME == "Adam":
            self.optimizer = torch.optim.Adam(self.parameters(),
                                         lr=self.cfg.SOLVER.BASE_LR,
                                         weight_decay=self.cfg.SOLVER.WEIGHT_DECAY)
        elif self.cfg.SOLVER.NAME == "AdamW":
            self.optimizer = torch.optim.AdamW(self.parameters(),
                                         lr=self.cfg.SOLVER.BASE_LR,
                                         weight_decay=self.cfg.SOLVER.WEIGHT_DECAY)
        elif self.cfg.SOLVER.NAME == "SGD":
            self.optimizer = torch.optim.SGD(self.parameters(),
                                        lr=self.cfg.SOLVER.BASE_LR,
                                        momentum=0.9,
                                        weight_decay=self.cfg.SOLVER.WEIGHT_DECAY)
        else:
            raise ValueError("Solver name is not supported, \
                Adam or SGD : {}".format(self.cfg.SOLVER.NAME))
        return {
            'optimizer': self.optimizer,
            'lr_scheduler': ReduceLROnPlateau(self.optimizer,
                                              mode='max',
                                              patience=3,
                                              factor=0.1,
                                              min_lr=1e-4),
            'monitor': 'PQ'
        }

    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_idx, closure, on_tpu=False, using_native_amp=False, using_lbfgs=False):
        # warm up lr
        if self.trainer.global_step < self.cfg.SOLVER.WARMUP_ITERS:
            lr_scale = min(1., float(self.trainer.global_step + 1) /
                                    float(self.cfg.SOLVER.WARMUP_ITERS))
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.cfg.SOLVER.BASE_LR

        # update params
        optimizer.step(closure=closure)
