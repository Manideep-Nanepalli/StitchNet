from math import ceil
import torch
import torch.nn as nn
from torch.nn import functional as F
from inplace_abn import InPlaceABN
import numpy as np

from efficientps.utils import DepthwiseSeparableConv

class DiceLoss(nn.Module):
    def __init__(self, n_classes, softmax=True):
        super(DiceLoss, self).__init__()
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
    
class SemanticHead(nn.Module):
    """
    Semantic Head compose of three main module DPC, LSFE and MC
    Args:
    - nb_class (int) : number of classes in the dataset
    """

    def __init__(self, nb_class):
        super().__init__()

        self.dpc_x32 = DPC()
        self.dpc_x16 = DPC()

        self.lsfe_x8 = LSFE()
        self.lsfe_x4 = LSFE()

        self.mc_16_to_8 = MC()
        self.mc_8_to_4 = MC()

        self.last_conv = nn.Conv2d(512, nb_class, 1)

        self.softmax = nn.Softmax(dim=1)
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')
        self.dice_loss = DiceLoss(n_classes=nb_class, softmax=True)


    def forward(self, inputs, targets={}):
        # TODO Make a loop
        # The forward is apply in a bottom up manner
        # x32 size
        p_32 = inputs['P_32']
        p_32 = self.dpc_x32(p_32)
        # [B, C, x32H, x32W] -> [B, C, x16H, x16W]
        p_32_to_merge = F.interpolate(
            p_32,
            scale_factor=(2, 2),
            mode='bilinear',
            align_corners=False)
        # [B, C, x16H, x16W] -> [B, C, x4H, x4W]
        p_32 = F.interpolate(
            p_32_to_merge,
            scale_factor=(4, 4),
            mode='bilinear',
            align_corners=False)

        # x16 size
        p_16 = inputs['P_16']
        p_16 = self.dpc_x16(p_16)
        p_16_to_merge = torch.add(p_32_to_merge, p_16)
        # [B, C, x16H, x16W] -> [B, C, x4H, x4W]
        p_16 = F.interpolate(
            p_16,
            scale_factor=(4, 4),
            mode='bilinear',
            align_corners=False)
        # [B, C, x16H, x16W] -> [B, C, x8H, x8W]
        p_16_to_merge = self.mc_16_to_8(p_16_to_merge)

        # x8 size
        p_8 = inputs['P_8']
        p_8 = self.lsfe_x8(p_8)
        p_8 = torch.add(p_16_to_merge, p_8)
        # [B, C, x8H, x8W] -> [B, C, x4H, x4W]
        p_8_to_merge = self.mc_8_to_4(p_8)
        # [B, C, x8H, x8W] -> [B, C, x4H, x4W]
        p_8 = F.interpolate(
            p_8,
            scale_factor=(2, 2),
            mode='bilinear',
            align_corners=False)

        # x4 size
        p_4 = inputs['P_4']
        p_4 = self.lsfe_x4(p_4)
        p_4 = torch.add(p_8_to_merge, p_4)

        # Create output
        # [B, 128, x4H, x4W] -> [B, 512, x4H, x4W]
        outputs = torch.cat((p_32, p_16, p_8, p_4), dim=1)
        outputs = self.last_conv(outputs)
        outputs = F.interpolate(
            outputs,
            scale_factor=(4, 4),
            mode='bilinear',
            align_corners=False)

        if 'semantic' in targets.keys():
            return self.softmax(outputs), self.loss(outputs, targets['semantic'])
        else:
            return self.softmax(outputs), {}

    
    def loss(self, inputs, targets):
        """
        Weighted pixel loss, described in the paper as :
        if loss \in worst 25% of per pixel loss then w = 4/(H*W)
        else w = 0
        We keep 25% of each image appy the weigth and then compute the mean.
        """
        # First apply cross entropy on the image.
        loss = self.cross_entropy_loss(inputs, targets)
        dice_loss = self.dice_loss(inputs, targets)
        # print(inputs.shape, targets.shape)
        # raise Exception
        # sort the loss and take 25 % worst pixel
        # [B, 1, H, W] -> [B, H * W]
        loss = loss.view(loss.shape[0], -1)
        size = loss.shape[1]
        max_id = int(ceil(size * 0.25))
        sorted_loss = torch.sort(loss, descending=True).values
        kept_loss = sorted_loss[:, : max_id]
        kept_loss = kept_loss * 4 / size
        kept_loss = torch.sum(kept_loss) / loss.shape[0]
        
        return {
            'semantic_loss': kept_loss,
            'dice_loss': dice_loss
        }

class LSFE(nn.Module):

    def __init__(self):
        super().__init__()
        # Separable Conv
        self.conv_1 = DepthwiseSeparableConv(256, 128, 3, padding=1)
        self.conv_2 = DepthwiseSeparableConv(128, 128, 3, padding=1)
        # Inplace BN + Leaky Relu
        self.abn_1 = InPlaceABN(128)
        self.abn_2 = InPlaceABN(128)

    def forward(self, inputs):
        # Apply first conv
        outputs = self.conv_1(inputs)
        outputs = self.abn_1(outputs)

        # Apply second conv
        outputs = self.conv_2(outputs)
        return self.abn_2(outputs)


class MC(nn.Module):

    def __init__(self):
        super().__init__()
        # Separable Conv
        self.conv_1 = DepthwiseSeparableConv(128, 128, 3, padding=1)
        self.conv_2 = DepthwiseSeparableConv(128, 128, 3, padding=1)
        # Inplace BN + Leaky Relu
        self.abn_1 = InPlaceABN(128)
        self.abn_2 = InPlaceABN(128)

    def forward(self, inputs):
        # Apply first conv
        outputs = self.conv_1(inputs)
        outputs = self.abn_1(outputs)

        # Apply second conv
        outputs = self.conv_2(outputs)
        outputs = self.abn_2(outputs)

        # Apply conv
        # outputs = self.lfse(inputs)

        # Return upsample features
        return F.interpolate(
            outputs,
            scale_factor=(2, 2),
            mode='bilinear',
            align_corners=False)

class DPC(nn.Module):

    def __init__(self):
        super().__init__()
        options = {
            'in_channels'   : 256,
            'out_channels'  : 256,
            'kernel_size'   : 3
        }
        self.conv_first = DepthwiseSeparableConv(dilation=(1, 6),
                                                 padding=(1, 6),
                                                 **options)
        self.iabn_first = InPlaceABN(256)
        # Branch 1
        self.conv_branch_1 = DepthwiseSeparableConv(padding=1,
                                                    **options)
        self.iabn_branch_1 = InPlaceABN(256)
        # Branch 2
        self.conv_branch_2 = DepthwiseSeparableConv(dilation=(6, 21),
                                                    padding=(6, 21),
                                                    **options)
        self.iabn_branch_2 = InPlaceABN(256)
        #Branch 3
        self.conv_branch_3 = DepthwiseSeparableConv(dilation=(18, 15),
                                                    padding=(18, 15),
                                                    **options)
        self.iabn_branch_3 = InPlaceABN(256)
        # Branch 4
        self.conv_branch_4 = DepthwiseSeparableConv(dilation=(6, 3),
                                                    padding=(6, 3),
                                                    **options)
        self.iabn_branch_4 = InPlaceABN(256)
        # Last conv
        # There is some mismatch in the paper about the dimension of this conv
        # In the paper it says "This tensor is then finally passed through a
        # 1×1 convolution with 256 output channels and forms the output of the
        # DPC module." But the overall schema shows an output of 128
        # The MC module schema also show an input of 256.
        # In order to have 512 channel at the concatenation of all layers,
        # I choosed 128 output channels
        self.conv_last = nn.Conv2d(1280, 128, 1)
        self.iabn_last = InPlaceABN(128)

    def forward(self, inputs):
        # First conv
        inputs = self.conv_first(inputs)
        inputs = self.iabn_first(inputs)
        # Branch 1
        branch_1 = self.conv_branch_1(inputs)
        branch_1 = self.iabn_branch_1(branch_1)
        # Branch 2
        branch_2 = self.conv_branch_2(inputs)
        branch_2 = self.iabn_branch_2(branch_2)
        # Branch 3
        branch_3 = self.conv_branch_3(inputs)
        branch_3 = self.iabn_branch_3(branch_3)
        # Branch 4 (take branch 3 as input)
        branch_4 = self.conv_branch_4(branch_3)
        branch_4 = self.iabn_branch_4(branch_4)
        # Concatenate
        # [B, 256, H, W] -> [B, 1280, H, W]
        concat = torch.cat(
            (inputs, branch_1, branch_2, branch_3, branch_4),
            dim=1)
        # Last conv
        outputs = self.conv_last(concat)
        return self.iabn_last(outputs)
