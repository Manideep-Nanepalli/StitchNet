
import torch
import torch.nn as nn
from torch.nn import Conv2d
import torch.nn.functional as F
# from inplace_abn import InPlaceABN

from efficientps.utils import DepthwiseSeparableConv

class OneWayFpn(nn.Module):
    """
    One-way (top-down) Feature Pyramid Network:
    - Takes EfficientNet-B7 reductions at x4, x8, x16, x32 levels
    - Applies standard top-down FPN fusion as described in literature
    - Matches TwoWayFPN input/output interface
    """
    def __init__(self, in_feature_shape):
        super().__init__()
        # 1×1 conv to unify channel dimensions
        self.lateral_conv_32 = nn.Conv2d(in_feature_shape[4], 256, 1)
        self.lateral_conv_16 = nn.Conv2d(in_feature_shape[3], 256, 1)
        self.lateral_conv_8  = nn.Conv2d(in_feature_shape[2], 256, 1)
        self.lateral_conv_4  = nn.Conv2d(in_feature_shape[1], 256, 1)

        # 3×3 conv to refine after fusion
        self.out_conv_32 = nn.Conv2d(256, 256, 3, padding=1)
        self.out_conv_16 = nn.Conv2d(256, 256, 3, padding=1)
        self.out_conv_8  = nn.Conv2d(256, 256, 3, padding=1)
        self.out_conv_4  = nn.Conv2d(256, 256, 3, padding=1)

    def forward(self, features):
        """
        Args:
            features (dict): Dictionary of backbone feature maps with keys:
                'reduction_2' (x4 resolution)
                'reduction_3' (x8)
                'reduction_4' (x16)
                'reduction_6' (x32)
        Returns:
            dict: Pyramidal features {P_4, P_8, P_16, P_32}, all with 256 channels
        """
        # Step 1: Lateral conv to unify channel dims
        lat_32 = self.lateral_conv_32(features['reduction_6'])
        lat_16 = self.lateral_conv_16(features['reduction_4'])
        lat_8  = self.lateral_conv_8(features['reduction_3'])
        lat_4  = self.lateral_conv_4(features['reduction_2'])

        # Step 2: Top-down fusion
        P_32 = lat_32
        P_16 = lat_16 + F.interpolate(P_32, size=lat_16.shape[2:], mode='nearest')
        P_8  = lat_8  + F.interpolate(P_16, size=lat_8.shape[2:], mode='nearest')
        P_4  = lat_4  + F.interpolate(P_8,  size=lat_4.shape[2:], mode='nearest')

        # Step 3: Final 3×3 conv refinement
        P_32 = self.out_conv_32(P_32)
        P_16 = self.out_conv_16(P_16)
        P_8  = self.out_conv_8(P_8)
        P_4  = self.out_conv_4(P_4)

        return {
            'P_4': P_4,
            'P_8': P_8,
            'P_16': P_16,
            'P_32': P_32
        }


class TwoWayFpn(nn.Module):
    """
    This FPN takes use feature from 4 level of the Backbone (x4, x8, x16,
    x32) corresponding to the size comparaison to the input.
    It applies lateral conv to set all channel to 256 and then concatenate
    in a descending way (bottom up) as well as ascending way (top bottom)
    to retrieve feature from diverse scales.
    """
    # TODO Reformat with functions
    def __init__(self, in_feature_shape):
        """
        Args:
        - in_feature_shape (List[int]) : size of feature at different levels
        """
        super().__init__()
        # Channel information are the one given in the EfficientPS paper
        # Depending on the EfficientNet model chosen the number of channel will
        # change
        # x4 size [B, 40, H, W] (input 40 channels)
        # Bottom up path layers
        self.conv_b_up_x4 = Conv2d(in_feature_shape[1], 256, 1)
        self.iabn_b_up_x4 = InPlaceABN(256)

        # Top down path layers
        self.conv_t_dn_x4 = Conv2d(in_feature_shape[1], 256, 1)
        self.iabn_t_dn_x4 = InPlaceABN(256)

        # x8 size [B, 64, H, W] (input 64 channels)
        # Bottom up path layers
        self.conv_b_up_x8 = Conv2d(in_feature_shape[2], 256, 1)
        self.iabn_b_up_x8 = InPlaceABN(256)

        # Top down path layers
        self.conv_t_dn_x8 = Conv2d(in_feature_shape[2], 256, 1)
        self.iabn_t_dn_x8 = InPlaceABN(256)

        # x16 size [B, 176, H, W] (input 176 channels)
        # In the paper they took the 5 block of efficient net ie 128 channels
        # But taking last block seem more pertinent and was already implemented
        # Skipping to id 3 since block 4 does not interest us
        # Bottom up path layers
        self.conv_b_up_x16 = Conv2d(in_feature_shape[3], 256, 1)
        self.iabn_b_up_x16 = InPlaceABN(256)

        # Top down path layers
        self.conv_t_dn_x16 = Conv2d(in_feature_shape[3], 256, 1)
        self.iabn_t_dn_x16 = InPlaceABN(256)

        # x32 size [B, 2048, H, W] (input 2048 channels)
        # Bottom up path layers
        self.conv_b_up_x32 = Conv2d(in_feature_shape[4], 256, 1)
        self.iabn_b_up_x32 = InPlaceABN(256)

        # Top down path layers
        self.conv_t_dn_x32 = Conv2d(in_feature_shape[4], 256, 1)
        self.iabn_t_dn_x32 = InPlaceABN(256)

        # Separable Conv and Inplace BN at the output of the FPN
        # x4
        self.depth_wise_conv_x4 = DepthwiseSeparableConv(
            in_channels=256,
            out_channels=256,
            kernel_size=3,
            padding=1)
        self.iabn_out_x4 = InPlaceABN(256)
        # x8
        self.depth_wise_conv_x8 = DepthwiseSeparableConv(
            in_channels=256,
            out_channels=256,
            kernel_size=3,
            padding=1)
        self.iabn_out_x8 = InPlaceABN(256)
        # x16
        self.depth_wise_conv_x16 = DepthwiseSeparableConv(
            in_channels=256,
            out_channels=256,
            kernel_size=3,
            padding=1)
        self.iabn_out_x16 = InPlaceABN(256)
        # x32
        self.depth_wise_conv_x32 = DepthwiseSeparableConv(
            in_channels=256,
            out_channels=256,
            kernel_size=3,
            padding=1)
        self.iabn_out_x32 = InPlaceABN(256)

    def forward(self, inputs):
        """
        Args:
        - inputs (dict[tensor]) : Features from the backbone
        Returns:
        - outputs (dict[tensor]) : The 4 levels of features
        """
        #################################
        # Bottom up part of the network #
        #################################
        # x4 size
        # [B, C, x4W, x4H]
        b_up_x4 = inputs['reduction_2']
        # [B, C, x4W, x4H] -> [B, 256, x4W, x4H]
        b_up_x4 = self.conv_b_up_x4(b_up_x4)
        b_up_x4 = self.iabn_b_up_x4(b_up_x4)
        # [B, 256, x4W, x4H] -> [B, 256, x8W, x8H]
        b_up_x4_to_merge = F.interpolate(
            b_up_x4,
            size=(inputs['reduction_3'].shape[2],
                  inputs['reduction_3'].shape[3]),
            mode='nearest'
        )

        # x8 size
        # [B, C, x8W, x8H]
        b_up_x8 = inputs['reduction_3']
        # [B, C, x8W, x8H] -> [B, 256, x8W, x8H]
        b_up_x8 = self.conv_b_up_x8(b_up_x8)
        b_up_x8 = self.iabn_b_up_x8(b_up_x8)
        b_up_x8 = torch.add(b_up_x4_to_merge, b_up_x8)
        # [B, 256, x8W, x8H] -> [B, 256, x16W, x16H]
        b_up_x8_to_merge = F.interpolate(
            b_up_x8,
            size=(inputs['reduction_4'].shape[2],
                  inputs['reduction_4'].shape[3]),
            mode='nearest'
        )

        #x16 size (reduction_4 since we don't need block 4)
        # [B, C, x16W, x16H]
        b_up_x16 = inputs['reduction_4']
        # [B, C, x16W, x16H] -> [B, 256, x16W, x16H]
        b_up_x16 = self.conv_b_up_x16(b_up_x16)
        b_up_x16 = self.iabn_b_up_x16(b_up_x16)
        b_up_x16 = torch.add(b_up_x8_to_merge, b_up_x16)
        # [B, 256, x16W, x16H] -> [B, 256, x32W, x32H]
        b_up_x16_to_merge = F.interpolate(
            b_up_x16,
            size=(inputs['reduction_6'].shape[2],
                  inputs['reduction_6'].shape[3]),
            mode='nearest'
        )

        #x32 size
        # [B, C, x32W, x32H]
        b_up_x32 = inputs['reduction_6']
        # [B, C, x32W, x32H] -> [B, 256, x32W, x32H]
        b_up_x32 = self.conv_b_up_x32(b_up_x32)
        b_up_x32 = self.iabn_b_up_x32(b_up_x32)
        b_up_x32 = torch.add(b_up_x16_to_merge, b_up_x32)

        ################################
        # Top down part of the network #
        ################################

        # x32 size
        # [B, C, x32W, x32H]
        t_dn_x32 = inputs['reduction_6']
        # [B, C, x32W, x32H] -> [B, 256, x32W, x32H]
        t_dn_x32 = self.conv_t_dn_x32(t_dn_x32)
        t_dn_x32 = self.iabn_t_dn_x32(t_dn_x32)
        # [B, 256, x32W, x32H] -> [B, 256, x16W, x16H]
        t_dn_x32_to_merge = F.interpolate(
            t_dn_x32,
            size=(inputs['reduction_4'].shape[2],
                  inputs['reduction_4'].shape[3]),
            mode='nearest'
        )
        # Create output
        p_32 = torch.add(t_dn_x32, b_up_x32)
        p_32 = self.depth_wise_conv_x32(p_32)
        p_32 = self.iabn_out_x32(p_32)

        # x16 size
        # [B, C, x16W, x16H]
        t_dn_x16 = inputs['reduction_4']
        # [B, C, x16W, x16H] -> [B, 256, x16W, x16H]
        t_dn_x16 = self.conv_t_dn_x16(t_dn_x16)
        t_dn_x16 = self.iabn_t_dn_x16(t_dn_x16)
        t_dn_x16 = torch.add(t_dn_x32_to_merge, t_dn_x16)
        # [B, 256, x16W, x16H] -> [B, 256, x32W, x32H]
        t_dn_x16_to_merge =  F.interpolate(
            t_dn_x16,
            size=(inputs['reduction_3'].shape[2],
                  inputs['reduction_3'].shape[3]),
            mode='nearest'
        )
        # Create output
        p_16 = torch.add(t_dn_x16, b_up_x16)
        p_16 = self.depth_wise_conv_x16(p_16)
        p_16 = self.iabn_out_x16(p_16)

        # x8 size
        # [B, C, x8W, x8H]
        t_dn_x8 = inputs['reduction_3']
        # [B, C, x8W, x8H] -> [B, 256, x8W, x8H]
        t_dn_x8 = self.conv_t_dn_x8(t_dn_x8)
        t_dn_x8 = self.iabn_t_dn_x8(t_dn_x8)
        t_dn_x8 = torch.add(t_dn_x16_to_merge, t_dn_x8)
        # [B, 256, x8W, x8H] -> [B, 256, x4W, x4H]
        t_dn_x8_to_merge = F.interpolate(
            t_dn_x8,
            size=(inputs['reduction_2'].shape[2],
                  inputs['reduction_2'].shape[3]),
            mode='nearest'
        )
        # Create output
        p_8 = torch.add(t_dn_x8, b_up_x8)
        p_8 = self.depth_wise_conv_x8(p_8)
        p_8 = self.iabn_out_x8(p_8)

        # x4 size
        # [B, C, x4W, x4H]
        t_dn_x4 = inputs['reduction_2']
        # [B, C, x4W, x4H] -> [B, 256, x4W, x4H]
        t_dn_x4 = self.conv_t_dn_x4(t_dn_x4)
        t_dn_x4 = self.iabn_t_dn_x4(t_dn_x4)
        t_dn_x4 = torch.add(t_dn_x8_to_merge, t_dn_x4)

        # Create outputs
        p_4 = torch.add(t_dn_x4, b_up_x4)
        p_4 = self.depth_wise_conv_x4(p_4)
        p_4 = self.iabn_out_x4(p_4)

        return {
            'P_4': p_4,
            'P_8': p_8,
            'P_16': p_16,
            'P_32': p_32
        }


# CBAM

# import torch
# import torch.nn as nn
# from torch.nn import Conv2d
# import torch.nn.functional as F
# from inplace_abn import InPlaceABN

# from efficientps.utils import DepthwiseSeparableConv
# from efficientps.cbam import CBAM


# class TwoWayFpn(nn.Module):
#     """
#     This FPN takes use feature from 4 level of the Backbone (x4, x8, x16,
#     x32) corresponding to the size comparaison to the input.
#     It applies lateral conv to set all channel to 256 and then concatenate
#     in a descending way (bottom up) as well as ascending way (top bottom)
#     to retrieve feature from diverse scales.
#     """
#     # TODO Reformat with functions
#     def __init__(self, in_feature_shape):
#         """
#         Args:
#         - in_feature_shape (List[int]) : size of feature at different levels
#         """
#         super().__init__()
#         # Channel information are the one given in the EfficientPS paper
#         # Depending on the EfficientNet model chosen the number of channel will
#         # change
#         # x4 size [B, 40, H, W] (input 40 channels)
#         # Bottom up path layers
#         self.conv_b_up_x4 = Conv2d(in_feature_shape[1], 256, 1)
#         self.iabn_b_up_x4 = InPlaceABN(256)

#         # Top down path layers
#         self.conv_t_dn_x4 = Conv2d(in_feature_shape[1], 256, 1)
#         self.iabn_t_dn_x4 = InPlaceABN(256)

#         # x8 size [B, 64, H, W] (input 64 channels)
#         # Bottom up path layers
#         self.conv_b_up_x8 = Conv2d(in_feature_shape[2], 256, 1)
#         self.iabn_b_up_x8 = InPlaceABN(256)

#         # Top down path layers
#         self.conv_t_dn_x8 = Conv2d(in_feature_shape[2], 256, 1)
#         self.iabn_t_dn_x8 = InPlaceABN(256)

#         # x16 size [B, 176, H, W] (input 176 channels)
#         # In the paper they took the 5 block of efficient net ie 128 channels
#         # But taking last block seem more pertinent and was already implemented
#         # Skipping to id 3 since block 4 does not interest us
#         # Bottom up path layers
#         self.conv_b_up_x16 = Conv2d(in_feature_shape[3], 256, 1)
#         self.iabn_b_up_x16 = InPlaceABN(256)

#         # Top down path layers
#         self.conv_t_dn_x16 = Conv2d(in_feature_shape[3], 256, 1)
#         self.iabn_t_dn_x16 = InPlaceABN(256)

#         # x32 size [B, 2048, H, W] (input 2048 channels)
#         # Bottom up path layers
#         self.conv_b_up_x32 = Conv2d(in_feature_shape[4], 256, 1)
#         self.iabn_b_up_x32 = InPlaceABN(256)

#         # Top down path layers
#         self.conv_t_dn_x32 = Conv2d(in_feature_shape[4], 256, 1)
#         self.iabn_t_dn_x32 = InPlaceABN(256)

#         # Separable Conv and Inplace BN at the output of the FPN
#         # x4
#         self.depth_wise_conv_x4 = DepthwiseSeparableConv(
#             in_channels=256,
#             out_channels=256,
#             kernel_size=3,
#             padding=1)
#         self.iabn_out_x4 = InPlaceABN(256)
#         # x8
#         self.depth_wise_conv_x8 = DepthwiseSeparableConv(
#             in_channels=256,
#             out_channels=256,
#             kernel_size=3,
#             padding=1)
#         self.iabn_out_x8 = InPlaceABN(256)
#         # x16
#         self.depth_wise_conv_x16 = DepthwiseSeparableConv(
#             in_channels=256,
#             out_channels=256,
#             kernel_size=3,
#             padding=1)
#         self.iabn_out_x16 = InPlaceABN(256)
#         # x32
#         self.depth_wise_conv_x32 = DepthwiseSeparableConv(
#             in_channels=256,
#             out_channels=256,
#             kernel_size=3,
#             padding=1)
#         self.iabn_out_x32 = InPlaceABN(256)
#         self.cbam_p4 = CBAM(256)
#         self.cbam_p8 = CBAM(256)
#         self.cbam_p16 = CBAM(256)
#         self.cbam_p32 = CBAM(256)
#         self.cbam_b_up_x4 = CBAM(256)
#         self.cbam_b_up_x8 = CBAM(256)
#         self.cbam_b_up_x16 = CBAM(256)
#         self.cbam_b_up_x32 = CBAM(256)


#     def forward(self, inputs):
#         """
#         Args:
#         - inputs (dict[tensor]) : Features from the backbone
#         Returns:
#         - outputs (dict[tensor]) : The 4 levels of features
#         """
#         #################################
#         # Bottom up part of the network #
#         #################################
#         # x4 size
#         # [B, C, x4W, x4H]
#         b_up_x4 = inputs['reduction_2']
#         # [B, C, x4W, x4H] -> [B, 256, x4W, x4H]
#         b_up_x4 = self.conv_b_up_x4(b_up_x4)
#         b_up_x4 = self.iabn_b_up_x4(b_up_x4)
#         b_up_x4 = self.cbam_b_up_x4(b_up_x4)
#         # [B, 256, x4W, x4H] -> [B, 256, x8W, x8H]
#         b_up_x4_to_merge = F.interpolate(
#             b_up_x4,
#             size=(inputs['reduction_3'].shape[2],
#                   inputs['reduction_3'].shape[3]),
#             mode='nearest'
#         )

#         # x8 size
#         # [B, C, x8W, x8H]
#         b_up_x8 = inputs['reduction_3']
#         # [B, C, x8W, x8H] -> [B, 256, x8W, x8H]
#         b_up_x8 = self.conv_b_up_x8(b_up_x8)
#         b_up_x8 = self.iabn_b_up_x8(b_up_x8)
#         b_up_x8 = self.cbam_b_up_x8(b_up_x8)
#         b_up_x8 = torch.add(b_up_x4_to_merge, b_up_x8)
#         # [B, 256, x8W, x8H] -> [B, 256, x16W, x16H]
#         b_up_x8_to_merge = F.interpolate(
#             b_up_x8,
#             size=(inputs['reduction_4'].shape[2],
#                   inputs['reduction_4'].shape[3]),
#             mode='nearest'
#         )

#         #x16 size (reduction_4 since we don't need block 4)
#         # [B, C, x16W, x16H]
#         b_up_x16 = inputs['reduction_4']
#         # [B, C, x16W, x16H] -> [B, 256, x16W, x16H]
#         b_up_x16 = self.conv_b_up_x16(b_up_x16)
#         b_up_x16 = self.iabn_b_up_x16(b_up_x16)
#         b_up_x16 = self.cbam_b_up_x16(b_up_x16)
#         b_up_x16 = torch.add(b_up_x8_to_merge, b_up_x16)
#         # [B, 256, x16W, x16H] -> [B, 256, x32W, x32H]
#         b_up_x16_to_merge = F.interpolate(
#             b_up_x16,
#             size=(inputs['reduction_6'].shape[2],
#                   inputs['reduction_6'].shape[3]),
#             mode='nearest'
#         )

#         #x32 size
#         # [B, C, x32W, x32H]
#         b_up_x32 = inputs['reduction_6']
#         # [B, C, x32W, x32H] -> [B, 256, x32W, x32H]
#         b_up_x32 = self.conv_b_up_x32(b_up_x32)
#         b_up_x32 = self.iabn_b_up_x32(b_up_x32)
#         b_up_x32 = self.cbam_b_up_x32(b_up_x32)
#         b_up_x32 = torch.add(b_up_x16_to_merge, b_up_x32)

#         ################################
#         # Top down part of the network #
#         ################################

#         # x32 size
#         # [B, C, x32W, x32H]
#         t_dn_x32 = inputs['reduction_6']
#         # [B, C, x32W, x32H] -> [B, 256, x32W, x32H]
#         t_dn_x32 = self.conv_t_dn_x32(t_dn_x32)
#         t_dn_x32 = self.iabn_t_dn_x32(t_dn_x32)
#         # [B, 256, x32W, x32H] -> [B, 256, x16W, x16H]
#         t_dn_x32_to_merge = F.interpolate(
#             t_dn_x32,
#             size=(inputs['reduction_4'].shape[2],
#                   inputs['reduction_4'].shape[3]),
#             mode='nearest'
#         )
#         # Create output
#         p_32 = torch.add(t_dn_x32, b_up_x32)
#         p_32 = self.depth_wise_conv_x32(p_32)
#         p_32 = self.iabn_out_x32(p_32)
#         p_32 = self.cbam_p32(p_32)

#         # x16 size
#         # [B, C, x16W, x16H]
#         t_dn_x16 = inputs['reduction_4']
#         # [B, C, x16W, x16H] -> [B, 256, x16W, x16H]
#         t_dn_x16 = self.conv_t_dn_x16(t_dn_x16)
#         t_dn_x16 = self.iabn_t_dn_x16(t_dn_x16)
#         t_dn_x16 = torch.add(t_dn_x32_to_merge, t_dn_x16)
#         # [B, 256, x16W, x16H] -> [B, 256, x32W, x32H]
#         t_dn_x16_to_merge =  F.interpolate(
#             t_dn_x16,
#             size=(inputs['reduction_3'].shape[2],
#                   inputs['reduction_3'].shape[3]),
#             mode='nearest'
#         )
#         # Create output
#         p_16 = torch.add(t_dn_x16, b_up_x16)
#         p_16 = self.depth_wise_conv_x16(p_16)
#         p_16 = self.iabn_out_x16(p_16)
#         p_16 = self.cbam_p16(p_16)

#         # x8 size
#         # [B, C, x8W, x8H]
#         t_dn_x8 = inputs['reduction_3']
#         # [B, C, x8W, x8H] -> [B, 256, x8W, x8H]
#         t_dn_x8 = self.conv_t_dn_x8(t_dn_x8)
#         t_dn_x8 = self.iabn_t_dn_x8(t_dn_x8)
#         t_dn_x8 = torch.add(t_dn_x16_to_merge, t_dn_x8)
#         # [B, 256, x8W, x8H] -> [B, 256, x4W, x4H]
#         t_dn_x8_to_merge = F.interpolate(
#             t_dn_x8,
#             size=(inputs['reduction_2'].shape[2],
#                   inputs['reduction_2'].shape[3]),
#             mode='nearest'
#         )
#         # Create output
#         p_8 = torch.add(t_dn_x8, b_up_x8)
#         p_8 = self.depth_wise_conv_x8(p_8)
#         p_8 = self.iabn_out_x8(p_8)
#         p_8 = self.cbam_p8(p_8)

#         # x4 size
#         # [B, C, x4W, x4H]
#         t_dn_x4 = inputs['reduction_2']
#         # [B, C, x4W, x4H] -> [B, 256, x4W, x4H]
#         t_dn_x4 = self.conv_t_dn_x4(t_dn_x4)
#         t_dn_x4 = self.iabn_t_dn_x4(t_dn_x4)
#         t_dn_x4 = torch.add(t_dn_x8_to_merge, t_dn_x4)

#         # Create outputs
#         p_4 = torch.add(t_dn_x4, b_up_x4)
#         p_4 = self.depth_wise_conv_x4(p_4)
#         p_4 = self.iabn_out_x4(p_4)
#         p_4 = self.cbam_p4(p_4)

#         return {
#             'P_4': p_4,
#             'P_8': p_8,
#             'P_16': p_16,
#             'P_32': p_32
#         }

# # ECA

# import torch
# import torch.nn as nn
# from torch.nn import Conv2d
# import torch.nn.functional as F
# from inplace_abn import InPlaceABN

# from efficientps.utils import DepthwiseSeparableConv
# from .eca_module import eca_layer


# class TwoWayFpn(nn.Module):
#     """
#     This FPN takes use feature from 4 level of the Backbone (x4, x8, x16,
#     x32) corresponding to the size comparaison to the input.
#     It applies lateral conv to set all channel to 256 and then concatenate
#     in a descending way (bottom up) as well as ascending way (top bottom)
#     to retrieve feature from diverse scales.
#     """
#     # TODO Reformat with functions
#     def __init__(self, in_feature_shape):
#         """
#         Args:
#         - in_feature_shape (List[int]) : size of feature at different levels
#         """
#         super().__init__()
#         # Channel information are the one given in the EfficientPS paper
#         # Depending on the EfficientNet model chosen the number of channel will
#         # change
#         # x4 size [B, 40, H, W] (input 40 channels)
#         # Bottom up path layers
#         self.conv_b_up_x4 = Conv2d(in_feature_shape[1], 256, 1)
#         self.iabn_b_up_x4 = InPlaceABN(256)
#         self.eca_b_up_x4 = eca_layer(256)

#         # Top down path layers
#         self.conv_t_dn_x4 = Conv2d(in_feature_shape[1], 256, 1)
#         self.iabn_t_dn_x4 = InPlaceABN(256)
#         self.eca_t_dn_x4 = eca_layer(256)

#         # x8 size [B, 64, H, W] (input 64 channels)
#         # Bottom up path layers
#         self.conv_b_up_x8 = Conv2d(in_feature_shape[2], 256, 1)
#         self.iabn_b_up_x8 = InPlaceABN(256)
#         self.eca_b_up_x8 = eca_layer(256)

#         # Top down path layers
#         self.conv_t_dn_x8 = Conv2d(in_feature_shape[2], 256, 1)
#         self.iabn_t_dn_x8 = InPlaceABN(256)
#         self.eca_t_dn_x8 = eca_layer(256)

#         # x16 size [B, 176, H, W] (input 176 channels)
#         # In the paper they took the 5 block of efficient net ie 128 channels
#         # But taking last block seem more pertinent and was already implemented
#         # Skipping to id 3 since block 4 does not interest us
#         # Bottom up path layers
#         self.conv_b_up_x16 = Conv2d(in_feature_shape[3], 256, 1)
#         self.iabn_b_up_x16 = InPlaceABN(256)
#         self.eca_b_up_x16 = eca_layer(256)

#         # Top down path layers
#         self.conv_t_dn_x16 = Conv2d(in_feature_shape[3], 256, 1)
#         self.iabn_t_dn_x16 = InPlaceABN(256)
#         self.eca_t_dn_x16 = eca_layer(256)

#         # x32 size [B, 2048, H, W] (input 2048 channels)
#         # Bottom up path layers
#         self.conv_b_up_x32 = Conv2d(in_feature_shape[4], 256, 1)
#         self.iabn_b_up_x32 = InPlaceABN(256)
#         self.eca_b_up_x32 = eca_layer(256)

#         # Top down path layers
#         self.conv_t_dn_x32 = Conv2d(in_feature_shape[4], 256, 1)
#         self.iabn_t_dn_x32 = InPlaceABN(256)
#         self.eca_t_dn_x32 = eca_layer(256)

#         # Separable Conv and Inplace BN at the output of the FPN
#         # x4
#         self.depth_wise_conv_x4 = DepthwiseSeparableConv(
#             in_channels=256,
#             out_channels=256,
#             kernel_size=3,
#             padding=1)
#         self.iabn_out_x4 = InPlaceABN(256)
#         self.eca_out_x4 = eca_layer(256)
#         # x8
#         self.depth_wise_conv_x8 = DepthwiseSeparableConv(
#             in_channels=256,
#             out_channels=256,
#             kernel_size=3,
#             padding=1)
#         self.iabn_out_x8 = InPlaceABN(256)
#         self.eca_out_x8 = eca_layer(256)
#         # x16
#         self.depth_wise_conv_x16 = DepthwiseSeparableConv(
#             in_channels=256,
#             out_channels=256,
#             kernel_size=3,
#             padding=1)
#         self.iabn_out_x16 = InPlaceABN(256)
#         self.eca_out_x16 = eca_layer(256)
#         # x32
#         self.depth_wise_conv_x32 = DepthwiseSeparableConv(
#             in_channels=256,
#             out_channels=256,
#             kernel_size=3,
#             padding=1)
#         self.iabn_out_x32 = InPlaceABN(256)
#         self.eca_out_x32 = eca_layer(256)

#     def forward(self, inputs):
#         # <unchanged code above...>
#         # Apply ECA after IABN at every layer (both bottom-up and top-down paths, and outputs)
#         b_up_x4 = self.eca_b_up_x4(self.iabn_b_up_x4(self.conv_b_up_x4(inputs['reduction_2'])))
#         b_up_x4_to_merge = F.interpolate(b_up_x4, size=inputs['reduction_3'].shape[2:], mode='nearest')

#         b_up_x8 = self.eca_b_up_x8(self.iabn_b_up_x8(self.conv_b_up_x8(inputs['reduction_3'])))
#         b_up_x8 = torch.add(b_up_x4_to_merge, b_up_x8)
#         b_up_x8_to_merge = F.interpolate(b_up_x8, size=inputs['reduction_4'].shape[2:], mode='nearest')

#         b_up_x16 = self.eca_b_up_x16(self.iabn_b_up_x16(self.conv_b_up_x16(inputs['reduction_4'])))
#         b_up_x16 = torch.add(b_up_x8_to_merge, b_up_x16)
#         b_up_x16_to_merge = F.interpolate(b_up_x16, size=inputs['reduction_6'].shape[2:], mode='nearest')

#         b_up_x32 = self.eca_b_up_x32(self.iabn_b_up_x32(self.conv_b_up_x32(inputs['reduction_6'])))
#         b_up_x32 = torch.add(b_up_x16_to_merge, b_up_x32)

#         t_dn_x32 = self.eca_t_dn_x32(self.iabn_t_dn_x32(self.conv_t_dn_x32(inputs['reduction_6'])))
#         t_dn_x32_to_merge = F.interpolate(t_dn_x32, size=inputs['reduction_4'].shape[2:], mode='nearest')

#         p_32 = self.eca_out_x32(self.iabn_out_x32(self.depth_wise_conv_x32(torch.add(t_dn_x32, b_up_x32))))

#         t_dn_x16 = self.eca_t_dn_x16(self.iabn_t_dn_x16(self.conv_t_dn_x16(inputs['reduction_4'])))
#         t_dn_x16 = torch.add(t_dn_x32_to_merge, t_dn_x16)
#         t_dn_x16_to_merge = F.interpolate(t_dn_x16, size=inputs['reduction_3'].shape[2:], mode='nearest')

#         p_16 = self.eca_out_x16(self.iabn_out_x16(self.depth_wise_conv_x16(torch.add(t_dn_x16, b_up_x16))))

#         t_dn_x8 = self.eca_t_dn_x8(self.iabn_t_dn_x8(self.conv_t_dn_x8(inputs['reduction_3'])))
#         t_dn_x8 = torch.add(t_dn_x16_to_merge, t_dn_x8)
#         t_dn_x8_to_merge = F.interpolate(t_dn_x8, size=inputs['reduction_2'].shape[2:], mode='nearest')

#         p_8 = self.eca_out_x8(self.iabn_out_x8(self.depth_wise_conv_x8(torch.add(t_dn_x8, b_up_x8))))

#         t_dn_x4 = self.eca_t_dn_x4(self.iabn_t_dn_x4(self.conv_t_dn_x4(inputs['reduction_2'])))
#         t_dn_x4 = torch.add(t_dn_x8_to_merge, t_dn_x4)

#         p_4 = self.eca_out_x4(self.iabn_out_x4(self.depth_wise_conv_x4(torch.add(t_dn_x4, b_up_x4))))

#         return {
#             'P_4': p_4,
#             'P_8': p_8,
#             'P_16': p_16,
#             'P_32': p_32
#         }


# FA

# import torch
# import torch.nn as nn
# from torch.nn import Conv2d
# import torch.nn.functional as F
# from inplace_abn import InPlaceABN
# from efficientps.flash_attn import flash_attn_qkvpacked_func
# from efficientps.utils import DepthwiseSeparableConv
# # from torch.amp import autocast

# # Used before
# class FlashAttention2D(nn.Module):
#     def __init__(self, dim, num_heads=8, dropout=0.0):
#         super().__init__()
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.qkv_proj = nn.Conv2d(dim, dim * 3, kernel_size=1)
#         self.out_proj = nn.Conv2d(dim, dim, kernel_size=1)
#         self.dropout = dropout

#     def forward(self, x):
#         # with autocast(device_type='cuda', dtype=torch.float16):
#         B, C, H, W = x.shape
#         N = H * W
#         qkv = self.qkv_proj(x).reshape(B, 3, self.num_heads, self.head_dim, H, W)
#         qkv = qkv.permute(0, 4, 5, 1, 2, 3).reshape(B, N, 3, self.num_heads, self.head_dim)
#         out = flash_attn_qkvpacked_func(qkv, dropout_p=self.dropout, softmax_scale=None, causal=False,
#                                         window_size=(-1, -1), softcap=0.0, alibi_slopes=None,
#                                         deterministic=True, return_attn_probs=False)
#         out = out.reshape(B, H, W, self.num_heads * self.head_dim).permute(0, 3, 1, 2)
#         out = self.out_proj(out)
#         return out
    
    

# class TwoWayFpn(nn.Module):
#     """
#     This FPN takes use feature from 4 level of the Backbone (x4, x8, x16,
#     x32) corresponding to the size comparaison to the input.
#     It applies lateral conv to set all channel to 256 and then concatenate
#     in a descending way (bottom up) as well as ascending way (top bottom)
#     to retrieve feature from diverse scales.
#     """
#     # TODO Reformat with functions
#     def __init__(self, in_feature_shape):
#         """
#         Args:
#         - in_feature_shape (List[int]) : size of feature at different levels
#         """
#         super().__init__()
#         self.flash_attn_x4 = FlashAttention2D(256)
#         self.flash_attn_x8 = FlashAttention2D(256)
#         self.flash_attn_x16 = FlashAttention2D(256)
#         self.flash_attn_x32 = FlashAttention2D(256)

#         # Channel information are the one given in the EfficientPS paper
#         # Depending on the EfficientNet model chosen the number of channel will
#         # change
#         # x4 size [B, 40, H, W] (input 40 channels)
#         # Bottom up path layers
#         self.conv_b_up_x4 = Conv2d(in_feature_shape[1], 256, 1)
#         self.iabn_b_up_x4 = InPlaceABN(256)

#         # Top down path layers
#         self.conv_t_dn_x4 = Conv2d(in_feature_shape[1], 256, 1)
#         self.iabn_t_dn_x4 = InPlaceABN(256)

#         # x8 size [B, 64, H, W] (input 64 channels)
#         # Bottom up path layers
#         self.conv_b_up_x8 = Conv2d(in_feature_shape[2], 256, 1)
#         self.iabn_b_up_x8 = InPlaceABN(256)

#         # Top down path layers
#         self.conv_t_dn_x8 = Conv2d(in_feature_shape[2], 256, 1)
#         self.iabn_t_dn_x8 = InPlaceABN(256)

#         # x16 size [B, 176, H, W] (input 176 channels)
#         # In the paper they took the 5 block of efficient net ie 128 channels
#         # But taking last block seem more pertinent and was already implemented
#         # Skipping to id 3 since block 4 does not interest us
#         # Bottom up path layers
#         self.conv_b_up_x16 = Conv2d(in_feature_shape[3], 256, 1)
#         self.iabn_b_up_x16 = InPlaceABN(256)

#         # Top down path layers
#         self.conv_t_dn_x16 = Conv2d(in_feature_shape[3], 256, 1)
#         self.iabn_t_dn_x16 = InPlaceABN(256)

#         # x32 size [B, 2048, H, W] (input 2048 channels)
#         # Bottom up path layers
#         self.conv_b_up_x32 = Conv2d(in_feature_shape[4], 256, 1)
#         self.iabn_b_up_x32 = InPlaceABN(256)

#         # Top down path layers
#         self.conv_t_dn_x32 = Conv2d(in_feature_shape[4], 256, 1)
#         self.iabn_t_dn_x32 = InPlaceABN(256)

#         # Separable Conv and Inplace BN at the output of the FPN
#         # x4
#         self.depth_wise_conv_x4 = DepthwiseSeparableConv(
#             in_channels=256,
#             out_channels=256,
#             kernel_size=3,
#             padding=1)
#         self.iabn_out_x4 = InPlaceABN(256)

#         # x8
#         self.depth_wise_conv_x8 = DepthwiseSeparableConv(
#             in_channels=256,
#             out_channels=256,
#             kernel_size=3,
#             padding=1)
#         self.iabn_out_x8 = InPlaceABN(256)

#         # x16
#         self.depth_wise_conv_x16 = DepthwiseSeparableConv(
#             in_channels=256,
#             out_channels=256,
#             kernel_size=3,
#             padding=1)
#         self.iabn_out_x16 = InPlaceABN(256)

#         # x32
#         self.depth_wise_conv_x32 = DepthwiseSeparableConv(
#             in_channels=256,
#             out_channels=256,
#             kernel_size=3,
#             padding=1)
#         self.iabn_out_x32 = InPlaceABN(256)
#         # self.bn_x4 = nn.BatchNorm2d(256)
#         # self.bn_x8 = nn.BatchNorm2d(256)
#         # self.bn_x16 = nn.BatchNorm2d(256)
#         # self.bn_x32 = nn.BatchNorm2d(256)

#         # Grouped conv after depthwise conv, before flash attention
#         # self.grouped_conv_x4 = ResNeXtGroupedConvBlock(in_channels=256)
#         # self.grouped_conv_x8 = ResNeXtGroupedConvBlock(in_channels=256)
#         # self.grouped_conv_x16 = ResNeXtGroupedConvBlock(in_channels=256)
#         # self.grouped_conv_x32 = ResNeXtGroupedConvBlock(in_channels=256)


#     def forward(self, inputs):
#         """
#         Args:
#         - inputs (dict[tensor]) : Features from the backbone
#         Returns:
#         - outputs (dict[tensor]) : The 4 levels of features
#         """
#         #################################
#         # Bottom up part of the network #
#         #################################
#         # x4 size
#         # [B, C, x4W, x4H]
#         b_up_x4 = inputs['reduction_2']
#         # [B, C, x4W, x4H] -> [B, 256, x4W, x4H]
#         b_up_x4 = self.conv_b_up_x4(b_up_x4)
#         b_up_x4 = self.iabn_b_up_x4(b_up_x4)
#         # [B, 256, x4W, x4H] -> [B, 256, x8W, x8H]
#         b_up_x4_to_merge = F.interpolate(
#             b_up_x4,
#             size=(inputs['reduction_3'].shape[2],
#                   inputs['reduction_3'].shape[3]),
#             mode='nearest'
#         )

#         # x8 size
#         # [B, C, x8W, x8H]
#         b_up_x8 = inputs['reduction_3']
#         # [B, C, x8W, x8H] -> [B, 256, x8W, x8H]
#         b_up_x8 = self.conv_b_up_x8(b_up_x8)
#         b_up_x8 = self.iabn_b_up_x8(b_up_x8)
#         b_up_x8 = torch.add(b_up_x4_to_merge, b_up_x8)
#         # [B, 256, x8W, x8H] -> [B, 256, x16W, x16H]
#         b_up_x8_to_merge = F.interpolate(
#             b_up_x8,
#             size=(inputs['reduction_4'].shape[2],
#                   inputs['reduction_4'].shape[3]),
#             mode='nearest'
#         )

#         #x16 size (reduction_4 since we don't need block 4)
#         # [B, C, x16W, x16H]
#         b_up_x16 = inputs['reduction_4']
#         # [B, C, x16W, x16H] -> [B, 256, x16W, x16H]
#         b_up_x16 = self.conv_b_up_x16(b_up_x16)
#         b_up_x16 = self.iabn_b_up_x16(b_up_x16)
#         b_up_x16 = torch.add(b_up_x8_to_merge, b_up_x16)
#         # [B, 256, x16W, x16H] -> [B, 256, x32W, x32H]
#         b_up_x16_to_merge = F.interpolate(
#             b_up_x16,
#             size=(inputs['reduction_6'].shape[2],
#                   inputs['reduction_6'].shape[3]),
#             mode='nearest'
#         )

#         #x32 size
#         # [B, C, x32W, x32H]
#         b_up_x32 = inputs['reduction_6']
#         # [B, C, x32W, x32H] -> [B, 256, x32W, x32H]
#         b_up_x32 = self.conv_b_up_x32(b_up_x32)
#         b_up_x32 = self.iabn_b_up_x32(b_up_x32)
#         b_up_x32 = torch.add(b_up_x16_to_merge, b_up_x32)

#         ################################
#         # Top down part of the network #
#         ################################

#         # x32 size
#         # [B, C, x32W, x32H]
#         t_dn_x32 = inputs['reduction_6']
#         # [B, C, x32W, x32H] -> [B, 256, x32W, x32H]
#         t_dn_x32 = self.conv_t_dn_x32(t_dn_x32)
#         t_dn_x32 = self.iabn_t_dn_x32(t_dn_x32)
#         # [B, 256, x32W, x32H] -> [B, 256, x16W, x16H]
#         t_dn_x32_to_merge = F.interpolate(
#             t_dn_x32,
#             size=(inputs['reduction_4'].shape[2],
#                   inputs['reduction_4'].shape[3]),
#             mode='nearest'
#         )
#         # Create output
#         p_32 = torch.add(t_dn_x32, b_up_x32)    
#         p_32 = self.depth_wise_conv_x32(p_32)
#         p_32 = self.iabn_out_x32(p_32)
#         # p_32 = self.grouped_conv_x32(p_32)
#         p_32 = self.flash_attn_x32(p_32)
#         # p_32 = self.bn_x32(p_32) 
#         # p_32 = self.iabn_after_x32(p_32)


#         # p_32 = self.iabn_out_x32(p_32)

#         # p_32 = self.flash_attn_x32(p_32.to(torch.float16)).to(torch.float32)
        
#         # x16 size
#         # [B, C, x16W, x16H]
#         t_dn_x16 = inputs['reduction_4']
#         # [B, C, x16W, x16H] -> [B, 256, x16W, x16H]
#         t_dn_x16 = self.conv_t_dn_x16(t_dn_x16)
#         t_dn_x16 = self.iabn_t_dn_x16(t_dn_x16)
#         t_dn_x16 = torch.add(t_dn_x32_to_merge, t_dn_x16)
#         # [B, 256, x16W, x16H] -> [B, 256, x32W, x32H]
#         t_dn_x16_to_merge =  F.interpolate(
#             t_dn_x16,
#             size=(inputs['reduction_3'].shape[2],
#                   inputs['reduction_3'].shape[3]),
#             mode='nearest'
#         )
#         # Create output
#         p_16 = torch.add(t_dn_x16, b_up_x16)
#         p_16 = self.depth_wise_conv_x16(p_16)
#         p_16 = self.iabn_out_x16(p_16)
#         # p_16 = self.grouped_conv_x16(p_16)
#         p_16 = self.flash_attn_x16(p_16)
#         # p_16 = self.bn_x16(p_16) 
#         # p_16 = self.iabn_after_x16(p_16)

#         # p_16 = self.iabn_out_x16(p_16)

#         # p_16 = self.flash_attn_x16(p_16.to(torch.float16)).to(torch.float32)


#         # x8 size
#         # [B, C, x8W, x8H]
#         t_dn_x8 = inputs['reduction_3']
#         # [B, C, x8W, x8H] -> [B, 256, x8W, x8H]
#         t_dn_x8 = self.conv_t_dn_x8(t_dn_x8)
#         t_dn_x8 = self.iabn_t_dn_x8(t_dn_x8)
#         # t_dn_x8 = torch.add(t_dn_x16_to_merge, t_dn_x8)
#         # [B, 256, x8W, x8H] -> [B, 256, x4W, x4H]
#         t_dn_x8_to_merge = F.interpolate(
#             t_dn_x8,
#             size=(inputs['reduction_2'].shape[2],
#                   inputs['reduction_2'].shape[3]),
#             mode='nearest'
#         )
#         # Create output
#         p_8 = torch.add(t_dn_x8, b_up_x8)
#         p_8 = self.depth_wise_conv_x8(p_8)
#         p_8 = self.iabn_out_x8(p_8)
#         # p_8 = self.grouped_conv_x8(p_8)
#         p_8 = self.flash_attn_x8(p_8)
#         # p_8 = self.bn_x8(p_8) 
#         # p_8 = self.iabn_after_x8(p_8)

#         # p_8 = self.iabn_out_x8(p_8)

#         # p_8 = self.flash_attn_x8(p_8.to(torch.float16)).to(torch.float32)


#         # x4 size
#         # [B, C, x4W, x4H]
#         t_dn_x4 = inputs['reduction_2']
#         # [B, C, x4W, x4H] -> [B, 256, x4W, x4H]
#         t_dn_x4 = self.conv_t_dn_x4(t_dn_x4)
#         t_dn_x4 = self.iabn_t_dn_x4(t_dn_x4)
#         t_dn_x4 = torch.add(t_dn_x8_to_merge, t_dn_x4)

#         # Create outputs
#         p_4 = torch.add(t_dn_x4, b_up_x4)
#         p_4 = self.depth_wise_conv_x4(p_4)
#         p_4 = self.iabn_out_x4(p_4)
#         # p_4 = self.grouped_conv_x4(p_4) 
#         p_4 = self.flash_attn_x4(p_4)
#         # p_4 = self.bn_x4(p_4) 
#         # p_4 = self.iabn_after_x4(p_4)

#         # p_4 = self.iabn_out_x4(p_4)
        
#         # p_4 = self.flash_attn_x4(p_4.to(torch.float16)).to(torch.float32)

#         # print(p_4.shape)
#         # print(p_8.shape)
#         # print(p_16.shape)
#         # print(p_32.shape)
#         # raise 
#         return {
#             'P_4': p_4,
#             'P_8': p_8,
#             'P_16': p_16,
#             'P_32': p_32
#         }
    