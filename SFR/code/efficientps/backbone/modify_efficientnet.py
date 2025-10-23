
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from inplace_abn import InPlaceABN

# Output channel from layers given by the `extract_endpoints` function of
# efficient net, use to initialize the fpn
output_feature_size = {
    0: [16, 24, 40, 112, 1280],
    1: [16, 24, 40, 112, 1280],
    2: [16, 24, 48, 120, 1408],
    3: [24, 32, 48, 136, 1536],
    4: [24, 32, 56, 160, 1792],
    5: [24, 40, 64, 176, 2048],
    6: [32, 40, 72, 200, 2304],
    7: [32, 48, 80, 224, 2560],
    8: [32, 56, 88, 248, 2816]
}

def generate_backbone_EfficientPS(cfg):
    """
    Create an EfficientNet model base on this repository:
    https://github.com/lukemelas/EfficientNet-PyTorch

    Modify the existing Efficientnet base on the EfficientPS paper,
    ie:
    - replace BN and swish with InplaceBN and LeakyRelu
    - remove se (squeeze and excite) blocks
    Args:
    - cdg (Config) : config object
    Return:
    - backbone (nn.Module) : Modify version of the EfficentNet
    """

    if cfg.MODEL_CUSTOM.BACKBONE.LOAD_PRETRAIN:
        backbone = EfficientNet.from_pretrained(
            'efficientnet-b{}'.format(cfg.MODEL_CUSTOM.BACKBONE.EFFICIENTNET_ID))
    else:
        backbone = EfficientNet.from_name(
            'efficientnet-b{}'.format(cfg.MODEL_CUSTOM.BACKBONE.EFFICIENTNET_ID))

    backbone._bn0 = InPlaceABN(num_features=backbone._bn0.num_features, eps=0.001)
    backbone._bn1 = InPlaceABN(num_features=backbone._bn1.num_features, eps=0.001)
    backbone._swish = nn.Identity()
    for i, block in enumerate(backbone._blocks):
        # Remove SE block
        block.has_se = False
        # Additional step to have the correct number of parameter on compute
        block._se_reduce =  nn.Identity()
        block._se_expand = nn.Identity()
        # Replace BN with Inplace BN (default activation is leaky relu)
        if '_bn0' in [name for name, layer in block.named_children()]:
            block._bn0 = InPlaceABN(num_features=block._bn0.num_features, eps=0.001)
        block._bn1 = InPlaceABN(num_features=block._bn1.num_features, eps=0.001)
        block._bn2 = InPlaceABN(num_features=block._bn2.num_features, eps=0.001)

        # Remove swish activation since Inplace BN contains the activation layer
        block._swish = nn.Identity()

    return backbone


#CBAM

# import torch.nn as nn
# from efficientnet_pytorch import EfficientNet
# from inplace_abn import InPlaceABN
# from efficientps.cbam import CBAM
# from efficientnet_pytorch.utils import drop_connect


# # Output channel from layers given by the `extract_endpoints` function of
# # efficient net, use to initialize the fpn
# output_feature_size = {
#     0: [16, 24, 40, 112, 1280],
#     1: [16, 24, 40, 112, 1280],
#     2: [16, 24, 48, 120, 1408],
#     3: [24, 32, 48, 136, 1536],
#     4: [24, 32, 56, 160, 1792],
#     5: [24, 40, 64, 176, 2048],
#     6: [32, 40, 72, 200, 2304],
#     7: [32, 48, 80, 224, 2560],
#     8: [32, 56, 88, 248, 2816]
# }

# # class BlockWithCBAM(nn.Module):
# #     def __init__(self, block, cbam):
# #         super().__init__()
# #         self.block = block
# #         self.cbam = cbam

# #     def forward(self, x, drop_connect_rate=None):
# #         x = self.block(x, drop_connect_rate=drop_connect_rate)
# #         x = self.cbam(x)
# #         return x

# class BlockWithCBAM(nn.Module):
#     def __init__(self, block, cbam):
#         super().__init__()
#         self.block = block
#         self.cbam = cbam

#     def forward(self, x, drop_connect_rate=None):
#         # Run first half of MBConv
#         x_input = x
#         x = self.block._expand_conv(x)
#         x = self.block._bn0(x)
#         x = self.block._depthwise_conv(x)
#         x = self.block._bn1(x)

#         # Apply CBAM *before* projection
#         x = self.cbam(x)

#         x = self.block._project_conv(x)
#         x = self.block._bn2(x)

#         if drop_connect_rate:
#             x = drop_connect(x, drop_connect_rate, self.training)

#         block_args = self.block._block_args
        
#         # if (block_args.stride == [1] or block_args.stride == (1, 1)) and \
#         #    (block_args.input_filters == block_args.output_filters):
#         #     x = x + x_input
        
#         if (block_args.id_skip and
#             block_args.stride == 1 and
#             block_args.input_filters == block_args.output_filters):
#             x = x + x_input

#         return x


# def generate_backbone_EfficientPS(cfg):
#     """
#     Create an EfficientNet model base on this repository:
#     https://github.com/lukemelas/EfficientNet-PyTorch

#     Modify the existing Efficientnet base on the EfficientPS paper,
#     ie:
#     - replace BN and swish with InplaceBN and LeakyRelu
#     - remove se (squeeze and excite) blocks
#     Args:
#     - cdg (Config) : config object
#     Return:
#     - backbone (nn.Module) : Modify version of the EfficentNet
#     """

#     if cfg.MODEL_CUSTOM.BACKBONE.LOAD_PRETRAIN:
#         backbone = EfficientNet.from_pretrained(
#             'efficientnet-b{}'.format(cfg.MODEL_CUSTOM.BACKBONE.EFFICIENTNET_ID))
#     else:
#         backbone = EfficientNet.from_name(
#             'efficientnet-b{}'.format(cfg.MODEL_CUSTOM.BACKBONE.EFFICIENTNET_ID))

#     backbone._bn0 = InPlaceABN(num_features=backbone._bn0.num_features, eps=0.001)
#     backbone._bn1 = InPlaceABN(num_features=backbone._bn1.num_features, eps=0.001)
#     backbone._swish = nn.Identity()

#     cbam_indices = [35, 45, 52]
    
#     for i, block in enumerate(backbone._blocks):
#         # print(block)
#         # Remove SE block
#         block.has_se = False
#         # Additional step to have the correct number of parameter on compute
#         block._se_reduce =  nn.Identity()
#         block._se_expand = nn.Identity()
#         # Replace BN with Inplace BN (default activation is leaky relu)
#         if '_bn0' in [name for name, layer in block.named_children()]:
#             block._bn0 = InPlaceABN(num_features=block._bn0.num_features, eps=0.001)
#         block._bn1 = InPlaceABN(num_features=block._bn1.num_features, eps=0.001)
#         block._bn2 = InPlaceABN(num_features=block._bn2.num_features, eps=0.001)

#         # Remove swish activation since Inplace BN contains the activation layer
#         block._swish = nn.Identity()
#         if i in cbam_indices:
#             # out_channels = block._project_conv.out_channels
#             out_channels = block._depthwise_conv.out_channels
#             cbam = CBAM(out_channels)
#             block = BlockWithCBAM(block, cbam)
#             backbone._blocks[i] = block
            
#     return backbone

# ECA


# import torch.nn as nn
# from efficientnet_pytorch import EfficientNet
# from inplace_abn import InPlaceABN
# from .eca_module import eca_layer  # Assumed to be available and correct

# # Output channel from layers given by the `extract_endpoints` function of
# # efficient net, use to initialize the fpn
# output_feature_size = {
#     0: [16, 24, 40, 112, 1280],
#     1: [16, 24, 40, 112, 1280],
#     2: [16, 24, 48, 120, 1408],
#     3: [24, 32, 48, 136, 1536],
#     4: [24, 32, 56, 160, 1792],
#     5: [24, 40, 64, 176, 2048],
#     6: [32, 40, 72, 200, 2304],
#     7: [32, 48, 80, 224, 2560],
#     8: [32, 56, 88, 248, 2816]
# }

# def generate_backbone_EfficientPS(cfg):
#     """
#     Create an EfficientNet model base on this repository:
#     https://github.com/lukemelas/EfficientNet-PyTorch

#     Modified:
#     - Replace BN and swish with InplaceBN and LeakyRelu
#     - Remove SE blocks
#     - Attach ECA layer after projection conv (not applied inside forward)
#     """
#     model_name = 'efficientnet-b{}'.format(cfg.MODEL_CUSTOM.BACKBONE.EFFICIENTNET_ID)
    
#     if cfg.MODEL_CUSTOM.BACKBONE.LOAD_PRETRAIN:
#         backbone = EfficientNet.from_pretrained(model_name)
#     else:
#         backbone = EfficientNet.from_name(model_name)

#     backbone._bn0 = InPlaceABN(num_features=backbone._bn0.num_features, eps=0.001)
#     backbone._bn1 = InPlaceABN(num_features=backbone._bn1.num_features, eps=0.001)
#     backbone._swish = nn.Identity()

#     for i, block in enumerate(backbone._blocks):
#         # Remove SE block
#         block.has_se = False
#         block._se_reduce = nn.Identity()
#         block._se_expand = nn.Identity()

#         # Replace BN with InPlaceABN
#         if '_bn0' in [name for name, _ in block.named_children()]:
#             block._bn0 = InPlaceABN(num_features=block._bn0.num_features, eps=0.001)
#         block._bn1 = InPlaceABN(num_features=block._bn1.num_features, eps=0.001)
#         block._bn2 = InPlaceABN(num_features=block._bn2.num_features, eps=0.001)

#         # Remove Swish activation
#         block._swish = nn.Identity()

#         # Attach ECA layer after the projection layer (but not applied here)
#         out_channels = block._project_conv.out_channels
#         block.eca = eca_layer(out_channels)  # You can manually apply block.eca(output) later if needed

#     return backbone
