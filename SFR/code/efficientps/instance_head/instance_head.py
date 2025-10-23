import torch.nn as nn
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.layers import ShapeSpec
from detectron2.modeling.roi_heads import build_roi_heads
import detectron2
from detectron2.structures import Instances, Boxes
import torch

class InstanceHead(nn.Module):
    """
    The Instance Head is a Mask RCNN with some modification, this implementation
    is based on detectron2
    Args:
    - cfg (Config) : Config object
    """

    def __init__(self, cfg):
        super().__init__()
        # Detectron 2 expects a dict of ShapeSpec object as input_shape
        input_shape = dict()
        for name, shape in zip(cfg.MODEL.RPN.IN_FEATURES, [4, 8, 16, 32]):
            input_shape[name] = ShapeSpec(channels=256, stride=shape)

        self.rpn = build_proposal_generator(cfg, input_shape=input_shape)

        self.roi_heads = build_roi_heads(cfg, input_shape)


    # def forward(self, inputs, targets={}):
    #     losses = {}
    #     proposals, losses_rpn = self.rpn(inputs, targets['instance'])
    #     if self.training:
    #         _, losses_head = self.roi_heads(inputs, proposals, targets['instance'])
    #         losses.update(losses_rpn)
    #         losses.update(losses_head)
    #         return {}, losses
    #     else:
    #         pred_instances , _ = self.roi_heads(inputs, proposals)
    #         return pred_instances, {}

    
    def forward(self, inputs, targets={}):
        losses = {}

        # Ensure instances exist
        instances = targets.get('instance', [])
        
        # Get RPN proposals
        proposals, losses_rpn = self.rpn(inputs, instances)

        # if self.training:
        #     if len(instances) == 0:
        #         print("Skipping RoI heads: No valid instances in batch")
        #         return {}, losses_rpn  # Only return RPN losses if no instances exist
            
        #     # Ensure proposals and instances have the same length
        #     valid_instances = [t for t in instances if t.get_fields()["gt_classes"].numel() > 0]
        #     valid_proposals = [p for i, p in enumerate(proposals) if instances[i].get_fields()["gt_classes"].numel() > 0]

        #     if len(valid_instances) == 0:
        #         print("Skipping RoI heads: No valid instances in batch")
        #         return {}, losses_rpn

        #     try:
        #         _, losses_head = self.roi_heads(inputs, valid_proposals, valid_instances)
        #     except Exception as e:
        #         print("Error in roi_heads:", str(e))
        #         print("Inputs keys:", inputs.keys())
        #         print("Proposals count:", len(valid_proposals))
        #         print("First Proposal:", valid_proposals[0] if valid_proposals else "None")
        #         print("Targets:", valid_instances)
        #         raise

        #     losses.update(losses_rpn)
        #     losses.update(losses_head)
        #     return {}, losses
        if self.training:
            if len(instances) == 0:
                print("Skipping RoI heads: No valid instances in batch")
                return {}, losses_rpn  # Only return RPN losses if no instances exist

            # Ensure proposals and instances have the same length
            valid_instances = []
            valid_proposals = []

            for i in range(len(instances)):
                if instances[i].get_fields()["gt_classes"].numel() > 0:
                    valid_instances.append(instances[i])
                    valid_proposals.append(proposals[i])
                else:
                    # Maintain batch size with empty instances
                    # empty_instance = detectron2.structures.Instances(image_size=instances[i].image_size)
                    # empty_proposal = detectron2.structures.Instances(image_size=instances[i].image_size)
                    # valid_instances.append(empty_instance)
                    # valid_proposals.append(empty_proposal)

                    # empty_instance = detectron2.structures.Instances(image_size=instances[i].image_size)
                    # empty_proposal = detectron2.structures.Instances(image_size=instances[i].image_size)

                    # empty_instance.gt_boxes = Boxes(torch.empty((0, 4)))  # No boxes
                    # empty_instance.gt_classes = torch.empty((0,), dtype=torch.int64)  # No classes
                    # empty_proposal.objectness_logits = torch.empty((0,), dtype=torch.float32) 
                    # empty_proposal.proposal_boxes = Boxes(torch.empty((0, 4)))  # No proposals

                    # valid_instances.append(empty_instance)
                    # valid_proposals.append(empty_proposal)

                    device = instances[i].gt_classes.device  # Get device of existing instances

                    empty_instance = detectron2.structures.Instances(image_size=instances[i].image_size)
                    empty_proposal = detectron2.structures.Instances(image_size=instances[i].image_size)

                    empty_instance.gt_boxes = Boxes(torch.empty((0, 4), device=device))  # No boxes
                    empty_instance.gt_classes = torch.empty((0,), dtype=torch.int64, device=device)  # No classes

                    # To avoid empty proposal issue, add a dummy proposal
                    dummy_proposal = torch.tensor([[0, 0, 1, 1]], dtype=torch.float32, device=device)  # Tiny box
                    empty_proposal.proposal_boxes = Boxes(dummy_proposal) if dummy_proposal.numel() > 0 else Boxes(torch.empty((0, 4), device=device))
                    empty_proposal.objectness_logits = torch.tensor([0.0], dtype=torch.float32, device=device)  # Dummy confidence

                    valid_instances.append(empty_instance)
                    valid_proposals.append(empty_proposal)



            try:
                _, losses_head = self.roi_heads(inputs, valid_proposals, valid_instances)
            except Exception as e:
                print("Error in roi_heads:", str(e))
                print("Inputs keys:", inputs.keys())
                print("Proposals count:", len(valid_proposals))
                print("First Proposal:", valid_proposals[0] if valid_proposals else "None")
                print("Targets:", valid_instances)
                raise

            losses.update(losses_rpn)
            losses.update(losses_head)
            return {}, losses

        else:
            # Inference mode
            pred_instances, _ = self.roi_heads(inputs, proposals)
            return pred_instances, {}
        

    