"""
Adapted from TUDelft-MAVLab https://github.com/tudelft/event_flow
"""

import torch
import torch.nn as nn

from .base import BaseModel
from .model_util import CropParameters

from .unet import (
    MultiResUNet,
    MultiResUNet_Segmentation,
    MultiResUNet_Flow,
)

class EVFlowSegNet(BaseModel):
    """
    Variant of EV-LayerSegNet architecture to only learn optical flow and not segmentation.
    """

    def __init__(self, unet_kwargs):
        super().__init__()

        EVFlowNet_kwargs = {
            "base_num_channels": unet_kwargs["base_num_channels"],
            "num_encoders": 4,
            "num_residual_blocks": 2,
            "num_output_channels": 5,   
            "skip_type": "concat",
            "norm": None,
            "use_upsample_conv": True,
            "kernel_size": unet_kwargs["kernel_size"],
            "channel_multiplier": 2,
            "final_activation": "tanh",


            "num_encoders_optical_flow_module": 3,
            "num_ff_layers_optical_flow_module": 4,
            
            "num_motion_models": 1
        }

        self.crop = None
        self.mask = unet_kwargs["mask_output"]
        self.norm_input = False if "norm_input" not in unet_kwargs.keys() else unet_kwargs["norm_input"]
        self.encoding = unet_kwargs["encoding"]
        self.num_bins = unet_kwargs["num_bins"]
        self.num_encoders = EVFlowNet_kwargs["num_encoders"]
        self.num_motion_models = EVFlowNet_kwargs["num_motion_models"]

        unet_kwargs.update(EVFlowNet_kwargs)
        unet_kwargs.pop("name", None)
        unet_kwargs.pop("eval", None)
        unet_kwargs.pop("encoding", None)
        unet_kwargs.pop("round_encoding", None)
        unet_kwargs.pop("mask_output", None)
        unet_kwargs.pop("norm_input", None)
        unet_kwargs.pop("spiking_neuron", None)

        self.multires_unet = MultiResUNet_Flow(unet_kwargs)

    def detach_states(self):
        pass

    def reset_states(self):
        pass

    def init_cropping(self, width, height, safety_margin=0):
        self.crop = CropParameters(width, height, self.num_encoders, safety_margin)

    def forward(self, event_voxel, event_cnt, log=False):
        """
        :param event_voxel: N x num_bins x H x W
        :param event_cnt: N x 4 x H x W per-polarity event cnt and average timestamp
        :param log: log activity
        :return: output dict with list of [N x 2 X H X W] (x, y) displacement within event_tensor.
        """

        # input encoding
        if self.encoding == "voxel":
            x = event_voxel
        elif self.encoding == "cnt" and self.num_bins == 2:
            x = event_cnt
        else:
            print("Model error: Incorrect input encoding.")
            raise AttributeError

        # normalize input
        if self.norm_input:
            mean, stddev = (
                x[x != 0].mean(),
                x[x != 0].std(),
            )
            x[x != 0] = (x[x != 0] - mean) / stddev

        # pad input
        if self.crop is not None:
            x = self.crop.pad(x)

        # forward pass  
        multires_flow = self.multires_unet.forward(x)
        
        # log activity
        if log:
            raise NotImplementedError("Activity logging not implemented")
        else:
            activity = None

        B = 8; N_classes = 1; H = 480; W = 640
        flow_list = torch.zeros(B, N_classes, 2, H, W).to(multires_flow["motion models"].device)

        affine_matrix = multires_flow["motion models"]

        # Calculate flow for each motion model
        for b in range(B):
            flow = nn.functional.affine_grid(
                affine_matrix[b,:,:,:].view(-1,2,3),
                (N_classes, 2, H, W)
                )
            flow_list[b,:,0,:,:] = flow[:,:,:,0]
            flow_list[b,:,1,:,:] = flow[:,:,:,1]
        
        flow_list =  flow_list.squeeze(dim=1)
        
        return {"flow": flow_list, "activity": activity}
class EVFlowNet_Segmentation(BaseModel):
    """
    EV-LayerSegNet architecture, as described in the thesis paper "EV-LayerSegNet: Self-supervised Motion Segmentation using Event-based Cameras"
    """
    

    def __init__(self, unet_kwargs):
        super().__init__()

        EVFlowNet_kwargs = {
            "base_num_channels": unet_kwargs["base_num_channels"],
            "num_encoders": 4,
            "num_residual_blocks": 2,
            "num_output_channels": 5,   
            "skip_type": "concat",
            "norm": None,
            "use_upsample_conv": True,
            "kernel_size": unet_kwargs["kernel_size"],
            "channel_multiplier": 2,
            "final_activation": "tanh",


            "num_encoders_optical_flow_module": 3,
            "num_ff_layers_optical_flow_module": 4,
            
            "num_motion_models": 2
        }

        self.crop = None
        self.mask = unet_kwargs["mask_output"]
        self.norm_input = False if "norm_input" not in unet_kwargs.keys() else unet_kwargs["norm_input"]
        self.encoding = unet_kwargs["encoding"]
        self.num_bins = unet_kwargs["num_bins"]
        self.num_encoders = EVFlowNet_kwargs["num_encoders"]
        self.num_motion_models = EVFlowNet_kwargs["num_motion_models"]

        unet_kwargs.update(EVFlowNet_kwargs)
        unet_kwargs.pop("name", None)
        unet_kwargs.pop("eval", None)
        unet_kwargs.pop("encoding", None)
        unet_kwargs.pop("round_encoding", None)
        unet_kwargs.pop("mask_output", None)
        unet_kwargs.pop("norm_input", None)
        unet_kwargs.pop("spiking_neuron", None)

        self.multires_unet = MultiResUNet_Segmentation(unet_kwargs)

    def detach_states(self):
        pass

    def reset_states(self):
        pass

    def init_cropping(self, width, height, safety_margin=0):
        self.crop = CropParameters(width, height, self.num_encoders, safety_margin)

    def forward(self, event_voxel, event_cnt, log=False):
        """
        :param event_voxel: N x num_bins x H x W
        :param event_cnt: N x 4 x H x W per-polarity event cnt and average timestamp
        :param log: log activity
        :return: output dict with list of [N x 2 X H X W] (x, y) displacement within event_tensor.
        """

        # input encoding
        if self.encoding == "voxel":
            x = event_voxel
        elif self.encoding == "cnt":
            x = event_cnt
        else:
            print("Model error: Incorrect input encoding.")
            raise AttributeError

        # normalize input
        if self.norm_input:
            mean, stddev = (
                x[x != 0].mean(),
                x[x != 0].std(),
            )
            x[x != 0] = (x[x != 0] - mean) / stddev

        # pad input
        if self.crop is not None:
            x = self.crop.pad(x)

        # forward pass  
        multires_flow = self.multires_unet.forward(x)
        
        # log activity
        if log:
            raise NotImplementedError("Activity logging not implemented")
        else:
            activity = None

        # Extract alpha mask size and initiate flow list
        B, N_classes, H, W = multires_flow["alpha mask"].shape 
        flow_list = torch.zeros(B, N_classes, 2, H, W).to(multires_flow["motion models"].device)
       
        # Calculate flow for each motion model
        for b in range(B):
            flow = nn.functional.affine_grid(
                multires_flow["motion models"][b,:,:,:].view(-1,2,3),
                (N_classes, 2, H, W)
                )

            flow_list[b,:,0,:,:] = flow[:,:,:,0] 
            flow_list[b,:,1,:,:] = flow[:,:,:,1]
        
        flow_total = flow_list.clone()

        # Find max values for each pixel in the alpha mask
        max_vals, _ = torch.max(multires_flow["alpha mask"], dim=1, keepdim=True)

        # Replace all values below the maximum with zeros
        multires_flow["alpha mask"] = torch.where(multires_flow["alpha mask"] < max_vals, 
                                                  torch.zeros_like(multires_flow["alpha mask"]), 
                                                  multires_flow["alpha mask"])
        
        # calculate combined flow
        flow_list = (flow_list * multires_flow["alpha mask"][:, :, None, :, :]).sum(1)
        
        # binarize alpha mask for visualization
        alpha_masks = multires_flow["alpha mask"].clone()
        alpha_masks[alpha_masks != 0] = 1

        return {"flow": flow_list, "alpha_masks": alpha_masks, "flow_total": flow_total, "activity": activity} 



class EVFlowNet(BaseModel):
    """
    EV-FlowNet architecture, as described in the paper "EV-FlowNet: Self-Supervised Optical
    Flow for Event-based Cameras", Zhu et al., RSS 2018.
    """

    def __init__(self, unet_kwargs):
        super().__init__()

        EVFlowNet_kwargs = {
            "base_num_channels": unet_kwargs["base_num_channels"],
            "num_encoders": 4,
            "num_residual_blocks": 2,
            "num_output_channels": 2,
            "skip_type": "concat",
            "norm": None,
            "use_upsample_conv": True,
            "kernel_size": unet_kwargs["kernel_size"],
            "channel_multiplier": 2,
            "final_activation": "tanh",
        }

        self.crop = None
        self.mask = unet_kwargs["mask_output"]
        self.norm_input = False if "norm_input" not in unet_kwargs.keys() else unet_kwargs["norm_input"]
        self.encoding = unet_kwargs["encoding"]
        self.num_bins = unet_kwargs["num_bins"]
        self.num_encoders = EVFlowNet_kwargs["num_encoders"]

        unet_kwargs.update(EVFlowNet_kwargs)
        unet_kwargs.pop("name", None)
        unet_kwargs.pop("eval", None)
        unet_kwargs.pop("encoding", None)
        unet_kwargs.pop("round_encoding", None)
        unet_kwargs.pop("mask_output", None)
        unet_kwargs.pop("norm_input", None)
        unet_kwargs.pop("spiking_neuron", None)

        self.multires_unet = MultiResUNet(unet_kwargs) 

    def detach_states(self):
        pass

    def reset_states(self):
        pass

    def init_cropping(self, width, height, safety_margin=0):
        self.crop = CropParameters(width, height, self.num_encoders, safety_margin)

    def forward(self, event_voxel, event_cnt, log=False):

        """
        :param event_voxel: N x num_bins x H x W
        :param event_cnt: N x 4 x H x W per-polarity event cnt and average timestamp
        :param log: log activity
        :return: output dict with list of [N x 2 X H X W] (x, y) displacement within event_tensor.
        """

        # input encoding
        if self.encoding == "voxel":
            x = event_voxel
        elif self.encoding == "cnt" and self.num_bins == 2:
            x = event_cnt
        else:
            print("Model error: Incorrect input encoding.")
            raise AttributeError

        # normalize input
        if self.norm_input:
            mean, stddev = (
                x[x != 0].mean(),
                x[x != 0].std(),
            )
            x[x != 0] = (x[x != 0] - mean) / stddev

        # pad input
        if self.crop is not None:
            x = self.crop.pad(x)

        # forward pass
        multires_flow = self.multires_unet.forward(x) 

        # log activity
        if log:
            raise NotImplementedError("Activity logging not implemented")
        else:
            activity = None

        # upsample flow estimates to the original input resolution
        flow_list = []
        for flow in multires_flow:
            flow_list.append(
                torch.nn.functional.interpolate(
                    flow,
                    scale_factor=(
                        multires_flow[-1].shape[2] / flow.shape[2],
                        multires_flow[-1].shape[3] / flow.shape[3],
                    ),
                )
            )

        # crop output
        if self.crop is not None:
            for i, flow in enumerate(flow_list):
                flow_list[i] = flow[:, :, self.crop.iy0 : self.crop.iy1, self.crop.ix0 : self.crop.ix1]
                flow_list[i] = flow_list[i].contiguous()

        return {"flow": flow_list, "activity": activity} 









