"""
Adapted from UZH-RPG https://github.com/uzh-rpg/rpg_e2vid
"""

import torch
import torch.nn as nn

from utils.softmax_binning import softmax_binning, get_masks

from .base import BaseModel
from .model_util import copy_states, CropParameters
from .spiking_submodules import (
    ConvALIF,
    ConvALIFRecurrent,
    ConvLIF,
    ConvLIFRecurrent,
    ConvPLIF,
    ConvPLIFRecurrent,
    ConvXLIF,
    ConvXLIFRecurrent,
)
from .submodules import ConvGRU, ConvLayer, ConvLayer_, ConvLeaky, ConvLeakyRecurrent, ConvRecurrent
from .unet import (
    #UNetRecurrent,
    MultiResUNet,
    MultiResUNet_Segmentation,
    #MultiResUNetRecurrent,
    #SpikingMultiResUNetRecurrent,
    #LeakyMultiResUNetRecurrent,
)

from utils.flow import get_optical_flow


# Youssef you need to use EVFlowNet now, since it a non-recurrent ANN
class EVFlowNet_Segmentation(BaseModel):
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
            "num_output_channels": 1,   # it was 2 {u_x,u_y}, but now it's 1 since each pixel has to output only 1 value for the alpha map
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

        self.multires_unet = MultiResUNet_Segmentation(unet_kwargs)  # NETWORK ARCHITECTURE 

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
        multires_flow = self.multires_unet.forward(x)      # OUTPUT

        # log activity
        if log:
            raise NotImplementedError("Activity logging not implemented")
        else:
            activity = None

        output_decoder = multires_flow["output_segmentation module"]
        motion_models = multires_flow["motion_models"]

        # calculate all estimated optical flows given the motion models
        H = output_decoder.shape[2]
        W = output_decoder.shape[3]
        flow_list = torch.empty(self.num_motion_models, H, W, 2)

        for i in range(self.num_motion_models):
            flow_list[i] = get_optical_flow(motion_models[i], H, W)
        
        #restrict last mask output to be between 0 and 1
        sigmoid = nn.Sigmoid()
        output_decoder = sigmoid(output_decoder)

        # compute masks

        output_decoder_softmax_binned = softmax_binning(self.num_motion_models, output_decoder)
        alpha_masks = get_masks(output_decoder_softmax_binned)

        

        """
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
        """
        
        return {"flow": flow_list, "alpha_maps": alpha_masks, "activity": activity} # OUTPUT flow_lists  -> what we should output is two alpha maps and two flow fields



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

        self.multires_unet = MultiResUNet(unet_kwargs)  # NETWORK ARCHITECTURE 

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
        multires_flow = self.multires_unet.forward(x)      # OUTPUT

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

        return {"flow": flow_list, "activity": activity} # OUTPUT flow_lists

