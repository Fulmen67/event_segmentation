"""
Adapted from UZH-RPG https://github.com/uzh-rpg/rpg_e2vid
"""

import torch
import torch.nn as nn
import logging

from .model_util import *
from .submodules import (
    ConvLayer,
    RecurrentConvLayer,
    ResidualBlock,
    TransposedConvLayer,
    UpsampleConvLayer,
    LeakyResidualBlock,
    LeakyUpsampleConvLayer,
    LeakyTransposedConvLayer,
    LeakyRecurrentConvLayer,

    Linear
)
from .spiking_submodules import (
    SpikingRecurrentConvLayer,
    SpikingResidualBlock,
    SpikingTransposedConvLayer,
    SpikingUpsampleConvLayer,
)



class BaseUNet(nn.Module):
    """
    Base class for conventional UNet architecture.
    Symmetric, skip connections on every encoding layer.
    """

    ff_type = ConvLayer
    res_type = ResidualBlock
    upsample_type = UpsampleConvLayer
    transpose_type = TransposedConvLayer
    w_scale_pred = None

    linear_type = Linear

    def __init__(
        self,
        base_num_channels,
        num_encoders,
        num_residual_blocks,
        num_output_channels,
        skip_type,
        norm,
        use_upsample_conv,
        num_bins,

        num_motion_models,
        num_encoders_optical_flow_module,
        num_ff_layers_optical_flow_module,

        recurrent_block_type=None,
        kernel_size=5,
        channel_multiplier=2,
        activations=["relu", None],
        spiking_feedforward_block_type=None,
        spiking_neuron=None,
        
    ):
        super(BaseUNet, self).__init__()
        self.base_num_channels = base_num_channels
        self.num_encoders = num_encoders
        self.num_residual_blocks = num_residual_blocks
        self.num_output_channels = num_output_channels
        self.kernel_size = kernel_size
        self.skip_type = skip_type
        self.norm = norm
        self.num_bins = num_bins
        self.recurrent_block_type = recurrent_block_type
        self.channel_multiplier = channel_multiplier
        self.ff_act, self.rec_act = activations

        self.num_motion_models = num_motion_models
        self.num_encoders_optical_flow_module = num_encoders_optical_flow_module
        self.num_ff_layers_optical_flow_module = num_ff_layers_optical_flow_module



        self.spiking_kwargs = {}
        if spiking_feedforward_block_type is not None:
            self.spiking_kwargs["spiking_feedforward_block_type"] = spiking_feedforward_block_type
        if type(spiking_neuron) is dict:
            self.spiking_kwargs.update(spiking_neuron)

        self.skip_ftn = eval("skip_" + skip_type)
        if use_upsample_conv:
            self.UpsampleLayer = self.upsample_type
        else:
            self.UpsampleLayer = self.transpose_type
        assert self.num_output_channels > 0

        self.encoder_input_sizes = [
            int(self.base_num_channels * pow(self.channel_multiplier, i)) for i in range(self.num_encoders)
        ]
        self.encoder_output_sizes = [
            int(self.base_num_channels * pow(self.channel_multiplier, i + 1)) for i in range(self.num_encoders)
        ]
        self.max_num_channels = self.encoder_output_sizes[-1]

        self.encoder_optical_flow_module_input_sizes = [
            int(self.max_num_channels * pow(self.channel_multiplier,i)) for i in range(self.num_encoders_optical_flow_module)
        ] 
        self.encoder_optical_flow_module_output_sizes = [
            int(self.max_num_channels * pow(self.channel_multiplier,i + 1)) for i in range(self.num_encoders_optical_flow_module)
        ]

        self.num_neurons_ff_optical_flow_module = [
          320, 256, 64, 6*self.num_motion_models
        ]
             
    def build_encoders(self):
        encoders = nn.ModuleList()
        for (input_size, output_size) in zip(self.encoder_input_sizes, self.encoder_output_sizes):
            encoders.append(
                self.ff_type(
                    input_size,
                    output_size,
                    kernel_size=self.kernel_size,
                    stride=2,
                    activation=self.ff_act,
                    norm=self.norm,
                    **self.spiking_kwargs
                )
            )
        return encoders

    def build_resblocks(self):
        resblocks = nn.ModuleList()
        for i in range(self.num_residual_blocks):
            resblocks.append(
                self.res_type(
                    self.max_num_channels,
                    self.max_num_channels,
                    activation=self.ff_act,
                    norm=self.norm,
                    **self.spiking_kwargs
                )
            )
        return resblocks

    def build_decoders(self):
        decoder_input_sizes = reversed(self.encoder_output_sizes)
        decoder_output_sizes = reversed(self.encoder_input_sizes)
        decoders = nn.ModuleList()
        for input_size, output_size in zip(decoder_input_sizes, decoder_output_sizes):
            decoders.append(
                self.UpsampleLayer(
                    input_size if self.skip_type == "sum" else 2 * input_size,
                    output_size,
                    kernel_size=self.kernel_size,
                    activation=self.ff_act,
                    norm=self.norm,
                    **self.spiking_kwargs
                )
            )
        return decoders

    def build_prediction_layer(self, num_output_channels, norm=None):
        return self.ff_type(
            self.base_num_channels if self.skip_type == "sum" else 2 * self.base_num_channels,
            num_output_channels,
            1,
            activation=None,
            norm=norm,
        )


class MultiResUNet_Segmentation(BaseUNet):
    """
    Original:
    Conventional UNet architecture.
    Symmetric, skip connections on every encoding layer.
    Predictions at each decoding layer.
    Predictions are added as skip connection (concat) to the input of the subsequent layer.
    """

    def __init__(self, unet_kwargs):
        self.final_activation = unet_kwargs.pop("final_activation", None)
        super().__init__(**unet_kwargs)

        self.encoders = self.build_encoders()
        self.resblocks = self.build_resblocks()
        self.decoders =  self.build_decoders_segmentation_module() 
        self.encoders_optical_flow_module = self.build_encoders_optical_flow_module()
        self.feedforward_optical_flow_module = self.build_feedforward_optical_flow_module()

    def build_encoders(self):       
        encoders = nn.ModuleList()
        for i, (input_size, output_size) in enumerate(zip(self.encoder_input_sizes, self.encoder_output_sizes)):
            if i == 0:
                input_size = self.num_bins
            encoders.append(
                self.ff_type(
                    input_size,
                    output_size,
                    kernel_size=self.kernel_size,
                    stride=2,
                    activation=self.ff_act,
                    norm=self.norm,
                    **self.spiking_kwargs
                )
            )
        return encoders
    
    def build_decoders_segmentation_module(self):
        decoder_input_sizes = reversed(self.encoder_output_sizes)
        decoder_output_sizes = reversed(self.encoder_input_sizes)
        decoder_input_sizes = list(decoder_input_sizes)
        decoder_output_sizes = list(decoder_output_sizes)
        decoder_input_sizes.append(32)
        decoder_output_sizes.append(5)
        
        
        decoders = nn.ModuleList()
        for input_size, output_size in zip(decoder_input_sizes, decoder_output_sizes):
            # the last layer of the decoder serves to bring the number of channels from 32 to 5, so that we have a mask
            if input_size == 32 and output_size == 5:
                decoders.append(
                    self.ff_type(
                        input_size,
                        output_size,
                        kernel_size=1,
                        activation=self.ff_act,
                        norm=self.norm,
                        **self.spiking_kwargs
                    )
                )
            else:
                decoders.append(
                    self.UpsampleLayer(
                        input_size if self.skip_type == "sum" else 2 * input_size,
                        output_size,
                        kernel_size=self.kernel_size ,
                        activation=self.ff_act,
                        norm=self.norm,
                        **self.spiking_kwargs
                    )
                )
        return decoders

    def build_encoders_optical_flow_module(self):

        encoders = nn.ModuleList()
        for i, (input_size, output_size) in enumerate(zip(self.encoder_optical_flow_module_input_sizes, self.encoder_optical_flow_module_output_sizes)):
            """if i == 0:
                input_size = self.num_bins"""
            encoders.append(
                self.ff_type(
                    input_size,
                    output_size,
                    kernel_size=self.kernel_size,
                    stride=2,
                    activation=self.ff_act,
                    norm=self.norm,
                    **self.spiking_kwargs
                )
            )
        return encoders

    def build_feedforward_optical_flow_module(self):
        feature_size = self.num_neurons_ff_optical_flow_module
        feedforward = nn.ModuleList()
        for i in range(1,len(feature_size)):
            
            feedforward.append(
                self.linear_type(
                    feature_size[i-1],
                    feature_size[i],    
                )
            )
        
        return feedforward

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W
        :return: [N x num_output_channels x H x W for i in range(self.num_encoders)]
        """

        # encoder
        blocks = []
        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            blocks.append(x)

        # residual blocks
        for resblock in self.resblocks:
            x, _ = resblock(x)
    
        # encoder optical flow module
        for i, encoder in enumerate(self.encoders_optical_flow_module):
            x_flow = encoder(x if i == 0 else x_flow)

        # apply max pooling to reduce number of elements to be passed to feedforward module
        pool1 = torch.nn.MaxPool2d(2)
        pool2 = torch.nn.MaxPool2d(4)
        pool3 = torch.nn.MaxPool2d(2)
        
        x_flow = pool1(x_flow.view(x_flow.shape[0], 1, 256, 320))
        x_flow = pool2(x_flow.view(x_flow.shape[0], 1, 128, 160))
        x_flow = pool3(x_flow.view(x_flow.shape[0], 1, 32, 40)).view(x_flow.shape[0], -1)
        
        # feedforward optical flow module
        for i, feedforward in enumerate(self.feedforward_optical_flow_module):
            x_flow = feedforward(x_flow)
         
        x_flow = x_flow.reshape(x_flow.shape[0], self.num_motion_models, 2, 3)
        
        # decoder segmentation module
        for i, decoder in enumerate(self.decoders):
            if i != 4:
                if i == 0:
                    x_seg = self.skip_ftn(x, blocks[self.num_encoders - i - 1]) 
                else: 
                    x_seg = self.skip_ftn(x_seg, blocks[self.num_encoders - i - 1]) 
            x_seg = decoder(x_seg)
        
        # apply sigmoid to restrict values to [0,1]
        sigmoid = nn.Sigmoid()
        x_seg = sigmoid(x_seg)   # [8 x 5 x 480 x 640]
        
        #print(x_seg)
        
        # apply softmax 
        softmax = nn.Softmax(dim=1)
        x_seg = softmax(x_seg)
        
        #print(x_seg)
        
        assert torch.sum(x_seg, dim=1).max() > 0.9999 and torch.sum(x_seg, dim=1).max() <= 1.0001, "Pixel probabilities do not sum to 1"
        return {'alpha mask': x_seg,'motion models': x_flow}
        
class MultiResUNet(BaseUNet):
    """
    Conventional UNet architecture.
    Symmetric, skip connections on every encoding layer.
    Predictions at each decoding layer.
    Predictions are added as skip connection (concat) to the input of the subsequent layer.
    """

    def __init__(self, unet_kwargs):
        self.final_activation = unet_kwargs.pop("final_activation", None)
        super().__init__(**unet_kwargs)

        self.encoders = self.build_encoders()
        self.resblocks = self.build_resblocks()
        self.decoders = self.build_multires_prediction_decoders()
        self.preds = self.build_multires_prediction_layer()

    def build_encoders(self):
        encoders = nn.ModuleList()
        for i, (input_size, output_size) in enumerate(zip(self.encoder_input_sizes, self.encoder_output_sizes)):
            if i == 0:
                input_size = self.num_bins
            encoders.append(
                self.ff_type(
                    input_size,
                    output_size,
                    kernel_size=self.kernel_size,
                    stride=2,
                    activation=self.ff_act,
                    norm=self.norm,
                    **self.spiking_kwargs
                )
            )
        return encoders

    def build_multires_prediction_layer(self):
        preds = nn.ModuleList()
        decoder_output_sizes = reversed(self.encoder_input_sizes)
        for output_size in decoder_output_sizes:
            preds.append(
                self.ff_type(output_size, self.num_output_channels, 1, activation=self.final_activation, norm=self.norm)
            )
        return preds

    def build_multires_prediction_decoders(self):
        decoder_input_sizes = reversed(self.encoder_output_sizes)
        decoder_output_sizes = reversed(self.encoder_input_sizes)
        decoders = nn.ModuleList()
        for i, (input_size, output_size) in enumerate(zip(decoder_input_sizes, decoder_output_sizes)):
            prediction_channels = 0 if i == 0 else self.num_output_channels
            decoders.append(
                self.UpsampleLayer(
                    2 * input_size + prediction_channels,
                    output_size,
                    kernel_size=self.kernel_size,
                    activation=self.ff_act,
                    norm=self.norm,
                    **self.spiking_kwargs
                )
            )
        return decoders

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W
        :return: [N x num_output_channels x H x W for i in range(self.num_encoders)]
        """

        # encoder
        blocks = []
        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            blocks.append(x)

        # residual blocks
        for resblock in self.resblocks:
            x, _ = resblock(x)

        # decoder and multires predictions
        predictions = []
        for i, (decoder, pred) in enumerate(zip(self.decoders, self.preds)):
            x = self.skip_ftn(x, blocks[self.num_encoders - i - 1])
            if i > 0:
                x = self.skip_ftn(predictions[-1], x)
            x = decoder(x)
            predictions.append(pred(x))

        return predictions

