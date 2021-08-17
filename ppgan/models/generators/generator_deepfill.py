#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# code was heavily based on https://github.com/open-mmlab/mmediting


import paddle
import paddle.nn.functional as F

from ppgan.modules.contextual_attentions import ContextualAttention


class GatedConv(paddle.nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 padding_mode,
                 norm=None,
                 act="ELU",
                 gated_act="Sigmoid",
                 **conv_args):
        super(GatedConv, self).__init__()
        self.conv = paddle.nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels * 2,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            padding_mode=padding_mode,
            **conv_args
        )
        if norm is not None:
            self.norm = getattr(paddle.nn, norm)
        else:
            self.norm = None
        if act is not None:
            self.act = getattr(paddle.nn, act)()
        else:
            self.act = None

        if gated_act is not None:
            self.gated_act = getattr(paddle.nn, gated_act)()
        else:
            self.gated_act = None

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        x, y = paddle.split(x, num_or_sections=2, axis=1)
        if self.act is not None:
            x = self.act(x)
        if self.gated_act is not None:
            y = self.gated_act(y)
        out = x * y
        return out


class SimpleConv(paddle.nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 padding_mode,
                 norm,
                 act="ELU",
                 gated_act=None,
                 **conv_args):
        super(SimpleConv, self).__init__()
        self.conv = paddle.nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            padding_mode=padding_mode,
            **conv_args
        )
        if norm is not None:
            self.norm = getattr(paddle.nn, norm)
        else:
            self.norm = None
        if act is not None:
            self.act = getattr(paddle.nn, act)()
        else:
            self.act = None

    def forward(self, x):
        out = self.conv(x)
        if self.norm is not None:
            out = self.norm(out)
        if self.act is not None:
            out = self.act(out)
        return out


class DeepFillv2Generator(paddle.nn.Layer):
    def __init__(self,
                 in_channels=5,
                 conv_type="gated",
                 padding_mode="reflect",
                 norm=None,
                 act="ELU",
                 gated_act="Sigmoid",
                 out_act="tanh",
                 channel_factor=0.75,
                 **conv_args):
        super(DeepFillv2Generator, self).__init__()
        if conv_type == "simple":
            Conv = SimpleConv
        elif conv_type == "gated":
            Conv = GatedConv
        else:
            raise NotImplementedError("conv type {} not implemented", conv_type)
        stage1_channels = [int(i * channel_factor) for i in [32, 64, 64, 128, 128, 128]]
        stage2_in_channels = in_channels
        stage2_conv_channels = [int(i * channel_factor) for i in [32, 32, 64, 64, 128, 128]]
        stage2_att_channels = [int(i * channel_factor) for i in [32, 32, 64, 128, 128, 128]]
        stage1_decoder_channels = [int(i * channel_factor) for i in [128, 128, 64, 64, 32, 16]]
        stage1_decoder_channels.append(3)
        stage2_decoder_channels = [128, 128, 64, 64, 32, 16, 3]
        kernel_sizes = [5, 3, 3, 3, 3, 3]
        strides = [1, 2, 1, 2, 1, 1]
        self.stage1_encoder = paddle.nn.LayerList()
        for i in range(6):
            self.stage1_encoder.append(
                Conv(in_channels=in_channels,
                     out_channels=stage1_channels[i],
                     kernel_size=kernel_sizes[i],
                     stride=strides[i],
                     padding=(kernel_sizes[i] - 1) // 2,
                     padding_mode=padding_mode,
                     norm=norm,
                     act=act,
                     gated_act=gated_act,
                     **conv_args))
            in_channels = stage1_channels[i]
        self.stage1_neck = paddle.nn.LayerList()
        for i in range(4):
            self.stage1_neck.append(
                Conv(in_channels=in_channels,
                     out_channels=in_channels,
                     kernel_size=3,
                     stride=1,
                     padding=int(2**(i+1)),
                     padding_mode=padding_mode,
                     dilation=int(2**(i+1)),
                     norm=norm,
                     act=act,
                     gated_act=gated_act,
                     **conv_args))
        self.stage1_decoder = paddle.nn.LayerList()
        for i in range(7):
            if i != 6:
                decoder_act = act
            else:
                decoder_act = None
            self.stage1_decoder.append(
                Conv(in_channels=in_channels,
                     out_channels=stage1_decoder_channels[i],
                     kernel_size=3,
                     stride=1,
                     padding=1,
                     padding_mode=padding_mode,
                     norm=norm,
                     act=decoder_act,
                     gated_act=gated_act,
                     **conv_args)
            )
            in_channels = stage1_decoder_channels[i]
        if out_act is not None:
            self.out_act = getattr(F, out_act)
        self.stage2_conv_encoder = paddle.nn.LayerList()
        in_channels = stage2_in_channels
        for i in range(6):
            self.stage2_conv_encoder.append(
                Conv(in_channels=in_channels,
                     out_channels=stage2_conv_channels[i],
                     kernel_size=kernel_sizes[i],
                     stride=strides[i],
                     padding=(kernel_sizes[i] - 1) // 2,
                     padding_mode=padding_mode,
                     norm=norm,
                     act=act,
                     gated_act=gated_act,
                     **conv_args))
            in_channels = stage2_conv_channels[i]
        self.stage2_neck = paddle.nn.LayerList()
        for i in range(4):
            self.stage2_neck.append(
                Conv(in_channels=in_channels,
                     out_channels=in_channels,
                     kernel_size=3,
                     stride=1,
                     padding=int(2 ** (i + 1)),
                     padding_mode=padding_mode,
                     dilation=int(2 ** (i + 1)),
                     norm=norm,
                     act=act,
                     gated_act=gated_act,
                     **conv_args))
        self.stage2_att_encoder = paddle.nn.LayerList()
        in_channels = stage2_in_channels
        for i in range(6):
            self.stage2_att_encoder.append(
                Conv(in_channels=in_channels,
                     out_channels=stage2_att_channels[i],
                     kernel_size=kernel_sizes[i],
                     stride=strides[i],
                     padding=(kernel_sizes[i] - 1) // 2,
                     padding_mode=padding_mode,
                     norm=norm,
                     act=act,
                     gated_act=gated_act,
                     **conv_args))
            in_channels = stage2_att_channels[i]
        self.contextual = ContextualAttention()
        self.contextual_conv = paddle.nn.LayerList()
        for i in range(2):
            self.contextual_conv.append(
                Conv(in_channels=in_channels,
                     out_channels=in_channels,
                     kernel_size=3,
                     stride=1,
                     padding=1,
                     padding_mode=padding_mode,
                     norm=norm,
                     act=act,
                     gated_act=gated_act,
                     **conv_args)
            )
        in_channels *= 2
        self.stage2_decoder = paddle.nn.LayerList()
        for i in range(7):
            if i != 6:
                decoder_act = act
            else:
                decoder_act = None
            self.stage2_decoder.append(
                Conv(in_channels=in_channels,
                     out_channels=stage2_decoder_channels[i],
                     kernel_size=3,
                     stride=1,
                     padding=1,
                     padding_mode=padding_mode,
                     norm=norm,
                     act=decoder_act,
                     gated_act=gated_act,
                     **conv_args)
            )
            in_channels = stage2_decoder_channels[i]

    def forward(self, x):
        input_x = x.clone()
        masked_img = x[:, :3]
        mask = input_x[:, -1:]
        for layer_i in self.stage1_encoder:
            x = layer_i(x)
        for layer_i in self.stage1_neck:
            x = layer_i(x)
        for i, layer_i in enumerate(self.stage1_decoder):
            x = layer_i(x)
            if i in (1, 3):
                x = F.interpolate(x, scale_factor=2)
        if self.out_act is not None:
            x = self.out_act(x)
        x = x * mask + masked_img * (1 - mask)
        x = paddle.concat([x, input_x[:, 3:]], axis=1)
        att_x = x
        for layer_i in self.stage2_conv_encoder:
            x = layer_i(x)
        for layer_i in self.stage2_neck:
            x = layer_i(x)
        for layer_i in self.stage2_att_encoder:
            att_x = layer_i(att_x)
        attention_size = att_x.shape[-2:]
        resized_mask = F.interpolate(mask, size=attention_size)
        att_x, offset = self.contextual(att_x, att_x, resized_mask)
        for layer_i in self.contextual_conv:
            att_x = layer_i(att_x)
        x = paddle.concat([x, att_x], axis=1)
        for i, layer_i in enumerate(self.stage2_decoder):
            x = layer_i(x)
            if i in (1, 3):
                x = F.interpolate(x, scale_factor=2)
        if self.out_act is not None:
            x = self.out_act(x)
        x = x * mask + masked_img * (1. - mask)
        x = paddle.concat([x, mask], axis=1)
        return x
