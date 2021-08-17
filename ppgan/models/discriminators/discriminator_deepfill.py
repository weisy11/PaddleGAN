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

from ...modules.utils import spectral_norm


class SNLinear(paddle.nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias_attr=True,
                 norm=None,
                 act=None,
                 act_attr=None,
                 with_spectral_norm=True):
        super(SNLinear, self).__init__()
        if not with_spectral_norm:
            self.fc = paddle.nn.Linear(in_channels, out_channels, bias_attr=bias_attr)
        else:
            self.fc = spectral_norm(paddle.nn.Linear(in_channels, out_channels))
        if norm is not None:
            self.norm = getattr(paddle.nn, norm)
        else:
            self.norm = None
        if act is not None:
            if act_attr is not None:
                self.act = getattr(paddle.nn, act)(**act_attr)
            else:
                self.act = getattr(paddle.nn, act)()
        else:
            self.act = None

    def forward(self, x):
        out = self.fc(x)
        if self.norm is not None:
            out = self.norm(out)
        if self.act is not None:
            out = self.act(out)
        return out


class SNConv(paddle.nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 norm,
                 act="ReLU",
                 act_attr=None,
                 with_spectral_norm=True,
                 **conv_args):
        super(SNConv, self).__init__()
        if not with_spectral_norm:
            self.conv = paddle.nn.Conv2D(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                **conv_args)
        else:
            self.conv = spectral_norm(paddle.nn.Conv2D(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                **conv_args))
        if norm is not None:
            self.norm = getattr(paddle.nn, norm)
        else:
            self.norm = None
        if act is not None:
            if act_attr is not None:
                self.act = getattr(paddle.nn, act)(**act_attr)
            else:
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


class DeepFillv2Discriminator(paddle.nn.Layer):
    def __init__(self,
                 in_channels=4,
                 max_channels=256,
                 conv_num=6,
                 kernel_size=5,
                 fc_in_channels=None,
                 fc_out_channels=1024,
                 act="LeakyReLU",
                 act_attr={"negative_slope": 0.2},
                 out_act="LeakyReLU",
                 out_act_attr={"negative_slope": 0.2},
                 with_spectral_norm=True,
                 norm=None,
                 with_input_norm=True,
                 with_out_conv=False,
                 **conv_args):
        super(DeepFillv2Discriminator, self).__init__()

        self.conv_layers = paddle.nn.LayerList()
        if with_out_conv:
            conv_num += 2
        stride = 2
        for i in range(conv_num):
            out_channels = min(64 * 2 ** i, max_channels)
            if i == 0 and not with_input_norm:
                norm_ = None
            elif i == conv_num - 1 and fc_in_channels is not None:
                norm_ = None
                act = out_act
                act_attr = out_act_attr
            else:
                norm_ = norm
            if with_out_conv and i >= conv_num - 2:
                stride = 1
            if with_out_conv and i == conv_num - 1:
                out_channels = 1
            self.conv_layers.append(
                SNConv(in_channels,
                       out_channels,
                       kernel_size=kernel_size,
                       stride=stride,
                       padding=kernel_size // 2,
                       norm=norm_,
                       act=act,
                       act_attr=act_attr,
                       with_spectral_norm=with_spectral_norm,
                       **conv_args))
            in_channels = out_channels
        if fc_in_channels is not None:
            self.fc = SNLinear(
                fc_in_channels,
                fc_out_channels,
                bias_attr=True,
                act=out_act,
                act_attr=out_act_attr,
                with_spectral_norm=with_spectral_norm)
        else:
            self.fc = None

    def forward(self, x):
        for layer_i in self.conv_layers:
            x = layer_i(x)
        if self.fc is not None:
            x = paddle.flatten(x)
            x = self.fc(x)
        return x
