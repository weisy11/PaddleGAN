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
from .base_model import BaseModel

from .builder import MODELS
from .generators.builder import build_generator
from .discriminators.builder import build_discriminator
from .criterions import build_criterion
from ..metrics import build_metric

from ..modules.init import init_weights
from ..utils.image_pool import ImagePool


class MaskedLoss(paddle.nn.Layer):
    def __init__(self, loss_cfg):
        super(MaskedLoss, self).__init__()
        _loss_cfg = loss_cfg.copy()
        self.sample_wise = _loss_cfg.pop("sample_wise", False)
        self.reduction = _loss_cfg.pop("reduction", "mean")
        assert self.reduction in ["none", "sum", "mean"]
        _loss_cfg["reduction"] = "none"
        self.loss = build_criterion(_loss_cfg)

    def forward(self, pred, target, mask=None):
        loss = self.loss(pred, target)
        if mask is None:
            mask = paddle.ones_like(loss)
        assert mask.dim() == loss.dim()
        assert mask.shape[1] == 1 or mask.shape[1] == loss.shape[1]
        loss = loss * mask
        if self.reduction == "none":
            return loss
        elif self.reduction == "sum":
            return loss.sum()
        # if reduction is mean, then compute mean over masked region
        elif self.reduction == 'mean':
            # expand mask from N1HW to NCHW
            if mask.shape[1] == 1:
                mask = mask.expand_as(loss)
            # small value to prevent division by zero
            eps = 1e-12

            # perform sample-wise mean
            if self.sample_wise:
                mask = mask.sum(axis=[1, 2, 3], keepdim=True)  # NCHW to N111
                loss = (loss / (mask + eps)).sum() / mask.size(0)
            # perform pixel-wise mean
            else:
                loss = loss.sum() / (mask.sum() + eps)
            return loss


@MODELS.register()
class Deepfillv2Model(BaseModel):
    def __init__(self,
                 generator,
                 discriminator=None,
                 loss_args=None,
                 disc_steps=1,
                 gen_input_with_ones=True,
                 disc_input_with_mask=False,
                 max_eval_steps=50000):
        super(Deepfillv2Model, self).__init__()
        self.nets["generator"] = build_generator(generator)
        if discriminator is not None:
            self.nets["discriminator"] = build_discriminator(discriminator)

        self.train_step = 0
        self.disc_step = disc_steps
        self.gen_input_with_ones = gen_input_with_ones
        self.disc_input_with_mask = disc_input_with_mask
        if loss_args is None:
            self.loss_args = {}
        else:
            self.loss_args = loss_args

        if "GAN_loss" in self.loss_args:
            self.GAN_loss = build_criterion(self.loss_args["GAN_loss"])
        else:
            self.GAN_loss = None

        if "l1_loss" in self.loss_args:
            l1_loss_config = self.loss_args["l1_loss"]
            if l1_loss_config.pop("masked", False):
                self.hole_weight = l1_loss_config.pop("hole_weight", 1.0)
                self.valid_weight = l1_loss_config.pop("valid_weight", 1.0)
                self.l1_loss = MaskedLoss(l1_loss_config)

            else:
                self.l1_loss = build_criterion(l1_loss_config)

    def forward(self):
        self.forward_G()

    def setup_input(self, input):
        self.mask = input["mask"].astype("float32")
        self.gt_img = input["img"].astype("float32")
        self.masked_img = self.gt_img * (1. - self.mask)

    def forward_G(self):
        if self.gen_input_with_ones:
            tmp_ones = paddle.ones_like(self.mask)
            x = paddle.concat([self.masked_img, tmp_ones, self.mask], axis=1)
        else:
            x = paddle.concat([self.masked_img, self.mask], axis=1)
        self.stage1_res, self.stage2_res = self.nets["generator"](x)
        self.fake_img = self.masked_img * (1. - self.mask) + self.stage2_res * self.mask

    def forward_D(self, is_disc=False):
        if is_disc:
            fake_img = self.fake_img.detach()
        else:
            fake_img = self.fake_img
        if self.disc_input_with_mask:
            fake_x = paddle.concat([fake_img, self.mask], axis=1)
            real_x = paddle.concat([self.gt_img, self.mask], axis=1)
        else:
            fake_x = fake_img
            real_x = self.gt_img
        self.disc_output_fake = self.nets["discriminator"](fake_x)
        self.disc_output_real = self.nets["discriminator"](real_x)

    def backward_G(self):
        loss_list = []
        if self.GAN_loss is not None:
            loss_list.append(self.GAN_loss(self.disc_output_fake, target_is_real=True, is_disc=True, is_updating_D=True))
        if isinstance(self.l1_loss, MaskedLoss):
            loss_list.append(self.l1_loss(self.stage1_res, self.gt_img))
            loss_list.append(self.l1_loss(self.stage2_res, self.gt_img))
        elif self.l1_loss is not None:
            loss_list.append(self.l1_loss(self.stage1_res, self.gt_img, mask=self.mask))
            loss_list.append(self.l1_loss(self.stage2_res, self.gt_img, mask=1-self.mask))
        self.losses["loss_G"] = paddle.sum(paddle.to_tensor(loss_list))
        for param in self.nets["discriminator"].parameters():
            param.trainable = False
        self.optimizers["optimG"].clear_grad()
        self.losses["loss_G"].backward()
        self.optimizers["optimG"].step()

    def backward_D(self):
        for param in self.nets["discriminator"].parameters():
            param.trainable = True
        if self.GAN_loss is not None:
            D_loss_real = self.GAN_loss(self.disc_output_real, target_is_real=True, is_disc=True, is_updating_D=True)
            self.optimizers["optimD"].clear_grad()
            D_loss_real.backward()
            D_loss_fake = self.GAN_loss(self.disc_output_fake, target_is_real=False, is_disc=True, is_updating_D=True)
            D_loss_fake.backward()
            self.optimizers["optimD"].step()
            loss = 0.5 * D_loss_real + 0.5 * D_loss_fake
            self.losses["loss_D"] = loss
        else:
            self.losses["loss_D"] = paddle.to_tensor([0])

    def train_iter(self, optimizers=None):
        self.forward_G()
        if not self.train_step % self.disc_step:
            self.forward_D(is_disc=True)
            self.backward_D()

        self.forward_D(is_disc=False)
        self.backward_G()
        self.train_step += 1
