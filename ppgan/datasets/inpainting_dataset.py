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

from paddle.io import Dataset
import numpy as np
import cv2
import os

from .preprocess import build_preprocess
from .builder import DATASETS


class MaskSynther(object):
    def __init__(self, mask_mode="brush_stroke_mask", **mask_config):
        self.mask_mode = mask_mode
        self.mask_config = mask_config
        preprocess = self.mask_config.get("preprocess", [{
            "name": "Transforms",
            "input_keys": ["mask"],
            "pipeline": {"name": "Transpose"}
        }])
        self.preprocess = build_preprocess(preprocess)
        if self.mask_mode == "file_mask":
            file_root = mask_config.get("mask_root", None)
            assert file_root is not None, "Please set mask_root for file_mode"
            mask_list_file = mask_config.get("mask_list_file", None)
            assert mask_list_file is not None, "Please set mask_list_file for file_mode"
            with open(mask_list_file, "r") as f:
                label_list = f.read().split("\n")[:-1]
            self.mask_list = [label.split("\t")[1] for label in label_list]

    def __getitem__(self, args):
        index, img = args
        return getattr(self, self.mask_mode)(index, img)

    def file_mask(self, index, img):
        mask_file = self.mask_list[index]
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        c, h, w = img.shape
        mask = cv2.resize(mask, (w, h), cv2.INTER_NEAREST)
        if self.preprocess is not None:
            mask = self.preprocess({"mask": mask})["mask"]
        return mask

    def brush_stroke_mask(self, index, img):
        c, h, w = img.shape
        mask = np.zeros((h, w))
        vert_num_range = self.mask_config.get("num_vertexes", (4, 12))
        assert isinstance(vert_num_range, tuple), \
            "The type of vert_num_range should be tuple, but got {}".format(type(vert_num_range))
        vert_num = np.random.randint(vert_num_range[0], vert_num_range[1])

        brush_width_range = self.mask_config.get("brush_width_range", (12, 40))
        assert isinstance(brush_width_range, tuple), \
            "The type of brush_width_range should be tuple, but got {}".format(type(brush_width_range))

        direction_num_range = self.mask_config.get("direction_num_range", (1, 6))
        assert isinstance(direction_num_range, tuple), \
            "The type of direction_num_range should be tuple, but got {}".format(type(direction_num_range))

        angle_mean = self.mask_config.get('angle_mean', np.pi * 2 / 5)
        assert isinstance(angle_mean, float), \
            "The type of angle_mean should be float, but got {}".format(type(angle_mean))

        length_mean_ratio = self.mask_config.get('length_mean_ratio', 1 / 8)
        assert isinstance(length_mean_ratio, float) and length_mean_ratio < 1, \
            "Length_mean_ratio should be <1, and it's type should be float, " \
            "but got {} with type {}".format(length_mean_ratio, type(length_mean_ratio))

        length_bias_ratio = self.mask_config.get('length_bias_ratio', 1 / 16)
        assert isinstance(length_bias_ratio, float) and length_bias_ratio < 1, \
            "Length_bias_ratio should be <1, and it's type should be float, " \
            "but got {} with type {}".format(length_bias_ratio, type(length_bias_ratio))

        angle_max_bias = self.mask_config.get('angle_max_bias', np.pi * 2 / 15)
        assert isinstance(angle_mean, float), \
            "The type of angle_mean should be float, but got {}".format(type(angle_mean))

        for vert_i in range(vert_num):
            start_x = np.random.randint(w)
            start_y = np.random.randint(h)
            direction_num = np.random.randint(direction_num_range[0], direction_num_range[1])
            max_length = np.sqrt(h * h + w * w)
            length_mean = max_length * length_mean_ratio
            length_bias = max_length * length_bias_ratio
            for direct_i in range(direction_num):
                angle = np.random.uniform(angle_mean - angle_max_bias, angle_mean + angle_max_bias)
                if not vert_i % 2:
                    angle = -angle
                length = np.clip(1, np.random.normal(length_mean, length_bias), max_length).astype(int)
                brush_width = np.random.randint(brush_width_range[0], brush_width_range[1])
                end_x = np.clip((start_x + length * np.sin(angle)).astype(np.int), 0, w)
                end_y = np.clip((start_y + length * np.cos(angle)).astype(np.int), 0, h)
                cv2.line(mask, (start_x, start_y), (end_x, end_y), 1, brush_width)
                start_x, start_y = end_x, end_y
        if self.preprocess is not None:
            mask = self.preprocess({"mask": mask})["mask"]
        return mask


@DATASETS.register()
class InpaintingDataset(Dataset):
    def __init__(self,
                 img_root,
                 img_list_path,
                 preprocess=None,
                 mask_mode="brush_stroke_mask",
                 mask_config=None):
        super(InpaintingDataset, self).__init__()
        self.img_root = img_root
        if preprocess is not None:
            self.preprocess = build_preprocess(preprocess)
        else:
            self.preprocess = None
        if mask_config is None:
            mask_config = {}
        with open(img_list_path, 'r') as f:
            self.img_list = f.read().split("\n")[:-1]
        if mask_mode is "file_mask":
            self.img_list = [label.split("\t")[0] for label in self.img_list]
        self.mask_synther = MaskSynther(mask_mode, **mask_config)

    def __getitem__(self, index):
        img_name = self.img_list[index]
        img_path = os.path.join(self.img_root, img_name)
        img = cv2.imread(img_path)
        if self.preprocess is not None:
            img = self.preprocess({"img": img})["img"]
        mask = self.mask_synther[(index, img)]
        img = img * (1 - mask)
        return {"img": img, "mask": mask}

    def __len__(self):
        return len(self.img_list)
