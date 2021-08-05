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


class MaskGenerator(object):
    def __init__(self, mask_mode="irregular_mask", **mask_config):
        self.mask_mode = mask_mode
        self.mask_config = mask_config

    def __getitem__(self, index, img):
        return getattr(self, self.mask_mode)(index, img)

    def file_mask(self, index, img):
        return

    def list_mask(self, index, img):
        return

    def brush_mask(self, index, img):
        return

    def irregular_mask(self, index, img):
        h, w, _ = img.shape
        mask = np.zeros((h, w))
        vert_num_range = self.mask_config.get("num_vertexes", (4, 12))
        assert isinstance(vert_num_range, tuple), \
            "The type of vert_num_range should be tuple, but got {}".format(type(vert_num_range))
        vert_num = np.random.randint(vert_num_range[0], vert_num_range[1])

        length_range = self.mask_config.get("length_range", (10, 100))
        assert isinstance(length_range, tuple), \
            "The type of length_range should be tuple, but got {}".format(type(length_range))

        brush_width_range = self.mask_config.get("brush_width_range", (12, 40))
        assert isinstance(brush_width_range, tuple), \
            "The type of brush_width_range should be tuple, but got {}".format(type(brush_width_range))

        direction_num_range = self.mask_config.get("direction_num_range", (1, 6))
        assert isinstance(direction_num_range, tuple), \
            "The type of direction_num_range should be tuple, but got {}".format(type(direction_num_range))

        angle_mean = self.mask_config.get('angle_mean', np.pi * 2 / 5)
        assert isinstance(angle_mean, float), \
            "The type of angle_mean should be float, but got {}".format(type(angle_mean))

        angle_max_bias = self.mask_config.get('angle_max_bias', np.pi * 2 / 15)
        assert isinstance(angle_mean, float), \
            "The type of angle_mean should be float, but got {}".format(type(angle_mean))

        for vert_i in range(vert_num):
            start_x = np.random.randint(w)
            start_y = np.random.randint(h)
            direction_num = np.random.randint(direction_num_range[0], direction_num_range[1])
            for direct_i in range(direction_num):
                angle = np.random.uniform(angle_mean - angle_max_bias, angle_mean + angle_max_bias)
                if not vert_i % 2:
                    angle = -angle
                length = np.random.randint(length_range[0], length_range[1])
                brush_width = np.random.randint(brush_width_range[0], brush_width_range[1])
                end_x = (start_x + length * np.sin(angle)).astype(np.int)
                end_y = (start_y + length * np.cos(angle)).astype(np.int)
                cv2.line(mask, (start_y, start_x), (end_y, end_x), 1, brush_width)
                start_x, start_y = end_x, end_y
        mask = np.expand_dims(mask, axis=2)
        return mask


class InpaintingDataset(Dataset):
    def __init__(self, img_root, img_list_path, mask_mode, ):
        super(InpaintingDataset, self).__init__()

    def __getitem__(self, index):
        return 0
