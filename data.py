import log
import logging
logger = logging.getLogger('root') 

import torch
from torch import Tensor

from torch.utils.data import Dataset

import pandas as pd
import numpy as np

from os import path
from PIL import Image

class MedicalImage:
    def __init__(self, img, class_name, class_id, rad_id, x_min, y_min, x_max, y_max):
        self.data = img
        self.class_name = class_name
        self.class_id = class_id
        self.rad_id = rad_id
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

    def split(self, num_x_splits, num_y_splits):
        list_X, list_Y = self.get_split_dims(num_x_splits, num_y_splits)
        return self.exact_split(list_X, list_Y)

    def get_split_dims(self, num_x_splits, num_y_splits, ignore_split_incompatability = False):
        def split_idx_generator(gstep, gmax):
            g = 0
            while g <= gmax:
                yield g
                g += gstep
        x_dim, y_dim = self.data.shape[:2]
        x_spacing, y_spacing = x_dim/num_x_splits, y_dim/num_y_splits
        x_spacing_int, y_spacing_int = int(x_spacing), int(y_spacing)
        if not ignore_split_incompatability and (x_spacing != x_spacing_int or y_spacing != y_spacing_int):
            if x_spacing != x_spacing_int:
                raise ValueError(f"x_dim={x_dim} and num_x_splits={num_x_splits} does not divide evenly ({x_spacing:.3f}).")
            else:
                raise ValueError(f"y_dim={y_dim} and num_y_splits={y_spacing} does not divide evenly ({y_spacing:.3f}).")
        list_X, list_Y = list(split_idx_generator(x_spacing_int, x_dim)), list(split_idx_generator(y_spacing_int, y_dim))
        return list_X, list_Y

    def exact_split(self, list_X, list_Y):
        """
                e.g. list_X=[0,10,15]; list_Y=[0,5,10] will produce images

                        [ [ (0,0),(10,5)], [ (0,6),(10,10)],
                          [(10,0),(15,5)], [(10,6),(15,10)]  ]
        """
        it_X = len(list_X) - 1
        it_Y = len(list_Y) - 1
        images = [None]*(it_X*it_Y)
        # c = 0
        # for i in range(it_X):
        #     for j in range(it_Y):
        #         images[c] = self.data[list_X[i]:list_X[i+1],
        #                               list_Y[j]:list_Y[j+1]]
        #         c +=1
        # print([(i, im.shape) for i, im in enumerate(images)])
        # images = Tensor(images)
        # return images
        return Tensor(
                  [self.data[list_X[i]:list_X[i+1],
                             list_Y[j]:list_Y[j+1]]
                    for i in range(it_X) for j in range(it_Y)]
        )


class ThoracicDataset(Dataset):
    def __init__(self, summary_csv, root_dir, transform=None, pre_split=True):
        self.summary_df = pd.read_csv(path.join(root_dir, summary_csv))
        self.root_dir = root_dir
        self.transform = transform
        self.summary_csv = summary_csv
        self.pre_split = pre_split
        self.split_list_X = None
        self.split_list_Y = None
        self._num_tiles = None
        self.num_x_splits = None
        self.num_y_splits = None
        self._shape = None
        self.split_ready = False

    def _register_X_split(self, list_X):
        self.split_list_X = list_X

    def _register_Y_split(self, list_Y):
        self.split_list_Y = list_Y

    def register_splits(self, list_X, list_Y):
        self._register_X_split(list_X)
        self._register_Y_split(list_Y)

        self.num_x_splits = len(list_X)-1
        self.num_y_splits = len(list_Y)-1
        self.split_ready  = True

    @property
    def num_tiles(self):
        if not self._num_tiles:
            if self.split_list_X and self.split_list_Y:
                self._num_tiles = (len(self.split_list_X)-1)*(len(self.split_list_Y)-1)
            else:
                self._num_tiles = 1
                # raise ValueError("Must register splits before number of splits can be determined.")
        return self._num_tiles

    @property
    def shape(self):
        """
            will be tile shape if pre_split is True
        """
        if not self._shape:
            self._shape = self[0].shape[-3:-1] # TODO work channels into the dimensions... This will change the tiling function too
        return self._shape

    def read_img(self, path):
        img = Image.open("data/pokemon/1.png")
        img.load()
        return np.asarray(img, dtype="int32")
    
    def __len__(self):
        return len(self.summary_df)

    def get_med_image(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_path = path.join(self.root_dir, self.summary_df.iloc[idx,0])
        img = self.read_img(img_path)

        if self.transform: 
            img = self.transform(img)

        class_name = self.summary_df.iloc[idx,1]
        class_id   = self.summary_df.iloc[idx,2]
        rad_id     = self.summary_df.iloc[idx,3]
        x_min = self.summary_df.iloc[idx,4]
        y_min = self.summary_df.iloc[idx,5]
        x_max = self.summary_df.iloc[idx,6]
        y_max = self.summary_df.iloc[idx,7]

        return MedicalImage(img, class_name, class_id, rad_id, x_min, y_min, x_max, y_max)


    def __getitem__(self, idx):
        med_img = self.get_med_image(idx)

        res = Tensor([med_img.data]) if not self.pre_split or not self.split_ready else med_img.exact_split(self.split_list_X, self.split_list_Y)

        return res




