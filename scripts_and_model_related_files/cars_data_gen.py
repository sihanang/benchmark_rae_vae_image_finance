# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 13:36:42 2023
Adapted from https://github.com/foamliu/Autoencoder/blob/69010395d4e34fe5f012cfc2eaf29f95fd1a1379/data_gen.py
"""
import os

import cv2 as cv
import numpy as np
from torch.utils.data import Dataset

def load_data(split, imsize, appendum = ''):
    # (num_samples, 320, 320, 4)
    num_samples = 8144
    train_split = 0.8
    num_train = int(round(num_samples * train_split))
    num_valid = num_samples - num_train

    if split == 'train':
        num_samples = num_train
        folder = '../data/image_data/stanford_cars/train' + appendum
    else:
        num_samples = num_valid
        folder = '../data/image_data/stanford_cars/valid' + appendum

    x = np.empty((num_samples, 3, imsize, imsize), dtype=np.float32)
    y = np.empty((num_samples, 3, imsize, imsize), dtype=np.float32)

    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.jpg')]

    for i, filename in enumerate(files):
        bgr_img = cv.imread(filename)
        rgb_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB)
        rgb_img = np.transpose(rgb_img, (2, 0, 1))
        # print('rgb_img.shape: ' + str(rgb_img.shape))
        assert rgb_img.shape == (3, imsize, imsize)
        assert np.max(rgb_img) <= 255

        x[i, :, :, :] = rgb_img / 255.
        y[i, :, :, :] = rgb_img / 255.

    return x, y


class VaeDataset(Dataset):
    def __init__(self, split, imsize, appendum):
        self.split = split
        self.x, self.y = load_data(split, imsize, appendum)

    def __getitem__(self, i):
        return self.x[i], self.y[i]

    def __len__(self):
        return len(self.x)
    
    

def load_test(imsize, appendum= ''):
    # (num_samples, 320, 320, 4)
    num_samples = 8041
    
    folder = '../data/image_data/stanford_cars/test' + appendum

    x = np.empty((num_samples, 3, imsize, imsize), dtype=np.float32)
    y = np.empty((num_samples, 3, imsize, imsize), dtype=np.float32)

    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.jpg')]

    for i, filename in enumerate(files):
        bgr_img = cv.imread(filename)
        rgb_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB)
        rgb_img = np.transpose(rgb_img, (2, 0, 1))
        # print('rgb_img.shape: ' + str(rgb_img.shape))
        assert rgb_img.shape == (3, imsize, imsize)
        assert np.max(rgb_img) <= 255

        x[i, :, :, :] = rgb_img / 255.
        y[i, :, :, :] = rgb_img / 255.

    return x, y

class VaeTestDataset(Dataset):
    def __init__(self, imsize, appendum):
        self.x, self.y = load_test(imsize, appendum)

    def __getitem__(self, i):
        return self.x[i], self.y[i]

    def __len__(self):
        return len(self.x)
