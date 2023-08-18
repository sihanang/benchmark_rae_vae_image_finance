# -*- coding: utf-8 -*-
"""
Adapted from https://github.com/foamliu/Autoencoder/blob/69010395d4e34fe5f012cfc2eaf29f95fd1a1379/pre_process.py#L20
"""

import os
import shutil
import random
import tarfile

import cv2 as cv
import numpy as np
import scipy.io
from tqdm import tqdm


def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
        

def reset_folder(folder):
    if not os.path.exists(folder):
        return
    else: 
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))


def save_train_data(fnames, class_ids, bboxes, img_height, img_width, appendum = ''):
    src_folder = '../data/image_data/stanford_cars/cars_train'
    num_samples = len(fnames)

    train_split = 0.8
    num_train = int(round(num_samples * train_split))
    train_indexes = random.sample(range(num_samples), num_train)
    print(len(train_indexes))
    print('train_indexes: '.format(str(train_indexes)))

    for i in tqdm(range(num_samples)):
        fname = fnames[i]
        (x1, y1, x2, y2) = bboxes[i]
        src_path = os.path.join(src_folder, fname)
        src_image = cv.imread(src_path)
        height, width = src_image.shape[:2]
        # margins of 16 pixels
        margin = 16
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(x2 + margin, width)
        y2 = min(y2 + margin, height)
        # print(fname)
        
        class_id = class_ids[i]
        
        # These are processed folders
        if i in train_indexes:
            dst_folder = '../data/image_data/stanford_cars/train{}/'.format(appendum)  #+ str(class_id)
        else:
            dst_folder = '../data/image_data/stanford_cars/valid{}/'.format(appendum) #+ str(class_id)
            
        
        
        if not os.path.exists(dst_folder):
            os.makedirs(dst_folder)
        
        dst_path = os.path.join(dst_folder, fname)

        crop_image = src_image[y1:y2, x1:x2]
        dst_img = cv.resize(src=crop_image, dsize=(img_height, img_width))
        cv.imwrite(dst_path, dst_img)
    print('\n')


def save_test_data(fnames, class_ids, bboxes, img_height, img_width, appendum=''):
    # Source folder
    src_folder = '../data/image_data/stanford_cars/cars_test'
    

    num_samples = len(fnames)

    for i in tqdm(range(num_samples)):
        fname = fnames[i]
        (x1, y1, x2, y2) = bboxes[i]
        src_path = os.path.join(src_folder, fname)
        src_image = cv.imread(src_path)
        height, width = src_image.shape[:2]
        # margins of 16 pixels
        margin = 16
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(x2 + margin, width)
        y2 = min(y2 + margin, height)
        # print(fname)
        
        class_id = class_ids[i]
        
        #destination folder
        dst_folder = '../data/image_data/stanford_cars/test{}/'.format(appendum)
        
        # This is only for different loading method
        # dst_folder += str(class_id)
        
        if not os.path.exists(dst_folder):
            os.makedirs(dst_folder)

        dst_path = os.path.join(dst_folder, fname)
        crop_image = src_image[y1:y2, x1:x2]
        dst_img = cv.resize(src=crop_image, dsize=(img_height, img_width))
        cv.imwrite(dst_path, dst_img)
    print('\n')


def process_data(usage, img_height, img_width, appendum = ''):
    print("Processing {} data...".format(usage))
    if usage == 'test':
        cars_annos = scipy.io.loadmat('../data/image_data/stanford_cars/cars_test_annos_withlabels')
    else:
        cars_annos = scipy.io.loadmat('../data/image_data/stanford_cars/devkit/cars_{}_annos'.format(usage))
    annotations = cars_annos['annotations']
    annotations = np.transpose(annotations)
    
    # for training labels
    cars_meta = scipy.io.loadmat('../data/image_data/stanford_cars/devkit/cars_meta')
    class_names = cars_meta['class_names']  # shape=(1, 196)
    class_names = np.transpose(class_names)

    fnames = []
    bboxes = []
    
    class_ids = []

    for annotation in annotations:
        bbox_x1 = annotation[0][0][0][0]
        bbox_y1 = annotation[0][1][0][0]
        bbox_x2 = annotation[0][2][0][0]
        bbox_y2 = annotation[0][3][0][0]
        class_id = annotation[0][4][0][0]
        fname = annotation[0][5][0]
            
        bboxes.append((bbox_x1, bbox_y1, bbox_x2, bbox_y2))
        fnames.append(fname)
        class_ids.append(class_id)
        

    if usage == 'train':
        save_train_data(fnames, class_ids, bboxes, img_height, img_width,appendum)
    else:
        save_test_data(fnames, class_ids, bboxes, img_height, img_width,appendum)