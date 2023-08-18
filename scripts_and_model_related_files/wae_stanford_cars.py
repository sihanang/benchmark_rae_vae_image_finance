# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 13:37:04 2023

@author: ang si han
"""
import os
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
# from torch.nn.utils.parametrizations import spectral_norm
from spectral_norm_layers import *

import torchvision
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms

from sklearn.model_selection import train_test_split
import argparse
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
from sklearn import preprocessing

#from ray import tune
#from ray.tune import CLIReporter
#from ray.tune.schedulers import ASHAScheduler

import torchvision.transforms.functional as transfunc
from cars_preprocess import *
from cars_data_gen import *


import json

from losses import *



import torchvision.transforms.functional as transfunc
class myContrastAdj:
    """Adjust image based on contrast factor."""

    def __init__(self, contrast_factor):
        self.contrast_factor = contrast_factor

    def __call__(self, x):
        return transfunc.adjust_contrast(x, self.contrast_factor)



# In[ ]:
## Setting up
epochs = 100
batch_size = 32 # the code uses 32
lr = 2e-5 # The learning rate from paper is 0.001

mmd_weight = 3e-2


# Stanford cars link broken... KIV TO CHECK AGAIN, for now, manual upload


# # This is using the dataset's mean and standard deviation from training dataset
# transform_proc = transforms.Compose([ #transforms.Resize((128, 128)),   # CELEBA is resized to 64 by 64, Stanford cars in github is 224 x 224
#                                      transforms.ToTensor(), lambda x: x*255])#,
#                                      # transforms.Normalize(mean=(0.4958, 0.4252, 0.3834), std = (0.3084, 0.2885, 0.2822))])
                                     
#                                      #transforms.Normalize(mean=(0.4627, 0.4550, 0.4450), std=(0.2745, 0.2722, 0.2839)), myContrastAdj(4)]) # For normalising training set of Stanfordcars


# # In[ ]:


# train_val_data = torchvision.datasets.StanfordCars('../data/image_data', 
#                                     split  = 'train',
#                                     download = True, transform = transform_proc)

# test_data = torchvision.datasets.StanfordCars('../data/image_data', 
#                                     split  = 'test',
#                                     download = True, transform = transform_proc)



# parameters
img_width, img_height = 64,64
print('Extracting data/cars_train.tgz...')
if not os.path.exists('../data/image_data/stanford_cars/cars_train'):
    with tarfile.open('../data/image_data/stanford_cars/cars_train.tgz', "r:gz") as tar:
        tar.extractall('../data/image_data/stanford_cars/')
print('Extracting data/image_data/stanford_cars/cars_test.tgz...')
if not os.path.exists('../data/image_data/stanford_cars/cars_test'):
    with tarfile.open('../data/image_data/stanford_cars/cars_test.tgz', "r:gz") as tar:
        tar.extractall('../data/image_data/stanford_cars/')
print('Extracting data/car_devkit.tgz...')
if not os.path.exists('../data/image_data/stanford_cars/devkit'):
    with tarfile.open('../data/image_data/stanford_cars/car_devkit.tgz', "r:gz") as tar:
        tar.extractall('../data/image_data/stanford_cars/')

cars_meta = scipy.io.loadmat('../data/image_data/stanford_cars/devkit/cars_meta')
class_names = cars_meta['class_names']  # shape=(1, 196)
class_names = np.transpose(class_names)
print('class_names.shape: ' + str(class_names.shape))
print('Sample class_name: [{}]'.format(class_names[8][0][0]))

reset_folder('../data/image_data/stanford_cars/train{}'.format(img_height))
reset_folder('../data/image_data/stanford_cars/valid{}'.format(img_height))
reset_folder('../data/image_data/stanford_cars/test{}'.format(img_height))

ensure_folder('../data/image_data/stanford_cars/train{}'.format(img_height))
ensure_folder('../data/image_data/stanford_cars/valid{}'.format(img_height))
ensure_folder('../data/image_data/stanford_cars/test{}'.format(img_height))

import random
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

process_data('train', img_height, img_width, str(img_height))
process_data('test', img_height, img_width,str(img_height))


train_dataloader = DataLoader(dataset=VaeDataset('train',img_height,str(img_height)), batch_size=batch_size, shuffle=True,
                          pin_memory=True, drop_last=True)
val_dataloader = DataLoader(dataset=VaeDataset('valid',img_height, str(img_height)), batch_size=batch_size, shuffle=False,
                        pin_memory=True, drop_last=True)
test_dataloader = DataLoader(dataset=VaeTestDataset(img_height, str(img_height)), batch_size=batch_size, shuffle=False,
                        pin_memory=True, drop_last=True)

# ## RAE

# In[ ]:

class conv2DBatchNormRelu(nn.Module):
    def __init__(
            self,
            in_channels,
            n_filters,
            k_size,
            stride,
            padding,
            bias=True,
            dilation=1,
            with_bn=True,
    ):
        super(conv2DBatchNormRelu, self).__init__()

        conv_mod = nn.Conv2d(int(in_channels),
                             int(n_filters),
                             kernel_size=k_size,
                             padding=padding,
                             stride=stride,
                             bias=bias,
                             dilation=dilation, )

        if with_bn:
            self.cbr_unit = nn.Sequential(conv_mod,
                                          nn.BatchNorm2d(int(n_filters)),
                                          nn.ReLU(inplace=True))
        else:
            self.cbr_unit = nn.Sequential(conv_mod, nn.ReLU(inplace=True))

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs
    
class conv2DBatchNormLeakyRelu(nn.Module):
    def __init__(
            self,
            in_channels,
            n_filters,
            k_size,
            stride,
            padding,
            bias=True,
            dilation=1,
            with_bn=True,
    ):
        super(conv2DBatchNormLeakyRelu, self).__init__()

        conv_mod = nn.Conv2d(int(in_channels),
                             int(n_filters),
                             kernel_size=k_size,
                             padding=padding,
                             stride=stride,
                             bias=bias,
                             dilation=dilation, )

        if with_bn:
            self.cbr_unit = nn.Sequential(conv_mod,
                                          nn.BatchNorm2d(int(n_filters)),
                                          nn.LeakyReLU(0.3, inplace=True))
        else:
            self.cbr_unit = nn.Sequential(conv_mod, nn.LeakyReLU(0.3, inplace=True))

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs
    
class conv2DTransBatchNormRelu(nn.Module):
    def __init__(
            self,
            in_channels,
            n_filters,
            k_size,
            stride,
            padding,
            output_padding=0,
            bias=True,
            dilation=1,
            with_bn=True,
    ):
        super(conv2DTransBatchNormRelu, self).__init__()

        conv_mod = nn.ConvTranspose2d(int(in_channels),
                             int(n_filters),
                             kernel_size=k_size,
                             padding=padding,
                             stride=stride,
                             output_padding = output_padding,
                             bias=bias,
                             dilation=dilation, )

        if with_bn:
            self.cbr_unit = nn.Sequential(conv_mod,
                                          nn.BatchNorm2d(int(n_filters)),
                                          nn.ReLU(inplace=True))
        else:
            self.cbr_unit = nn.Sequential(conv_mod, nn.ReLU(inplace=True))

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs


class conv2DTransBatchNormLeakyRelu(nn.Module):
    def __init__(
            self,
            in_channels,
            n_filters,
            k_size,
            stride,
            padding,
            output_padding=0,
            bias=True,
            dilation=1,
            with_bn=True,
    ):
        super(conv2DTransBatchNormLeakyRelu, self).__init__()

        conv_mod = nn.ConvTranspose2d(int(in_channels),
                             int(n_filters),
                             kernel_size=k_size,
                             padding=padding,
                             stride=stride,
                             output_padding = output_padding,
                             bias=bias,
                             dilation=dilation, )

        if with_bn:
            self.cbr_unit = nn.Sequential(conv_mod,
                                          nn.BatchNorm2d(int(n_filters)),
                                          nn.LeakyReLU(0.3, inplace=True))
        else:
            self.cbr_unit = nn.Sequential(conv_mod, nn.LeakyReLU(0.3,inplace=True))

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs


class conv2DTransBatchNormSigmoid(nn.Module):
    def __init__(
            self,
            in_channels,
            n_filters,
            k_size,
            stride,
            padding,
            output_padding=0,
            bias=True,
            dilation=1,
            with_bn=True,
    ):
        super(conv2DTransBatchNormSigmoid, self).__init__()

        conv_mod = nn.ConvTranspose2d(int(in_channels),
                             int(n_filters),
                             kernel_size=k_size,
                             padding=padding,
                             stride=stride,
                             output_padding = output_padding,
                             bias=bias,
                             dilation=dilation, )

        if with_bn:
            self.cbr_unit = nn.Sequential(conv_mod,
                                          nn.BatchNorm2d(int(n_filters)),
                                          nn.Sigmoid())
        else:
            self.cbr_unit = nn.Sequential(conv_mod, nn.Sigmoid())

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs


class RegularisedEncoder(nn.Module):
    def __init__(self, latent_dims, SN =False):
        super(RegularisedEncoder, self).__init__()

        ## Try another architecture
        self.conv1_1 =  conv2DBatchNormLeakyRelu(3, 64, 3, 1, 1)
        self.conv1_2 =  conv2DBatchNormLeakyRelu(64, 64, 3, 2, 1)
        self.conv2_1 =  conv2DBatchNormLeakyRelu(64, 128, 3, 1, 1)
        self.conv2_2 =  conv2DBatchNormLeakyRelu(128, 128, 3, 2, 1)
        self.conv3_1 =  conv2DBatchNormLeakyRelu(128, 256, 3, 1, 1)
        self.conv3_2 =  conv2DBatchNormLeakyRelu(256, 256, 3, 2, 1)
        self.conv4_1 =  conv2DBatchNormLeakyRelu(256, 512, 3, 1, 1)
        self.conv4_2 =  conv2DBatchNormLeakyRelu(512, 512, 3, 2, 1)
        
        # self.linear = nn.Linear(8192, latent_dims)
        self.dense = nn.Sequential(nn.Linear(8192, latent_dims))#,

        self.SN = SN

    def forward(self, x):

        ## Trying another one
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        
        x = torch.flatten(x, start_dim=1)
        z = self.dense(x)
        
        return z
    

class Decoder(nn.Module):
    def __init__(self, latent_dims, SN = False):
        super(Decoder, self).__init__()
        self.fc_layer = nn.Sequential(
            nn.Linear(latent_dims, 8*8*512) # archi 2: 8*8*256 instead of 512
        )
        # self.batch_norm1 = nn.BatchNorm2d(1024)
        self.batch_norm1 = nn.BatchNorm2d(512)
        
        self.lrelu = nn.LeakyReLU(0.3)
 
        self.ctbr1 = conv2DTransBatchNormLeakyRelu(512,256,3,2,1,1)
        self.ctbr2 = conv2DTransBatchNormLeakyRelu(256,128,3,2,1,1)
        self.ctbr3 = conv2DTransBatchNormLeakyRelu(128,64,3,2,1,1)
        self.convtransposefinal = nn.ConvTranspose2d(64, out_channels = 3, kernel_size = 3, stride = (1,1), padding = 1)
        
        
        self.SN = SN

    def forward(self, z):


        x = self.fc_layer(z)
        x = x.view(x.size(0),512,8,8)
        x = self.lrelu(self.batch_norm1(x))
        x = self.ctbr1(x)
        x = self.ctbr2(x)
        x = self.ctbr3(x)
        x = self.convtransposefinal(x)
    
        x = F.sigmoid(x)
        # print(x.shape)
        return x
    
class RegularisedAutoencoder(nn.Module):
    def __init__(self, latent_dims=64, SN = False):
        super(RegularisedAutoencoder, self).__init__()
        self.encoder = RegularisedEncoder(latent_dims, SN)
        self.decoder = Decoder(latent_dims, SN)
            

    def forward(self, x):
        self.z = self.encoder(x)
        return self.decoder(self.z)
    
    
def wae_train(autoencoder, data, mmd_weight, lr, device, opt, latent_dims,
              regularisation_loss_type = 'l2'):
    
    '''
    regularisation_loss_type either l2 or l1
    '''

    running_loss = 0.0
    
    if isinstance(autoencoder, nn.DataParallel):
        model_attr_accessor = autoencoder.module
        #print('data parallel')
    else:
        model_attr_accessor = autoencoder
        #print('not data parallel')
    
    
    # Set the loss computation method
    loss_compute = WAETotalLoss(device = device, mmd_weight=mmd_weight, recon_loss_name=regularisation_loss_type)
    
    
    for i, x_y in enumerate(data):
        if i == 10000:
            print(i)
        x, y = x_y
        x = x.to(device) # GPU
        y = y.to(device)
        opt.zero_grad()
        
        if isinstance(autoencoder, nn.DataParallel):
            model_attr_accessor = autoencoder.module
            #print('data parallel')
        else:
            model_attr_accessor = autoencoder
            #print('not data parallel')
            
        z = model_attr_accessor.encoder(x.float())
        x_hat = model_attr_accessor.decoder(z.float())
        
        # print(dir(model_attr_accessor))
        
        loss = loss_compute(x_hat, x, z, latent_dims)
        
        loss.to(device)
        running_loss += loss.item()
        loss.backward()
        opt.step()
    # scheduler.step(val_loss)
    train_loss = running_loss/len(data.dataset)
    return autoencoder, train_loss, opt #, scheduler


def wae_validate(autoencoder, data, mmd_weight, lr, device, opt, latent_dims,
              regularisation_loss_type = 'l2'):
    autoencoder.eval()
    running_loss = 0.0
    
    if isinstance(autoencoder, nn.DataParallel):
        model_attr_accessor = autoencoder.module
        #print('data parallel')
    else:
        model_attr_accessor = autoencoder
        #print('not data parallel')
      
    # Set the loss computation method
    loss_compute = WAETotalLoss(device = device, mmd_weight=mmd_weight, recon_loss_name=regularisation_loss_type)
    
    with torch.no_grad():
    
        for i, x_y in enumerate(data):
            x, y = x_y
            x = x.to(device) # GPU
            y = y.to(device)
            # x_hat = autoencoder(x.float())
            
            if isinstance(autoencoder, nn.DataParallel):
                model_attr_accessor = autoencoder.module
            else:
                model_attr_accessor = autoencoder
            
            z = model_attr_accessor.encoder(x.float())
            x_hat = model_attr_accessor.decoder(z.float())
            
            # print(dir(model_attr_accessor))
            
            loss = loss_compute(x_hat, x, z, latent_dims)
            
            loss.to(device)
            running_loss += loss.item()

    val_loss = running_loss/len(data.dataset)
    return autoencoder, val_loss


def wae_model_run(config, train_loader = train_dataloader, val_loader = val_dataloader, epochs = 20, lr=0.005, 
                  regularisation_loss_type = 'l2',
                  PATH = './', latent_dims = 128, appendum = ''):
    train_loss = []
    val_loss = []
    
    
    mmd_weight = config['mmd_weight']
    
    # Only implement the one without spectral normalisation
    autoencoder = RegularisedAutoencoder(latent_dims)
    
    PATH_latest = None
    val_epoch_loss = None
    flag = 0
    dataparallelornot = False
    
    # Check if the directory for saved model exists to determine how many more epochs to go
    for starting_epoch in reversed(range(epochs)):
        
        # save checkpoint
        PATH_new = PATH + 'wae_cars' + '_' +str(starting_epoch) + '{}.pth'.format(appendum)
        
        # Check if this directory doesn't exists, continue
        if not os.path.exists(PATH_new):
            continue
        else:
            PATH_latest = PATH_new
            print('Latest path found: ', PATH_latest)
            flag = 1
            starting_epoch+=1
            break
        
    
    # Load the model if latest is found, otherwise use initial model
    if flag == 1:
        
        checkpoint = torch.load(PATH_latest)
        autoencoder.load_state_dict(checkpoint['model_state_dict'])
        
        # set devices
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(autoencoder)
            print('No. of GPUs: {}'.format(torch.cuda.device_count()))
            model.to(device)
            dataparallelornot = True
        
        else:
            model = autoencoder.to(device) #GPU
            
        # For model training
        model.train()
        
        
        # Initialise optimiser and scheduler
        opt = torch.optim.Adam(model.parameters(), lr=lr, betas = (0.9, 0.999)) #, weight_decay = 1e-4)
        
        # Reduce learning rate by half with each plateau
        scheduler = lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)
        
        
        # opt = torch.optim.Adam(model.parameters(), lr=lr)
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        
        train_loss = checkpoint['train_loss']
        val_loss = checkpoint['val_loss']
        
        val_epoch_loss = val_loss[-1]
        
    else:
        # Reset starting_epoch to 0 as no model was saved
        starting_epoch = 0
        
        # set devices
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(autoencoder)
            print('No. of GPUs: {}'.format(torch.cuda.device_count()))
            model.to(device)
            dataparallelornot = True
        
        else:
            model = autoencoder.to(device) #GPU
            
        # For model training
        model.train()
        
        # Initialise optimiser and scheduler
        opt = torch.optim.Adam(model.parameters(), lr=lr, betas = (0.9, 0.999)) #, weight_decay = 1e-4)
        
        # Reduce learning rate by half with each plateau
        scheduler = lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience = 5)
        
        if val_epoch_loss == None:
            val_epoch_loss = 0
    
    
    
    for epoch in range(starting_epoch, epochs):
    
        print(f"Epoch {epoch+1} of {epochs}")
        model, train_epoch_loss, opt = wae_train(model, train_loader, mmd_weight, lr, device, opt, latent_dims,
                                                 regularisation_loss_type)
     
        model, val_epoch_loss = wae_validate(model, val_loader, mmd_weight, lr, device, opt, latent_dims,
                                             regularisation_loss_type)

        train_loss.append(train_epoch_loss)
        val_loss.append(val_epoch_loss)
        print(f"Train Loss: {train_epoch_loss:.8f}")
        print(f"Val Loss: {val_epoch_loss:.8f}")
        
        # save checkpoint
        PATH_new = PATH + 'wae_cars' + '_' +str(epoch) + '{}.pth'.format(appendum)
        
        # Save model
        if dataparallelornot:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss
                }, PATH_new)
            print('Saving model details complete for epoch {} at: {}'.format(epoch, PATH_new))
            
        else:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss
                }, PATH_new)
            print('Saving model details complete for epoch {} at: {}'.format(epoch, PATH_new))
            
        scheduler.step(val_epoch_loss)
        
    return model, train_loss, val_loss



print('Running WAE')

# wae, train_loss, val_loss = wae_model_run({'mmd_weight':3e-2}, train_loader = train_dataloader, val_loader = val_dataloader, epochs = epochs, lr=lr, 
#                                       regularisation_loss_type = 'l2', PATH = './', latent_dims = 512, appendum = 'config_1')

# wae, train_loss, val_loss = wae_model_run({'mmd_weight':0.006}, train_loader = train_dataloader, val_loader = val_dataloader, epochs = epochs, lr=lr, 
#                                       regularisation_loss_type = 'l2', PATH = './', latent_dims = 512, appendum = 'config_2')

# wae, train_loss, val_loss = wae_model_run({'mmd_weight':0.001}, train_loader = train_dataloader, val_loader = val_dataloader, epochs = epochs, lr=lr, 
#                                       regularisation_loss_type = 'l2', PATH = './', latent_dims = 512, appendum = 'config_3')

wae, train_loss, val_loss = wae_model_run({'mmd_weight':0.1}, train_loader = train_dataloader, val_loader = val_dataloader, epochs = epochs, lr=lr, 
                                      regularisation_loss_type = 'l2', PATH = './', latent_dims = 512, appendum = 'config_8')

# wae, train_loss, val_loss = wae_model_run({'mmd_weight':mmd_weight}, train_loader = train_dataloader, val_loader = val_dataloader, epochs = epochs, lr=lr, 
#                                       regularisation_loss_type = 'l2', PATH = './', latent_dims = 512, appendum = 'config_7')