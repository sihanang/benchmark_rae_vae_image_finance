# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 17:52:20 2023

@author: angsi
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

kl_weight = 0.09024


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


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)
    

class VaeEncoder(nn.Module):
    def __init__(self, latent_dims, constant_sigma = None):
        super(VaeEncoder, self).__init__()
        
        self.conv_layers1 = nn.Sequential(
            
            nn.Conv2d(1, out_channels = 128, kernel_size = (4,4), stride = (2,2), padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # nn.LeakyReLU(0.1),
            nn.Conv2d(128, out_channels = 256, kernel_size = (4,4), stride = (2,2), padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # nn.LeakyReLU(0.1),
            nn.Conv2d(256, out_channels = 512, kernel_size = (4,4), stride = (2,2), padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # nn.LeakyReLU(0.1),
            nn.Conv2d(512, out_channels = 1024, kernel_size = (4,4), stride = (2,2), padding = 1),
            nn.BatchNorm2d(1024),
            nn.ReLU()
            # nn.LeakyReLU(0.1)
        )

        self.conv_layers2 = nn.Sequential(
            
            nn.Conv2d(1, out_channels = 128, kernel_size = (4,4), stride = (2,2), padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # nn.LeakyReLU(0.1),
            nn.Conv2d(128, out_channels = 256, kernel_size = (4,4), stride = (2,2), padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # nn.LeakyReLU(0.1),
            nn.Conv2d(256, out_channels = 512, kernel_size = (4,4), stride = (2,2), padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # nn.LeakyReLU(0.1),
            nn.Conv2d(512, out_channels = 1024, kernel_size = (4,4), stride = (2,2), padding = 1),
            nn.BatchNorm2d(1024),
            nn.ReLU()
            # nn.LeakyReLU(0.1)
        )

        self.linear = nn.Linear(4096, latent_dims) #  128*72 = 9216
        self.dense_for_sigma = nn.Sequential( 
            
            nn.Linear(4096, latent_dims),
            nn.Tanh()
            )
        
        # Initialise variables
        self.constant_sigma = constant_sigma
        

    def forward(self, x):
    
        x1 = self.conv_layers1(x)
        x1 = torch.flatten(x1, start_dim=1)
        mu = self.linear(x1)

        x2 = self.conv_layers2(x)
        x2 = torch.flatten(x2, start_dim=1)
        
        # From the implementation in keras, it is using the 2nd last layer for log_sigma
        if self.constant_sigma is None:
            log_sigma = 5*self.dense_for_sigma(x2)
            #print('log_sigma:',log_sigma)
            sigma = torch.exp(log_sigma)
            
        else:
            # constant variance variant, using similar method as in the keras implementation
            log_sigma = LambdaLayer(lambda var: torch.log(self.constant_sigma))(x2)
            sigma = self.constant_sigma
            
        z = mu + sigma*torch.randn_like(sigma)
        
        return z, mu, log_sigma
    

class CarsVaeEncoder(nn.Module):
    def __init__(self, latent_dims, constant_sigma = None):
        super(CarsVaeEncoder, self).__init__()
        
        # For mu
        self.conv_layers1 = nn.Sequential(
            conv2DBatchNormLeakyRelu(3, 64, 3, 1, 1),
            conv2DBatchNormLeakyRelu(64, 64, 3, 2, 1),
            conv2DBatchNormLeakyRelu(64, 128, 3, 1, 1),
            conv2DBatchNormLeakyRelu(128, 128, 3, 2, 1),
            conv2DBatchNormLeakyRelu(128, 256, 3, 1, 1),
            conv2DBatchNormLeakyRelu(256, 256, 3, 2, 1),
            conv2DBatchNormLeakyRelu(256, 512, 3, 1, 1),
            conv2DBatchNormLeakyRelu(512, 512, 3, 2, 1)
        )
        

        self.dense = nn.Sequential(nn.Linear(8192, latent_dims))#,
                                   #nn.LeakyReLU(0.3, inplace = True)) #,
                                   #nn.Linear(1024, latent_dims))
                                   
        self.dense_for_sigma = nn.Sequential( 
            
            nn.Linear(8192, latent_dims),
            nn.Tanh()
            )
        
        # Initialise variables
        self.constant_sigma = constant_sigma

    def forward(self, x):
        
        x1 = self.conv_layers1(x)
        x1 = torch.flatten(x1, start_dim=1)
        mu = self.dense(x1)


        
        # From the implementation in keras, it is using the 2nd last layer for log_sigma
        if self.constant_sigma is None:
            log_sigma = 5*self.dense_for_sigma(x1)
            #print('log_sigma:',log_sigma)
            sigma = torch.exp(log_sigma)
            
        else:
            # constant variance variant, using similar method as in the keras implementation
            log_sigma = LambdaLayer(lambda var: torch.log(self.constant_sigma))(x1)
            sigma = self.constant_sigma
            
        z = mu + sigma*torch.randn_like(sigma)
        
        return z, mu, log_sigma
    

class CarsVaeDecoder(nn.Module):
    def __init__(self, latent_dims):
        super(CarsVaeDecoder, self).__init__()
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
    
class CarsVae(nn.Module):
    def __init__(self, latent_dims=64, SN = False):
        super(CarsVae, self).__init__()
        self.encoder = CarsVaeEncoder(latent_dims)
        self.decoder = CarsVaeDecoder(latent_dims)
            

    def forward(self, x):
        self.z, mu, log_sigma = self.encoder(x)
        return self.decoder(self.z), mu, log_sigma




def vae_train(autoencoder, data, lr, device, opt, kl_weight = 1, recon_loss_func = None):

    running_loss = 0.0
    
    if isinstance(autoencoder, nn.DataParallel):
        model_attr_accessor = autoencoder.module
        print('data parallel')
    else:
        model_attr_accessor = autoencoder
        #print('not data parallel')
    
    
    # Set the loss computation method
    loss_compute = VAETotalLoss(kl_weight, recon_loss_func)
    
    
    for i, x_y in enumerate(data):
        if i == 10000:
            print(i)
        x, y = x_y
        x = x.to(device) # GPU
        y = y.to(device)
        opt.zero_grad()
        x_hat, mu, log_sigma = autoencoder(x.float())
        
        
        if isinstance(autoencoder, nn.DataParallel):
            model_attr_accessor = autoencoder.module
            #print('data parallel')
        else:
            model_attr_accessor = autoencoder
            #print('not data parallel')
            
        
        loss = loss_compute(x_hat, x, mu, log_sigma)
        
        loss.to(device)
        running_loss += loss.item()
        loss.backward()
        opt.step()

    train_loss = running_loss/len(data.dataset)
    return autoencoder, train_loss, opt #, scheduler


def vae_validate(autoencoder, data, lr, device, opt, kl_weight = 1, recon_loss_func = None):
    autoencoder.eval()
    running_loss = 0.0
    
    if isinstance(autoencoder, nn.DataParallel):
        model_attr_accessor = autoencoder.module
        #print('data parallel')
    else:
        model_attr_accessor = autoencoder
        #print('not data parallel')
      
    # Set the loss computation method
    loss_compute = VAETotalLoss(kl_weight, recon_loss_func)
    
    with torch.no_grad():
    
        for i, x_y in enumerate(data):
            x, y = x_y
            x = x.to(device) # GPU
            y = y.to(device)
            x_hat, mu, log_sigma = autoencoder(x.float())
            
            if isinstance(autoencoder, nn.DataParallel):
                model_attr_accessor = autoencoder.module
            else:
                model_attr_accessor = autoencoder
            
            
            # mu = model_attr_accessor.mu
            # log_sigma = model_attr_accessor.log_sigma
            
            loss = loss_compute(x_hat, x, mu, log_sigma)
            loss.to(device)
            running_loss += loss.item()

    val_loss = running_loss/len(data.dataset)
    return autoencoder, val_loss


def vae_model_run(config, train_loader = train_dataloader, val_loader = val_dataloader, epochs = 20, lr=0.005, 
              recon_loss_func = None,
              PATH = './', latent_dims = 128, addendum = ''):
    train_loss = []
    val_loss = []
    
    autoencoder = CarsVae(latent_dims)
    kl_weight = config['kl_weight']

    
    PATH_latest = None
    val_epoch_loss = None
    flag = 0
    dataparallelornot = False
    
    # Check if the directory for saved model exists to determine how many more epochs to go
    for starting_epoch in reversed(range(epochs)):
        
        # save checkpoint
        PATH_new = PATH + 'vae_cars_' +str(starting_epoch) + '{}.pth'.format(addendum)
        
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
        scheduler = lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)
        
        if val_epoch_loss == None:
            val_epoch_loss = 0
    
    
    
    for epoch in range(starting_epoch, epochs):
    
        print(f"Epoch {epoch+1} of {epochs}")
        model, train_epoch_loss, opt = vae_train(model, train_loader, lr, device, opt, kl_weight, recon_loss_func)
        model, val_epoch_loss = vae_validate(model, val_loader, lr, device, opt, kl_weight, recon_loss_func)
        
        train_loss.append(train_epoch_loss)
        val_loss.append(val_epoch_loss)
        print(f"Train Loss: {train_epoch_loss:.8f}")
        print(f"Val Loss: {val_epoch_loss:.8f}")
        
        # save checkpoint
        PATH_new = PATH + 'vae_cars_' +str(epoch) + '{}.pth'.format(addendum)
        
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


# So far the following seems the best
vae, train_loss, val_loss = vae_model_run({'kl_weight':0.024}, lr =lr, epochs = 100, latent_dims=512, addendum='latent_512_try5')
