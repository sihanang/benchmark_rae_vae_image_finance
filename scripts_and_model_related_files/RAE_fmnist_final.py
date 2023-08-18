#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
from torch.nn.utils.parametrizations import spectral_norm
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

import json

from losses import *

import torchvision.transforms.functional as transfunc
class myContrastAdj:
    """Adjust image based on contrast factor."""

    def __init__(self, contrast_factor):
        self.contrast_factor = contrast_factor

    def __call__(self, x):
        return transfunc.adjust_contrast(x, self.contrast_factor)



epochs = 85
batch_size = 64 # paper uses 100, for try 5 i did 500..
lr = 0.0005 # The learning rate from paper is 0.001


lamb = 1e-8
beta = 1e-5

# This is using the dataset's mean and standard deviation from training dataset
transform_proc = transforms.Compose([transforms.Pad(2), transforms.ToTensor()]) #, myContrastAdj(4)])# , transforms.Normalize(mean=(0.2192,), std=(0.3318,))])
# transform_proc = transforms.Compose([transforms.Pad(2), transforms.ToTensor(), transforms.Normalize(mean=(0.1307,), std=(0.3081,))])

# In[ ]:


train_val_data = torchvision.datasets.FashionMNIST('../data/image_data', 
                                  train = True,
                                  download = True, transform = transform_proc)

test_data = torchvision.datasets.FashionMNIST('../data/image_data', 
                                  train = False,
                                  download = True, transform = transform_proc)

# train_val_data = torchvision.datasets.MNIST('../data/image_data', 
#                                   train = True,
#                                   download = True, transform = transform_proc)

# test_data = torchvision.datasets.MNIST('../data/image_data', 
#                                   train = False,
#                                   download = True, transform = transform_proc)


# Training and validation split
training_data, val_data = torch.utils.data.random_split(train_val_data, [50000, 10000], generator=torch.Generator().manual_seed(1))



# labels_map = {
#     0: "T-Shirt",
#     1: "Trouser",
#     2: "Pullover",
#     3: "Dress",
#     4: "Coat",
#     5: "Sandal",
#     6: "Shirt",
#     7: "Sneaker",
#     8: "Bag",
#     9: "Ankle Boot",
# }



# Check the size of the one instance
# next(iter(training_data))[0].shape



### Prepare for Training
import random
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)



# ## RAE

class RegularisedEncoder(nn.Module):
    def __init__(self, latent_dims, SN =False):
        super(RegularisedEncoder, self).__init__()
        
        self.conv_layers = nn.Sequential(
            
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
        
     
        self.sn_conv_layers = nn.Sequential(
            
            SNConv2d(1, out_channels = 128, kernel_size = (4,4), stride = (2,2), padding = (1,1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # nn.LeakyReLU(0.02),
            SNConv2d(128, out_channels = 256, kernel_size = (4,4), stride = (2,2), padding = (1,1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # nn.LeakyReLU(0.02),
            SNConv2d(256, out_channels = 512, kernel_size = (4,4), stride = (2,2), padding = (1,1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # nn.LeakyReLU(0.02),
            SNConv2d(512, out_channels = 1024, kernel_size = (4,4), stride = (2,2), padding = (1,1)),
            nn.BatchNorm2d(1024),
            nn.ReLU()
            # nn.LeakyReLU(0.02)
        )
        

        self.sn_linear = SNLinear(4096, latent_dims)
        
        self.SN = SN

    def forward(self, x):
        
        # Check if spectral norm is implemented
        if self.SN:
            x = self.sn_conv_layers(x)
            x = torch.flatten(x, start_dim=1)
            z = self.sn_linear(x)
        else:
            x = self.conv_layers(x)
            x = torch.flatten(x, start_dim=1)
            z = self.linear(x)
        
        
        return z
    

class Decoder(nn.Module):
    def __init__(self, latent_dims, SN = False):
        super(Decoder, self).__init__()
        self.fc_layer = nn.Sequential(
            nn.Linear(latent_dims, 8*8*1024)
        )
        self.batch_norm1 = nn.BatchNorm2d(1024)
        
        
        self.conv_layers = nn.Sequential(
            
            nn.ConvTranspose2d(1024, out_channels = 512, kernel_size = (4,4), stride = (2,2), padding = (1,1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(512, out_channels = 256, kernel_size = (4,4), stride = (2,2), padding = (1,1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(256, out_channels = 1, kernel_size = (5,5), stride = (1,1), padding = 2)
        )
        
        
        self.sn_conv_layers = nn.Sequential(
            
            SNConvTranspose2d(1024, out_channels = 512, kernel_size = (4,4), stride = (2,2), padding = (1,1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # nn.LeakyReLU(0.02),
            SNConvTranspose2d(512, out_channels = 256, kernel_size = (4,4), stride = (2,2), padding = (1,1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # nn.LeakyReLU(0.02),
            
            # Using kernel size of 9 by 9 instead to get 32
            SNConvTranspose2d(256, out_channels = 1, kernel_size = (5,5), stride = (1,1), padding = 2)
        )

        
        self.sn_fc_layer = nn.Sequential(
            SNLinear(latent_dims, 8*8*1024)
        )
        
        self.SN = SN

    def forward(self, z):

        # Check if spectral norm is implemented
        if self.SN:
            x = self.sn_fc_layer(z)
            x = x.view(x.size(0),1024,8,8)
            x = F.relu(self.batch_norm1(x))
            x = self.sn_conv_layers(x)

        else:
            x = self.fc_layer(z)
            x = x.view(x.size(0),1024,8,8)
            # print(x.shape)
            x = F.relu(self.batch_norm1(x))
            x = self.conv_layers(x)
        
        x = F.sigmoid(x)
        # print(x.shape)
        return x
    
class RegularisedAutoencoder(nn.Module):
    def __init__(self, latent_dims=32, SN = False):
        super(RegularisedAutoencoder, self).__init__()
        self.encoder = RegularisedEncoder(latent_dims, SN)
        self.decoder = Decoder(latent_dims, SN)
            

    def forward(self, x):
        self.z = self.encoder(x)
        return self.decoder(self.z)
    
    
def rae_train(autoencoder, data, beta, lamb, lr, device, opt,
              apply_grad_pen = False,
              grad_pen_weight = None,
              entropy_qz=None,
              regularization_loss=None, 
              loss='l2',
              regularisation_loss_type = 'l2'):

    running_loss = 0.0
    
    if isinstance(autoencoder, nn.DataParallel):
        model_attr_accessor = autoencoder.module
        #print('data parallel')
    else:
        model_attr_accessor = autoencoder
        #print('not data parallel')
    
    
    # Set the loss computation method
    loss_compute = TotalLoss(apply_grad_pen=apply_grad_pen, grad_pen_weight=grad_pen_weight,
                             entropy_qz=entropy_qz,
                             regularization_loss=regularization_loss, beta=beta, lamb = lamb, loss='l2')
    
    
    for i, x_y in enumerate(data):
        if i == 10000:
            print(i)
        x, y = x_y
        x = x.to(device) # GPU
        y = y.to(device)
        opt.zero_grad()
        # x_hat = autoencoder(x.float())
        
        
        if isinstance(autoencoder, nn.DataParallel):
            model_attr_accessor = autoencoder.module
            #print('data parallel')
        else:
            model_attr_accessor = autoencoder
            #print('not data parallel')
                
                
        z = model_attr_accessor.encoder(x.float())
        x_hat = model_attr_accessor.decoder(z.float())
        
        
        loss = loss_compute(x_hat, x, z, model_attr_accessor.decoder.parameters())
        
        loss.to(device)
        running_loss += loss.item()
        loss.backward()
        opt.step()
    # scheduler.step(val_loss)
    train_loss = running_loss/len(data.dataset)
    return autoencoder, train_loss, opt #, scheduler


def rae_validate(autoencoder, data, beta, lamb, device,
                 apply_grad_pen = False,
                 grad_pen_weight = None,
                 entropy_qz=None,
                 regularization_loss=None,
                 loss='l2',
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
    loss_compute = TotalLoss(apply_grad_pen=apply_grad_pen, grad_pen_weight=grad_pen_weight,
                             entropy_qz=entropy_qz,
                             regularization_loss=regularization_loss, beta=beta, lamb = lamb, loss='l2')
    
    if regularisation_loss_type != 'grad_pen':
        with torch.no_grad():
        
            for i, x_y in enumerate(data):
                x, y = x_y
                x = x.to(device) # GPU
                y = y.to(device)
                x_hat = autoencoder(x.float())
                # l_reg = rae_reg_loss(autoencoder)
                
                if isinstance(autoencoder, nn.DataParallel):
                    model_attr_accessor = autoencoder.module
                else:
                    model_attr_accessor = autoencoder
                
                # loss = ((x.float() - x_hat.float())**2).sum() + beta*(model_attr_accessor.z.float()**2).sum()/2 + lamb*l_reg
                # loss = ((x.float() - x_hat.float())**2).mean() + beta*(model_attr_accessor.encoder(x.float()).float()**2).mean() + lamb*l_reg
                
                loss = loss_compute(x_hat, x, model_attr_accessor.encoder(x.float()), model_attr_accessor.decoder.parameters())
                
                loss.to(device)
                running_loss += loss.item()
        
    else:
        for i, x_y in enumerate(data):
            x, y = x_y
            x = x.to(device) # GPU
            y = y.to(device)
            x_hat = autoencoder(x.float())
            # l_reg = rae_reg_loss(autoencoder)
            
            if isinstance(autoencoder, nn.DataParallel):
                model_attr_accessor = autoencoder.module
            else:
                model_attr_accessor = autoencoder
            
            # z, add_dict = model_attr_accessor.encoder(x.float())
            # z = torch.flatten(z, start_dim = 1)
            
            z = model_attr_accessor.encoder(x.float())
            x_hat = model_attr_accessor.decoder(z.float())
            
            loss = loss_compute(x_hat, x, z, model_attr_accessor.decoder.parameters())
            
            loss.to(device)
            running_loss += loss.item()

    val_loss = running_loss/len(data.dataset)
    return autoencoder, val_loss

def model_run(config, train_loader = train_dataloader, val_loader = val_dataloader, epochs = 20, lr=0.005, 
              reg_type = 'l2',
              PATH = './', latent_dims = 128, appendum = ''):
    train_loss = []
    val_loss = []
    
    
    lamb = config['lamb']
    beta = config['beta']
    
    # Set up implementation for regularisation_loss
    if reg_type == 'l2':
        apply_grad_pen = False
        grad_pen_weight = None
        regularization_loss = L2RegLoss()
        
    elif reg_type == 'grad_pen':
        apply_grad_pen = True
        grad_pen_weight = lamb
        regularization_loss = None # Not using the L2 Loss
    elif reg_type == 'spec_norm':
        apply_grad_pen = False
        grad_pen_weight = None
        regularization_loss = None # Not using the L2 Loss
    else:
        print('Such regularisation is currently not implemented.')
    
    # Use the spectral norm version of RegularisedAutoencoder
    if reg_type == 'spec_norm':
        autoencoder = RegularisedAutoencoder(latent_dims, SN = True)
    else:
        autoencoder = RegularisedAutoencoder(latent_dims)

    
    PATH_latest = None
    val_epoch_loss = None
    flag = 0
    dataparallelornot = False
    
    # Check if the directory for saved model exists to determine how many more epochs to go
    for starting_epoch in reversed(range(epochs)):
        
        # save checkpoint
        PATH_new = PATH + 'rae_fmnist'+ reg_type + '_' +str(starting_epoch) + '{}.pth'.format(appendum)
        
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
        scheduler = lr_scheduler.ReduceLROnPlateau(opt, factor=0.5)
        
        
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
        scheduler = lr_scheduler.ReduceLROnPlateau(opt, factor=0.5)
        
        if val_epoch_loss == None:
            val_epoch_loss = 0
    
    
    
    for epoch in range(starting_epoch, epochs):
    
        print(f"Epoch {epoch+1} of {epochs}")
        model, train_epoch_loss, opt = rae_train(model, train_loader, beta, lamb, lr, device, opt, regularisation_loss_type=reg_type,
                                                 apply_grad_pen = apply_grad_pen,
                                                 grad_pen_weight = grad_pen_weight,
                                                 regularization_loss = regularization_loss)#, scheduler, val_epoch_loss)
        model, val_epoch_loss = rae_validate(model, val_loader, beta, lamb, device, regularisation_loss_type=reg_type,
                                             apply_grad_pen = apply_grad_pen,
                                             grad_pen_weight = grad_pen_weight,
                                             regularization_loss = regularization_loss)
        train_loss.append(train_epoch_loss)
        val_loss.append(val_epoch_loss)
        print(f"Train Loss: {train_epoch_loss:.8f}")
        print(f"Val Loss: {val_epoch_loss:.8f}")
        
        # save checkpoint
        PATH_new = PATH + 'rae_fmnist'+ reg_type + '_' +str(epoch) + '{}.pth'.format(appendum)
        
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



################### RAE RUNS ##################################
rae, train_loss, val_loss= model_run({'beta':beta, 'lamb':lamb}, train_dataloader, val_dataloader, epochs = epochs, lr = lr, 
                                        reg_type = 'l2',
                                        PATH = './', latent_dims = 32, appendum='lat32_batchsize64aa_final')

rae, train_loss, val_loss= model_run({'beta':1e-5, 'lamb':0}, train_dataloader, val_dataloader, epochs = epochs, lr = 5e-4, 
                                        reg_type = 'spec_norm',
                                        PATH = './', latent_dims = 32, appendum='lat32_batchsize64_8_rerun')

rae, train_loss, val_loss= model_run({'beta':1e-6, 'lamb':1e-9}, train_dataloader, val_dataloader, epochs = epochs, lr = lr, 
                                        reg_type = 'grad_pen',
                                        PATH = './', latent_dims = 32, appendum='lat32_batchsize64ab_final')


############## AE is run here ###################################
ae, train_loss, val_loss= model_run({'beta':0, 'lamb':0}, train_dataloader, val_dataloader, epochs = epochs, lr = lr, 
                                        reg_type = 'l2',
                                        PATH = './', latent_dims=32, appendum='lat32_batchsize64_AE')
