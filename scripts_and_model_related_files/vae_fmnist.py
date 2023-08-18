# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 17:21:52 2023

@author: angsi
"""
import os
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

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


import json

from losses import *



# In[ ]:
## Setting up
epochs = 85
batch_size = 64 # paper uses 100, for try 5 i did 500..
lr = 1e-5 #1e-6

kl_weight = 0.312
# kl_weight = 0.024

# This is using the dataset's mean and standard deviation from training dataset
transform_proc = transforms.Compose([transforms.Pad(2), transforms.ToTensor()]) #, transforms.Normalize(mean=(0.2192,), std=(0.3318,))])
# transform_proc = transforms.Compose([transforms.Pad(2), transforms.ToTensor(), transforms.Normalize(mean=(0.1307,), std=(0.3081,))])

# In[ ]:


train_val_data = torchvision.datasets.FashionMNIST('../data/image_data', 
                                  train = True,
                                  download = True, transform = transform_proc)

test_data = torchvision.datasets.FashionMNIST('../data/image_data', 
                                  train = False,
                                  download = True, transform = transform_proc)

#train_val_data = torchvision.datasets.MNIST('../data/image_data', 
#                                  train = True,
#                                  download = True, transform = transform_proc)

#test_data = torchvision.datasets.MNIST('../data/image_data', 
#                                  train = False,
#                                  download = True, transform = transform_proc)


# Training and validation split
training_data, val_data = torch.utils.data.random_split(train_val_data, [50000, 10000], generator=torch.Generator().manual_seed(1))

# In[ ]:
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

# In[ ]:
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

        # self.conv_layers2 = nn.Sequential(
            
        #     nn.Conv2d(1, out_channels = 128, kernel_size = (4,4), stride = (2,2), padding = 1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),
        #     # nn.LeakyReLU(0.1),
        #     nn.Conv2d(128, out_channels = 256, kernel_size = (4,4), stride = (2,2), padding = 1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),
        #     # nn.LeakyReLU(0.1),
        #     nn.Conv2d(256, out_channels = 512, kernel_size = (4,4), stride = (2,2), padding = 1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(),
        #     # nn.LeakyReLU(0.1),
        #     nn.Conv2d(512, out_channels = 1024, kernel_size = (4,4), stride = (2,2), padding = 1),
        #     nn.BatchNorm2d(1024),
        #     nn.ReLU()
        #     # nn.LeakyReLU(0.1)
        # )

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

        # x2 = self.conv_layers2(x)
        # x2 = torch.flatten(x2, start_dim=1)
        
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
    

class Decoder(nn.Module):
    def __init__(self, latent_dims):
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


    def forward(self, z):


        x = self.fc_layer(z)
        x = x.view(x.size(0),1024,8,8)
        # print(x.shape)
        x = F.relu(self.batch_norm1(x))
        x = self.conv_layers(x)
    
        x = F.sigmoid(x)
        # print(x.shape)
        return x
    
class VAE(nn.Module):
    def __init__(self, latent_dims=128, constant_sigma = None):
        super(VAE, self).__init__()
        self.encoder = VaeEncoder(latent_dims, constant_sigma)
        self.decoder = Decoder(latent_dims)

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
    # scheduler.step(val_loss)
    # print(len(data.dataset))
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
    
    '''
    Runs the VAE Procedure
    '''
    train_loss = []
    val_loss = []
    
    autoencoder = VAE(latent_dims)
    kl_weight = config['kl_weight']

    
    PATH_latest = None
    val_epoch_loss = None
    flag = 0
    dataparallelornot = False
    
    # Check if the directory for saved model exists to determine how many more epochs to go
    for starting_epoch in reversed(range(epochs)):
        
        # save checkpoint
        PATH_new = PATH + 'vae_fmnist_' +str(starting_epoch) + '{}.pth'.format(addendum)
        
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
        model, train_epoch_loss, opt = vae_train(model, train_loader, lr, device, opt, kl_weight, recon_loss_func)
        model, val_epoch_loss = vae_validate(model, val_loader, lr, device, opt, kl_weight, recon_loss_func)
        
        train_loss.append(train_epoch_loss)
        val_loss.append(val_epoch_loss)
        print(f"Train Loss: {train_epoch_loss:.8f}")
        print(f"Val Loss: {val_epoch_loss:.8f}")
        
        # save checkpoint
        PATH_new = PATH + 'vae_fmnist_' +str(epoch) + '{}.pth'.format(addendum)
        
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



vae, train_loss, val_loss = vae_model_run({'kl_weight':kl_weight}, lr =lr, latent_dims=32, epochs = epochs, addendum= 'try_7_rerun') 

