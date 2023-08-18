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
from torch.utils.data import DataLoader, TensorDataset
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

import json

from losses import *

import datetime


# Load in data
spx_df = pd.read_csv('../data/finance_data/spx_implied_vol_interpolated.csv')

def struct_data(df, date_array):
    '''
    For the dates in date_array, pick them out from df
    and put them in a 4D array corresponding to the following dimensions:
    (date, 1, days to expiration, delta)
    
    This is so that we could reuse some of the loss setup from images
    '''
    
    data_lst =[]
    for date in date_array:
        subset_df = df[df['date'] == date]
        
        data_lst.append(np.array([subset_df.pivot(index = 'days', columns = ['date','delta'], values = 'impl_volatility').values]))
        
    return np.array(data_lst)


## Setting up
epochs = 150
batch_size = 32 # paper uses 100, for try 5 i did 500..
lr = 0.0005 # 0.0001 & 3 layers so far best

kl_weight = 0.01 # 0.01 quite good

# Convert date to datetime
spx_df['date'] = spx_df['date'].apply(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d'))

# Get unique dates
date_series_all = spx_df['date'].drop_duplicates().reset_index(drop = True)

# The pivot will sort columns and rows by ascending order, so just need to keep track of sorted unique arrays of columns and rows
delta_arr= np.unique(spx_df['delta'])
daysexp_arr= np.unique(spx_df['days'])

# Define date arrays for training, validation and test
train_date_series = date_series_all[(date_series_all >= datetime.datetime(2006,1,1)) & (date_series_all < datetime.datetime(2020,1,1))]
val_date_series = date_series_all[(date_series_all >= datetime.datetime(2020,1,1)) & (date_series_all < datetime.datetime(2021,7,1))]
test_date_series = date_series_all[(date_series_all >= datetime.datetime(2021,7,1))]

# Structure the data
train_tensor = torch.from_numpy(struct_data(spx_df, train_date_series))
val_tensor = torch.from_numpy(struct_data(spx_df, val_date_series))
test_tensor = torch.from_numpy(struct_data(spx_df, test_date_series))

# Normalise the tensors by subtracting mean from train_tensor and divide by std from train_tensor
train_mean = train_tensor.mean()
train_std = train_tensor.std()

train_tensor = (train_tensor - train_mean)/train_std
val_tensor = (val_tensor - train_mean)/train_std
test_tensor = (test_tensor - train_mean)/train_std

# In[ ]:

import random
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

train_dataloader = DataLoader(TensorDataset(train_tensor), batch_size = batch_size, shuffle = True)
val_dataloader = DataLoader(TensorDataset(val_tensor), batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(TensorDataset(test_tensor), batch_size=batch_size, shuffle=False)


### VAE
class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)
    
class VaeEncoder(nn.Module):
    def __init__(self, latent_dims, constant_sigma = None):
        super(VaeEncoder, self).__init__()
        
        self.dense = nn.Sequential(
            # nn.Linear(25, 8),
            # nn.ReLU(),
            # nn.Linear(8, 8),
            # nn.ReLU(),
            # nn.Linear(8, 16),
            # nn.ReLU(),
            # nn.Linear(16, 32),
            # nn.ReLU()
            
            nn.Linear(25, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU()
            

        )
        
        self.dense_for_mu = nn.Sequential(
            nn.Linear(32, latent_dims)
            )
        self.dense_for_sigma = nn.Sequential( 
            
            nn.Linear(32, latent_dims),
            nn.Tanh()
            )
        
        # Initialise variables
        self.constant_sigma = constant_sigma
        

    def forward(self, x):
        # x = self.conv_layers(x)
        x = torch.flatten(x, start_dim=1)
        x1 = self.dense(x)
        mu = self.dense_for_mu(x1)


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
    
class VaeDecoder(nn.Module):
    def __init__(self, latent_dims):
        super(VaeDecoder, self).__init__()

        
        self.dense = nn.Sequential(
            # nn.Linear(latent_dims, 32),
            # nn.ReLU(),
            # nn.Linear(32, 16),
            # nn.ReLU(),
            # nn.Linear(16, 8),
            # nn.ReLU(),
            # nn.Linear(8, 8),
            # nn.ReLU(),
            # nn.Linear(8, 25)
            
            # CUrrent
            nn.Linear(latent_dims, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 25)
            
        )
        


    def forward(self, z):

        x = self.dense(z)
        
        
        x = x.view(x.size(0),1,5,5)
        
        # x = F.sigmoid(x)
        # print(x.shape)
        return x
    

class VaeDecoderPoint(nn.Module):
    def __init__(self, latent_dims):
        super(VaeDecoderPoint, self).__init__()

        self.dense = nn.Sequential(
            # nn.Linear(latent_dims + 2, 32),
            # nn.ReLU(),
            # nn.Linear(32, 16),
            # nn.ReLU(),
            # nn.Linear(16, 8),
            # nn.ReLU(),
            # nn.Linear(8, 8),
            # nn.ReLU(),
            # nn.Linear(8, 1)
            
            # Current
            nn.Linear(latent_dims + 2, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
            
        )
        


    def forward(self, z, dayexp, delta):
        
        # To make dayexp and delta to be min-max scaled based on permissible values
        dayexp = (dayexp - 10)/(365-10)
        delta = (delta - 10)/(90-10)
        
        # Concat the dayexp (days to expiration) and delta to z
        z_expanded = torch.cat((z, dayexp, delta), axis = 1)
        

        x = self.dense(z_expanded)
     
        return x


class VAEGrid(nn.Module):
    def __init__(self, latent_dims=2, constant_sigma = None):
        super(VAEGrid, self).__init__()
        self.encoder = VaeEncoder(latent_dims, constant_sigma)
        self.decoder = VaeDecoder(latent_dims)

    def forward(self, x):
        self.z, mu, log_sigma = self.encoder(x)
        return self.decoder(self.z), mu, log_sigma


class VAEPoint(nn.Module):
    def __init__(self, latent_dims=2, constant_sigma = None):
        super(VAEPoint, self).__init__()
        self.encoder = VaeEncoder(latent_dims, constant_sigma)
        self.decoder = VaeDecoderPoint(latent_dims)

    def forward(self, x, dayexp, delta):
        self.z, mu, log_sigma = self.encoder(x)
        return self.decoder(self.z, dayexp, delta), mu, log_sigma
    

    
    
def vae_train(autoencoder, data, lr, device, opt, kl_weight = 1, recon_loss_func = None, delta_arr = None,
              daysexp_arr = None, approach = 'grid'):
    
    '''
    approach refers to either 'point' - point-based or 'grid' - grid-based approach
    '''

    running_loss = 0.0
    
    if isinstance(autoencoder, nn.DataParallel):
        model_attr_accessor = autoencoder.module
        #print('data parallel')
    else:
        model_attr_accessor = autoencoder
        #print('not data parallel')
    
    
    # Set the loss computation method
    loss_compute = VAETotalLoss(kl_weight, recon_loss_func)
    
    
    for i, x_y in enumerate(data):
        if i == 10:
            print(i)
        x = x_y[0]
        x = x.to(device) # GPU
        
        opt.zero_grad()
        # x_hat = autoencoder(x.float())
        
        if isinstance(autoencoder, nn.DataParallel):
            model_attr_accessor = autoencoder.module
            #print('data parallel')
        else:
            model_attr_accessor = autoencoder
            #print('not data parallel')
                
                
        z, mu, log_sigma = model_attr_accessor.encoder(x.float())
        
        if approach == 'point':
            # form the "image" with days to expiration as rows, and delta as columns
            
            vol_mat_outer_lst = []
            vol_mat_inner_lst = []
            
            for row_index, daysexp in enumerate(daysexp_arr):
                for col_index, delta in enumerate(delta_arr):
                    
                    ###### NEED TO MAKE SURE ASSIGNED PROPERLY
                    dayexp_tensor = torch.full((z.size(0),1), daysexp).to(device)
                    delta_tensor = torch.full((z.size(0),1), delta).to(device)
                    x_pt = model_attr_accessor.decoder(z.float(), dayexp_tensor, delta_tensor)
                    
                    # print(x_pt.shape)
                    
                    vol_mat_inner_lst.append(x_pt)
                    
                vol_mat_row = torch.cat(vol_mat_inner_lst, axis = 1)
                vol_mat_row = torch.unsqueeze(vol_mat_row, 1)
                # print(vol_mat_row.shape)
                
                vol_mat_outer_lst.append(vol_mat_row)
                
                # Reset inner list
                vol_mat_inner_lst = []
            
            volmatrix = torch.cat(vol_mat_outer_lst, axis = 1)
            volmatrix = torch.unsqueeze(volmatrix, 1)
            
            # print(volmatrix.shape, volmatrix.requires_grad)
            
            x_hat = volmatrix.to(device)
        else:
            # For grid implementation
            x_hat = model_attr_accessor.decoder(z.float())

        loss = loss_compute(x_hat, x, mu, log_sigma)
        
        loss.to(device)
        running_loss += loss.item()
        loss.backward()
        opt.step()
    # scheduler.step(val_loss)
    train_loss = running_loss/(i+1)
    return autoencoder, train_loss, opt #, scheduler


def vae_validate(autoencoder, data, device, kl_weight = 1, recon_loss_func = None, delta_arr = None,
                 daysexp_arr = None, approach = 'grid'):
    
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
    
    mae = 0
    mae_pre = 0
    
    with torch.no_grad():
        for i, x_y in enumerate(data):
            x = x_y[0]
            x = x.to(device) # GPU
            
            z, mu, log_sigma = model_attr_accessor.encoder(x.float())
            
            if isinstance(autoencoder, nn.DataParallel):
                model_attr_accessor = autoencoder.module
            else:
                model_attr_accessor = autoencoder
            
            if approach == 'point':
                # form the "image" with days to expiration as rows, and delta as columns
                
                # initialise the matrix
                volmatrix = torch.zeros((x.size(0),1, len(daysexp_arr), len(delta_arr)))
                
                for row_index, daysexp in enumerate(daysexp_arr):
                    for col_index, delta in enumerate(delta_arr):
                        
                        dayexp_tensor = torch.full((z.size(0),1), daysexp).to(device)
                        delta_tensor = torch.full((z.size(0),1), delta).to(device)
 
                        x_pt = model_attr_accessor.decoder(z.float(), dayexp_tensor, delta_tensor)
                        
                        volmatrix[:,:,row_index, col_index] = x_pt
                        
                x_hat = volmatrix.to(device)
            else:
                # For grid implementation
                x_hat = model_attr_accessor.decoder(z.float())
            
            
            loss = loss_compute(x_hat, x, mu, log_sigma)
            
            # Also check MAE
            mae += torch.mean(torch.abs(x_hat - x)).cpu().detach().numpy()
            mae_pre += torch.mean(torch.abs(x_hat - x)*train_std).cpu().detach().numpy()
            
            # print('MAE', torch.mean(torch.abs(x_hat - x)).cpu().detach().numpy())
            
            loss.to(device)
            running_loss += loss.item()

    val_loss = running_loss/(i+1)
    
    print('Val Transformed MAE:', mae/(i+1))
    print('Val MAE:', mae_pre/(i+1))
    
    return autoencoder, val_loss

def vae_finance_model_run(config, train_loader = train_dataloader, val_loader = val_dataloader, epochs = 20, lr=0.005, recon_loss_func = None,
              PATH = './', latent_dims = 128, appendum = '', delta_arr = None, daysexp_arr = None, approach='grid'):
    train_loss = []
    val_loss = []
    
    
    kl_weight = config['kl_weight']
    
    if approach == 'grid':
        autoencoder = VAEGrid(latent_dims)
    else:
        autoencoder = VAEPoint(latent_dims)
    
    PATH_latest = None
    val_epoch_loss = None
    flag = 0
    dataparallelornot = False
    
    # Check if the directory for saved model exists to determine how many more epochs to go
    for starting_epoch in reversed(range(epochs)):
        
        # save checkpoint
        PATH_new = PATH + 'vae_vol_{}'.format(approach) + '_' +str(starting_epoch) + '{}.pth'.format(appendum)
        
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
        model, train_epoch_loss, opt = vae_train(model, train_loader, lr, device, opt, kl_weight, recon_loss_func, delta_arr = delta_arr,
                                                 daysexp_arr = daysexp_arr, approach = approach) 
        model, val_epoch_loss = vae_validate(model, val_loader, device, kl_weight, recon_loss_func, delta_arr=delta_arr, daysexp_arr = daysexp_arr, approach=approach)
        train_loss.append(train_epoch_loss)
        val_loss.append(val_epoch_loss)
        print(f"Train Loss: {train_epoch_loss:.8f}")
        print(f"Val Loss: {val_epoch_loss:.8f}")
        
        # save checkpoint
        PATH_new = PATH + 'vae_vol_{}'.format(approach) + '_' +str(epoch) + '{}.pth'.format(appendum)
        
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



##### Point based approach
vae, train_loss, val_loss= vae_finance_model_run({'kl_weight': kl_weight}, train_dataloader, val_dataloader, epochs = epochs, lr = lr, recon_loss_func = None,
                                        PATH = './', latent_dims = 2, appendum='lat2_batchsize32_try_2', delta_arr = delta_arr, daysexp_arr = daysexp_arr, approach='point')

vae, train_loss, val_loss= vae_finance_model_run({'kl_weight': kl_weight}, train_dataloader, val_dataloader, epochs = epochs, lr = lr, recon_loss_func = None,
                                        PATH = './', latent_dims = 3, appendum='lat3_batchsize32_try_2', delta_arr = delta_arr, daysexp_arr = daysexp_arr, approach='point')

vae, train_loss, val_loss= vae_finance_model_run({'kl_weight': kl_weight}, train_dataloader, val_dataloader, epochs = epochs, lr = lr, recon_loss_func = None,
                                        PATH = './', latent_dims = 4, appendum='lat4_batchsize32_try_2', delta_arr = delta_arr, daysexp_arr = daysexp_arr, approach='point')


##### Grid based approach
vae, train_loss, val_loss= vae_finance_model_run({'kl_weight': kl_weight}, train_dataloader, val_dataloader, epochs = epochs, lr = lr, recon_loss_func = None,
                                        PATH = './', latent_dims = 2, appendum='lat2_batchsize32_try_2', delta_arr = None, daysexp_arr = None, approach='grid')

vae, train_loss, val_loss= vae_finance_model_run({'kl_weight': kl_weight}, train_dataloader, val_dataloader, epochs = epochs, lr = lr, recon_loss_func = None,
                                        PATH = './', latent_dims = 3, appendum='lat3_batchsize32_try_2', delta_arr = None, daysexp_arr = None, approach='grid')

vae, train_loss, val_loss= vae_finance_model_run({'kl_weight': kl_weight}, train_dataloader, val_dataloader, epochs = epochs, lr = lr, recon_loss_func = None,
                                        PATH = './', latent_dims = 4, appendum='lat4_batchsize32_try_2', delta_arr = None, daysexp_arr = None, approach='grid')


