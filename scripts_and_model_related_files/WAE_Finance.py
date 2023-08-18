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
lr = 0.0005 # The learning rate from paper is 0.001

mmd_weight = 2e-3

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


# ## RAE Architecture
class RegularisedEncoder(nn.Module):
    def __init__(self, latent_dims, SN =False):
        super(RegularisedEncoder, self).__init__()
        
        self.dense = nn.Sequential(
            
            nn.Linear(25, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dims)
            
            
            # nn.Linear(25, 32),
            # nn.SiLU(),
            # nn.Linear(32, 32),
            # nn.SiLU(),
            # nn.Linear(32, latent_dims)
        )
        

        
        self.sn_dense = nn.Sequential(
            # SNLinear(25, 32),
            # nn.SiLU(),
            # SNLinear(32, 32),
            # nn.SiLU(),
            # SNLinear(32, latent_dims)
            
            SNLinear(25, 16),
            nn.ReLU(),
            SNLinear(16, 16),
            nn.ReLU(),
            SNLinear(16, 32),
            nn.ReLU(),
            SNLinear(32, latent_dims)
            
        )
        
        self.SN = SN

    def forward(self, x):
        
        # Check if spectral norm is implemented
        if self.SN:
            # x = self.sn_conv_layers(x)
            x = torch.flatten(x, start_dim=1)
            z = self.sn_dense(x)
        else:
            # x = self.conv_layers(x)
            x = torch.flatten(x, start_dim=1)
            z = self.dense(x)
        
        
        return z
    

class Decoder(nn.Module):
    def __init__(self, latent_dims, SN = False):
        super(Decoder, self).__init__()
      
        
        self.dense = nn.Sequential(
            # nn.Linear(latent_dims, 32),
            # nn.SiLU(),
            # nn.Linear(32, 32),
            # nn.SiLU(),
            # nn.Linear(32, 25)
            
            nn.Linear(latent_dims, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 25)
        )
        
        self.sn_dense = nn.Sequential(
            # SNLinear(latent_dims, 32),
            # nn.SiLU(),
            # SNLinear(32, 32),
            # nn.SiLU(),
            # SNLinear(32, 25)
            
            SNLinear(latent_dims, 32),
            nn.ReLU(),
            SNLinear(32, 16),
            nn.ReLU(),
            SNLinear(16, 16),
            nn.ReLU(),
            SNLinear(16, 25)
        )
        
        
        
        self.SN = SN

    def forward(self, z):

        # Check if spectral norm is implemented
        if self.SN:
            # x = self.sn_fc_layer(z)
            # x = x.view(x.size(0),1024,8,8)
            # x = F.relu(self.batch_norm1(x))
            # x = self.sn_conv_layers(x)
            
            x = self.sn_dense(z)
            x = x.view(x.size(0),1,5,5)

        else:
            # x = self.fc_layer(z)
            # x = x.view(x.size(0),1024,8,8)
            # x = F.relu(self.batch_norm1(x))
            # x = self.conv_layers(x)
            
            x = self.dense(z)
            x = x.view(x.size(0),1,5,5)
        
        # x = F.sigmoid(x)
        # print(x.shape)
        return x
    

class DecoderPoint(nn.Module):
    def __init__(self, latent_dims, SN = False):
        super(DecoderPoint, self).__init__()
   
        
        self.dense = nn.Sequential(
            # nn.Linear(latent_dims + 2, 32),
            # nn.SiLU(),
            # nn.Linear(32, 32),
            # nn.SiLU(),
            # nn.Linear(32, 1)
            
            nn.Linear(latent_dims + 2, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
        self.sn_dense = nn.Sequential(
            # SNLinear(latent_dims + 2, 32),
            # nn.SiLU(),
            # SNLinear(32, 32),
            # nn.SiLU(),
            # SNLinear(32, 1)
            
            SNLinear(latent_dims + 2, 32),
            nn.ReLU(),
            SNLinear(32, 16),
            nn.ReLU(),
            SNLinear(16, 16),
            nn.ReLU(),
            SNLinear(16, 1)
        )
        
        self.SN = SN

    def forward(self, z, dayexp, delta):
        
        # To make dayexp and delta to be min-max scaled based on permissible values
        dayexp = (dayexp - 10)/(365-10)
        delta = (delta - 10)/(90-10)
        
        
        # Concat the dayexp (days to expiration) and delta to z
        
        z_expanded = torch.cat((z, dayexp, delta), axis = 1)
        
        # Check if spectral norm is implemented
        if self.SN:
            # x = self.sn_fc_layer(z)
            # x = x.view(x.size(0),1024,8,8)
            # x = F.relu(self.batch_norm1(x))
            # x = self.sn_conv_layers(x)
            
            x = self.sn_dense(z_expanded)

        else:
            # x = self.fc_layer(z)
            # x = x.view(x.size(0),1024,8,8)
            # x = F.relu(self.batch_norm1(x))
            # x = self.conv_layers(x)
            
            x = self.dense(z_expanded)
        
        
        # x = F.sigmoid(x)
        # print(x.shape)
        return x

    
    
class RegularisedAutoencoderGrid(nn.Module):
    def __init__(self, latent_dims=2, SN = False):
        '''RAE using Grid approach'''
        super(RegularisedAutoencoderGrid, self).__init__()
        self.encoder = RegularisedEncoder(latent_dims, SN)
        self.decoder = Decoder(latent_dims, SN)
            

    def forward(self, x):
        self.z = self.encoder(x)
        return self.decoder(self.z)
    
    
    
class RegularisedAutoencoderPoint(nn.Module):
    '''
    RAE using pointwise approach, returns a point in the volatility surface
    '''
    def __init__(self, latent_dims=2, SN = False):
        super(RegularisedAutoencoderPoint, self).__init__()
        self.encoder = RegularisedEncoder(latent_dims, SN)
        self.decoder = DecoderPoint(latent_dims, SN)
        
            
    def forward(self, x, dayexp, delta):
        self.z = self.encoder(x)
        return self.decoder(self.z, dayexp, delta)
    

def wae_train(autoencoder, data, mmd_weight, lr, device, opt, latent_dims, daysexp_arr =None, delta_arr=None,
              regularisation_loss_type = 'l2', approach = 'grid'):
    
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
        x = x_y[0]
        x = x.to(device) # GPU
        opt.zero_grad()
        
        if isinstance(autoencoder, nn.DataParallel):
            model_attr_accessor = autoencoder.module
            #print('data parallel')
        else:
            model_attr_accessor = autoencoder
            #print('not data parallel')
        
        
        z = model_attr_accessor.encoder(x.float())
        
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
        
        
        # print(dir(model_attr_accessor))
        
        loss = loss_compute(x_hat, x, z, latent_dims)
        
        loss.to(device)
        running_loss += loss.item()
        loss.backward()
        opt.step()
    # scheduler.step(val_loss)
    train_loss = running_loss/(i+1)
    return autoencoder, train_loss, opt #, scheduler


def wae_validate(autoencoder, data, mmd_weight, device, opt, latent_dims, daysexp_arr =None, delta_arr=None,
              regularisation_loss_type = 'l2', approach = 'grid', train_std = train_std):
    autoencoder.eval()
    running_loss = 0.0
    
    mae = 0.0
    mae_pre = 0.0
    
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
            x = x_y[0]
            x = x.to(device) # GPU
            # x_hat = autoencoder(x.float())
            
            if isinstance(autoencoder, nn.DataParallel):
                model_attr_accessor = autoencoder.module
            else:
                model_attr_accessor = autoencoder
            
            z = model_attr_accessor.encoder(x.float())
            
            if approach == 'point':
                # form the "image" with days to expiration as rows, and delta as columns
                
                vol_mat_outer_lst = []
                vol_mat_inner_lst = []
                
                for row_index, daysexp in enumerate(daysexp_arr):
                    for col_index, delta in enumerate(delta_arr):
                        
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
            
            # print(dir(model_attr_accessor))
            
            loss = loss_compute(x_hat, x, z, latent_dims)
            
            loss.to(device)
            running_loss += loss.item()
            
            # Also check MAE
            mae += torch.mean(torch.abs(x_hat - x)).cpu().detach().numpy()
            mae_pre += torch.mean(torch.abs(x_hat - x)*train_std).cpu().detach().numpy()

    val_loss = running_loss/(i+1)
    print('Val Transformed MAE:', mae/(i+1))
    print('Val MAE:', mae_pre/(i+1))
    
    
    return autoencoder, val_loss


def wae_model_run(config, train_loader = train_dataloader, val_loader = val_dataloader, epochs = 20, lr=0.005, daysexp_arr =None, delta_arr=None, approach = 'grid',
                  regularisation_loss_type = 'l2',
                  PATH = './', latent_dims = 128, appendum = ''):
    train_loss = []
    val_loss = []
    
    
    mmd_weight = config['mmd_weight']
    
    # Only implement the one without spectral normalisation
    if approach == 'grid':
        autoencoder = RegularisedAutoencoderGrid(latent_dims)
    else:
        autoencoder = RegularisedAutoencoderPoint(latent_dims)

    PATH_latest = None
    val_epoch_loss = None
    flag = 0
    dataparallelornot = False
    
    # Check if the directory for saved model exists to determine how many more epochs to go
    for starting_epoch in reversed(range(epochs)):
        
        # save checkpoint
        PATH_new = PATH + 'wae_vol_{}'.format(approach) + '_' +str(starting_epoch) + '{}.pth'.format(appendum)
        
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
        model, train_epoch_loss, opt = wae_train(model, train_loader, mmd_weight, lr, device, opt, latent_dims, daysexp_arr, delta_arr,
                                                 regularisation_loss_type, approach)

     
        model, val_epoch_loss = wae_validate(model, val_loader, mmd_weight, device, opt, latent_dims, daysexp_arr, delta_arr,
                                             regularisation_loss_type, approach)
       
        train_loss.append(train_epoch_loss)
        val_loss.append(val_epoch_loss)
        print(f"Train Loss: {train_epoch_loss:.8f}")
        print(f"Val Loss: {val_epoch_loss:.8f}")
        
        # save checkpoint
        PATH_new = PATH + 'wae_vol_{}'.format(approach) + '_' +str(epoch) + '{}.pth'.format(appendum)
        
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

# Point-based Approach
wae, train_loss, val_loss = wae_model_run({'mmd_weight':mmd_weight}, train_loader = train_dataloader, val_loader = val_dataloader, epochs = epochs, lr=lr, daysexp_arr =daysexp_arr, delta_arr=delta_arr, approach = 'point',
                                      regularisation_loss_type = 'l2', PATH = './', latent_dims = 2, appendum = 'latent2_batchsize32_config_1')

wae, train_loss, val_loss = wae_model_run({'mmd_weight':1e-3}, train_loader = train_dataloader, val_loader = val_dataloader, epochs = epochs, lr=lr, daysexp_arr =daysexp_arr, delta_arr=delta_arr, approach = 'point',
                                      regularisation_loss_type = 'l2', PATH = './', latent_dims = 3, appendum = 'latent3_batchsize32_config_1')

wae, train_loss, val_loss = wae_model_run({'mmd_weight': 8e-9}, train_loader = train_dataloader, val_loader = val_dataloader, epochs = epochs, lr=lr, daysexp_arr =daysexp_arr, delta_arr=delta_arr, approach = 'point',
                                      regularisation_loss_type = 'l2', PATH = './', latent_dims = 4, appendum = 'latent4_batchsize32_config_1')


# Grid Approach
wae, train_loss, val_loss = wae_model_run({'mmd_weight':mmd_weight}, train_loader = train_dataloader, val_loader = val_dataloader, epochs = epochs, lr=lr, daysexp_arr =None, delta_arr=None, approach = 'grid',
                                      regularisation_loss_type = 'l2', PATH = './', latent_dims = 2, appendum = 'latent2_batchsize32_config_1')

wae, train_loss, val_loss = wae_model_run({'mmd_weight':1e-3}, train_loader = train_dataloader, val_loader = val_dataloader, epochs = epochs, lr=lr, daysexp_arr =None, delta_arr=None, approach = 'grid',
                                      regularisation_loss_type = 'l2', PATH = './', latent_dims = 3, appendum = 'latent3_batchsize32_config_1')

wae, train_loss, val_loss = wae_model_run({'mmd_weight': 8e-9}, train_loader = train_dataloader, val_loader = val_dataloader, epochs = epochs, lr=lr, daysexp_arr =None, delta_arr=None, approach = 'grid',
                                      regularisation_loss_type = 'l2', PATH = './', latent_dims = 4, appendum = 'latent4_batchsize32_config_1')


