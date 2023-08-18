# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 18:22:54 2023

@author: angsi
This is for ex-post density estimation and generation of new samples
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import random
import datetime


from torch.optim import lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
from random import sample
from spectral_norm_layers import *

from model_repo import *

import torchvision
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms

from sklearn import mixture




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


def initialise_model(model_name, latent_dims, SN, approach = 'point'):
    '''
    Initialises the autoencoder models based on the model_name, it should be having
    model type (i.e. ae, rae, vae ). This is implied that it is for the implied volatility data set
    '''
    
    initial_model = None 
    
    # To select the appropriate model class to initiatise
    if approach == 'point':
        
        if (model_name == 'rae') or (model_name == 'ae') or (model_name == 'wae'):
            initial_model = FinRegularisedAutoencoderPoint(latent_dims, SN)
    
        elif model_name == 'vae':
            initial_model = FinVAEPoint(latent_dims)
        
        else:
            print('Not Implemented')
    
    else:
        if (model_name == 'rae') or (model_name == 'ae') or (model_name == 'wae'):
            initial_model = FinRegularisedAutoencoderGrid(latent_dims, SN)
    
        elif model_name == 'vae':
            initial_model = FinVAEGrid(latent_dims)
        
        else:
            print('Not Implemented')
    
    return initial_model


def load_model(model_name, saved_model_path, latent_dims, SN, approach):
    
    
    model = initialise_model(model_name, latent_dims, SN, approach)
    checkpoint = torch.load(saved_model_path)
    
    print('loading model success')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print('loaded model')
    
    model.eval()
    
    return model


def get_ztrain(train_tensor, model, vae, device):
    '''
    For a train_tensor and autoencoder model, it will the z from training set in numpy array
    Suitable only for smaller datasets, for larger datasets the alternate version in compute_fid.py should be used
    '''
    train_tensor = train_tensor.float().to(device)
    if vae == True:
        z, _, __ = model.encoder(train_tensor)
    else:
        z = model.encoder(train_tensor)
        
    z_trained = z.cpu().detach().numpy()
    
    return z_trained


def rae_rndm_smpl(z_trained, model, sample_size, device, approach = 'grid', daysexp_arr=None, delta_arr=None):
    '''
    Estimates covariance, mean and generates random sample from multivariate normal
    '''
    est_cov = np.cov(z_trained.T)
    est_mean = np.mean(z_trained.T, axis = 1)
    
    # Generate sampled latent variables from fitted multivariate normal distribution
    latent = torch.from_numpy(np.random.multivariate_normal(est_mean, cov=est_cov, size=sample_size)).to(torch.float32).to(device)
    
    # Generate the samples based on normal distribution in accordance to the approach
    rndm_smpl = decode_z(latent, model.decoder, approach, sample_size, device, daysexp_arr, delta_arr)
    
    rndm_smpl = rndm_smpl.cpu().detach().numpy()
    
    return rndm_smpl

def gmm_rndmn_train(z_trained, n_dist, device):
    '''
    To train the GMM model
    '''
    gmmmodel = mixture.GaussianMixture(n_components=n_dist, covariance_type='full', max_iter=500,
                                                 verbose=2, tol=1e-3)

    gmmmodel.fit(z_trained)
    
    return gmmmodel

def gmm_rndm_smpl(model, sample_size, gmmmodel, device, approach = 'grid', daysexp_arr=None, delta_arr=None):

    # Randomly shuffle indices
    scrmb_idx = np.array(range(sample_size))
    np.random.shuffle(scrmb_idx)
    
    gmm_sample = gmmmodel.sample(sample_size)[0][scrmb_idx, :]
    
    # Send to decoder
    latent = torch.from_numpy(gmm_sample).to(torch.float32).to(device)
    
    # Generates the samples based on fitted GMM-10 of latent variables.
    gmm_sample_out = decode_z(latent, model.decoder, approach, sample_size, device, daysexp_arr, delta_arr)
    
    # Convert to numpy array
    gmm_sample_out = gmm_sample_out.cpu().detach().numpy()
    
    return gmm_sample_out

def decode_z(z, decoder, approach, test_size, device, daysexp_arr=None, delta_arr=None):
    '''
    Decodes z based on the decoder provided, approach and other pertinent parameters
    '''
    if approach == 'point':
        # form the "image" with days to expiration as rows, and delta as columns
        try:
            # initialise the matrix
            volmatrix = torch.zeros((test_size,1, len(daysexp_arr), len(delta_arr)))
            
            for row_index, daysexp in enumerate(daysexp_arr):
                for col_index, delta in enumerate(delta_arr):
                    
                    dayexp_tensor = torch.full((test_size,1), daysexp).to(device)
                    delta_tensor = torch.full((test_size,1), delta).to(device)
    
                    x_pt = decoder(z, dayexp_tensor, delta_tensor)
                    
                    volmatrix[:,:,row_index, col_index] = x_pt
                    
            x_hat = volmatrix.to(device)
        except:
            print('Pointwise method failed, please check inputs.')
            return None
    else:
        # For grid implementation
        x_hat = decoder(z)
        
    return x_hat


def generation_mae(model_path, model_name, SN, train_tensor, test_tensor, latent_dims,
                              daysexp_arr, delta_arr,
                              device_choose, approach, test_tensor_original = None, train_mean = None, train_std =None):
    
    print('Running {} model with latent dimensions={}'.format(model_name, latent_dims))
    print('Approach selected: {}'.format(approach))
    # Load Model
    model = load_model(model_name, model_path, latent_dims, SN, approach)
    
    # Try to use data parallel if available
    if device_choose =='cuda':
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            print('No. of GPUs: {}'.format(torch.cuda.device_count()))
            model.to(device)
        
        else:
            model = model.to(device)
    else:
        device = torch.device('cpu') 
        model = model.to(device)
    
    train_tensor = train_tensor.to(device)
    test_tensor = test_tensor.to(device)
    
    test_size = len(test_tensor)
    
    if isinstance(model, nn.DataParallel):
        model_attr_accessor = model.module
    else:
        model_attr_accessor = model
        
    if model_name == 'vae':
        vae = True
        z_test, mu, log_sigma = model_attr_accessor.encoder(test_tensor.float())
    else:
        vae = False
        z_test = model_attr_accessor.encoder(test_tensor.float())
        
    ##################################################
    ### Ex-post Density Estimation
    #################################################
    # Get z train
    z_train = get_ztrain(train_tensor, model_attr_accessor, vae, device)
    
    ######### Generate random samples based on Gaussian Distributions or Priors #########
    # For VAE and WAE
    if model_name.lower() == 'vae' or model_name.lower() == 'wae':
        z_rndmN = np.random.normal(loc=0.0, scale=1.0, size = z_test.cpu().detach().numpy().shape)
        z_rndmN = torch.from_numpy(z_rndmN).to(torch.float32).to(device)
        
        # Fetch the x generated
        x_rndmN = decode_z(z_rndmN, model_attr_accessor.decoder, approach, test_size, device, daysexp_arr, delta_arr).cpu().detach().numpy()

    else:
        # Get random samples from Gaussian Distribution for non-VAE and non-WAE
        x_rndmN = rae_rndm_smpl(z_train, model_attr_accessor, test_size, device, approach, daysexp_arr, delta_arr) 
    
    ######### Generate random sample based on GMM-10 #########################
    # Train GMM Model on z_train
    gmm_model = gmm_rndmn_train(z_train, 10, device)
    
    # Generate random samples from fitted GMM of z
    x_gmm = gmm_rndm_smpl(model_attr_accessor, test_size, gmm_model, device,approach,daysexp_arr,delta_arr)
    
    ################ Compute the paired t-test statistics for results #######################
    # First create a matrix with test_size number of rows and m x p number of columns where m is the len(daysexp_arr) and p is len(delta_arr)
    # and convert back inverse of normalisation
    x_rndmN_reshaped = x_rndmN.reshape((test_size,len(daysexp_arr) * len(delta_arr)))
    x_gmm_reshaped = x_gmm.reshape((test_size,len(daysexp_arr) * len(delta_arr)))
    test_tensor_reshaped = test_tensor.cpu().detach().numpy().reshape((test_size,len(daysexp_arr) * len(delta_arr)))
    

    
    if test_tensor_original is None or train_mean is None or train_std is None:


        # COMPUTE MAE of the mean of each point on volatility surface (bps)
        mae_N = np.mean(np.abs(np.mean(test_tensor_reshaped, axis = 0) - np.mean(x_rndmN_reshaped, axis = 0))) * 10000
        mae_GMM = np.mean(np.abs(np.mean(test_tensor_reshaped, axis = 0) - np.mean(x_gmm_reshaped, axis = 0))) * 10000
        
        
    else:
        # Reshape original test tensor
        test_tensor_original_reshaped = test_tensor_original.cpu().detach().numpy().reshape((test_size,len(daysexp_arr) * len(delta_arr)))
        
        # Apply inverse of normalisation
        x_rndmN_reshaped = x_rndmN_reshaped * train_std.item() + train_mean.item()
        x_gmm_reshaped = x_gmm_reshaped * train_std.item() + train_mean.item()
        
        # COMPUTE MAE of the mean of each point on volatility surface (bps)
        mae_N = np.mean(np.abs(np.mean(test_tensor_original_reshaped, axis = 0) - np.mean(x_rndmN_reshaped, axis = 0))) * 10000
        mae_GMM = np.mean(np.abs(np.mean(test_tensor_original_reshaped, axis = 0) - np.mean(x_gmm_reshaped, axis = 0))) * 10000
        
    
    # Store as pandas df to output
    df = pd.DataFrame([[mae_N, mae_GMM]])
    df.columns = ['MAE of Random Samples from Normal', 'MAE of Random Samples from GMM']
    

    return df

    

if __name__ == '__main__':
    
    ############################################### DATA LOADING AND PREPROCESSING #######################################################
    ## Loading in data
    spx_df = pd.read_csv('../data/finance_data/spx_implied_vol_interpolated.csv')
    
    batch_size = 32
    
    # Convert date to datetime
    spx_df['date'] = spx_df['date'].apply(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d'))
    
    # Get unique dates
    date_series_all = spx_df['date'].drop_duplicates().reset_index(drop = True)
    
    # Define date arrays for training, validation and test
    train_date_series = date_series_all[(date_series_all >= datetime.datetime(2006,1,1)) & (date_series_all < datetime.datetime(2020,1,1))]
    val_date_series = date_series_all[(date_series_all >= datetime.datetime(2020,1,1)) & (date_series_all < datetime.datetime(2021,7,1))]
    test_date_series = date_series_all[(date_series_all >= datetime.datetime(2021,7,1))]
    
    # The pivot will sort columns and rows by ascending order, so just need to keep track of sorted unique arrays of columns and rows
    delta_arr= np.unique(spx_df['delta'])
    daysexp_arr= np.unique(spx_df['days'])
    
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
    
    # Keep original copy for MAE evaluation
    train_tensor_original = torch.from_numpy(struct_data(spx_df, train_date_series))
    val_tensor_original = torch.from_numpy(struct_data(spx_df, val_date_series))
    test_tensor_original = torch.from_numpy(struct_data(spx_df, test_date_series))
    
    
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    
    train_dataloader = DataLoader(TensorDataset(train_tensor), batch_size = batch_size, shuffle = True)
    val_dataloader = DataLoader(TensorDataset(val_tensor), batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(TensorDataset(test_tensor), batch_size=batch_size, shuffle=False)
    
    ################################################################################################################
    # Evaluation of the Ex-post density estimation generated samples
    #################################################################################################################
    
    # For ease of exposition, we will also use latent dimension = 4 models only for both approaches
    
    # For the grid-based approach
    vae_grid = generation_mae('vae_vol_grid_149lat4_batchsize32_try_2.pth', 'vae', False, train_tensor,test_tensor, 4,daysexp_arr,delta_arr,'cuda','grid',test_tensor_original,train_mean, train_std)
    rael2_grid = generation_mae('rae_vol_gridl2_149lat4_batchsize32_try_1.pth', 'rae', False, train_tensor,test_tensor, 4,daysexp_arr,delta_arr,'cuda','grid',test_tensor_original,train_mean, train_std)
    raesn_grid = generation_mae('rae_vol_gridspec_norm_149lat4_batchsize32_try_1.pth', 'rae', True, train_tensor,test_tensor, 4,daysexp_arr,delta_arr,'cuda','grid',test_tensor_original,train_mean, train_std)
    raegp_grid = generation_mae('rae_vol_gridgrad_pen_149lat4_batchsize32_try_1.pth', 'rae', False, train_tensor,test_tensor, 4,daysexp_arr,delta_arr,'cuda','grid',test_tensor_original,train_mean, train_std)
    wae_grid = generation_mae('wae_vol_grid_149latent4_batchsize32_config_1.pth', 'wae', False, train_tensor,test_tensor, 4,daysexp_arr,delta_arr,'cuda','grid',test_tensor_original,train_mean, train_std)
    ae_grid = generation_mae('rae_vol_gridl2_149lat4_batchsize32_ae.pth', 'ae', False, train_tensor,test_tensor, 4,daysexp_arr,delta_arr,'cuda','grid',test_tensor_original,train_mean, train_std)
    
    grid_results_df = pd.concat([vae_grid, rael2_grid, raesn_grid, raegp_grid, wae_grid, ae_grid])
    grid_results_df.insert(0, 'Model', ['VAE','RAE-L2', 'RAE-SN','RAE-GP','WAE','AE'])
    
    # For the pointwise approach
    vae_point = generation_mae('vae_vol_point_149lat4_batchsize32_try_2.pth', 'vae', False, train_tensor,test_tensor, 4,daysexp_arr,delta_arr,'cuda','point',test_tensor_original,train_mean, train_std)
    rael2_point = generation_mae('rae_vol_pointl2_149lat4_batchsize32_try_1.pth', 'rae', False, train_tensor,test_tensor, 4,daysexp_arr,delta_arr,'cuda','point',test_tensor_original,train_mean, train_std)
    raesn_point = generation_mae('rae_vol_pointspec_norm_149lat4_batchsize32_try_1.pth', 'rae', True, train_tensor,test_tensor, 4,daysexp_arr,delta_arr,'cuda','point',test_tensor_original,train_mean, train_std)
    raegp_point = generation_mae('rae_vol_pointgrad_pen_149lat4_batchsize32_try_1.pth', 'rae', False, train_tensor,test_tensor, 4,daysexp_arr,delta_arr,'cuda','point',test_tensor_original,train_mean, train_std)
    wae_point = generation_mae('wae_vol_point_149latent4_batchsize32_config_1.pth', 'wae', False, train_tensor,test_tensor, 4,daysexp_arr,delta_arr,'cuda','point',test_tensor_original,train_mean, train_std)
    ae_point = generation_mae('rae_vol_pointl2_149lat4_batchsize32_ae.pth', 'ae', False, train_tensor,test_tensor, 4,daysexp_arr,delta_arr,'cuda','point',test_tensor_original,train_mean, train_std)
    
    point_results_df = pd.concat([vae_point, rael2_point, raesn_point, raegp_point, wae_point, ae_point])
    point_results_df.insert(0, 'Model', ['VAE','RAE-L2', 'RAE-SN','RAE-GP','WAE','AE'])

    # Export results
    grid_results_df.to_csv('finance_rndm_smpl_test_grid.csv', index = False)
    point_results_df.to_csv('finance_rndm_smpl_test_point.csv', index = False)
    
    
    