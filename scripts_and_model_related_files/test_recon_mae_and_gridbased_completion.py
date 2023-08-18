# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 18:22:54 2023

@author: angsi
This is for test reconstruction for both grid and point based approaches
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
    
    # TO Design a way to initialise the models accordingly
    # Maybe I should load all the scripts (import) with model functions then have a way to serialise them
    # like a dictionary or if else; can call to initialise them 
    
    model = initialise_model(model_name, latent_dims, SN, approach)
    checkpoint = torch.load(saved_model_path)
    
    print('loading model success')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print('loaded model')
    
    model.eval()
    
    return model

def run_test_recon_eval(model_path, model_name, SN, test_tensor, latent_dims,
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
    
    
    test_tensor = test_tensor.to(device)
    
    test_size = len(test_tensor)
    
        
    ####################################################################
    ## TO GET PREDICTION        
    ####################################################################
    
    if isinstance(model, nn.DataParallel):
        model_attr_accessor = model.module
    else:
        model_attr_accessor = model
    
    if model_name == 'vae':
        z, mu, log_sigma = model_attr_accessor.encoder(test_tensor.float())
    else:
        z = model_attr_accessor.encoder(test_tensor.float())

    if approach == 'point':
        # form the "image" with days to expiration as rows, and delta as columns
        
        # initialise the matrix
        volmatrix = torch.zeros((test_tensor.size(0),1, len(daysexp_arr), len(delta_arr)))
        
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
        
    #### Compute MAE
    if test_tensor_original is None or train_mean is None or train_std is None:
        mae = torch.mean(torch.abs(x_hat - test_tensor))
    else:
        x_hat_original = x_hat*train_std + train_mean
        x_hat_original = x_hat_original.to(device)
        test_tensor_original = test_tensor_original.to(device)
        mae = torch.mean(torch.abs(x_hat_original - test_tensor_original))
        # print(x_hat_original[0])
        # print(test_tensor_original[0])
    return mae.cpu().detach().numpy()


def grid_based_complete_vol_surface(model_path, model_type, SN, test_tensor, ntrials, sample_size_grid, latent_dims, device_choose, normalizing_std):
    '''
    Using a linear interpolation followed by grid-based autoencoders to complete the volatility surface with partially sampled datapoints
    '''
    
    print('Running {} model with latent dimensions={}'.format(model_type, latent_dims))
    # Load Model
    model = load_model(model_type, model_path, latent_dims, SN, approach='grid')
    
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
    
    test_tensor = test_tensor.to(device)
    
    test_size = len(test_tensor)
    
    # Get the configuration index list
    config_ind_lst = [(a, b) for a in range(len(daysexp_arr)) for b in range(len(delta_arr))]
    
    # initialise mae
    mae = torch.zeros(1, device = device)
    mae_trials = []
    mae_lst = []
    
    
    # Each sample size selection, run ntrials and evaluate mean mae
    for sample_size in sample_size_grid:
        
        print('Running Partially Known Data Points:{}'.format(sample_size))
        mae_trials = []
        
        for ntrial in range(ntrials):
            print('Running Trial ', ntrial)
            
            # Sample without replacement
            config_ind_sample = sample(config_ind_lst, sample_size)
            
            # initialise mae
            mae = torch.zeros(1, device = device)
            
            for test_set in test_tensor:
                            
                ### Interpolation procedure
                x = torch.zeros((1,1,5,5), requires_grad= False, device=device)
                
                for days_ind, delta_ind in config_ind_sample:
                    target = test_set[0, days_ind, delta_ind].to(device)
                    x[:,:,days_ind, delta_ind] = target
                
                # Replace 0 with np.nan
                x_df = pd.DataFrame(x.squeeze().cpu().detach().numpy()).replace(0, np.nan)
                
                #Interpolate
                x_df = x_df.interpolate(limit_direction = 'both', axis = 0).interpolate(limit_direction = 'both', axis = 1)
                x_df = x_df.values
                
                # Assign to x
                x = torch.from_numpy(x_df).unsqueeze(0).unsqueeze(0)
                x.requires_grad = False
                x = x.to(device)
                
                ###### Apply Autoencoder on x
                if model_type == 'vae':
                    x_hat, mu, log_sigma = model(x.float())
                else:
                    x_hat = model(x.float())
                
                ###### Evaluate MAE and first add to MAE (subsequently will be divided by the size of test sample)
                if normalizing_std is not None:
                    mae += torch.mean(torch.abs(x_hat - test_set) * normalizing_std)
                else:
                    mae += torch.mean(torch.abs(x_hat - test_set))
                    
                
            ## Compute and store mae
            mae_trials.append(mae.cpu().detach().numpy()[0]/test_size)
            
            # Append MAE for each trial
            mae_trials.append(mae.cpu().detach().numpy()[0]/test_size)
            print('MAE for trial ', ntrial,'is ' , mae.cpu().detach().numpy()[0]/test_size)
            
        
        print('Mean MAE across trials for {} points is {}'.format(sample_size, np.mean(mae_trials)))
        
        # Take average of all mae_trials and append to mae_lst
        mae_lst.append(np.mean(mae_trials))
    
    # This is the output df where MAEs are kept
    df = pd.DataFrame(mae_lst).T
    
    # Assign sample size grid to columns
    df.columns = sample_size_grid
    
    # Assign latent_dims to index
    df.index = [latent_dims]
    
    print('Volatility surface completed for {}\n\n'.format(model_path))
    
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
    # Grid Based Completion of Volatility Surface
    #################################################################################################################
    ntrials = 3
    sample_size_grid = [5,10,15,20,25]
    max_steps = 50
    
    random.seed(888)
    
    ##### Complete Volatility Surface using grid-based approach 
    # VAE
    vae_mae_grid_result_lat2 = grid_based_complete_vol_surface('./model_files/finance_models/vae_vol_grid_149lat2_batchsize32_try_2.pth', 'vae', False, test_tensor, ntrials, sample_size_grid,  2, 
      device_choose='cuda', normalizing_std = train_std)
    vae_mae_grid_result_lat3 = grid_based_complete_vol_surface('./model_files/finance_models/vae_vol_grid_149lat3_batchsize32_try_2.pth', 'vae', False, test_tensor, ntrials, sample_size_grid,  3, 
      device_choose='cuda', normalizing_std = train_std)
    
    vae_mae_grid_result_lat4 = grid_based_complete_vol_surface('./model_files/finance_models/vae_vol_grid_149lat4_batchsize32_try_2.pth', 'vae', False, test_tensor, ntrials, sample_size_grid,  4, 
      device_choose='cuda', normalizing_std = train_std)
    
    vae_mae_grid_result = pd.concat((vae_mae_grid_result_lat2,vae_mae_grid_result_lat3, vae_mae_grid_result_lat4))
    vae_mae_grid_result.to_csv('vae_mae_grid.csv')
    # RAE
    random.seed(888)
    rae_l2_mae_grid_result_lat2 = grid_based_complete_vol_surface('./model_files/finance_models/rae_vol_gridl2_149lat2_batchsize32_try_1.pth', 'rae', False, test_tensor, ntrials, sample_size_grid,  2, 
      device_choose='cuda', normalizing_std = train_std)
    rae_l2_mae_grid_result_lat3 = grid_based_complete_vol_surface('./model_files/finance_models/rae_vol_gridl2_149lat3_batchsize32_try_1.pth', 'rae', False, test_tensor, ntrials, sample_size_grid,  3, 
      device_choose='cuda', normalizing_std = train_std)
    
    rae_l2_mae_grid_result_lat4 = grid_based_complete_vol_surface('./model_files/finance_models/rae_vol_gridl2_149lat4_batchsize32_try_1.pth', 'rae', False, test_tensor, ntrials, sample_size_grid,  4, 
      device_choose='cuda', normalizing_std = train_std)
    
    rae_l2_mae_grid_result = pd.concat((rae_l2_mae_grid_result_lat2,rae_l2_mae_grid_result_lat3, rae_l2_mae_grid_result_lat4))
    rae_l2_mae_grid_result.to_csv('rae_l2_mae_grid.csv')
    # RAE Spec Norm
    random.seed(888)
    rae_sn_mae_grid_result_lat2 = grid_based_complete_vol_surface('./model_files/finance_models/rae_vol_gridspec_norm_149lat2_batchsize32_try_1.pth', 'rae', True, test_tensor, ntrials, sample_size_grid,  2, 
      device_choose='cuda', normalizing_std = train_std)
    rae_sn_mae_grid_result_lat3 = grid_based_complete_vol_surface('./model_files/finance_models/rae_vol_gridspec_norm_149lat3_batchsize32_try_1.pth', 'rae', True, test_tensor, ntrials, sample_size_grid,  3, 
      device_choose='cuda', normalizing_std = train_std)
    
    rae_sn_mae_grid_result_lat4 = grid_based_complete_vol_surface('./model_files/finance_models/rae_vol_gridspec_norm_149lat4_batchsize32_try_1.pth', 'rae', True, test_tensor, ntrials, sample_size_grid,  4, 
      device_choose='cuda', normalizing_std = train_std)
    
    rae_sn_mae_grid_result = pd.concat((rae_sn_mae_grid_result_lat2,rae_sn_mae_grid_result_lat3, rae_sn_mae_grid_result_lat4))
    rae_sn_mae_grid_result.to_csv('rae_sn_mae_grid.csv')
    # RAE Gradient Penalty
    random.seed(888)
    rae_gp_mae_grid_result_lat2 = grid_based_complete_vol_surface('./model_files/finance_models/rae_vol_gridgrad_pen_149lat2_batchsize32_try_1.pth', 'rae', False, test_tensor, ntrials, sample_size_grid,  2, 
      device_choose='cuda', normalizing_std = train_std)
    rae_gp_mae_grid_result_lat3 = grid_based_complete_vol_surface('./model_files/finance_models/rae_vol_gridgrad_pen_149lat3_batchsize32_try_1.pth', 'rae', False, test_tensor, ntrials, sample_size_grid,  3, 
      device_choose='cuda', normalizing_std = train_std)
    
    rae_gp_mae_grid_result_lat4 = grid_based_complete_vol_surface('./model_files/finance_models/rae_vol_gridgrad_pen_149lat4_batchsize32_try_1.pth', 'rae', False, test_tensor, ntrials, sample_size_grid,  4, 
      device_choose='cuda', normalizing_std = train_std)
    rae_gp_mae_grid_result = pd.concat((rae_gp_mae_grid_result_lat2,rae_gp_mae_grid_result_lat3, rae_gp_mae_grid_result_lat4))
    rae_gp_mae_grid_result.to_csv('rae_gp_mae_grid.csv')
    # AE
    random.seed(888)
    ae_mae_grid_result_lat2 = grid_based_complete_vol_surface('./model_files/finance_models/rae_vol_gridl2_149lat2_batchsize32_ae.pth', 'ae', False, test_tensor, ntrials, sample_size_grid,  2, 
      device_choose='cuda', normalizing_std = train_std)
    
    ae_mae_grid_result_lat3 = grid_based_complete_vol_surface('./model_files/finance_models/rae_vol_gridl2_149lat3_batchsize32_ae.pth', 'ae', False, test_tensor, ntrials, sample_size_grid,  3, 
      device_choose='cuda', normalizing_std = train_std)
    
    ae_mae_grid_result_lat4 = grid_based_complete_vol_surface('./model_files/finance_models/rae_vol_gridl2_149lat4_batchsize32_ae.pth', 'ae', False, test_tensor, ntrials, sample_size_grid,  4, 
      device_choose='cuda', normalizing_std = train_std)
    
    ae_mae_grid_result = pd.concat((ae_mae_grid_result_lat2,ae_mae_grid_result_lat3, ae_mae_grid_result_lat4))
    ae_mae_grid_result.to_csv('ae_mae_grid.csv')
    # WAE
    random.seed(888)
    wae_mae_grid_result_lat2 = grid_based_complete_vol_surface('./model_files/finance_models/wae_vol_grid_149latent2_batchsize32_config_1.pth', 'wae', False, test_tensor, ntrials, sample_size_grid,  2, 
      device_choose='cuda', normalizing_std = train_std)
    wae_mae_grid_result_lat3 = grid_based_complete_vol_surface('./model_files/finance_models/wae_vol_grid_149latent3_batchsize32_config_1.pth', 'wae', False, test_tensor, ntrials, sample_size_grid,  3, 
      device_choose='cuda', normalizing_std = train_std)
    
    wae_mae_grid_result_lat4 = grid_based_complete_vol_surface('./model_files/finance_models/wae_vol_grid_149latent4_batchsize32_config_1.pth', 'wae', False, test_tensor, ntrials, sample_size_grid,  4, 
      device_choose='cuda', normalizing_std = train_std)
    
    wae_mae_grid_result = pd.concat((wae_mae_grid_result_lat2,wae_mae_grid_result_lat3, wae_mae_grid_result_lat4))
    wae_mae_grid_result.to_csv('wae_mae_grid.csv')




    
    # ########################################################
    # # Test Recon Results
    # ########################################################
    
    
    # Getting results
    random.seed(888)
    ae_l2_point_test_recon_mae_result_lat2 = run_test_recon_eval('./model_files/finance_models/rae_vol_pointl2_149lat2_batchsize32_ae.pth','ae',False, test_tensor, 2, daysexp_arr, delta_arr, 'cuda', 'point', test_tensor_original, train_mean, train_std)
    ae_l2_point_test_recon_mae_result_lat3 = run_test_recon_eval('./model_files/finance_models/rae_vol_pointl2_149lat3_batchsize32_ae.pth','ae',False, test_tensor, 3, daysexp_arr, delta_arr, 'cuda', 'point', test_tensor_original, train_mean, train_std)
    ae_l2_point_test_recon_mae_result_lat4 = run_test_recon_eval('./model_files/finance_models/rae_vol_pointl2_149lat4_batchsize32_ae.pth','ae',False, test_tensor, 4, daysexp_arr, delta_arr, 'cuda', 'point', test_tensor_original, train_mean, train_std)
    rae_l2_point_test_recon_mae_result_lat2 = run_test_recon_eval('./model_files/finance_models/rae_vol_pointl2_149lat2_batchsize32_try_1.pth','rae',False, test_tensor, 2, daysexp_arr, delta_arr, 'cuda', 'point', test_tensor_original, train_mean, train_std)
    rae_l2_point_test_recon_mae_result_lat3 = run_test_recon_eval('./model_files/finance_models/rae_vol_pointl2_149lat3_batchsize32_try_1.pth','rae',False, test_tensor, 3, daysexp_arr, delta_arr, 'cuda', 'point', test_tensor_original, train_mean, train_std)
    rae_l2_point_test_recon_mae_result_lat4 = run_test_recon_eval('./model_files/finance_models/rae_vol_pointl2_149lat4_batchsize32_try_1.pth','rae',False, test_tensor, 4, daysexp_arr, delta_arr, 'cuda', 'point', test_tensor_original, train_mean, train_std)
    rae_gp_point_test_recon_mae_result_lat2 = run_test_recon_eval('./model_files/finance_models/rae_vol_pointgrad_pen_149lat2_batchsize32_try_1.pth','rae',False, test_tensor, 2, daysexp_arr, delta_arr, 'cuda', 'point', test_tensor_original, train_mean, train_std)
    rae_gp_point_test_recon_mae_result_lat3 = run_test_recon_eval('./model_files/finance_models/rae_vol_pointgrad_pen_149lat3_batchsize32_try_1.pth','rae',False, test_tensor, 3, daysexp_arr, delta_arr, 'cuda', 'point', test_tensor_original, train_mean, train_std)
    rae_gp_point_test_recon_mae_result_lat4 = run_test_recon_eval('./model_files/finance_models/rae_vol_pointgrad_pen_149lat4_batchsize32_try_1.pth','rae',False, test_tensor, 4, daysexp_arr, delta_arr, 'cuda', 'point', test_tensor_original, train_mean, train_std)
    rae_sn_point_test_recon_mae_result_lat2 = run_test_recon_eval('./model_files/finance_models/rae_vol_pointspec_norm_149lat2_batchsize32_try_1.pth','rae',True, test_tensor, 2, daysexp_arr, delta_arr, 'cuda', 'point', test_tensor_original, train_mean, train_std)
    rae_sn_point_test_recon_mae_result_lat3 = run_test_recon_eval('./model_files/finance_models/rae_vol_pointspec_norm_149lat3_batchsize32_try_1.pth','rae',True, test_tensor, 3, daysexp_arr, delta_arr, 'cuda', 'point', test_tensor_original, train_mean, train_std)
    rae_sn_point_test_recon_mae_result_lat4 = run_test_recon_eval('./model_files/finance_models/rae_vol_pointspec_norm_149lat4_batchsize32_try_1.pth','rae',True, test_tensor, 4, daysexp_arr, delta_arr, 'cuda', 'point', test_tensor_original, train_mean, train_std)
    wae_point_test_recon_mae_result_lat2 = run_test_recon_eval('./model_files/finance_models/wae_vol_point_149latent2_batchsize32_config_1.pth','wae',False, test_tensor, 2, daysexp_arr, delta_arr, 'cuda', 'point', test_tensor_original, train_mean, train_std)
    wae_point_test_recon_mae_result_lat3 = run_test_recon_eval('./model_files/finance_models/wae_vol_point_149latent3_batchsize32_config_1.pth','wae',False, test_tensor, 3, daysexp_arr, delta_arr, 'cuda', 'point', test_tensor_original, train_mean, train_std)
    wae_point_test_recon_mae_result_lat4 = run_test_recon_eval('./model_files/finance_models/wae_vol_point_149latent4_batchsize32_config_1.pth','wae',False, test_tensor, 4, daysexp_arr, delta_arr, 'cuda', 'point', test_tensor_original, train_mean, train_std)
    vae_point_test_recon_mae_result_lat2 = run_test_recon_eval('./model_files/finance_models/vae_vol_point_149lat2_batchsize32_try_2.pth','vae',False, test_tensor, 2, daysexp_arr, delta_arr, 'cuda', 'point', test_tensor_original, train_mean, train_std)
    vae_point_test_recon_mae_result_lat3 = run_test_recon_eval('./model_files/finance_models/vae_vol_point_149lat3_batchsize32_try_2.pth','vae',False, test_tensor, 3, daysexp_arr, delta_arr, 'cuda', 'point', test_tensor_original, train_mean, train_std)
    vae_point_test_recon_mae_result_lat4 = run_test_recon_eval('./model_files/finance_models/vae_vol_point_149lat4_batchsize32_try_2.pth','vae',False, test_tensor, 4, daysexp_arr, delta_arr, 'cuda', 'point', test_tensor_original, train_mean, train_std)
    ae_l2_grid_test_recon_mae_result_lat2 = run_test_recon_eval('./model_files/finance_models/rae_vol_gridl2_149lat2_batchsize32_ae.pth','ae',False, test_tensor, 2, daysexp_arr, delta_arr, 'cuda', 'grid', test_tensor_original, train_mean, train_std)
    ae_l2_grid_test_recon_mae_result_lat3 = run_test_recon_eval('./model_files/finance_models/rae_vol_gridl2_149lat3_batchsize32_ae.pth','ae',False, test_tensor, 3, daysexp_arr, delta_arr, 'cuda', 'grid', test_tensor_original, train_mean, train_std)
    ae_l2_grid_test_recon_mae_result_lat4 = run_test_recon_eval('./model_files/finance_models/rae_vol_gridl2_149lat4_batchsize32_ae.pth','ae',False, test_tensor, 4, daysexp_arr, delta_arr, 'cuda', 'grid', test_tensor_original, train_mean, train_std)
    rae_l2_grid_test_recon_mae_result_lat2 = run_test_recon_eval('./model_files/finance_models/rae_vol_gridl2_149lat2_batchsize32_try_1.pth','rae',False, test_tensor, 2, daysexp_arr, delta_arr, 'cuda', 'grid', test_tensor_original, train_mean, train_std)
    rae_l2_grid_test_recon_mae_result_lat3 = run_test_recon_eval('./model_files/finance_models/rae_vol_gridl2_149lat3_batchsize32_try_1.pth','rae',False, test_tensor, 3, daysexp_arr, delta_arr, 'cuda', 'grid', test_tensor_original, train_mean, train_std)
    rae_l2_grid_test_recon_mae_result_lat4 = run_test_recon_eval('./model_files/finance_models/rae_vol_gridl2_149lat4_batchsize32_try_1.pth','rae',False, test_tensor, 4, daysexp_arr, delta_arr, 'cuda', 'grid', test_tensor_original, train_mean, train_std)
    rae_gp_grid_test_recon_mae_result_lat2 = run_test_recon_eval('./model_files/finance_models/rae_vol_gridgrad_pen_149lat2_batchsize32_try_1.pth','rae',False, test_tensor, 2, daysexp_arr, delta_arr, 'cuda', 'grid', test_tensor_original, train_mean, train_std)
    rae_gp_grid_test_recon_mae_result_lat3 = run_test_recon_eval('./model_files/finance_models/rae_vol_gridgrad_pen_149lat3_batchsize32_try_1.pth','rae',False, test_tensor, 3, daysexp_arr, delta_arr, 'cuda', 'grid', test_tensor_original, train_mean, train_std)
    rae_gp_grid_test_recon_mae_result_lat4 = run_test_recon_eval('./model_files/finance_models/rae_vol_gridgrad_pen_149lat4_batchsize32_try_1.pth','rae',False, test_tensor, 4, daysexp_arr, delta_arr, 'cuda', 'grid', test_tensor_original, train_mean, train_std)
    rae_sn_grid_test_recon_mae_result_lat2 = run_test_recon_eval('./model_files/finance_models/rae_vol_gridspec_norm_149lat2_batchsize32_try_1.pth','rae',True, test_tensor, 2, daysexp_arr, delta_arr, 'cuda', 'grid', test_tensor_original, train_mean, train_std)
    rae_sn_grid_test_recon_mae_result_lat3 = run_test_recon_eval('./model_files/finance_models/rae_vol_gridspec_norm_149lat3_batchsize32_try_1.pth','rae',True, test_tensor, 3, daysexp_arr, delta_arr, 'cuda', 'grid', test_tensor_original, train_mean, train_std)
    rae_sn_grid_test_recon_mae_result_lat4 = run_test_recon_eval('./model_files/finance_models/rae_vol_gridspec_norm_149lat4_batchsize32_try_1.pth','rae',True, test_tensor, 4, daysexp_arr, delta_arr, 'cuda', 'grid', test_tensor_original, train_mean, train_std)
    wae_grid_test_recon_mae_result_lat2 = run_test_recon_eval('./model_files/finance_models/wae_vol_grid_149latent2_batchsize32_config_1.pth','wae',False, test_tensor, 2, daysexp_arr, delta_arr, 'cuda', 'grid', test_tensor_original, train_mean, train_std)
    wae_grid_test_recon_mae_result_lat3 = run_test_recon_eval('./model_files/finance_models/wae_vol_grid_149latent3_batchsize32_config_1.pth','wae',False, test_tensor, 3, daysexp_arr, delta_arr, 'cuda', 'grid', test_tensor_original, train_mean, train_std)
    wae_grid_test_recon_mae_result_lat4 = run_test_recon_eval('./model_files/finance_models/wae_vol_grid_149latent4_batchsize32_config_1.pth','wae',False, test_tensor, 4, daysexp_arr, delta_arr, 'cuda', 'grid', test_tensor_original, train_mean, train_std)
    vae_grid_test_recon_mae_result_lat2 = run_test_recon_eval('./model_files/finance_models/vae_vol_grid_149lat2_batchsize32_try_2.pth','vae',False, test_tensor, 2, daysexp_arr, delta_arr, 'cuda', 'grid', test_tensor_original, train_mean, train_std)
    vae_grid_test_recon_mae_result_lat3 = run_test_recon_eval('./model_files/finance_models/vae_vol_grid_149lat3_batchsize32_try_2.pth','vae',False, test_tensor, 3, daysexp_arr, delta_arr, 'cuda', 'grid', test_tensor_original, train_mean, train_std)
    vae_grid_test_recon_mae_result_lat4 = run_test_recon_eval('./model_files/finance_models/vae_vol_grid_149lat4_batchsize32_try_2.pth','vae',False, test_tensor, 4, daysexp_arr, delta_arr, 'cuda', 'grid', test_tensor_original, train_mean, train_std)

    
    test_recon_df = pd.DataFrame([['ae_l2_point_lat2', 'ae_l2_point_lat3', 'ae_l2_point_lat4', 'rae_l2_point_lat2', 'rae_l2_point_lat3', 'rae_l2_point_lat4', 'rae_gp_point_lat2', 'rae_gp_point_lat3', 'rae_gp_point_lat4', 'rae_sn_point_lat2', 'rae_sn_point_lat3', 'rae_sn_point_lat4', 'wae_point_lat2', 'wae_point_lat3', 'wae_point_lat4', 'vae_point_lat2', 'vae_point_lat3', 'vae_point_lat4', 'ae_l2_grid_lat2', 'ae_l2_grid_lat3', 'ae_l2_grid_lat4', 'rae_l2_grid_lat2', 'rae_l2_grid_lat3', 'rae_l2_grid_lat4', 'rae_gp_grid_lat2', 'rae_gp_grid_lat3', 'rae_gp_grid_lat4', 'rae_sn_grid_lat2', 'rae_sn_grid_lat3', 'rae_sn_grid_lat4', 'wae_grid_lat2', 'wae_grid_lat3', 'wae_grid_lat4', 'vae_grid_lat2', 'vae_grid_lat3', 'vae_grid_lat4'], [ae_l2_point_test_recon_mae_result_lat2,ae_l2_point_test_recon_mae_result_lat3,ae_l2_point_test_recon_mae_result_lat4,rae_l2_point_test_recon_mae_result_lat2,rae_l2_point_test_recon_mae_result_lat3,rae_l2_point_test_recon_mae_result_lat4,rae_gp_point_test_recon_mae_result_lat2,rae_gp_point_test_recon_mae_result_lat3,rae_gp_point_test_recon_mae_result_lat4,rae_sn_point_test_recon_mae_result_lat2,rae_sn_point_test_recon_mae_result_lat3,rae_sn_point_test_recon_mae_result_lat4,wae_point_test_recon_mae_result_lat2,wae_point_test_recon_mae_result_lat3,wae_point_test_recon_mae_result_lat4,vae_point_test_recon_mae_result_lat2,vae_point_test_recon_mae_result_lat3,vae_point_test_recon_mae_result_lat4,ae_l2_grid_test_recon_mae_result_lat2,ae_l2_grid_test_recon_mae_result_lat3,ae_l2_grid_test_recon_mae_result_lat4,rae_l2_grid_test_recon_mae_result_lat2,rae_l2_grid_test_recon_mae_result_lat3,rae_l2_grid_test_recon_mae_result_lat4,rae_gp_grid_test_recon_mae_result_lat2,rae_gp_grid_test_recon_mae_result_lat3,rae_gp_grid_test_recon_mae_result_lat4,rae_sn_grid_test_recon_mae_result_lat2,rae_sn_grid_test_recon_mae_result_lat3,rae_sn_grid_test_recon_mae_result_lat4,wae_grid_test_recon_mae_result_lat2,wae_grid_test_recon_mae_result_lat3,wae_grid_test_recon_mae_result_lat4,vae_grid_test_recon_mae_result_lat2,vae_grid_test_recon_mae_result_lat3,vae_grid_test_recon_mae_result_lat4]])
    
    # Export results
    test_recon_df.to_csv('test_recon_results.csv')
    
    
    
    