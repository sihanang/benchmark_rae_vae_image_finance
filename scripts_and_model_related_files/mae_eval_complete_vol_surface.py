# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 22:04:12 2023

@author: angsi
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


def calibrate_vol(test_set, decoder, sample_size, max_steps, latent_dims, device, encoder, model_type):
    
    input_shape = (1, latent_dims)
    
    # Get the configuration index list
    config_ind_lst = [(a, b) for a in range(len(daysexp_arr)) for b in range(len(delta_arr))]
        
    # Sample without replacement
    config_ind_sample = sample(config_ind_lst, sample_size)
    
    if encoder is not None:
        # Use encoder to get initial estimate of z by feeding in x values with missing values filled with interpolation
        x_initial = torch.zeros((1,1,5,5), requires_grad= False, device=device)
        
        for days_ind, delta_ind in config_ind_sample:
            target = test_set[0, days_ind, delta_ind].to(device)
            x_initial[:,:,days_ind, delta_ind] = target
        
        # Replace 0 with np.nan
        x_initial_df = pd.DataFrame(x_initial.squeeze().cpu().detach().numpy()).replace(0, np.nan)
        
        #Interpolate
        x_initial_df = x_initial_df.interpolate(limit_direction = 'both', axis = 0).interpolate(limit_direction = 'both', axis = 1)
        x_initial_df = x_initial_df.values
        
        # Assign to x_initial
        x_initial = torch.from_numpy(x_initial_df).unsqueeze(0).unsqueeze(0)
        x_initial.requires_grad = False
        x_initial = x_initial.to(device)
            
        # Initial estimate of z
        if model_type == 'vae':
            z_est, mu, log_sigma = encoder(x_initial.float())
        else:
            z_est = encoder(x_initial.float())
            
        z = torch.tensor(z_est.cpu().detach().numpy(), requires_grad=True, device=device)
        
    else:
        z = torch.randn(input_shape, requires_grad= True, device=device)
    
    # Using RMSprop Optimiser by default
    optimizer = optim.RMSprop([z])
    
    
    # def closure():
    #     lbfgs.zero_grad()
    #     loss = (decoder(z.float(), dayexp_tensor, delta_tensor) - target) **2
    #     loss.backward()
    #     return loss
    
    # lbfgs = optim.LBFGS([z],
    #                 history_size=10, 
    #                 max_iter=5, 
    #                 line_search_fn="strong_wolfe")
    
    # Initialise best ever loss
    best_ever_loss = np.inf
    best_ever_z = None
    
    # print('Starting z: ', z)
    
    for step in range(max_steps):
        # clear out the gradients of all Variables in the optimiser
        optimizer.zero_grad()
        
        mseloss = torch.zeros(1, requires_grad=True, device =device)

        # Loop through each configuration and evaluate loss
        for days_ind, delta_ind in config_ind_sample:
            daysexp = daysexp_arr[days_ind]
            delta = delta_arr[delta_ind]
            target = test_set[0, days_ind, delta_ind].to(device)
            
            dayexp_tensor = torch.full((z.size(0),1), daysexp).to(device)
            delta_tensor = torch.full((z.size(0),1), delta).to(device)
            output = decoder(z.float(), dayexp_tensor, delta_tensor)

            loss = (output - target) ** 2
            
            mseloss = mseloss + loss
            
            # print('Days to expiry: {}, Delta: {}; Output: {}, Target: {}'.format(daysexp, delta, output, target))
            
            # loss = (decoder(z.float(), dayexp_tensor, delta_tensor) - target) **2
            # lbfgs.step(closure)

        mseloss = mseloss/sample_size
        
        
        # if step % 50 == 0 or step == 1149:
        #     print('For step:', step, 'Loss is ' ,loss)
        
        # Keep track of best ever loss
        if mseloss.cpu().detach().numpy()[0] <  best_ever_loss:
            best_ever_loss = mseloss.cpu().detach().numpy()[0]
            best_ever_z = z.clone().detach()
        #     # print('Best ever Loss updated')
        
        mseloss.backward()
        optimizer.step()
        
        # print('Step{} z: {}'.format(step, z))
        
    # print(best_ever_z)
    # print(z)
    # print(best_ever_mae)
    
    # print('Best ever loss is: ', best_ever_loss)
    
    
    ## TO CHECK
    # diff=0
    # for days_ind, delta_ind in config_ind_sample:
    #     daysexp = daysexp_arr[days_ind]
    #     delta = delta_arr[delta_ind]
    #     target = test_set[0, days_ind, delta_ind].to(device).cpu().detach().numpy()
        
    #     dayexp_tensor = torch.full((best_ever_z.size(0),1), daysexp).to(device)
    #     delta_tensor = torch.full((best_ever_z.size(0),1), delta).to(device)
    #     output = decoder(best_ever_z.float(), dayexp_tensor, delta_tensor).cpu().detach().numpy()[0]
        
    #     diff += abs(output - target)
        
    #     print('Days to expiry: {}, Delta: {}; Output: {}, Target: {}'.format(daysexp, delta, output, target))
    #     print('Diff ', abs(output - target))
    
    # print('MAE:',diff/sample_size)
        
    
    return best_ever_z

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
    
    try:
        checkpoint = torch.load(saved_model_path)
    except:
        checkpoint = torch.load(saved_model_path, map_location=torch.device('cpu')) 
    
    print('loading model success')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print('loaded model')
    
    model.eval()
    
    return model

def vol_surface_recon(decoder, z_calibrated, daysexp_arr, delta_arr, device):
    '''
    Given decoder and calibrated latent variable, reconstruct the whole volaltility surface
    '''
    vol_mat_outer_lst = []
    vol_mat_inner_lst = []
    
    
    #KIV THE DIMENSIONS
    for row_index, daysexp in enumerate(daysexp_arr):
        for col_index, delta in enumerate(delta_arr):
            
            ###### NEED TO MAKE SURE ASSIGNED PROPERLY
            dayexp_tensor = torch.full((z_calibrated.size(0),1), daysexp).to(device)
            delta_tensor = torch.full((z_calibrated.size(0),1), delta).to(device)
            x_pt = decoder(z_calibrated.float(), dayexp_tensor, delta_tensor)

            vol_mat_inner_lst.append(x_pt)
            
        vol_mat_row = torch.cat(vol_mat_inner_lst, axis = 1)
        vol_mat_row = torch.unsqueeze(vol_mat_row, 1)
        # print(vol_mat_row.shape)
        
        vol_mat_outer_lst.append(vol_mat_row)
        
        # Reset inner list
        vol_mat_inner_lst = []
        
        volmatrix = torch.cat(vol_mat_outer_lst, axis = 1)
        
    return volmatrix

def complete_vol_surf_mae(model_path, model_name, SN, test_tensor, ntrials, sample_size_grid, max_steps, latent_dims,
                          daysexp_arr, delta_arr,
                          device_choose, approach='point', normalizing_std = None):
    
    '''
    Function to complete volatility surface 
    '''
    print('Running {} model with latent dimensions={}'.format(model_name, latent_dims))
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
    
    # MAE List
    mae_lst = []
    mae_trials = []
    
    test_tensor = test_tensor.to(device)
    
    test_size = len(test_tensor)
    
    
    # To assign decoder based on the instance
    if isinstance(model, nn.DataParallel):
        decoder = model.module.decoder
        encoder = model.module.encoder
    else:
        decoder = model.decoder
        encoder = model.encoder
    
    # Loop through the possible subsets defined
    for sample_size in sample_size_grid:
        
        print('Running Partially Known Data Points:{}'.format(sample_size))
        
        # Reset mae_trials
        mae_trials = []
        
        # For each ntrial
        for ntrial in range(ntrials):
            
            print('Running Trial ', ntrial)
        
            # Initialise mae_metric
            mae = torch.zeros(1, device = device)
            
            for i, test_set in enumerate(test_tensor):
                
                # Calibration to find best z 
                z_calibrated = calibrate_vol(test_set, decoder, sample_size, max_steps, latent_dims, device, encoder=encoder, model_type=model_name)
                
                # Get volatility surface reconstructed
                volmatrix = vol_surface_recon(decoder, z_calibrated, daysexp_arr, delta_arr, device)
                
                # Compute MAE:
                if normalizing_std is not None:
                    mae += torch.mean(torch.abs(volmatrix - test_set) * normalizing_std)
                    # print(torch.mean(torch.abs(volmatrix - test_set) * normalizing_std))
                else:
                    mae += torch.mean(torch.abs(volmatrix - test_set))
                    # print(torch.mean(torch.abs(volmatrix - test_set)))
                
                if ((i+1) % 50 == 0):
                    print('{} test samples out of {} are processed.'.format(i+1, test_size))
                    
                # print(mae.cpu().detach().numpy()[0]/test_size)

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
    
    ################################################  EVALUATION STARTS ################################################################
    ntrials = 3
    sample_size_grid = [5,10,15,20,25]
    max_steps = 50
    
    random.seed(888)
    
    fin_mod_dir = './model_files/finance_models/'
    
    ###### Complete Volatility Surface
    
    # VAE
    vae_mae_result_lat2 = complete_vol_surf_mae(fin_mod_dir+'vae_vol_point_149lat2_batchsize32_try_2.pth', 'vae', False, test_tensor, ntrials, sample_size_grid, max_steps, 2, 
                                                daysexp_arr, delta_arr, device_choose='cuda', normalizing_std = train_std)
    
    # vae_mae_result_lat2.to_csv('vae_mae_lat2.csv')
    
    vae_mae_result_lat3 = complete_vol_surf_mae(fin_mod_dir+'vae_vol_point_149lat3_batchsize32_try_2.pth', 'vae', False, test_tensor, ntrials, sample_size_grid, max_steps, 3, 
                                                daysexp_arr, delta_arr, device_choose='cuda', normalizing_std = train_std)
    
    # vae_mae_result_lat3.to_csv('vae_mae_lat3.csv')
    vae_mae_result_lat4 = complete_vol_surf_mae(fin_mod_dir+'vae_vol_point_149lat4_batchsize32_try_2.pth', 'vae', False, test_tensor, ntrials, sample_size_grid, max_steps, 4, 
                                                daysexp_arr, delta_arr, device_choose='cuda', normalizing_std = train_std)
    
    # vae_mae_result_lat4.to_csv('vae_mae_lat4.csv')
    
    vae_mae_result = pd.concat((vae_mae_result_lat2,vae_mae_result_lat3, vae_mae_result_lat4))
    vae_mae_result.to_csv('vae_mae.csv')
    
    
    # RAE
    random.seed(888)
    rae_l2_mae_result_lat2 = complete_vol_surf_mae(fin_mod_dir+'rae_vol_pointl2_149lat2_batchsize32_try_1.pth', 'rae', False, test_tensor, ntrials, sample_size_grid, max_steps, 2, 
                                                daysexp_arr, delta_arr, device_choose='cuda', normalizing_std = train_std)
    
    # rae_l2_mae_result_lat2.to_csv('rae_l2_mae_lat2.csv')
    
    rae_l2_mae_result_lat3 = complete_vol_surf_mae(fin_mod_dir+'rae_vol_pointl2_149lat3_batchsize32_try_1.pth', 'rae', False, test_tensor, ntrials, sample_size_grid, max_steps, 3, 
                                                daysexp_arr, delta_arr, device_choose='cuda', normalizing_std = train_std)
    
    # rae_l2_mae_result_lat3.to_csv('rae_l2_mae_lat3.csv')
    rae_l2_mae_result_lat4 = complete_vol_surf_mae(fin_mod_dir+'rae_vol_pointl2_149lat4_batchsize32_try_1.pth', 'rae', False, test_tensor, ntrials, sample_size_grid, max_steps, 4, 
                                                daysexp_arr, delta_arr, device_choose='cuda', normalizing_std = train_std)
    
    # rae_l2_mae_result_lat4.to_csv('rae_l2_mae_lat4.csv')
    
    rae_l2_mae_result = pd.concat((rae_l2_mae_result_lat2,rae_l2_mae_result_lat3, rae_l2_mae_result_lat4))
    rae_l2_mae_result.to_csv('rae_l2_mae.csv')
    
    
    # RAE Spec Norm
    random.seed(888)
    rae_sn_mae_result_lat2 = complete_vol_surf_mae(fin_mod_dir+'rae_vol_pointspec_norm_149lat2_batchsize32_try_1.pth', 'rae', True, test_tensor, ntrials, sample_size_grid, max_steps, 2, 
                                                daysexp_arr, delta_arr, device_choose='cuda', normalizing_std = train_std)
    
    # rae_sn_mae_result_lat2.to_csv('rae_sn_mae_lat2.csv')
    
    rae_sn_mae_result_lat3 = complete_vol_surf_mae(fin_mod_dir+'rae_vol_pointspec_norm_149lat3_batchsize32_try_1.pth', 'rae', True, test_tensor, ntrials, sample_size_grid, max_steps, 3, 
                                                daysexp_arr, delta_arr, device_choose='cuda', normalizing_std = train_std)
    
    # rae_sn_mae_result_lat3.to_csv('rae_sn_mae_lat3.csv')
    rae_sn_mae_result_lat4 = complete_vol_surf_mae(fin_mod_dir+'rae_vol_pointspec_norm_149lat4_batchsize32_try_1.pth', 'rae', True, test_tensor, ntrials, sample_size_grid, max_steps, 4, 
                                                daysexp_arr, delta_arr, device_choose='cuda', normalizing_std = train_std)
    
    # rae_sn_mae_result_lat4.to_csv('rae_sn_mae_lat4.csv')
    
    rae_sn_mae_result = pd.concat((rae_sn_mae_result_lat2,rae_sn_mae_result_lat3, rae_sn_mae_result_lat4))
    rae_sn_mae_result.to_csv('rae_sn_mae.csv')
    
    
    # RAE Gradient Penalty
    random.seed(888)
    rae_gp_mae_result_lat2 = complete_vol_surf_mae(fin_mod_dir+'rae_vol_pointgrad_pen_149lat2_batchsize32_try_1.pth', 'rae', False, test_tensor, ntrials, sample_size_grid, max_steps, 2, 
                                                daysexp_arr, delta_arr, device_choose='cuda', normalizing_std = train_std)
    
    # rae_gp_mae_result_lat2.to_csv('rae_gp_mae_lat2.csv')
    
    rae_gp_mae_result_lat3 = complete_vol_surf_mae(fin_mod_dir+'rae_vol_pointgrad_pen_149lat3_batchsize32_try_1.pth', 'rae', False, test_tensor, ntrials, sample_size_grid, max_steps, 3, 
                                                daysexp_arr, delta_arr, device_choose='cuda', normalizing_std = train_std)
    
    # rae_gp_mae_result_lat3.to_csv('rae_gp_mae_lat3.csv')
    rae_gp_mae_result_lat4 = complete_vol_surf_mae(fin_mod_dir+'rae_vol_pointgrad_pen_149lat4_batchsize32_try_1.pth', 'rae', False, test_tensor, ntrials, sample_size_grid, max_steps, 4, 
                                                daysexp_arr, delta_arr, device_choose='cuda', normalizing_std = train_std)
    
    # rae_gp_mae_result_lat4.to_csv('rae_gp_mae_lat4.csv')
    
    rae_gp_mae_result = pd.concat((rae_gp_mae_result_lat2,rae_gp_mae_result_lat3, rae_gp_mae_result_lat4))
    rae_gp_mae_result.to_csv('rae_gp_mae.csv')
    
    
    # AE
    random.seed(888)
    ae_mae_result_lat2 = complete_vol_surf_mae(fin_mod_dir+'rae_vol_pointl2_149lat2_batchsize32_ae.pth', 'ae', False, test_tensor, ntrials, sample_size_grid, max_steps, 2, 
                                                daysexp_arr, delta_arr, device_choose='cuda', normalizing_std = train_std)
    
    # ae_mae_result_lat2.to_csv('ae_mae_lat2.csv')
    
    ae_mae_result_lat3 = complete_vol_surf_mae(fin_mod_dir+'rae_vol_pointl2_149lat3_batchsize32_ae.pth', 'ae', False, test_tensor, ntrials, sample_size_grid, max_steps, 3, 
                                                daysexp_arr, delta_arr, device_choose='cuda', normalizing_std = train_std)
    
    # ae_mae_result_lat3.to_csv('ae_mae_lat3.csv')
    ae_mae_result_lat4 = complete_vol_surf_mae(fin_mod_dir+'rae_vol_pointl2_149lat4_batchsize32_ae.pth', 'ae', False, test_tensor, ntrials, sample_size_grid, max_steps, 4, 
                                                daysexp_arr, delta_arr, device_choose='cuda', normalizing_std = train_std)
    
    # ae_mae_result_lat4.to_csv('ae_mae_lat4.csv')
    
    ae_mae_result = pd.concat((ae_mae_result_lat2,ae_mae_result_lat3, ae_mae_result_lat4))
    ae_mae_result.to_csv('ae_mae.csv')
    
    
    # WAE
    random.seed(888)
    wae_mae_result_lat2 = complete_vol_surf_mae(fin_mod_dir+'wae_vol_point_149latent2_batchsize32_config_1.pth', 'wae', False, test_tensor, ntrials, sample_size_grid, max_steps, 2, 
                                                daysexp_arr, delta_arr, device_choose='cuda', normalizing_std = train_std)
    
    # wae_mae_result_lat2.to_csv('wae_mae_lat2.csv')
    
    wae_mae_result_lat3 = complete_vol_surf_mae(fin_mod_dir+'wae_vol_point_149latent3_batchsize32_config_1.pth', 'wae', False, test_tensor, ntrials, sample_size_grid, max_steps, 3, 
                                                daysexp_arr, delta_arr, device_choose='cuda', normalizing_std = train_std)
    
    # wae_mae_result_lat3.to_csv('wae_mae_lat3.csv')
    wae_mae_result_lat4 = complete_vol_surf_mae(fin_mod_dir+'wae_vol_point_149latent4_batchsize32_config_1.pth', 'wae', False, test_tensor, ntrials, sample_size_grid, max_steps, 4, 
                                                daysexp_arr, delta_arr, device_choose='cuda', normalizing_std = train_std)
    
    # wae_mae_result_lat4.to_csv('wae_mae_lat4.csv')
    
    wae_mae_result = pd.concat((wae_mae_result_lat2,wae_mae_result_lat3, wae_mae_result_lat4))
    wae_mae_result.to_csv('wae_mae.csv')
    
    
    
    