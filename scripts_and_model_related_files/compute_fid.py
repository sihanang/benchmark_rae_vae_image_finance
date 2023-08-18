"""
This script does the FID computation for the different reconstructed images-
"""
import os
import pathlib
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np
import torch
import torchvision.transforms as TF
from PIL import Image
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
import pandas as pd

from imageio import imwrite

from fid import *
from model_repo import *

try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x

from inception import InceptionV3
import random
from cars_preprocess import *
from cars_data_gen import *

from sklearn import mixture


def get_ztrain(train_dataloader, model, vae, device):
    '''
    For a train_dataloader and autoencoder model, it will the z from training set in numpy array
    '''
    z_lst = []
    for i, x_y in enumerate(train_dataloader):
        x, y = x_y
        x = x.to(device)
        
        if vae == True:
            z, _, __ = model.encoder(x)
        else:
            z = model.encoder(x)
        
        # Append to z_lst
        z_lst.append(z.cpu().detach().numpy())
        
        if (i+1)%10 == 0:
            print('{} batches done'.format(i+1))
        
    z_trained = np.vstack(z_lst)
    
    return z_trained


def rae_rndm_smpl(z_trained, model, sample_size, device):
    '''
    Estimates covariance, mean and generates random sample from multivariate normal
    '''
    est_cov = np.cov(z_trained.T)
    est_mean = np.mean(z_trained.T, axis = 1)
    
    latent = torch.from_numpy(np.random.multivariate_normal(est_mean, cov=est_cov, size=sample_size)).to(torch.float32).to(device)
    
    rndm_smpl = model.decoder(latent)

    # Convert to numpy and make sure it is within 255
    if rndm_smpl.shape[1] == 3:
        rndm_smpl = (np.clip(rndm_smpl.cpu().detach().permute(0,2,3,1).numpy(), 0, 1) * 255).astype('uint8')
    else:
        rndm_smpl = (np.clip(rndm_smpl.cpu().detach().numpy(), 0, 1) * 255).astype('uint8')
    
    return rndm_smpl

def gmm_rndmn_train(z_trained, n_dist, device):
    '''
    To train the GMM model
    '''
    gmmmodel = mixture.GaussianMixture(n_components=n_dist, covariance_type='full', max_iter=500,
                                                 verbose=2, tol=1e-3)

    gmmmodel.fit(z_trained)
    
    return gmmmodel

def gmm_rndm_smpl(model, sample_size, gmmmodel, device):

    # Randomly shuffle indices
    scrmb_idx = np.array(range(sample_size))
    np.random.shuffle(scrmb_idx)
    
    gmm_sample = gmmmodel.sample(sample_size)[0][scrmb_idx, :]
    
    # Send to decoder
    latent = torch.from_numpy(gmm_sample).to(torch.float32).to(device)
    
    gmm_sample_out = model.decoder(latent)
    
    # Convert to numpy array
    if gmm_sample_out.shape[1] == 3:
        gmm_sample_out = (np.clip(gmm_sample_out.cpu().detach().permute(0,2,3,1).numpy(), 0, 1) * 255).astype('uint8')
    else:
        gmm_sample_out = (np.clip(gmm_sample_out.cpu().detach().numpy(), 0, 1) * 255).astype('uint8')
    return gmm_sample_out



def fmnist_compute_fid_main(modelfile, latent_dims,  model_type = 'rae', model_select = 'rae_fmnist', device_choose = 'cpu', SN = False):
    
    random.seed(911)
    np.random.seed(911)
    torch.manual_seed(911)
    torch.cuda.manual_seed_all(911)
    
    # Getting the paths
    image_path = '../data/image_data/'
    
    if not os.path.exists(image_path):
        os.mkdir(image_path)
        
    if not os.path.exists(image_path + 'FashionMNIST'):
        os.mkdir(image_path + 'FashionMNIST')

    
    fashionmnist_path = image_path + 'FashionMNIST/proc/'
    
    if not os.path.exists(fashionmnist_path):
        os.mkdir(fashionmnist_path)
    
    
    fashionmnist_org_path = fashionmnist_path + 'test'
    fashionmnist_rndmN_path = fashionmnist_path + 'rndmN'
    fashionmnist_recon_path = fashionmnist_path + 'test_recon'
    fashionmnist_gmm_path = fashionmnist_path + 'GMM'
    
    
    # Ensure that all the paths are made available
    if not os.path.exists(fashionmnist_org_path):
        os.mkdir(fashionmnist_org_path)
    
    if not os.path.exists(fashionmnist_rndmN_path):
        os.mkdir(fashionmnist_rndmN_path)
    
    if not os.path.exists(fashionmnist_recon_path):
        os.mkdir(fashionmnist_recon_path)
    
    if not os.path.exists(fashionmnist_gmm_path):
        os.mkdir(fashionmnist_gmm_path)
    
    
    # To get test images
    print('Starting to save Test images')
    if len(os.listdir(fashionmnist_org_path)) == 10000:
        print('10K Test images previously loaded, skipping save')
    else:    
        save_test_images(fashionmnist_org_path, 'fmnist')
        print('Saved Test images')
    
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    
    fmnist_mod = load_model(model_select, modelfile, latent_dims, SN)
    print('model loaded')
    
    if torch.cuda.device_count() > 1:
        fmnist_mod = nn.DataParallel(fmnist_mod)
        
    if device_choose =='cuda':
        device = torch.device("cuda")
    else:
        device = torch.device('cpu')
    fmnist_mod = fmnist_mod.to(device)
    
    
    if isinstance(fmnist_mod, nn.DataParallel):
        model_attr_accessor = fmnist_mod.module
        print('data parallel')
    else:
        model_attr_accessor = fmnist_mod

    
    print('Model load success!')
    
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    transform_proc = transforms.Compose([TF.Pad(2), TF.ToTensor()])
    train_data = torchvision.datasets.FashionMNIST('../data/image_data', 
                                                  train = True,
                                                  download = True, transform = transform_proc)
    test_data = torchvision.datasets.FashionMNIST('../data/image_data', 
                                      train = False,
                                      download = True, transform = transform_proc)
    
    training_data, val_data = torch.utils.data.random_split(train_data, [50000, 10000], generator=torch.Generator().manual_seed(1))
    train_dataloader = DataLoader(training_data, batch_size=500, shuffle = False)
    test_dataloader = DataLoader(test_data, batch_size=500, shuffle=False)
    
    
    # Get Z_trained
    vae_flag = True if model_type == 'vae' else False
    z_trained = get_ztrain(train_dataloader, model_attr_accessor, vae_flag, device)
    
    # If RAE, then for random sample, have to check test fit
    if model_type == 'rae' or model_type == 'ae':
        print('Running RAE Random Sample from Normal')
        
        # break into smaller arrays
        for i in range(10):
            rndmN_ = rae_rndm_smpl(z_trained, model_attr_accessor, sample_size=1000, device=device)
            if i == 0:
                fmnist_randN_img = rndmN_
            else:
                fmnist_randN_img = np.vstack([fmnist_randN_img, rndmN_])
        
        # Save random normal sampled images
        # Overwrites existing ones in folder
        for j, img in enumerate(fmnist_randN_img):
            img = img.squeeze()
            imwrite(os.path.join(fashionmnist_rndmN_path, '%08d.png' % j), img)
    
    
    # Compute GMM-10
    gmm_trained_model = gmm_rndmn_train(z_trained=z_trained, n_dist=10, device=device)
    
    print('Running GMM Model')
    
    # Break into smaller partitions to run
    for i in range(10):
        gmm_sample_ = gmm_rndm_smpl(model_attr_accessor, 1000, gmm_trained_model, device)
        if i == 0:
            gmm_sample_img = gmm_sample_
        else:
            gmm_sample_img = np.vstack([gmm_sample_img, gmm_sample_])
    
    # Save GMM-10 generated images
    # Overwrites existing ones
    print('Saving GMM Sample Generated')
    for j, img in enumerate(gmm_sample_img):
        img = img.squeeze()
        imwrite(os.path.join(fashionmnist_gmm_path, '%08d.png' % j), img)
    
    
    # This is to infer latent dimensions
    fmnist_mod_first_parameters = next(model_attr_accessor.decoder.parameters())
    fmnist_mod_input_shape = fmnist_mod_first_parameters.shape
    
    
    # To cater to the issue
    if device_choose == 'cuda':
        torch.cuda.set_device('cuda:0')
        fmnist_mod.cuda()
    
    
    # For test recon
    for i, x_y in enumerate(test_dataloader):
        x, y = x_y
        x = x.to(device)
        
        # Test Recon
        if vae_flag:
            ztest, _, __ = model_attr_accessor.encoder(x.float())
        else:
            ztest = model_attr_accessor.encoder(x.float())
        test_recon_img = model_attr_accessor.decoder(ztest)
        test_recon_img = (np.clip(test_recon_img.cpu().detach().numpy(), 0, 1) * 255).astype('uint8')
        
        # Save test recon
        for j, img in enumerate(test_recon_img):
            img = img.squeeze()
            # print(img)
            # Overwrites existing ones in folder
            imwrite(os.path.join(fashionmnist_recon_path, '%08d.png' % (j+i*500)), img)
            
        # This is for vae/wae random normal samples
        if model_type == 'vae' or model_type == 'wae':
            randnorm_tensor = torch.from_numpy(np.random.normal(loc=0.0, scale=1.0, size=(500, list(fmnist_mod_input_shape)[1]))).to(torch.float32).to(device)
            
            # print('These are the random tensors: ', randnorm_tensor)
            fmnist_randN_img = model_attr_accessor.decoder(randnorm_tensor).cpu().detach()
            fmnist_randN_img = (np.clip(fmnist_randN_img.cpu().detach().numpy(), 0, 1) * 255).astype('uint8')
            
    
            # Save random normal
            for j, img in enumerate(fmnist_randN_img):
                img = img.squeeze()
                # print(img)
                imwrite(os.path.join(fashionmnist_rndmN_path, '%08d.png' % (j+i*500)), img)
        
        print(i, ' is done')
    
    # fmnist_randN_img_all =  torch.cat(fmnist_mod_lst,0)
    # test_recon_img_all = torch.cat(test_recon_img_lst,0)
    
    # # Save generated images
    # save_generated_images(fashionmnist_rndmN_path, fmnist_randN_img_all)
    # save_generated_images(fashionmnist_recon_path, test_recon_img_all)
    
    fid_score_fashionmnist_rndmN = get_fid(fashionmnist_org_path, fashionmnist_rndmN_path)
    fid_score_fashionmnist_recon = get_fid(fashionmnist_org_path,fashionmnist_recon_path)
    fid_score_fashionmnist_gmm = get_fid(fashionmnist_org_path,fashionmnist_gmm_path)
    
    print('FID Score for Random Normal Sample is: {}'.format(fid_score_fashionmnist_rndmN))
    print('FID Score for Test Recon is {}'.format(fid_score_fashionmnist_recon))
    print('FID Score for GMM is {}'.format(fid_score_fashionmnist_gmm))
    
    return fid_score_fashionmnist_recon, fid_score_fashionmnist_rndmN, fid_score_fashionmnist_gmm

def cars_compute_fid_main(modelfile, latent_dims,  model_type = 'rae', model_select = 'rae_cars', img_height = 64, device_choose = 'cpu', SN = False):
    random.seed(911)
    np.random.seed(911)
    torch.manual_seed(911)
    torch.cuda.manual_seed_all(911)
    
    # Getting the paths
    image_path = '../data/image_data/'
    cars_path = image_path + 'stanford_cars/'
    
    if not os.path.exists(cars_path):
        os.mkdir(cars_path)
        
    cars_proc_path = cars_path + 'proc/'
    
    if not os.path.exists(cars_proc_path):
        os.mkdir(cars_proc_path)
    
    cars_org_path = cars_path + 'test{}'.format(img_height)
    cars_rndmN_path = cars_proc_path + 'rndmN'
    cars_recon_path = cars_proc_path + 'test_recon'
    cars_gmm_path = cars_proc_path + 'gmm'
    
    if not os.path.exists(cars_recon_path):
        os.mkdir(cars_recon_path)
        
    if not os.path.exists(cars_rndmN_path):
        os.mkdir(cars_rndmN_path)
        
    if not os.path.exists(cars_gmm_path):
        os.mkdir(cars_gmm_path)

    # To get test images
    save_test_images(cars_org_path, 'cars')
    
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    
    mod = load_model(model_select, modelfile ,latent_dims = latent_dims, SN = SN)
    print('model loaded')
    
    if torch.cuda.device_count() > 1:
        mod = nn.DataParallel(mod)
        
    if device_choose =='cuda':
        device = torch.device("cuda")
    else:
        device = torch.device('cpu')
    mod = mod.to(device)
    
    
    if isinstance(mod, nn.DataParallel):
        model_attr_accessor = mod.module
        print('data parallel')
    else:
        model_attr_accessor = mod

    print('Model load success!')
    
    # This is to infer latent dimensions
    mod_first_parameters = next(model_attr_accessor.decoder.parameters())
    mod_input_shape = mod_first_parameters.shape
    
    cars_rndm_sample_lst = []
    cars_test_recon_lst = []
    
    train_dataloader = DataLoader(dataset=VaeDataset('train',img_height,str(img_height)), batch_size=200, shuffle=True,
                              pin_memory=True, drop_last=True)
    val_dataloader = DataLoader(dataset=VaeDataset('valid',img_height, str(img_height)), batch_size=1, shuffle=False,
                            pin_memory=True, drop_last=True)
    test_dataloader = DataLoader(dataset=VaeTestDataset(img_height, str(img_height)), batch_size=1, shuffle=False,
                            pin_memory=True, drop_last=True)
    
    

    # Get Z_trained
    vae_flag = True if model_type == 'vae' else False
    z_trained = get_ztrain(train_dataloader, model_attr_accessor, vae_flag, device)
    
    #####################################################################################
    # Random Normal Sample for RAE or AE, then for random sample, have to check test fit
    ######################################################################################
    if model_type == 'rae' or model_type == 'ae':
        print('Running RAE Random Sample from Normal')
        
        # Sample same sample size as training data
        for i, x_y in enumerate(train_dataloader):
            x, y = x_y
            batch_size = len(x)
            rndmN_ = rae_rndm_smpl(z_trained, model_attr_accessor, sample_size=batch_size, device=device)
            
            if i == 0:
                cars_randN_img = rndmN_
            else:
                cars_randN_img = np.vstack([cars_randN_img, rndmN_])
                
        # Save random normal sampled images
        for j, img in enumerate(cars_randN_img):
            img = img.squeeze()
            imwrite(os.path.join(cars_rndmN_path, '%08d.png' % j), img)
    
    #################################################################################
    # Compute GMM-10
    #################################################################################
    gmm_trained_model = gmm_rndmn_train(z_trained=z_trained, n_dist=10, device=device)
    
    print('Running GMM Model')
    
    # Sample same sample size as training data
    for i, x_y in enumerate(train_dataloader):
        x, y = x_y
        batch_size = len(x)
        gmm_sample_ = gmm_rndm_smpl(model_attr_accessor, batch_size, gmm_trained_model, device)
        if i == 0:
            cars_gmm_img = gmm_sample_
        else:
            cars_gmm_img = np.vstack([cars_gmm_img, gmm_sample_])
    
    
    # Save GMM-10 generated images
    print('Saving GMM Sample Generated')
    for j, img in enumerate(cars_gmm_img):
        img = img.squeeze()
        imwrite(os.path.join(cars_gmm_path, '%08d.png' % j), img)
    

    # For test recon
    for i, x_y in enumerate(test_dataloader):
        x, y = x_y
        x = x.to(device)
        
        # Test Recon
        if vae_flag:
            ztest, _, __ = model_attr_accessor.encoder(x.float())
        else:
            ztest = model_attr_accessor.encoder(x.float())
        test_recon_img = model_attr_accessor.decoder(ztest)
        test_recon_img = test_recon_img[0].cpu().detach().permute(1, 2, 0).numpy().squeeze()
        
        # Save test recon
        imwrite(os.path.join(cars_recon_path, '%08d.png' % i), test_recon_img)
            
        # This is for vae/wae random normal samples
        if model_type == 'vae' or model_type == 'wae':
            randnorm_tensor = torch.from_numpy(np.random.normal(loc=0.0, scale=1.0, size=(1, list(mod_input_shape)[1]))).to(torch.float32).to(device)
            cars_randN_img = model_attr_accessor.decoder(randnorm_tensor)[0].cpu().detach().permute(1, 2, 0).numpy().squeeze()
            
            imwrite(os.path.join(cars_rndmN_path, '%08d.png' % i), cars_randN_img)
            
        print(i, ' is done')

            
    fid_score_cars_rndmN = get_fid(cars_org_path, cars_rndmN_path)
    fid_score_cars_recon = get_fid(cars_org_path, cars_recon_path)
    fid_score_cars_gmm = get_fid(cars_org_path, cars_gmm_path)
    
    print('FID Score for Test Reconstruction for Cars is {}'.format(fid_score_cars_recon))
    print('RNDMN FID Score for Cars is: {}'.format(fid_score_cars_rndmN))
    print('GMM FID Score for Cars is: {}'.format(fid_score_cars_rndmN))
    
    
    return fid_score_cars_recon, fid_score_cars_rndmN, fid_score_cars_gmm

if __name__ == '__main__':
    
    print('Running FID Computations')
    
    
    ###############################################################################################
    ## Fashion MNIST
    ###############################################################################################

    fid_fmnist_dict = {}
    fid_fmnist_recon_lst = []
    fid_fmnist_rndmN_lst =[]
    fid_fmnist_name_lst = []
    fid_fmnist_gmm_lst = []

    fmnist_dir = './model_files/fmnist_models/'
    #### RUNS#####
    fid_score_fmnist_recon,  fid_score_fmnist_rndmN, fid_score_fashionmnist_gmm = fmnist_compute_fid_main(fmnist_dir+'rae_fmnistspec_norm_84lat32_batchsize64_8_rerun.pth', 32, 'rae','rae_fmnist','cuda', SN = True)
    fid_fmnist_name_lst.append('RAE-SN'); fid_fmnist_recon_lst.append(fid_score_fmnist_recon); fid_fmnist_rndmN_lst.append(fid_score_fmnist_rndmN); fid_fmnist_gmm_lst.append(fid_score_fashionmnist_gmm)
    
    fid_score_fmnist_recon,  fid_score_fmnist_rndmN, fid_score_fashionmnist_gmm = fmnist_compute_fid_main(fmnist_dir+'rae_fmnistl2_84lat32_batchsize64aa_final.pth', 32, 'rae','rae_fmnist','cuda')
    fid_fmnist_name_lst.append('RAE-L2'); fid_fmnist_recon_lst.append(fid_score_fmnist_recon); fid_fmnist_rndmN_lst.append(fid_score_fmnist_rndmN); fid_fmnist_gmm_lst.append(fid_score_fashionmnist_gmm)
    
    fid_score_fmnist_recon,  fid_score_fmnist_rndmN, fid_score_fashionmnist_gmm = fmnist_compute_fid_main(fmnist_dir+'rae_fmnistgrad_pen_76lat32_batchsize64ab_final.pth', 32, 'rae','rae_fmnist','cuda')
    fid_fmnist_name_lst.append('RAE-GP'); fid_fmnist_recon_lst.append(fid_score_fmnist_recon); fid_fmnist_rndmN_lst.append(fid_score_fmnist_rndmN); fid_fmnist_gmm_lst.append(fid_score_fashionmnist_gmm)
    
    fid_score_fmnist_recon,  fid_score_fmnist_rndmN, fid_score_fashionmnist_gmm = fmnist_compute_fid_main(fmnist_dir+'rae_fmnistl2_84lat32_batchsize64_AE.pth', 32, 'ae', 'ae_fmnist','cuda')
    fid_fmnist_name_lst.append('Autoencoder'); fid_fmnist_recon_lst.append(fid_score_fmnist_recon); fid_fmnist_rndmN_lst.append(fid_score_fmnist_rndmN); fid_fmnist_gmm_lst.append(fid_score_fashionmnist_gmm)
    
    fid_score_fmnist_recon,  fid_score_fmnist_rndmN, fid_score_fashionmnist_gmm = fmnist_compute_fid_main(fmnist_dir+'vae_fmnist_84try_7_rerun.pth', 32, 'vae','vae_fmnist', 'cuda')
    fid_fmnist_name_lst.append('VAE'); fid_fmnist_recon_lst.append(fid_score_fmnist_recon); fid_fmnist_rndmN_lst.append(fid_score_fmnist_rndmN); fid_fmnist_gmm_lst.append(fid_score_fashionmnist_gmm)
    
    fid_score_fmnist_recon,  fid_score_fmnist_rndmN, fid_score_fashionmnist_gmm = fmnist_compute_fid_main(fmnist_dir+'wae_fmnist_84try_6.pth', 32, 'wae','wae_fmnist','cuda')
    fid_fmnist_name_lst.append('WAE'); fid_fmnist_recon_lst.append(fid_score_fmnist_recon); fid_fmnist_rndmN_lst.append(fid_score_fmnist_rndmN); fid_fmnist_gmm_lst.append(fid_score_fashionmnist_gmm)
  
    
    fid_fmnist_dict['model_name'] = fid_fmnist_name_lst
    fid_fmnist_dict['fid_recon_test'] = fid_fmnist_recon_lst
    fid_fmnist_dict['fid_rndmN'] = fid_fmnist_rndmN_lst
    fid_fmnist_dict['fid_gmm'] = fid_fmnist_gmm_lst
    
    
    #Export to csv file
    pd.DataFrame(fid_fmnist_dict).to_csv('fid_fmnist.csv', index = False)
    
    
    #################################################################################################
    ## Stanford Cars Dataset
    #################################################################################################
    fid_cars_dict = {}
    fid_cars_recon_lst = []
    fid_cars_rndmN_lst =[]
    fid_cars_name_lst = []
    fid_cars_gmm_lst = []
    
    cars_dir = './model_files/cars_models/'
    
    ###################### NEW STUFF BELOW ##################################
    fid_score_cars_recon,  fid_score_cars_rndmN, fid_score_cars_gmm = cars_compute_fid_main(cars_dir+'rae_carsl2_99paper_msvae_2layers_enc_v2_512ab_final.pth', 512,  'rae', 'rae_cars', 64, 'cuda')
    fid_cars_name_lst.append('RAE_L2'); fid_cars_recon_lst.append(fid_score_cars_recon); fid_cars_rndmN_lst.append(fid_score_cars_rndmN); fid_cars_gmm_lst.append(fid_score_cars_gmm)
    
    fid_score_cars_recon,  fid_score_cars_rndmN, fid_score_cars_gmm = cars_compute_fid_main(cars_dir+'rae_carsgrad_pen_99paper_msvae_2layers_enc_v2_512ab_final.pth', 512,  'rae', 'rae_cars', 64, 'cuda')
    fid_cars_name_lst.append('RAE_GP'); fid_cars_recon_lst.append(fid_score_cars_recon); fid_cars_rndmN_lst.append(fid_score_cars_rndmN); fid_cars_gmm_lst.append(fid_score_cars_gmm)
    
    fid_score_cars_recon,  fid_score_cars_rndmN, fid_score_cars_gmm = cars_compute_fid_main(cars_dir+'rae_carsspec_norm_99paper_msvae_2layers_enc_v2_512aa_final.pth', 512,  'rae', 'rae_cars', 64, 'cuda', SN = True)
    fid_cars_name_lst.append('RAE_SN'); fid_cars_recon_lst.append(fid_score_cars_recon); fid_cars_rndmN_lst.append(fid_score_cars_rndmN); fid_cars_gmm_lst.append(fid_score_cars_gmm)
    
    fid_score_cars_recon,  fid_score_cars_rndmN, fid_score_cars_gmm = cars_compute_fid_main(cars_dir+'vae_cars_99latent_512_try5.pth', 512,  'vae', 'vae_cars', 64, 'cuda')
    fid_cars_name_lst.append('VAE'); fid_cars_recon_lst.append(fid_score_cars_recon); fid_cars_rndmN_lst.append(fid_score_cars_rndmN); fid_cars_gmm_lst.append(fid_score_cars_gmm)
  
    fid_score_cars_recon,  fid_score_cars_rndmN, fid_score_cars_gmm = cars_compute_fid_main(cars_dir+'wae_cars_99config_8.pth', 512,  'wae', 'wae_cars', 64, 'cuda')
    fid_cars_name_lst.append('WAE'); fid_cars_recon_lst.append(fid_score_cars_recon); fid_cars_rndmN_lst.append(fid_score_cars_rndmN); fid_cars_gmm_lst.append(fid_score_cars_gmm)
    
    fid_score_cars_recon,  fid_score_cars_rndmN, fid_score_cars_gmm = cars_compute_fid_main(cars_dir+'rae_carsl2_99paper_msvae_2layers_enc_v2_512ab_ae.pth', 512,  'ae', 'ae_cars', 64, 'cuda')
    fid_cars_name_lst.append('AE'); fid_cars_recon_lst.append(fid_score_cars_recon); fid_cars_rndmN_lst.append(fid_score_cars_rndmN); fid_cars_gmm_lst.append(fid_score_cars_gmm)
    
    
    
    fid_cars_dict['model_name'] = fid_cars_name_lst
    fid_cars_dict['fid_recon_test'] = fid_cars_recon_lst
    fid_cars_dict['fid_rndmN'] = fid_cars_rndmN_lst
    fid_cars_dict['fid_gmm'] = fid_cars_gmm_lst
    
    #Export to csv file
    pd.DataFrame(fid_cars_dict).to_csv('fid_cars.csv', index = False)
    


    