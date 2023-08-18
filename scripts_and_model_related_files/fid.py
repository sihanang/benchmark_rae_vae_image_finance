"""
Code adapted from: https://github.com/mseitzer/pytorch-fid/blob/3d604a25516746c3a4a5548c8610e99010b2c819/src/pytorch_fid/fid_score.py
which is python library

Calculates the Frechet Inception Distance (FID) to evalulate GANs

The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.

When run as a stand-alone program, it compares the distribution of
images that are stored as PNG/JPEG at a specified location with a
distribution given by summary statistics (in pickle format).

The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectively.

See --help to see further details.

Code apapted from https://github.com/bioinf-jku/TTUR to use PyTorch instead
of Tensorflow

Copyright 2018 Institute of Bioinformatics, JKU Linz

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
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
import random

from imageio import imwrite
from cars_preprocess import *
from cars_data_gen import *


try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x

from inception import InceptionV3
from model_repo import *

# parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
# parser.add_argument('--batch-size', type=int, default=50,
#                     help='Batch size to use')
# parser.add_argument('--num-workers', type=int,
#                     help=('Number of processes to use for data loading. '
#                           'Defaults to `min(8, num_cpus)`'))
# parser.add_argument('--device', type=str, default=None,
#                     help='Device to use. Like cuda, cuda:0 or cpu')
# parser.add_argument('--dims', type=int, default=2048,
#                     choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
#                     help=('Dimensionality of Inception features to use. '
#                           'By default, uses pool3 features'))
# parser.add_argument('path', type=str, nargs=2,
#                     help=('Paths to the generated images or '
#                           'to .npz statistic files'))

IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp'}


class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img


def get_activations(files, model, batch_size=50, dims=2048, device='cpu',
                    num_workers=1):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers

    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    if batch_size > len(files):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = len(files)

    dataset = ImagePathDataset(files, transforms=TF.ToTensor())
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=num_workers)

    pred_arr = np.empty((len(files), dims))

    start_idx = 0

    for batch in tqdm(dataloader):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()

        pred_arr[start_idx:start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(files, model, batch_size=50, dims=2048,
                                    device='cpu', num_workers=1):
    """Calculation of the statistics used by the FID.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers

    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations(files, model, batch_size, dims, device, num_workers)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def compute_statistics_of_path(path, model, batch_size, dims, device,
                               num_workers=1):
    if path.endswith('.npz'):
        with np.load(path) as f:
            m, s = f['mu'][:], f['sigma'][:]
    else:
        path = pathlib.Path(path)
        files = sorted([file for ext in IMAGE_EXTENSIONS
                       for file in path.glob('*.{}'.format(ext))])
        m, s = calculate_activation_statistics(files, model, batch_size,
                                               dims, device, num_workers)

    return m, s


def calculate_fid_given_paths(paths, batch_size, device, dims, num_workers=1):
    """Calculates the FID of two paths"""
    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError('Invalid path: %s' % p)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx]).to(device)

    m1, s1 = compute_statistics_of_path(paths[0], model, batch_size,
                                        dims, device, num_workers)
    m2, s2 = compute_statistics_of_path(paths[1], model, batch_size,
                                        dims, device, num_workers)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value


def get_fid(path1, path2):
    '''
    Wrapper function to set the default values for batch_size, device, dims and num_workers
    '''
    batch_size = 50
    device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    dims = 2048
    num_workers = 1
    
    return calculate_fid_given_paths([path1,path2], batch_size, device, dims, num_workers)

def initialise_model(model_name, latent_dims, SN):
    '''
    Initialises the autoencoder models based on the model_name, it should be having
    model type (i.e. ae, rae, vae ) followed by _ and the dataset name
    '''
    
    initial_model = None 
    
    # To select the appropriate model class to initiatise
    if (model_name == 'rae_fmnist') or (model_name == 'ae_fmnist') or (model_name == 'wae_fmnist'):
        initial_model = RegularisedAutoencoder(latent_dims, SN)
    
    elif model_name == 'vae_fmnist':
        initial_model = VAEFmnist(latent_dims)
        
    elif (model_name == 'rae_cars') or (model_name == 'ae_cars') or (model_name == 'wae_cars'):
        if SN == False:
            initial_model = CarsRegularisedAutoencoder(latent_dims)
        else:
            initial_model = SNCarsRegularisedAutoencoder(latent_dims)
        
    elif model_name == 'vae_cars':
        initial_model = CarsVae(latent_dims)
    
    else:
        print('Not Implemented')
    
    return initial_model


def load_model(model_name, saved_model_path, latent_dims, SN):
    
    # TO Design a way to initialise the models accordingly
    # Maybe I should load all the scripts (import) with model functions then have a way to serialise them
    # like a dictionary or if else; can call to initialise them 
    
    model = initialise_model(model_name, latent_dims, SN)
    checkpoint = torch.load(saved_model_path)
    
    print('loading model success')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print('loaded model')
    
    model.eval()
    
    return model


def save_generated_images(path, images):
    '''
    Adapted from https://github.com/ParthaEth/Regularized_autoencoders-RAE-/blob/9478b8f781f7229807a0d7c4ea92a7c9c7994bfa/my_utility/save_batches_of_images.py#L6
    '''
    if not os.path.exists(path):
        os.mkdir(path)

    images = (np.clip(images.cpu().detach().numpy(), 0, 1) * 255).astype('uint8')
    

    for i, img in enumerate(images):
        img = img.squeeze()
        # print(img)
        imwrite(os.path.join(path, '%08d.png' % i), img)
        
        
def save_test_images(path, dataset_choice):
    
    if not os.path.exists(path):
        os.mkdir(path)
        not_exist = True
    else:
        not_exist = False
    
    # KIV TO ADD THE LOGIC FOR OTHER ONES
    if dataset_choice == 'fmnist':
        transform_proc = transforms.Compose([transforms.Pad(2), transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])
        test_data = torchvision.datasets.FashionMNIST('../data/image_data', 
                                                      train = False,
                                                      download = True, transform = transform_proc)
        test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)
        
        
        i = 0
        for test_features, test_labels in test_dataloader:
            img = test_features[0].squeeze()
            # label = test_labels.numpy()[0]
            img = (np.clip(img.detach().numpy(),0,1) *255).astype('uint8')
            
            imwrite(os.path.join(path, '%08d.png' % i), img)
            i+= 1
        
    elif dataset_choice == 'cars':
        # print(not_exist)
        # print(len(os.listdir(path)))
        if not_exist or len(os.listdir(path)) < 8041:
            # if path doesn't exist or if test data has a smaller size than what is expected, reload the data
            transform_proc = transforms.Compose([transforms.ToTensor(), lambda x: x*255])
            
            try:
                train_val_data = torchvision.datasets.StanfordCars('../data/image_data', 
                                                    split  = 'train',
                                                    download = True, transform = transform_proc)
    
                test_data = torchvision.datasets.StanfordCars('../data/image_data', 
                                                    split  = 'test',
                                                    download = True, transform = transform_proc)
            except:
                print('Pytorch datasets Stanford Cars connection is still not working! \nMake sure to upload the files cars_train.tgz, cars_test.tgz and car_devkit.tgz to ../data/image_data/stanford_cars/ folder. They can be found on Kaggle datasets')
            
            
            img_width, img_height = 64,64
            print('Extracting data/cars_train.tgz...')
            if not os.path.exists('../data/image_data/stanford_cars/cars_train'):
                with tarfile.open('../data/image_data/stanford_cars/cars_train.tgz', "r:gz") as tar:
                    tar.extractall('data')
            print('Extracting data/image_data/stanford_cars/cars_test.tgz...')
            if not os.path.exists('../data/image_data/stanford_cars/cars_test'):
                with tarfile.open('../data/image_data/stanford_cars/cars_test.tgz', "r:gz") as tar:
                    tar.extractall('data')
            print('Extracting data/car_devkit.tgz...')
            if not os.path.exists('../data/image_data/stanford_cars/devkit'):
                with tarfile.open('../data/image_data/stanford_cars/car_devkit.tgz', "r:gz") as tar:
                    tar.extractall('data')

            reset_folder('../data/image_data/stanford_cars/train{}'.format(img_height))
            reset_folder('../data/image_data/stanford_cars/valid{}'.format(img_height))
            reset_folder('../data/image_data/stanford_cars/test{}'.format(img_height))

            ensure_folder('../data/image_data/stanford_cars/train{}'.format(img_height))
            ensure_folder('../data/image_data/stanford_cars/valid{}'.format(img_height))
            ensure_folder('../data/image_data/stanford_cars/test{}'.format(img_height))

            
            random.seed(1)
            np.random.seed(1)
            torch.manual_seed(1)
            torch.cuda.manual_seed_all(1)

            process_data('train', img_height, img_width, str(img_height))
            process_data('test', img_height, img_width,str(img_height))

        
        else:
            print('Cars Test Data Already Available.')
    

    else:
        print('Currently not implemented')

    

# def main():
#     args = parser.parse_args()

#     if args.device is None:
#         device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
#     else:
#         device = torch.device(args.device)

#     if args.num_workers is None:
#         num_avail_cpus = len(os.sched_getaffinity(0))
#         num_workers = min(num_avail_cpus, 8)
#     else:
#         num_workers = args.num_workers

#     fid_value = calculate_fid_given_paths(args.path,
#                                           args.batch_size,
#                                           device,
#                                           args.dims,
#                                           num_workers)
#     print('FID: ', fid_value)



    