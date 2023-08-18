"""
This script contains the model classes
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
from spectral_norm_layers import *

import torchvision
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms

from sklearn.model_selection import train_test_split
import argparse
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
from sklearn import preprocessing

from losses import *



''' For Fashion Mnist'''
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
        
        
        # self.sn_conv_layers = nn.Sequential(
            
        #     spectral_norm(nn.Conv2d(1, out_channels = 128, kernel_size = (4,4), stride = (2,2), padding = 1)),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),
        #     # nn.LeakyReLU(0.1),
        #     spectral_norm(nn.Conv2d(128, out_channels = 256, kernel_size = (4,4), stride = (2,2), padding =1)),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),
        #     # nn.LeakyReLU(0.1),
        #     spectral_norm(nn.Conv2d(256, out_channels = 512, kernel_size = (4,4), stride = (2,2), padding = 1)),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(),
        #     # nn.LeakyReLU(0.1),
        #     spectral_norm(nn.Conv2d(512, out_channels = 1024, kernel_size = (4,4), stride = (2,2), padding = 1)),
        #     nn.BatchNorm2d(1024),
        #     nn.ReLU()
        #     # nn.LeakyReLU(0.1)
        # )
        
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
        
        
        # self.sn_linear = spectral_norm(nn.Linear(4096, latent_dims))
        
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
        
        # self.sn_conv_layers = nn.Sequential(
            
        #     spectral_norm(nn.ConvTranspose2d(1024, out_channels = 512, kernel_size = (4,4), stride = (2,2), padding = (1,1))),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(),
        #     # nn.LeakyReLU(0.1),
        #     spectral_norm(nn.ConvTranspose2d(512, out_channels = 256, kernel_size = (4,4), stride = (2,2), padding = (1,1))),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),
        #     # nn.LeakyReLU(0.1)
        #     spectral_norm(nn.ConvTranspose2d(256, out_channels = 1, kernel_size = (5,5), stride = (1,1), padding = 2))
        # )
        
        
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
        
        # self.sn_fc_layer = nn.Sequential(
        #     spectral_norm(nn.Linear(latent_dims, 8*8*1024))
        # )
        
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
    def __init__(self, latent_dims=128, SN = False):
        super(RegularisedAutoencoder, self).__init__()
        self.encoder = RegularisedEncoder(latent_dims, SN)
        self.decoder = Decoder(latent_dims, SN)
            

    def forward(self, x):
        self.z = self.encoder(x)
        return self.decoder(self.z)
    
##############################################################################################
## For VAE Fmnist
##############################################################################################
class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)
    

class VaeEncoderFmnist(nn.Module):
    def __init__(self, latent_dims, constant_sigma = None):
        super(VaeEncoderFmnist, self).__init__()
        
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
    

class VAEDecoderFmnist(nn.Module):
    def __init__(self, latent_dims):
        super(VAEDecoderFmnist, self).__init__()
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
    
class VAEFmnist(nn.Module):
    def __init__(self, latent_dims=128, constant_sigma = None):
        super(VAEFmnist, self).__init__()
        self.encoder = VaeEncoderFmnist(latent_dims, constant_sigma)
        self.decoder = VAEDecoderFmnist(latent_dims)

    def forward(self, x):
        self.z, mu, log_sigma = self.encoder(x)
        return self.decoder(self.z), mu, log_sigma

###############################################
''' For Cars below '''
################################################
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

# class encoderConv(nn.Module):
#     def __init__(self, in_size, out_size):
#         super(encoderConv, self).__init__()
#         self.conv1 = conv2DBatchNormRelu(in_size, out_size, 3,2,1)
        
#     def forward(self, inputs):
#         outputs = self.conv1(inputs)
#         return outputs
    
# class dencoderDeconv(nn.Module):
#     def __init__(self, in_size, out_size):
#         super(dencoderDeconv, self).__init__()
#         self.conv1 = conv2DTransBatchNormRelu(in_size, out_size, 3,2,1,1)
        
#     def forward(self, inputs):
#         outputs = self.conv1(inputs)
#         return outputs


class CarsRegularisedEncoder(nn.Module):
    def __init__(self, latent_dims, SN =False):
        super(CarsRegularisedEncoder, self).__init__()
        
  
        ## Try another architecture
        self.conv1_1 =  conv2DBatchNormLeakyRelu(3, 64, 3, 1, 1)
        self.conv1_2 =  conv2DBatchNormLeakyRelu(64, 64, 3, 2, 1)
        self.conv2_1 =  conv2DBatchNormLeakyRelu(64, 128, 3, 1, 1)
        self.conv2_2 =  conv2DBatchNormLeakyRelu(128, 128, 3, 2, 1)
        self.conv3_1 =  conv2DBatchNormLeakyRelu(128, 256, 3, 1, 1)
        self.conv3_2 =  conv2DBatchNormLeakyRelu(256, 256, 3, 2, 1)
        self.conv4_1 =  conv2DBatchNormLeakyRelu(256, 512, 3, 1, 1)
        self.conv4_2 =  conv2DBatchNormLeakyRelu(512, 512, 3, 2, 1)
        
        # self.linear = nn.Linear(8192, latent_dims)
        self.dense = nn.Sequential(nn.Linear(8192, latent_dims))#,
                                   #nn.LeakyReLU(0.3, inplace = True)) #,
                                   #nn.Linear(1024, latent_dims))
        
        self.SN = SN

    def forward(self, x):
 
        
        ## Trying another one
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        
        x = torch.flatten(x, start_dim=1)
        z = self.dense(x)
        
        return z
    

class CarsDecoder(nn.Module):
    def __init__(self, latent_dims, SN = False):
        super(CarsDecoder, self).__init__()
        self.fc_layer = nn.Sequential(
            # nn.Linear(latent_dims, 8*8*1024)
            # nn.Linear(latent_dims, 1024),
            #nn.LeakyReLU(0.3, inplace = True),
            nn.Linear(latent_dims, 8*8*512) # archi 2: 8*8*256 instead of 512
        )
        # self.batch_norm1 = nn.BatchNorm2d(1024)
        self.batch_norm1 = nn.BatchNorm2d(512)
        
        self.lrelu = nn.LeakyReLU(0.3)
   
        
        self.ctbr1 = conv2DTransBatchNormLeakyRelu(512,256,3,2,1,1)
        self.ctbr2 = conv2DTransBatchNormLeakyRelu(256,128,3,2,1,1)
        self.ctbr3 = conv2DTransBatchNormLeakyRelu(128,64,3,2,1,1)
        self.convtransposefinal = nn.ConvTranspose2d(64, out_channels = 3, kernel_size = 3, stride = (1,1), padding = 1)
        
  
        self.SN = SN

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
    
class CarsRegularisedAutoencoder(nn.Module):
    def __init__(self, latent_dims=64, SN = False):
        super(CarsRegularisedAutoencoder, self).__init__()
        self.encoder = CarsRegularisedEncoder(latent_dims, SN)
        self.decoder = CarsDecoder(latent_dims, SN)
            

    def forward(self, x):
        self.z = self.encoder(x)
        return self.decoder(self.z)


### Spectral Norm for Cars
class conv2DSNBatchNormLeakyRelu(nn.Module):
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
        super(conv2DSNBatchNormLeakyRelu, self).__init__()

        conv_mod = SNConv2d(int(in_channels),
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
  

class conv2DTransSNBatchNormLeakyRelu(nn.Module):
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
        super(conv2DTransSNBatchNormLeakyRelu, self).__init__()

        conv_mod = SNConvTranspose2d(int(in_channels),
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


class conv2DTransSNBatchNormSigmoid(nn.Module):
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
        super(conv2DTransSNBatchNormSigmoid, self).__init__()

        conv_mod = SNConvTranspose2d(int(in_channels),
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



class SNCarsRegularisedEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(SNCarsRegularisedEncoder, self).__init__()
        
        ## Try another architecture
        self.conv1_1 =  conv2DSNBatchNormLeakyRelu(3, 64, 3, 1, 1)
        self.conv1_2 =  conv2DSNBatchNormLeakyRelu(64, 64, 3, 2, 1)
        self.conv2_1 =  conv2DSNBatchNormLeakyRelu(64, 128, 3, 1, 1)
        self.conv2_2 =  conv2DSNBatchNormLeakyRelu(128, 128, 3, 2, 1)
        self.conv3_1 =  conv2DSNBatchNormLeakyRelu(128, 256, 3, 1, 1)
        self.conv3_2 =  conv2DSNBatchNormLeakyRelu(256, 256, 3, 2, 1)
        self.conv4_1 =  conv2DSNBatchNormLeakyRelu(256, 512, 3, 1, 1)
        self.conv4_2 =  conv2DSNBatchNormLeakyRelu(512, 512, 3, 2, 1)
        
        # self.linear = nn.Linear(8192, latent_dims)
        self.dense = nn.Sequential(SNLinear(8192, latent_dims))#,
                                   #nn.LeakyReLU(0.3, inplace = True)) #,
                                   #nn.Linear(1024, latent_dims))
        

    def forward(self, x):
        
        ## Trying another one
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        
        x = torch.flatten(x, start_dim=1)
        z = self.dense(x)
        
        return z
    

class SNCarsDecoder(nn.Module):
    def __init__(self, latent_dims):
        super(SNCarsDecoder, self).__init__()
        self.fc_layer = nn.Sequential(
            # nn.Linear(latent_dims, 8*8*1024)
            # nn.Linear(latent_dims, 1024),
            #nn.LeakyReLU(0.3, inplace = True),
            SNLinear(latent_dims, 8*8*512) # archi 2: 8*8*256 instead of 512
        )
        # self.batch_norm1 = nn.BatchNorm2d(1024)
        self.batch_norm1 = nn.BatchNorm2d(512)
        
        self.lrelu = nn.LeakyReLU(0.3)
        
        
        self.ctbr1 = conv2DTransSNBatchNormLeakyRelu(512,256,3,2,1,1)
        self.ctbr2 = conv2DTransSNBatchNormLeakyRelu(256,128,3,2,1,1)
        self.ctbr3 = conv2DTransSNBatchNormLeakyRelu(128,64,3,2,1,1)
        self.convtransposefinal = SNConvTranspose2d(64, out_channels = 3, kernel_size = 3, stride = (1,1), padding = 1)
 

    def forward(self, z):

       
        x = self.fc_layer(z)
        x = x.view(x.size(0),512,8,8)
        x = self.lrelu(self.batch_norm1(x))
        x = self.ctbr1(x)
        x = self.ctbr2(x)
        x = self.ctbr3(x)
        x = self.convtransposefinal(x)
    
        x = F.sigmoid(x)
        return x
    
class SNCarsRegularisedAutoencoder(nn.Module):
    def __init__(self, latent_dims=64):
        super(SNCarsRegularisedAutoencoder, self).__init__()
        self.encoder = SNCarsRegularisedEncoder(latent_dims)
        self.decoder = SNCarsDecoder(latent_dims)
            

    def forward(self, x):
        self.z = self.encoder(x)
        return self.decoder(self.z)
    

#######################################################
''' The following for VAE Cars'''
#######################################################
class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)
    

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
        
        # # For sigma
        # self.conv_layers2 = nn.Sequential(
        #     conv2DBatchNormLeakyRelu(3, 64, 3, 1, 1),
        #     conv2DBatchNormLeakyRelu(64, 64, 3, 2, 1),
        #     conv2DBatchNormLeakyRelu(64, 128, 3, 1, 1),
        #     conv2DBatchNormLeakyRelu(128, 128, 3, 2, 1),
        #     conv2DBatchNormLeakyRelu(128, 256, 3, 1, 1),
        #     conv2DBatchNormLeakyRelu(256, 256, 3, 2, 1),
        #     conv2DBatchNormLeakyRelu(256, 512, 3, 1, 1),
        #     conv2DBatchNormLeakyRelu(512, 512, 3, 2, 1)
        # )
        
        # self.linear = nn.Linear(8192, latent_dims)
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

#################################################################################
##  Model for Finance Dataset 
##################################################################################
class FinRegularisedEncoder(nn.Module):
    def __init__(self, latent_dims, SN =False):
        super(FinRegularisedEncoder, self).__init__()
        
        self.dense = nn.Sequential(
            
            nn.Linear(25, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dims)
            
        )
        

        
        self.sn_dense = nn.Sequential(

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
    

class FinDecoder(nn.Module):
    def __init__(self, latent_dims, SN = False):
        super(FinDecoder, self).__init__()
      
        
        self.dense = nn.Sequential(
            
            nn.Linear(latent_dims, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 25)
        )
        
        self.sn_dense = nn.Sequential(
            
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

            
            x = self.sn_dense(z)
            x = x.view(x.size(0),1,5,5)

        else:

            
            x = self.dense(z)
            x = x.view(x.size(0),1,5,5)
        
        return x
    

class FinDecoderPoint(nn.Module):
    def __init__(self, latent_dims, SN = False):
        super(FinDecoderPoint, self).__init__()
   
        
        self.dense = nn.Sequential(
            
            nn.Linear(latent_dims + 2, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
        self.sn_dense = nn.Sequential(
            
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

            
            x = self.sn_dense(z_expanded)

        else:
            
            x = self.dense(z_expanded)
   
        return x

    
    
class FinRegularisedAutoencoderGrid(nn.Module):
    def __init__(self, latent_dims=2, SN = False):
        '''RAE using Grid approach'''
        super(FinRegularisedAutoencoderGrid, self).__init__()
        self.encoder = FinRegularisedEncoder(latent_dims, SN)
        self.decoder = FinDecoder(latent_dims, SN)
            

    def forward(self, x):
        self.z = self.encoder(x)
        return self.decoder(self.z)
    
    
    
class FinRegularisedAutoencoderPoint(nn.Module):
    '''
    RAE using pointwise approach, returns a point in the volatility surface
    '''
    def __init__(self, latent_dims=2, SN = False):
        super(FinRegularisedAutoencoderPoint, self).__init__()
        self.encoder = FinRegularisedEncoder(latent_dims, SN)
        self.decoder = FinDecoderPoint(latent_dims, SN)
        
            
    def forward(self, x, dayexp, delta):
        self.z = self.encoder(x)
        return self.decoder(self.z, dayexp, delta)

    
############ Model for Finance VAE ####################
class FinVaeEncoder(nn.Module):
    def __init__(self, latent_dims, constant_sigma = None):
        super(FinVaeEncoder, self).__init__()
        
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
    
class FinVaeDecoder(nn.Module):
    def __init__(self, latent_dims):
        super(FinVaeDecoder, self).__init__()
    
        
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
        
        # x = x.view(x.size(0), 4, 5,5)
        # x = self.conv_layers(x)
        
        
        x = x.view(x.size(0),1,5,5)
        
        # x = F.sigmoid(x)
        # print(x.shape)
        return x
    

class FinVaeDecoderPoint(nn.Module):
    def __init__(self, latent_dims):
        super(FinVaeDecoderPoint, self).__init__()



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
  
        
        # x = F.sigmoid(x)
        # print(x.shape)
        return x


class FinVAEGrid(nn.Module):
    def __init__(self, latent_dims=2, constant_sigma = None):
        super(FinVAEGrid, self).__init__()
        self.encoder = FinVaeEncoder(latent_dims, constant_sigma)
        self.decoder = FinVaeDecoder(latent_dims)

    def forward(self, x):
        self.z, mu, log_sigma = self.encoder(x)
        return self.decoder(self.z), mu, log_sigma


class FinVAEPoint(nn.Module):
    def __init__(self, latent_dims=2, constant_sigma = None):
        super(FinVAEPoint, self).__init__()
        self.encoder = FinVaeEncoder(latent_dims, constant_sigma)
        self.decoder = FinVaeDecoderPoint(latent_dims)

    def forward(self, x, dayexp, delta):
        self.z, mu, log_sigma = self.encoder(x)
        return self.decoder(self.z, dayexp, delta), mu, log_sigma

