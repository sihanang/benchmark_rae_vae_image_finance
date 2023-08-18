# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 18:54:50 2023

@author: angsi

RAE and VAE Losses functions and classes

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

#from .ssim import ssim
#from .percep_loss import VGG19

'''
Classes and Functions for RAE losses
Adapted from https://github.com/tomguluson92/Regularized-AutoEncoder/tree/master/utils/losses
'''
def get_loss_from_name(name):
    if name == "l1":
        return L1LossWrapper()
    elif name == 'l2':
        return L2LossWrapper()
    else:
        return L2LossWrapper()


class TotalLoss(nn.Module):
    def __init__(self,
                 apply_grad_pen=False,
                 grad_pen_weight=None,
                 entropy_qz=None,
                 regularization_loss=None,
                 lamb=1e-7,
                 beta=1e-4, 
                 loss='l2'):
        super(TotalLoss, self).__init__()

        # Get the losses
        self.loss = get_loss_from_name(loss)
        self.embed_loss = EmbeddingLoss()
        self.grad_loss = GradPenLoss()

        # Extra parameters
        self.apply_grad_pen = apply_grad_pen
        self.grad_pen_weight = grad_pen_weight
        self.entropy_qz = entropy_qz
        self.regularization_loss = regularization_loss
        self.beta = beta
        self.lamb = lamb
        # if torch.cuda.is_available():
        #     return loss.cuda()

    def forward(self, pred_img, gt_img, embedding, parameters):

        # print('prediction size', pred_img.shape)
        # print('img size', gt_img.shape)
        loss = self.loss(pred_img, gt_img).mean(dim=[1, 2])
        # print('loss shape', loss.shape)
        # print('embedding shape', embedding.shape)
        loss += self.beta * self.embed_loss(embedding)

        if self.apply_grad_pen:
            loss += self.grad_pen_weight * self.grad_loss(self.entropy_qz, embedding, pred_img)
        if self.entropy_qz is not None:
            loss -= self.beta * self.entropy_qz
        if self.regularization_loss is not None:
            loss += self.regularization_loss(self.lamb, parameters)

        return loss.mean()


# Wrapper of the L1Loss so that the format matches what is expected

class L1LossWrapper(nn.Module):
    def __init__(self):
        super(L1LossWrapper, self).__init__()

    def forward(self, pred_img, gt_img):
        return torch.mean(torch.abs(pred_img - gt_img), dim=1)


class L2LossWrapper(nn.Module):
    def __init__(self):
        super(L2LossWrapper, self).__init__()

    def forward(self, pred_img, gt_img):
        return torch.mean((pred_img - gt_img) ** 2, dim=1)


class EmbeddingLoss(nn.Module):
    def forward(self, embedding):
        return (embedding ** 2).mean(dim=1)


class GradPenLoss(nn.Module):
    def forward(self, entropy_qz, embedding, y_pred):
        if entropy_qz is not None:
            return torch.mean((entropy_qz * torch.autograd.grad(y_pred ** 2,
                                                                embedding, retain_graph=True, grad_outputs=torch.ones_like(y_pred)
                                                                )) ** 2)  # No batch shape is there so mean accross everything is ok
            
        else:
            return torch.mean((torch.autograd.grad(y_pred ** 2, embedding, retain_graph=True, grad_outputs=torch.ones_like(y_pred), allow_unused=True))[0] ** 2)
    
    
class L2RegLoss(nn.Module):
    def forward(self, lamb, parameters):
        return lamb*sum(torch.linalg.norm(p.flatten(), 2)**2 for p in parameters)
    

'''
VAE losses classes below
'''

class LossKLDiverg(nn.Module):
    def forward(self, mu, log_sigma):
        # print(mu.shape)
        return 0.5*torch.sum(torch.exp(log_sigma*2) + mu.pow(2) -2*log_sigma  - 1, dim = 1)
    
    
class VAETotalLoss(nn.Module):
    '''
    Depends on associated Loss wrapper classes above
    '''
    def __init__(self,
                 kl_weight=1, recon_loss_func=None):
        super(VAETotalLoss, self).__init__()

        self.kl_weight = kl_weight
        self.kl_diverg = LossKLDiverg()
        
        # Initialise kl_loss and recon_loss
        # self.kl_loss = 0
        # self.recon_loss = 0
        
        # By default uses the mse
        if recon_loss_func is None:
            self.lossfunc = L2LossWrapper()
        else:
            self.lossfunc = recon_loss_func
        
    def forward(self, pred_img, gt_img, mu, log_sigma):

        # print('prediction size', pred_img.shape)
        
        # This is by default the MSE implementation
        self.recon_loss = self.lossfunc(pred_img, gt_img).sum(dim=[1, 2])
        
        # Compute KL divergence loss
        self.kl_loss = self.kl_diverg(mu, log_sigma)
        
        #print('mean recon_loss: ', self.recon_loss.mean())
        #print('mean kl_loss: ', self.kl_loss.mean())
        
        # Add the KL divergence loss to reconstruction loss
        loss = self.recon_loss + self.kl_loss * self.kl_weight
        
        return loss.mean()

'''
WAE - MMD losses below
Adatped from https://github.com/schelotto/Wasserstein-AutoEncoders/blob/master/wae_mmd.py
'''
class ImqKernelLoss(nn.Module):
    def forward(self, X: torch.Tensor,
               Y: torch.Tensor,
               h_dim: int):
        
        batch_size = X.size(0)
    
        norms_x = X.pow(2).sum(1, keepdim=True)  # batch_size x 1
        prods_x = torch.mm(X, X.t())  # batch_size x batch_size
        dists_x = norms_x + norms_x.t() - 2 * prods_x
    
        norms_y = Y.pow(2).sum(1, keepdim=True)  # batch_size x 1
        prods_y = torch.mm(Y, Y.t())  # batch_size x batch_size
        dists_y = norms_y + norms_y.t() - 2 * prods_y
    
        dot_prd = torch.mm(X, Y.t())
        dists_c = norms_x + norms_y.t() - 2 * dot_prd
    
        stats = 0
        for scale in [.1, .2, .5, 1., 2., 5., 10.]:
            # Here this is already hardcoded for sigma = 1, assuming normal
            C = 2 * h_dim * 1.0 * scale
            res1 = C / (C + dists_x)
            res1 += C / (C + dists_y)
    
            if torch.cuda.is_available():
                res1 = (1 - torch.eye(batch_size).cuda()) * res1
            else:
                res1 = (1 - torch.eye(batch_size)) * res1
    
            res1 = res1.sum() / (batch_size - 1)
            res2 = C / (C + dists_c)
            res2 = res2.sum() * 2. / (batch_size)
            stats += res1 - res2
    
        return stats

class RbfKenelLoss(nn.Module):
    def forward(self, X: torch.Tensor,
                Y: torch.Tensor,
                h_dim: int):
    
        batch_size = X.size(0)

        norms_x = X.pow(2).sum(1, keepdim=True)  # batch_size x 1
        prods_x = torch.mm(X, X.t())  # batch_size x batch_size
        dists_x = norms_x + norms_x.t() - 2 * prods_x
    
        norms_y = Y.pow(2).sum(1, keepdim=True)  # batch_size x 1
        prods_y = torch.mm(Y, Y.t())  # batch_size x batch_size
        dists_y = norms_y + norms_y.t() - 2 * prods_y
    
        dot_prd = torch.mm(X, Y.t())
        dists_c = norms_x + norms_y.t() - 2 * dot_prd
    
        stats = 0
        for scale in [.1, .2, .5, 1., 2., 5., 10.]:
            C = 2 * h_dim * 1.0 / scale
            res1 = torch.exp(-C * dists_x)
            res1 += torch.exp(-C * dists_y)
    
            if torch.cuda.is_available():
                res1 = (1 - torch.eye(batch_size).cuda()) * res1
            else:
                res1 = (1 - torch.eye(batch_size)) * res1
    
            res1 = res1.sum() / (batch_size - 1)
            res2 = torch.exp(-C * dists_c)
            res2 = res2.sum() * 2. / batch_size
            stats += res1 - res2
    
        return stats
    
def get_loss_from_name_WAE(name):
    if name == "l1":
        return L1LossWrapperWAE()
    elif name == 'l2':
        return L2LossWrapperWAE()
    else:
        return L2LossWrapperWAE()

class L1LossWrapperWAE(nn.Module):
    '''
    Following the loss function in https://github.com/tolstikhin/wae/blob/63515656201eb6e3c3f32f6d38267401ed8ade8f/wae.py
    '''
    def __init__(self):
        super(L1LossWrapperWAE, self).__init__()

    def forward(self, pred_img, gt_img):
        l1loss = torch.sum(torch.abs(pred_img - gt_img), dim=[1,2,3])
        l1loss = 0.02* l1loss.mean()
        return l1loss


class L2LossWrapperWAE(nn.Module):
    '''
    Following loss function in https://github.com/tolstikhin/wae/blob/63515656201eb6e3c3f32f6d38267401ed8ade8f/wae.py
    '''
    def __init__(self):
        super(L2LossWrapperWAE, self).__init__()

    def forward(self, pred_img, gt_img):
        l2loss = torch.sum((pred_img - gt_img) ** 2, dim=[1,2,3])
        l2loss = 0.05 * l2loss.mean()
        return l2loss

    

class WAETotalLoss(nn.Module):
    '''
    Depends on get_loss_from_name function and associated Loss wrapper classes above
    Only IMQ Kernel implemented
    '''
    def __init__(self,device,
                 mmd_weight=1, recon_loss_name = 'l2'):
        super(WAETotalLoss, self).__init__()
        
        # Initialise recon loss and the other losses
        self.recon_loss = get_loss_from_name_WAE(recon_loss_name)
        self.imq = ImqKernelLoss()
        self.mmd_weight = mmd_weight
        # self.rbf = RbfKenelLoss()
        
        self.device = device
        
    
    def forward(self, pred_img, gt_img, embedding, latent_dims, sigma = 1):
        ''' 
        gt_img: original input image
        embedding: z
        latent_dims: latent dimensions of z
        sigma: std deviation
        '''
        
        z_fake = Variable(torch.randn(gt_img.size()[0], latent_dims) * sigma).to(self.device)
        z_real = embedding.to(self.device)
        
        mmd_loss = self.imq(z_real, z_fake, h_dim=latent_dims)
        mmd_loss = mmd_loss # / batch_size, to see if need to divide by batch size
        
        
        recon_loss = self.recon_loss(pred_img, gt_img).mean()
        
        total_loss = recon_loss + mmd_loss*self.mmd_weight
        
        # print(recon_loss.shape)
        # print(mmd_loss.shape)
        # print(total_loss.shape)
        
        # print(mmd_loss)
        
        return total_loss
                
        

    

