import math
import random
import numpy as np

import torch
from torch.nn import functional as F
from torch.utils import data as data

from model.model_utils.diffjpeg import DiffJPEG
from model.model_utils.img_process_utils import filter2D
from model.model_utils.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt, circular_lowpass_kernel, random_mixed_kernels

class KernelGeneration(object):
    def __init__(self):
        super(object, self).__init__()
        self.kernel_range = [2 * v + 1 for v in range(3, 11)]
        """ The First Kernel Generating Parameters """
        self.blur_kernel_size1 = 21
        self.kernel_list1 = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
        self.kernel_prob1 = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
        self.sinc_prob1 = 0.1
        self.blur_sigma1 = [0.2, 3]
        self.betag_range1 = [0.5, 4]
        self.betap_range1 = [1, 2]
        self.kernel_size1 = random.choice(self.kernel_range)
        self.pad_size1 = (21 - self.kernel_size1) // 2
        """ The Second Kernel Generating Parameters """
        self.blur_kernel_size2 = 21
        self.kernel_list2 = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
        self.kernel_prob2 = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
        self.sinc_prob2 = 0.1
        self.blur_sigma2 = [0.2, 1.5]
        self.betag_range2 = [0.5, 4]
        self.betap_range2 = [1, 2]
        self.kernel_size2 = random.choice(self.kernel_range)
        self.pad_size2 = (21 - self.kernel_size2) // 2
        
        
    def make_kernel(self, img_gt):
        if np.random.uniform() < self.sinc_prob1:
            if self.kernel_size1 < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel1 = circular_lowpass_kernel(omega_c, self.kernel_size1, pad_to=False)
        else:
            kernel1 = random_mixed_kernels(
                self.kernel_list1,
                self.kernel_prob1,
                self.kernel_size1,
                self.blur_sigma1,
                self.blur_sigma1, [-math.pi, math.pi],
                self.betag_range1,
                self.betap_range1,
                noise_range=None)
        kernel1 = np.pad(kernel1, ((self.pad_size1, self.pad_size1), (self.pad_size1, self.pad_size1)))
        
        if np.random.uniform() < self.sinc_prob2:
            if self.kernel_size2 < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, self.kernel_size2, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                self.kernel_list2,
                self.kernel_prob2,
                self.kernel_size2,
                self.blur_sigma2,
                self.blur_sigma2, [-math.pi, math.pi],
                self.betag_range2,
                self.betap_range2,
                noise_range=None)
        kernel2 = np.pad(kernel2, ((self.pad_size2, self.pad_size2), (self.pad_size2, self.pad_size2)))

        kernel1 = torch.FloatTensor(kernel1)
        kernel2 = torch.FloatTensor(kernel2)
        return_d = {'gt': img_gt, 'kernel1': kernel1, 'kernel2': kernel2}
        return return_d
        

class Degradation_Methods(object):
    def __init__(self):
        super(Degradation_Methods, self).__init__()
        self.device = 'cuda'
        self.fixed_resize_dict = {
                'ori_h':512,
                'ori_w':512,
                'opt_scale':4
                }
        self.random_resize_dict = {
                'resize_prob':[0.2, 0.7, 0.1],
                'lowerlimit':0.3,
                'upperlimit':1.2
                }
        self.noise_dict = {'gaussian_noise_prob':0.5,
                'gaussian_noise_range':[1, 25],
                'poisson_scale_range':[0.05, 2.5],
                'gray_noise_prob':0.4,
                'jpeg_quality_range':[30, 95]
                }
        
        
    def add_blur(self, img, kernel): 
        out = filter2D(img, kernel)
        return out
    
    
    def add_fixed_resize(self, img, parameter_dict):
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(img, size=(parameter_dict['ori_h'] // parameter_dict['opt_scale'], parameter_dict['ori_w'] // parameter_dict['opt_scale']), mode=mode)
        return out
    
    
    def add_random_resize(self, img, parameter_dict):
        updown_type = random.choices(['up', 'down', 'keep'], parameter_dict['resize_prob'])[0]
        if updown_type == 'up':
            scale = np.random.uniform(1, parameter_dict['upperlimit'])
        elif updown_type == 'down':
            scale = np.random.uniform(parameter_dict['lowerlimit'], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(img, scale_factor=scale, mode=mode)
        return out


    def add_noise(self, img, parameter_dict):
        if np.random.uniform() < parameter_dict['gaussian_noise_prob']:
            out = random_add_gaussian_noise_pt(
                img, 
                sigma_range=parameter_dict['gaussian_noise_range'], 
                clip=True, 
                rounds=False, 
                gray_prob=parameter_dict['gray_noise_prob'])
        else:
            out = random_add_poisson_noise_pt(
                img,
                scale_range=parameter_dict['poisson_scale_range'],
                gray_prob=parameter_dict['gray_noise_prob'],
                clip=True,
                rounds=False)
        return out
        
        
    def add_JPEG_noise(self, img, parameter_dict):
        jpeger = DiffJPEG(differentiable=False).cuda()
        jpeg_p = img.new_zeros(img.size(0)).uniform_(*parameter_dict['jpeg_quality_range'])
        out = torch.clamp(img, 0, 1)  
        out = jpeger(out, quality=jpeg_p)
        return out


    def classical_degradation(self, data):
        img = data['gt'].to(self.device)
        kernel = data['kernel1'].to(self.device)
        
        img = self.add_blur(img, kernel)
        img = self.add_random_resize(img, self.random_resize_dict)
        img = self.add_noise(img, self.noise_dict)
        img = self.add_fixed_resize(img, self.fixed_resize_dict)
        img = self.add_JPEG_noise(img, self.noise_dict)
        return img
    
    
    def BSRGAN(self, data):
        img = data['gt'].to(self.device)
        kernel = data['kernel1'].to(self.device)
        
        shuffle_order = random.sample(range(3), 3)
        for i in shuffle_order:
            if i == 0:
                img = self.add_blur(img, kernel)
            elif i == 1:
                img = self.add_random_resize(img, self.random_resize_dict)
            elif i == 2:
                img = self.add_noise(img, self.noise_dict)
        img = self.add_fixed_resize(img, self.fixed_resize_dict)
        img = self.add_JPEG_noise(img, self.noise_dict)
        return img
    
    
    def RealESRGAN(self, data):
        img = data['gt'].to(self.device)
        kernel1 = data['kernel1'].to(self.device)
        kernel2 = data['kernel2'].to(self.device)
        
        img = self.add_blur(img, kernel1)
        img = self.add_random_resize(img, self.random_resize_dict)
        img = self.add_noise(img, self.noise_dict)
        img = self.add_JPEG_noise(img, self.noise_dict)
        
        img = self.add_blur(img, kernel2)
        img = self.add_fixed_resize(img, self.fixed_resize_dict)
        img = self.add_noise(img, self.noise_dict)
        img = self.add_JPEG_noise(img, self.noise_dict)
        return img
    
    
    def RealBESRGAN(self, data):
        img = data['gt'].to(self.device)
        kernel1 = data['kernel1'].to(self.device)
        kernel2 = data['kernel2'].to(self.device)
        
        shuffle_order1 = random.sample(range(3), 3)
        shuffle_order2 = random.sample(range(3), 3)
        for i in shuffle_order1:
            if i == 0:
                img = self.add_blur(img, kernel1)
            elif i == 1:
                img = self.add_random_resize(img, self.random_resize_dict)
            elif i == 2:
                img = self.add_noise(img, self.noise_dict)
        img = self.add_JPEG_noise(img, self.noise_dict)
        
        for i in shuffle_order2:
            if i == 0:
                img = self.add_blur(img, kernel2)
            elif i == 1:
                img = self.add_fixed_resize(img, self.fixed_resize_dict)
            elif i == 2:
                img = self.add_noise(img, self.noise_dict)
        img = self.add_JPEG_noise(img, self.noise_dict)
        return img