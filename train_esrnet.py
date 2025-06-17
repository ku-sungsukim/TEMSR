import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import cv2
import math
import tqdm
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from IQA_pytorch import SSIM

from SRloader import make_data_loader
from utils.saver import Saver
from utils.logger import get_tqdm_config
from model.rrdnet import RRDBNet
from model.losses.basic_loss import L1Loss
from model.model_utils.img_utils import RGB2Gray
from degradation.degradation_utils import KernelGeneration, Degradation_Methods
from enhancing.unsharpmasking import UnsharpMasking
from enhancing.highboostfiltering import HighBoostFiltering
from enhancing.laplaciansharpening import LaplacianSharpening


""" Super Resolution Trainer """
class ESRNetTrainer(object):
    def __init__(self,
                 args: argparse,
                 check_path: str):
        
        # Argument
        self.args = args
        self.check_path = check_path

        # DataLoader
        self.train_loader, self.test_loader1, self.test_loader2, self.test_loader3, self.test_loader4, self.test_loader5 = make_data_loader(args)
        
        # Saver
        self.saver = Saver(path=check_path)
        
        # Degradation Methods
        self.degradations = Degradation_Methods()
        self.kernelgenerator = KernelGeneration()
        
        # Model
        net_g = RRDBNet(num_in_ch=1, num_out_ch=1, scale=4, num_feat=64, num_block=23, num_grow_ch=32)

        # Optimizer
        self.optimizer_g = torch.optim.Adam(net_g.parameters(),
                                            lr=0.0001,
                                            weight_decay=0,
                                            betas = (0.9, 0.99))
    
        # Criterion
        self.criterion = nn.L1Loss().to(device=args.gpu)

        # Allocate GPU
        self.net_g = net_g.to(device=args.gpu)

        # Loss
        self.cri_pix = L1Loss(loss_weight=1.0, reduction='mean')
        
        
    """ Training & Evaluation Protocol """  
    def run(self, epochs):
        result = pd.DataFrame()
        with tqdm.tqdm(**get_tqdm_config(total=epochs, leave=True, color='red')) as pbar:
            train_best_loss = float('inf')
            for epoch in range(1, epochs+1):
                
                # Training
                train_history = self.train(self.train_loader, current_epoch=epoch)
                epoch_history = {
                    'L1Loss': {
                        'train_loss': train_history.get('train_loss'),
                    },
                }

                # Save Best Model using Train Loss
                train_mse = epoch_history['L1Loss']['train_loss']
                if train_mse < train_best_loss:
                    train_best_loss = train_mse
                    self.saver.checkpoint(f'best_SR_model_train', self.net_g)
                    test_history = self.test(self.test_loader1, self.test_loader2, self.test_loader3, self.test_loader4, self.test_loader5, current_epoch=epoch)
                    
                    ### Save Result
                    test_history['epoch'] = [epoch]
                    test_history['train_loss'] = [train_mse]
                    result = pd.concat([result, pd.DataFrame(test_history)]).reset_index(drop=True)
                    result.to_csv(f'{self.check_path}/result.csv', index=False, encoding='utf-8-sig')
                    
                # Save Model & Testing per 50 Epoch
                if epoch % 50 == 0:                    
                    self.saver.checkpoint(f'best_SR_model_{epoch}', self.net_g)
                    test_history = self.test(self.test_loader1, self.test_loader2, self.test_loader3, self.test_loader4, self.test_loader5, current_epoch=epoch)
                    
                    ### Save Result
                    test_history['epoch'] = [epoch]
                    test_history['train_loss'] = [train_mse]
                    result = pd.concat([result, pd.DataFrame(test_history)]).reset_index(drop=True)
                    result.to_csv(f'{self.check_path}/result.csv', index=False, encoding='utf-8-sig')

                # Save Model in Last Epoch
                if epoch == epochs:                    
                    self.saver.checkpoint(f'best_SR_model_last', self.net_g)

                # Logging
                desc = f" Epoch [{epoch:>04}/{epochs:>04} |"
                for metric_name, metric_dict in epoch_history.items():
                    for k, v in metric_dict.items():
                        desc += f" {k}_{metric_name}: {v:.4f} |"

                pbar.set_description_str(desc)
                pbar.update(1)
                    
        return epoch_history
    
    
    """ Training """     
    def train(self,
              train_data_loader,              
              current_epoch: int):
        
        # Change Status
        self.net_g.train()
        
        # Reset Variable        
        step_per_epoch = len(train_data_loader)
        train_loss = 0.0
        
        # Training
        with tqdm.tqdm(**get_tqdm_config(total=step_per_epoch, leave=False, color='cyan')) as pbar:
            for i, sample in enumerate(train_data_loader):
                inputs = sample['lr'].float().to(device=self.args.gpu)
                outputs = sample['hr'].float().to(device=self.args.gpu)
                
                # Degradating LR Image
                if self.args.lr_degradation == 'None':
                    lr_data = inputs.clone()
                elif self.args.lr_degradation == 'Classic':
                    pre_lr_data = self.kernelgenerator.make_kernel(outputs)
                    lr_data = self.degradations.classical_degradation(pre_lr_data)
                elif self.args.lr_degradation == 'BSRGAN':
                    pre_lr_data = self.kernelgenerator.make_kernel(outputs)
                    lr_data = self.degradations.BSRGAN(pre_lr_data)
                elif self.args.lr_degradation == 'RealESRGAN':
                    pre_lr_data = self.kernelgenerator.make_kernel(outputs)
                    lr_data = self.degradations.RealESRGAN(pre_lr_data)
                elif self.args.lr_degradation == 'RealBESRGAN':
                    pre_lr_data = self.kernelgenerator.make_kernel(outputs)
                    lr_data = self.degradations.RealBESRGAN(pre_lr_data)
                    
                # change to grayscale                
                lr_data = RGB2Gray(lr_data, self.args)
                outputs = RGB2Gray(outputs, self.args)  
                    
                # Enhancing HR Image
                if self.args.hr_enhancing == 'None':
                    l1_gt = outputs
                elif self.args.hr_enhancing == 'UnsharpMasking':
                    outputs_enhanced = UnsharpMasking(outputs)
                    l1_gt = outputs_enhanced
                elif self.args.hr_enhancing == 'HighBoostFiltering':
                    outputs_enhanced = HighBoostFiltering(outputs)
                    l1_gt = outputs_enhanced
                elif self.args.hr_enhancing == 'LaplacianSharpening':
                    outputs_enhanced = LaplacianSharpening(outputs)
                    l1_gt = outputs_enhanced
                    
                # optimize parameters
                self.optimizer_g.zero_grad()
                prediction = self.net_g(lr_data)
                
                loss = self.cri_pix(prediction, l1_gt) # Pixel Loss (L1 loss)
                loss.backward()
                self.optimizer_g.step()
                
                train_loss += loss.item()
                                    
                # Print Log
                desc = f" Batch [{i + 1:>04}/{len(self.train_loader):>04}"
                pbar.set_description_str(desc)
                pbar.update(1)
        
        # Overall Log                
        train_loss /= (i + 1)
        total_train_history = {
            'train_loss': train_loss
        }
                
        return total_train_history


    """ Inference """
    @torch.no_grad()
    def test(self,
             test_data_loader1,
             test_data_loader2,
             test_data_loader3,
             test_data_loader4,
             test_data_loader5,
             current_epoch: int
             ):

        # Change Status
        self.net_g.eval()
        
        # Reset Variable & Model        
        ssim_model = SSIM(channels=self.args.n_colors)
        steps_per_epoch = len(test_data_loader1)
        
        
        """ TestLoader1-Type1 """
        total_psnr1 = 0.0
        total_ssim1 = 0.0
        
        with torch.no_grad():
            with tqdm.tqdm(**get_tqdm_config(total=steps_per_epoch, leave=False, color='yellow')) as pbar:
                for i, sample in enumerate(test_data_loader1):
                    # Load data from dataloader
                    img_name = sample['data_name'][0]
                    inputs = sample['lr'].float().to(device=self.args.gpu)
                    inputs = RGB2Gray(inputs, self.args)
                    outputs = sample['hr'].float().to(device=self.args.gpu)
                    outputs = RGB2Gray(outputs, self.args)
                    
                    # image restoration
                    logits = self.net_g(inputs)              
                    
                    # calculate metric
                    psnr = self.psnr(outputs*255, logits*255)
                    ssim_score = ssim_model(outputs, logits, as_loss=False)
                    total_psnr1 += psnr
                    total_ssim1 += ssim_score
                    
                    # log                                            
                    desc = f" Batch [{i + 1:>04}/{len(self.test_loader1):>04}"
                    pbar.set_description_str(desc)
                    pbar.update(1)

                    # save fig
                    check_data = logits[0].permute(1,2,0).detach().cpu().numpy()
                    check_data = (check_data*255).astype(np.uint8)
                    os.makedirs(f"./save_plt/test_type1/{self.args.dataset}/{self.args.lr_degradation}-{self.args.hr_enhancing}-{self.args.model_architecture}/SEED{self.args.seed}/{self.args.exp_name}/Epoch{current_epoch}", exist_ok=True)
                    cv2.imwrite(f"./save_plt/test_type1/{self.args.dataset}/{self.args.lr_degradation}-{self.args.hr_enhancing}-{self.args.model_architecture}/SEED{self.args.seed}/{self.args.exp_name}/Epoch{current_epoch}/{img_name}_{psnr}_{ssim_score.item()}.png", check_data)

        # overall log
        total_psnr1 /= (i+1)
        total_ssim1 /= (i+1)
        
        
        """ TestLoader-Type2 """
        total_psnr2 = 0.0
        total_ssim2 = 0.0
        
        # Inference                        
        with torch.no_grad():
            with tqdm.tqdm(**get_tqdm_config(total=steps_per_epoch, leave=False, color='yellow')) as pbar:
                for i, sample in enumerate(test_data_loader2):
                    # Load data from dataloader
                    img_name = sample['data_name'][0]
                    inputs = sample['lr'].float().to(device=self.args.gpu)
                    inputs = RGB2Gray(inputs, self.args)
                    outputs = sample['hr'].float().to(device=self.args.gpu)
                    outputs = RGB2Gray(outputs, self.args)
                    
                    # image restoration
                    logits = self.net_g(inputs)              
                    
                    # calculate metric
                    psnr = self.psnr(outputs*255, logits*255)
                    ssim_score = ssim_model(outputs, logits, as_loss=False)
                    total_psnr2 += psnr
                    total_ssim2 += ssim_score
                    
                    # log                                            
                    desc = f" Batch [{i + 1:>04}/{len(self.test_loader2):>04}"
                    pbar.set_description_str(desc)
                    pbar.update(1)

                    # save fig
                    check_data = logits[0].permute(1,2,0).detach().cpu().numpy()
                    check_data = (check_data*255).astype(np.uint8)
                    os.makedirs(f"./save_plt/test_type2/{self.args.dataset}/{self.args.lr_degradation}-{self.args.hr_enhancing}-{self.args.model_architecture}/Seed{self.args.seed}/{self.args.exp_name}/Epoch{current_epoch}", exist_ok=True)
                    cv2.imwrite(f"./save_plt/test_type2/{self.args.dataset}/{self.args.lr_degradation}-{self.args.hr_enhancing}-{self.args.model_architecture}/Seed{self.args.seed}/{self.args.exp_name}/Epoch{current_epoch}/{img_name}_{psnr}_{ssim_score.item()}.png", check_data)

        # overall log
        total_psnr2 /= (i+1)
        total_ssim2 /= (i+1)
        
        
        """ TestLoader3-Type3 """
        total_psnr3 = 0.0
        total_ssim3 = 0.0
        
        with torch.no_grad():
            with tqdm.tqdm(**get_tqdm_config(total=steps_per_epoch, leave=False, color='yellow')) as pbar:
                for i, sample in enumerate(test_data_loader3):
                    # Load data from dataloader
                    img_name = sample['data_name'][0]
                    inputs = sample['lr'].float().to(device=self.args.gpu)
                    inputs = RGB2Gray(inputs, self.args)
                    outputs = sample['hr'].float().to(device=self.args.gpu)
                    outputs = RGB2Gray(outputs, self.args)
                    
                    # image restoration
                    logits = self.net_g(inputs)              
                    
                    # calculate metric
                    psnr = self.psnr(outputs*255, logits*255)
                    ssim_score = ssim_model(outputs, logits, as_loss=False)
                    total_psnr3 += psnr
                    total_ssim3 += ssim_score
                    
                    # log                                            
                    desc = f" Batch [{i + 1:>04}/{len(self.test_loader1):>04}"
                    pbar.set_description_str(desc)
                    pbar.update(1)

                    # save fig
                    check_data = logits[0].permute(1,2,0).detach().cpu().numpy()
                    check_data = (check_data*255).astype(np.uint8)
                    os.makedirs(f"./save_plt/test_type3/{self.args.dataset}/{self.args.lr_degradation}-{self.args.hr_enhancing}-{self.args.model_architecture}/SEED{self.args.seed}/{self.args.exp_name}/Epoch{current_epoch}", exist_ok=True)
                    cv2.imwrite(f"./save_plt/test_type3/{self.args.dataset}/{self.args.lr_degradation}-{self.args.hr_enhancing}-{self.args.model_architecture}/SEED{self.args.seed}/{self.args.exp_name}/Epoch{current_epoch}/{img_name}_{psnr}_{ssim_score.item()}.png", check_data)

        # overall log
        total_psnr3 /= (i+1)
        total_ssim3 /= (i+1)
        
        
        """ TestLoader4-Type4 """
        total_psnr4 = 0.0
        total_ssim4 = 0.0
        
        with torch.no_grad():
            with tqdm.tqdm(**get_tqdm_config(total=steps_per_epoch, leave=False, color='yellow')) as pbar:
                for i, sample in enumerate(test_data_loader4):
                    # Load data from dataloader
                    img_name = sample['data_name'][0]
                    inputs = sample['lr'].float().to(device=self.args.gpu)
                    inputs = RGB2Gray(inputs, self.args)
                    outputs = sample['hr'].float().to(device=self.args.gpu)
                    outputs = RGB2Gray(outputs, self.args)
                    
                    # image restoration
                    logits = self.net_g(inputs)              
                    
                    # calculate metric
                    psnr = self.psnr(outputs*255, logits*255)
                    ssim_score = ssim_model(outputs, logits, as_loss=False)
                    total_psnr4 += psnr
                    total_ssim4 += ssim_score
                    
                    # log                                            
                    desc = f" Batch [{i + 1:>04}/{len(self.test_loader1):>04}"
                    pbar.set_description_str(desc)
                    pbar.update(1)

                    # save fig
                    check_data = logits[0].permute(1,2,0).detach().cpu().numpy()
                    check_data = (check_data*255).astype(np.uint8)
                    os.makedirs(f"./save_plt/test_type4/{self.args.dataset}/{self.args.lr_degradation}-{self.args.hr_enhancing}-{self.args.model_architecture}/SEED{self.args.seed}/{self.args.exp_name}/Epoch{current_epoch}", exist_ok=True)
                    cv2.imwrite(f"./save_plt/test_type4/{self.args.dataset}/{self.args.lr_degradation}-{self.args.hr_enhancing}-{self.args.model_architecture}/SEED{self.args.seed}/{self.args.exp_name}/Epoch{current_epoch}/{img_name}_{psnr}_{ssim_score.item()}.png", check_data)

        # overall log
        total_psnr4 /= (i+1)
        total_ssim4 /= (i+1)
        
        
        """ TestLoader5-Real """
        with torch.no_grad():
            with tqdm.tqdm(**get_tqdm_config(total=steps_per_epoch, leave=False, color='yellow')) as pbar:
                for i, sample in enumerate(test_data_loader5):
                    # Load data from dataloader
                    img_name = sample['data_name'][0]
                    inputs = sample['lr'].float().to(device=self.args.gpu)
                    inputs = RGB2Gray(inputs, self.args)
                    outputs = sample['hr'].float().to(device=self.args.gpu)
                    outputs = RGB2Gray(outputs, self.args)
                    
                    # image restoration
                    logits = self.net_g(inputs)              
                    
                    # log                                            
                    desc = f" Batch [{i + 1:>04}/{len(self.test_loader5):>04}"
                    pbar.set_description_str(desc)
                    pbar.update(1)

                    # save fig
                    check_data = logits[0].permute(1,2,0).detach().cpu().numpy()
                    check_data = (check_data*255).astype(np.uint8)
                    os.makedirs(f"./save_plt/test_type5/{self.args.dataset}/{self.args.lr_degradation}-{self.args.hr_enhancing}-{self.args.model_architecture}/SEED{self.args.seed}/{self.args.exp_name}/Epoch{current_epoch}", exist_ok=True)
                    cv2.imwrite(f"./save_plt/test_type5/{self.args.dataset}/{self.args.lr_degradation}-{self.args.hr_enhancing}-{self.args.model_architecture}/SEED{self.args.seed}/{self.args.exp_name}/Epoch{current_epoch}/{img_name}.png", check_data)
        
        print("\n *** Test Results ***")
        print(f"   - [Type1]     PSNR = {round(total_psnr1,3):.3f},     SSIM = {round(total_ssim1.item(),3):.3f}")
        print(f"   - [Type2]     PSNR = {round(total_psnr2,3):.3f},     SSIM = {round(total_ssim2.item(),3):.3f}")
        print(f"   - [Type3]     PSNR = {round(total_psnr3,3):.3f},     SSIM = {round(total_ssim3.item(),3):.3f}")
        print(f"   - [Type4]     PSNR = {round(total_psnr4,3):.3f},     SSIM = {round(total_ssim4.item(),3):.3f}")
        
        total_test_history = {
            'Type1_Degradation_psnr': [total_psnr1],
            'Type1_Degradation_ssim': [total_ssim1.item()],
            'Type2_Degradation_psnr': [total_psnr2],
            'Type2_Degradation_ssim': [total_ssim2.item()],
            'Type3_Degradation_psnr': [total_psnr3],
            'Type3_Degradation_ssim': [total_ssim3.item()],
            'Type4_Degradation_psnr': [total_psnr4],
            'Type4_Degradation_ssim': [total_ssim4.item()],
        }

        return total_test_history


    """ Calculate PSNR """
    def psnr(self, label, outputs, max_val=255.):
        """
        Compute Peak Signal to Noise Ratio (the higher the better).
        PSNR = 20 * log10(MAXp) - 10 *log10(MSE)
        PSNR: label and outputs must be in [0,255]
        SSIM: label and outputs must be in [0,1]
        """
        label = label.cpu().detach().numpy()
        outputs = outputs.cpu().detach().numpy()
        img_diff = outputs - label
        rmse = math.sqrt(np.mean((img_diff)**2))
        
        if rmse == 0:
            return 100
        else:
            PSNR = 20 * math.log10(max_val/rmse)
            return PSNR