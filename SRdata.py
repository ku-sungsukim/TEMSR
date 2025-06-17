import os
import cv2
import warnings

from torch.utils.data import Dataset

warnings.filterwarnings("ignore")


""" SRdataset """
class SRdataset(Dataset):
    
    def __init__(self, root, degradation_type, need_lr, seed, augmentation, augmentation2, transform_input, transform_output):
        self.root = root
        self.degradation_type = degradation_type
        self.need_lr = need_lr
        self.seed = seed
        self.augmentation = augmentation
        self.augmentation2 = augmentation2
        self.transform_input = transform_input
        self.transform_output = transform_output
    
    def __len__(self):
        return len(os.listdir(os.path.join(self.root, 'hr_512')))
    
    def __getitem__(self, idx):
        
        """ Define HR Image Path """        
        hr_root = os.path.join(self.root, 'hr_512')
        hr_list = os.listdir(hr_root)
        hr_root = os.path.join(hr_root, hr_list[idx])
        data_name = hr_root.split('/')[-1]            
        
        """ Load HR Image """
        hr_data = cv2.imread(hr_root)            
        # hr_data = cv2.cvtColor(hr_data, cv2.COLOR_BGR2GRAY)
        
        """ Define LR Image Path & Load LR Image """
        if self.need_lr:
            if self.degradation_type == 'None':
                lr_path = 'lr_bicubic_128'
            elif self.degradation_type == 'human_labeling':
                lr_path = 'lr_128'
            elif self.degradation_type == 'test_type1':
                lr_path = 'lr_test_degradation_type1'
            elif self.degradation_type == 'test_type2':
                lr_path = 'lr_test_degradation_type2'
            elif self.degradation_type == 'test_type3':
                lr_path = 'lr_test_degradation_type3'
            elif self.degradation_type == 'test_type4':
                lr_path = 'lr_test_degradation_type4'
            
            lr_root = os.path.join(self.root, lr_path)
            lr_root = os.path.join(lr_root, data_name)
            lr_data = cv2.imread(lr_root)            
            # lr_data = cv2.cvtColor(lr_data, cv2.COLOR_BGR2GRAY)
            if self.augmentation is not None:
                hr_lr_data = self.augmentation(image=hr_data, image2=lr_data)
                hr_data = hr_lr_data['image']
                lr_data = hr_lr_data['image2']
        else:
            lr_data = cv2.imread(hr_root)            
            # lr_data = cv2.cvtColor(lr_data, cv2.COLOR_BGR2GRAY)
            if self.augmentation is not None:
                hr_lr_data = self.augmentation(image=hr_data, image2=hr_data)
                lr_data = hr_lr_data['image']
                hr_data = hr_lr_data['image2']

        if self.transform_output is not None:
            lr_data = self.transform_output(lr_data)
            hr_data = self.transform_output(hr_data)
                
        return {'hr': hr_data,
                'lr': lr_data,
                'data_name': data_name,
                'index': idx}