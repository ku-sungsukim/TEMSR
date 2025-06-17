# Comprehensive Analysis and Evaluation of Super-Resolution for Wafer Transmission Electron Microscopy Images

Official PyTorch Implementation of the paper: **Comprehensive Analysis and Evaluation of Super-Resolution for Wafer Transmission Electron Microscopy Images (2025, Under Review)**

Sungsu Kim, Insung Baek, Hansam Cho, Yongwon Jo, Heejoong Roh, Kyunghye Kim, Munki Jo, Jaeung Tae, and Seoung Bum Kim

![image](https://github.com/user-attachments/assets/fbd63d84-c458-40c1-9bc2-eeecbe47b120)

Abstract: *High-resolution wafer transmission electron microscopy (TEM) images are vital for nano-level analysis in semiconductor manufacturing. While super-resolution offers significant potential for obtaining high-resolution images, training models for wafer TEM images remains challenging because of the absence of paired low-resolution and high-resolution data and the presence of wafer-specific noise caused by unstable magnetic fields and scattered electrons.  Moreover, super-resolved images must exhibit well-defined edges because wafer analyses focus on boundary regions. These challenges have limited research on super-resolution for wafer TEM images, despite its importance. This study performs a comprehensive analysis of wafer TEM images based on three stages, essential for super-resolution modeling: ‘practical degradation’, ‘image enhancement’, and ‘super-resolution modeling.’ We evaluate all 48 possible combinations of representative methods from each stage, identifying ‘RealBESRGAN’–‘HighboostFiltering’–‘ESRGAN’ as the best-performing combination. In addition, we conduct a detailed analysis to evaluate the impact of practical degradation, image enhancement and super-resolution architectures on the overall performance of super-resolution modeling. Our findings highlight the significance in addressing wafer TEM-specific challenges and provide a strong foundation for advancing super-resolution modeling of wafer TEM images.*

## Installation
we used python 3.9, pytorch 2.2.1, and torchvision 0.17.1. For specific package details, refer to the requirements.txt file. You can install the packages as follows
```
pip install -r requirements.txt
```

## Dataset Preparation
Training data: data/dataset1/128_512_train/SEED1/hr_512/
 - Put your personal images for training in the directory (only high-resolution images, ex) 512x512)
Testing data: data/dataset1/128_512_test/SEED1/lr_512/
 - Put your personal images for testing in the directory (only low-resolution images, ex) 128x128)
 - You can other folders for testing 
 - You can place a separate dataset for evaluation purposes in a different folder. (ex. lr_bicubic_128, But these folders are not used in our study.)

## Training
You can set practical degradation, image enhancement, super-resolution architecture in python script.
```
sh script/train_esrgan.sh
```
 
