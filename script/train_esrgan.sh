#!/bin/bash

for ENHANCING in 'None' 'LaplacianSharpening' 'UnsharpMasking' 'HighBoostFiltering'
    do
        for DEGRADATION in 'None' 'Classic' 'BSRGAN' 'RealESRGAN' 'RealBESRGAN'
            do
                CUDA_VISIBLE_DEVICES='1' python main.py \
                    --dataset 'dataset1' \
                    --lr_degradation ${DEGRADATION} \
                    --hr_enhancing ${ENHANCING} \
                    --model_architecture 'ESRGAN' \
                    --seed '1'
            done
    done