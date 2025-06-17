import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import json
import torch
import warnings
import argparse
import numpy as np
from datetime import datetime
from utils.args import SR_parser
from utils.mypath import mypath

from train_esrnet import ESRNetTrainer
from train_esrgan import ESRGANTrainer

warnings.filterwarnings(action='ignore')


""" Main """
def main(args: argparse):

    ### Basic Path ###
    save_path = mypath(args=args)

    ### Random ###
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    ### Save Argparse ###
    with open(os.path.join(save_path, 'arg_parser.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    ### Training ###
    if args.model_architecture == 'ESRNet':
        SRTrain = ESRNetTrainer(args=args, check_path=save_path)
    elif args.model_architecture == 'ESRGAN':
        SRTrain = ESRGANTrainer(args=args, check_path=save_path)
    else:
        Exception
    SRHistory = SRTrain.run(epochs=args.epochs)

""" __name__ == '__main__' """
if __name__ == '__main__':
    parser = SR_parser()
    args = parser.parse_args()
    args.exp_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    if args.lr_degradation in ['None']:
        args.train_need_lr = True
    elif args.lr_degradation in ['Classic', 'BSRGAN', 'RealESRGAN', 'RealBESRGAN']:
        args.train_need_lr = False
    
    main(args)