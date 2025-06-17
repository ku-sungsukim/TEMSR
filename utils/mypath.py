import os
import argparse


def mypath(args: argparse):
    
    """ Define Top Save Path """    
    top_save_path = f'./results/'
    if not os.path.exists(top_save_path):
        os.makedirs(top_save_path)
    
    """ Define Result Save Path """
    result_save_path = f'./results/{args.dataset}/{args.lr_degradation}-{args.hr_enhancing}-{args.model_architecture}/SEED{args.seed}/{args.exp_name}'
    if not os.path.exists(result_save_path):
        os.makedirs(result_save_path)

    return result_save_path