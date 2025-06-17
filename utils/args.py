import argparse

def SR_parser():
    parser = argparse.ArgumentParser(description='PyTorch SR')
    
    # Define argument of data loader
    parser.add_argument('--dataset', default='dataset1', type=str,
                        help='Type for Train/Test data', choices=['dataset1', 'dataset2'])
    parser.add_argument('--lr_degradation', default='RealBESRGAN', type=str,
                        help='Type for degradating HR image', choices=['None', 'Classic', 'BSRGAN', 'RealESRGAN', 'RealBESRGAN']) 
    parser.add_argument('--hr_enhancing', default='UnsharpMasking', type=str,
                        help='Type for Enhancing HR Image', choices=['None', 'LaplacianSharpening', 'UnsharpMasking', 'HighBoostFiltering']) 
    parser.add_argument('--model_architecture', default='ESRGAN', type=str,
                        help='Type for Train/Test data', choices=['ESRNet', 'ESRGAN', 'SR3'])  ###
    parser.add_argument('--resize', default=512, type=int,
                        help='resize input image data') 
    parser.add_argument('--crop-resize', default=32, type=int,
                        help='random resize crop input image data')
    parser.add_argument('--train-batch-size', default=4, type=int,
                        help='Batch size of train data')
    parser.add_argument('--test-batch-size', default=1, type=int,
                        help='Batch size of test data')

    # seed, gpu, workers
    parser.add_argument('--seed', default=1, type=int,
                        help='fix seed number')
    parser.add_argument('--gpu', default='cuda', type=str,
                        help='Use cuda or not')
    parser.add_argument('--num-workers', default=8, type=int,
                        help='The number of cpu core')
       
    # optimzer parameters
    parser.add_argument('--learning-rate', default=0.001, type=float,
                        help='initial learning rate')
    parser.add_argument('--weight-decay', default=5e-5, type=float,
                        help='weight decay')

    # training hyper parameters
    parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                        help='number of epochs to train (default: auto)')

    parser.add_argument('--debug', action='store_true',
                        help='Enables debug mode')
    parser.add_argument('--template', default='.',
                        help='You can set various templates in option.py')

    # Data specifications
    parser.add_argument('--scale', default=4,
                        help='super resolution scale')
    parser.add_argument('--patch_size', type=int, default=192,
                        help='output patch size')
    parser.add_argument('--rgb_range', type=int, default=255,
                        help='maximum value of RGB')
    parser.add_argument('--n_colors', type=int, default=1,
                        help='number of color channels to use')
    parser.add_argument('--noise', type=str, default='.',
                        help='Gaussian noise std.')
    parser.add_argument('--chop', action='store_true',
                        help='enable memory-efficient forward')

    # Model specifications
    parser.add_argument('--model', default='san',
                        help='model name')

    parser.add_argument('--act', type=str, default='relu',
                        help='activation function')
    parser.add_argument('--pre_train', type=str, default='.',
                        help='pre-trained model directory')
    parser.add_argument('--extend', type=str, default='.',
                        help='pre-trained model directory')
    parser.add_argument('--n_resblocks', type=int, default=4,
                        help='number of residual blocks')
    parser.add_argument('--n_feats', type=int, default=32,
                        help='number of feature maps')
    parser.add_argument('--res_scale', type=float, default=1,
                        help='residual scaling')
    parser.add_argument('--shift_mean', default=True,
                        help='subtract pixel mean from the input')
    parser.add_argument('--precision', type=str, default='single',
                        choices=('single', 'half'),
                        help='FP precision for test (single | half)')

    # options for residual group and feature channel reduction
    parser.add_argument('--n_resgroups', type=int, default=8,
                        help='number of residual groups')
    parser.add_argument('--reduction', type=int, default=16,
                        help='number of feature maps reduction')

    args = parser.parse_args()
    # template.set_template(args)

    # args.scale = list(map(lambda x: int(x), args.scale.split('+')))

    if args.epochs == 0:
        args.epochs = 1e8

    for arg in vars(args):
        if vars(args)[arg] == 'True':
            vars(args)[arg] = True
        elif vars(args)[arg] == 'False':
            vars(args)[arg] = False
    
    return parser