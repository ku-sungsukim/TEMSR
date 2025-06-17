import argparse 
import albumentations as A

from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from SRdata import SRdataset


""" Data Loader """
def make_data_loader(args: argparse):

    """ Data Augmentation """
    augment_transform=A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5)],
        additional_targets={'image':'image', 'image2':'image'},
        is_check_shapes=False
    )

    """ Data Transformation """
    input_transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize((int(args.resize/args.scale), int(args.resize/args.scale)))]
    )

    output_transform = transforms.Compose(
        [transforms.ToTensor()]
    )

    """ DataLoader """
    train_set = SRdataset(root=f'data/{args.dataset}/128_512_train/SEED{args.seed}/',
                          degradation_type=args.lr_degradation,
                          need_lr=args.train_need_lr,
                          seed=args.seed,
                          augmentation=augment_transform, # augment_transform
                          augmentation2=None,
                          transform_input=input_transform,
                          transform_output=output_transform)

    train_loader = DataLoader(train_set,
                              batch_size=args.train_batch_size,
                              shuffle=True,
                              num_workers=args.num_workers
                              )

    test_set1 = SRdataset(root=f'data/{args.dataset}/128_512_test/SEED{args.seed}/',
                        degradation_type='test_type1',
                        need_lr=True,
                        seed=args.seed,
                        augmentation=None,
                        augmentation2=None,
                        transform_input=input_transform,
                        transform_output=output_transform)

    test_loader1 = DataLoader(test_set1,
                            batch_size=args.test_batch_size,
                            shuffle=False,
                            num_workers=args.num_workers
                            )
    
    test_set2 = SRdataset(root=f'data/{args.dataset}/128_512_test/SEED{args.seed}/',
                        degradation_type='test_type2',
                        need_lr=True,
                        seed=args.seed,
                        augmentation=None,
                        augmentation2=None,
                        transform_input=input_transform,
                        transform_output=output_transform)

    test_loader2 = DataLoader(test_set2,
                            batch_size=args.test_batch_size,
                            shuffle=False,
                            num_workers=args.num_workers
                            )
    
    test_set3 = SRdataset(root=f'data/{args.dataset}/128_512_test/SEED{args.seed}/',
                        degradation_type='test_type3',
                        need_lr=True,
                        seed=args.seed,
                        augmentation=None,
                        augmentation2=None,
                        transform_input=input_transform,
                        transform_output=output_transform)

    test_loader3 = DataLoader(test_set3,
                            batch_size=args.test_batch_size,
                            shuffle=False,
                            num_workers=args.num_workers
                            )
    
    test_set4 = SRdataset(root=f'data/{args.dataset}/128_512_test/SEED{args.seed}/',
                        degradation_type='test_type4',
                        need_lr=True,
                        seed=args.seed,
                        augmentation=None,
                        augmentation2=None,
                        transform_input=input_transform,
                        transform_output=output_transform)

    test_loader4 = DataLoader(test_set4,
                            batch_size=args.test_batch_size,
                            shuffle=False,
                            num_workers=args.num_workers
                            )
    
    test_set5 = SRdataset(root=f'data/{args.dataset}/128_512_test/SEED{args.seed}/',
                        degradation_type='human_labeling',
                        need_lr=True,
                        seed=args.seed,
                        augmentation=None,
                        augmentation2=None,
                        transform_input=input_transform,
                        transform_output=output_transform)

    test_loader5 = DataLoader(test_set5,
                            batch_size=args.test_batch_size,
                            shuffle=False,
                            num_workers=args.num_workers
                            )

    print(f'Model = {args.lr_degradation}-{args.hr_enhancing}-{args.model_architecture}')
    print(f'Seed = {args.seed}')
    print(f'Train data number =', len(train_set))
    print(f'Test data number =', len(test_set1))

    return train_loader, test_loader1, test_loader2, test_loader3, test_loader4, test_loader5

