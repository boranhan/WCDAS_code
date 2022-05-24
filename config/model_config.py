############### configuration file ###############
import numpy as np

import torchvision.transforms as transforms
import utils.mytransforms as mytransforms

#- Augmentation -#
train_transform = {
            'imagenet': 
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.ToTensor(),
                mytransforms.Lighting(0.1, mytransforms.IMAGENET_PCA['eigval'], mytransforms.IMAGENET_PCA['eigvec']),
                transforms.Normalize(mytransforms.IMAGENET_STATS['mean'], mytransforms.IMAGENET_STATS['std'])
                ]),
            'inat': 
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mytransforms.INAT_STATS['mean'], mytransforms.INAT_STATS['std'])
            ]),
            'cifar100': 
            transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ]),
            'cifar10': 
            transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                ]),
}
test_transform = {
            'imagenet':
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mytransforms.IMAGENET_STATS['mean'], mytransforms.IMAGENET_STATS['std'])
                ]),
            'inat': 
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mytransforms.INAT_STATS['mean'], mytransforms.INAT_STATS['std'])
            ]),
            'places365': 
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mytransforms.PLACES_STATS['mean'], mytransforms.PLACES_STATS['std'])
            ]),
            'cifar100': 
            transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ]),
            'cifar10': 
            transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                ]), 
}

#----------Learning Rate-------------#
cos_lr = lambda lr, T: ( 0.5*lr*np.cos(np.pi*np.arange(T)/T) + 0.5*lr).tolist()
cos_lr_annealing = lambda lr, T: ( 0.5*lr*np.cos(np.pi*np.arange(T)/T*2) + 0.5*lr).tolist()
linear_warmup = lambda lr, T: (lr*np.arange(T)/T+lr/T).tolist()

ResNet50Feature = {
    'arch' : 'ResNet50Feature',
    'batch_size' : 512,
    'lrs' : cos_lr(0.4, 200), 
    'opt_params': {'weight_decay' : 0.0001, 'momentum': 0.9},
}

ResNet50Feature_finetune = {
    'arch' : 'ResNet50Feature',
    'batch_size' : 512,
    'lrs' : cos_lr(0.2, 30), 
    'opt_params': {'weight_decay' : 0.0001, 'momentum': 0.9},
}

ResNet10Feature = {
    'arch' : 'ResNet10Feature',
    'batch_size' : 256,
    'lrs' : cos_lr(0.4, 90), 
    'opt_params': {'weight_decay' : 0.0001, 'momentum': 0.9},
}

ResNet10Feature_finetune = {
    'arch' : 'ResNet10Feature',
    'batch_size' : 256,
    'lrs' : cos_lr(0.2, 30),
    'opt_params': {'weight_decay' : 0.0001, 'momentum': 0.9},
}

ResNet32Feature = {
    'arch' : 'ResNet32Feature',
    'batch_size' : 256,
    'lrs' : cos_lr(0.2, 700), 
    'opt_params': {'weight_decay' : 0.0001, 'momentum': 0.9},
}

ResNet32Feature_finetune = {
    'arch' : 'ResNet32Feature',
    'batch_size' : 256,
    'lrs' : cos_lr(0.005, 30),
    'opt_params': {'weight_decay' : 0.0001, 'momentum': 0.9},
}
