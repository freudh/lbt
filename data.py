import os

import torchvision.datasets as datasets
import torchvision.transforms as transforms


def get_dataset(args):
    if args.data == 'cifar10':
        normalize = transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010],
        )
        train_dataset = datasets.CIFAR10(
            root=os.path.expanduser('~/data'),
            train=True,
            transform=transforms.Compose([
                transforms.RandomCrop(size=32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]),
            download=True,
        )
        val_dataset = datasets.CIFAR10(
            root=os.path.expanduser('~/data'),
            train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]),
            download=True,
        )
    elif args.data == 'mnist':
        normalize = transforms.Normalize(
            (0.1307,), # mean
            (0.3081,), # std
        )
        train_dataset = datasets.MNIST(
            root=os.path.expanduser('~/data'),
            train=True,
            transform=transforms.Compose([
                # transforms.RandomCrop(size=28, padding=4),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]),
            download=True,
        )
        val_dataset = datasets.MNIST(
            root=os.path.expanduser('~/data'),
            train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]),
            download=True,
        )
    elif args.data == 'imagenet':
        if args.arch == 'inception_v3':
            crop_size, resize_size = 299, 340
            normalize = transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            )
        else:
            crop_size, resize_size = 224, 256
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        train_dataset = datasets.ImageFolder(
            args.train_dir,
            transforms.Compose([
                transforms.RandomResizedCrop(crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.4,
                ),
                transforms.ToTensor(),
                normalize,
            ])
        )
        val_dataset = datasets.ImageFolder(
            args.val_dir,
            transforms.Compose([
                transforms.Resize(resize_size),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                normalize,
            ])
        )
    else:
        assert False, 'Invalid dataset %s' % args.data

    return train_dataset, val_dataset
