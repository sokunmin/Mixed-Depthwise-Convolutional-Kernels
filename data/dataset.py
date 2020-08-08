import torch
from torchvision import datasets, transforms

import os

from data.tiny_imagenet import TinyImageNet

norm = dict(
    MNIST=transforms.Normalize(
        mean=(0.1307,),
        std=(0.3081,)
    ),
    CIFAR10=transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    ),
    CIFAR100=transforms.Normalize(
        mean=(0.5071, 0.4865, 0.4409),
        std=(0.2673, 0.2564, 0.2762)
    ),
    IMAGENET=transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),
    TINY_IMAGENET=transforms.Normalize(
        mean=(0.480, 0.448, 0.398),
        std=(0.230, 0.227, 0.226)
        # [0.277, 0.269, 0.282]
    )
)


def load_data(args):
    print('Load Dataset :: {}'.format(args.dataset))
    if args.dataset == 'CIFAR10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            norm[args.dataset]
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            norm[args.dataset]
        ])

        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('data', train=True, download=True, transform=transform_train),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )

        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('data', train=False, transform=transform_test),
            batch_size=100,
            shuffle=False,
            num_workers=args.num_workers
        )

    elif args.dataset == 'CIFAR100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            norm[args.dataset]
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            norm[args.dataset]
        ])

        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('data', train=True, download=True, transform=transform_train),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )

        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('data', train=False, transform=transform_test),
            batch_size=100,
            shuffle=False,
            num_workers=args.num_workers
        )

    elif args.dataset == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            norm[args.dataset]
        ])
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=True, download=True, transform=transform),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=False, transform=transform),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )
    elif args.dataset == 'TINY_IMAGENET':
        transform_train = transforms.Compose([
            transforms.RandomCrop(64, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            norm[args.dataset]
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            norm[args.dataset]
        ])
        train_dataset = TinyImageNet(root=args.data, mode="train", transform=transform_train)
        test_dataset = TinyImageNet(root=args.data, mode="val", transform=transform_test)
        # Check class labels
        # print(train_dataset.classes)
        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            num_workers=args.num_workers,
            pin_memory=True,
            sampler=train_sampler
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
    elif args.dataset == 'IMAGENET':
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val')
        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                norm[args.dataset]
            ]))

        # Check class labels
        # print(train_dataset.classes)

        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            num_workers=args.num_workers,
            pin_memory=True,
            sampler=train_sampler
        )

        test_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                norm[args.dataset]
            ])),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
    else:
        raise NotImplementedError

    return train_loader, test_loader
