import os
import time
from operator import itemgetter

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.distributed as dist


# From https://github.com/catalyst-team/catalyst/blob/ea3fadbaa6034dabeefbbb53ab8c310186f6e5d0/catalyst/data/sampler.py#L522
class DatasetFromSampler(torch.utils.data.Dataset):
    """Dataset to create indexes from `Sampler`.
    Args:
        sampler: PyTorch sampler
    """

    def __init__(self, sampler):
        """Initialisation for DatasetFromSampler."""
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index):
        """Gets element of the dataset.
        Args:
            index: index of the element in the dataset
        Returns:
            Single element by index
        """
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self):
        """
        Returns:
            int: length of the dataset
        """
        return len(self.sampler)


class DistributedSamplerWrapper(torch.utils.data.distributed.DistributedSampler):
    """
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.
    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.
    .. note::
        Sampler is assumed to be of constant size.
    """

    def __init__(
        self,
        sampler,
        num_replicas=None,
        rank=None,
        shuffle=True,
    ):
        """
        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
              distributed training
            rank (int, optional): Rank of the current process
              within ``num_replicas``
            shuffle (bool, optional): If true (default),
              sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
        self.sampler = sampler

    def __iter__(self):
        """@TODO: Docs. Contribution is welcome."""
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))


def get_dataloaders(args):
    train_loader, val_loader, test_loader = None, None, None
    if args.dataset == 'cifar10':
        normalize = transforms.Normalize(mean=[0.4914, 0.4824, 0.4467],
                                         std=[0.2471, 0.2435, 0.2616])
        train_set = datasets.CIFAR10(args.data_root, train=True,
                                     transform=transforms.Compose([
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        normalize
                                     ]))
        val_set = datasets.CIFAR10(args.data_root, train=False,
                                   transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    normalize
                                   ]))
    elif args.dataset == 'cifar100':
        normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                         std=[0.2675, 0.2565, 0.2761])
        train_set = datasets.CIFAR100(args.data_root, train=True, download=True,
                                      transform=transforms.Compose([
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        normalize
                                      ]))
        val_set = datasets.CIFAR100(args.data_root, train=False, download=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        normalize
                                    ]))
    elif args.dataset == 'imagenet':
        traindir = os.path.join(args.data_root, 'train')
        # traindir = os.path.join(args.data_root, 'train_subset')
        valdir = os.path.join(args.data_root, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_set = datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]))
        val_set = datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ]))
    else:
        raise Exception('Invalid dataset name')

    if args.use_valid:
        if os.path.exists(os.path.join(args.result_dir, 'index.pth')):
            # print('!!!!!! Load train_set_index !!!!!!')
            time.sleep(30)
            train_set_index = torch.load(os.path.join(args.result_dir, 'index.pth'))
        else:
            if not args.distributed or dist.get_rank() == 0:
                train_set_index = torch.randperm(len(train_set))
                torch.save(train_set_index, os.path.join(args.result_dir, 'index.pth'))
            # print('!!!!!! Save train_set_index !!!!!!')
            time.sleep(30)
            train_set_index = torch.load(os.path.join(args.result_dir, 'index.pth'))

        if args.dataset.startswith('cifar'):
            num_sample_valid = 5000
        else:
            num_sample_valid = 50000

        if 'train' in args.splits:
            train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_set_index[:-num_sample_valid])
            if args.distributed:
                train_sampler  = DistributedSamplerWrapper(train_sampler, shuffle=True)
            train_loader = torch.utils.data.DataLoader(
                train_set,
                batch_size=args.batch_size,
                sampler=train_sampler,
                num_workers=args.workers,
                pin_memory=True)
        if 'val' in args.splits:
            val_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_set_index[-num_sample_valid:])
            if args.distributed:
                val_sampler  = DistributedSamplerWrapper(val_sampler, shuffle=False)
            val_loader = torch.utils.data.DataLoader(
                train_set, batch_size=args.batch_size,
                sampler=val_sampler,
                num_workers=args.val_workers,
                pin_memory=True)
        if 'test' in args.splits:
            if args.distributed:
                test_sampler = torch.utils.data.distributed.DistributedSampler(val_set)
                additional_args = {'shuffle': False, 'sampler': test_sampler}
            else:
                additional_args = {'shuffle': False}
            test_loader = torch.utils.data.DataLoader(
                val_set,
                batch_size=args.batch_size,
                num_workers=args.val_workers,
                pin_memory=True,
                **additional_args)
    else:
        if 'train' in args.splits:
            if args.distributed:
                train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
                additional_args = {'shuffle': False, 'sampler': train_sampler}
            else:
                additional_args = {'shuffle': True}
            train_loader = torch.utils.data.DataLoader(
                train_set,
                batch_size=args.batch_size,
                num_workers=args.workers,
                pin_memory=True,
                **additional_args)
        if 'val' in args.splits:
            if args.distributed:
                val_sampler = torch.utils.data.distributed.DistributedSampler(val_set)
                additional_args = {'shuffle': False, 'sampler': val_sampler}
            else:
                additional_args = {'shuffle': False}
            val_loader = torch.utils.data.DataLoader(
                val_set,
                batch_size=args.batch_size,
                num_workers=args.val_workers,
                pin_memory=True,
                **additional_args)
            test_loader = val_loader

    return train_loader, val_loader, test_loader
