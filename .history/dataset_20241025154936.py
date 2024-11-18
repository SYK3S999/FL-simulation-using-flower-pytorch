import torch
from torch.utils.data import random_split, DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.datasets import MNIST



def get_mnist(data_path: str= './data'):

    tr = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

    trainset = MNIST(root=data_path, train=True, download=True , transform=tr)
    testset = MNIST(root=data_path, train=False, download=True, transform=tr)

    return trainset, testset





def prepare_dataset(num_partition: int,batch_size: int, val_ratio: float = 0.1,):
    ## for the partitioning of the dataset, there are three common ways to do it:
    ## iid, non-iid, and pathological.
    trainset, testset = get_mnist()

    #split the dataset into "num_partitions" trainsets
    num_images = len(trainset) // num_partition

    partition_len = [num_images] * num_partition

    trainsets = random_split(trainset, partition_len,torch.Generator().manual_seed(2023))

    #create data loaders with train+val support
    trainloaders = []
    validationloaders = []
    for trainset_ in trainsets:
        num_total = len(trainset)
        num_val = int(val_ratio * num_total)
        num_train = num_total - num_val
        
        for_train, for_val = random_split(trainset_, [num_train, num_val],torch.Generator().manual_seed(2023))
        
        trainloaders.append(DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=2))
        validationloaders.append(DataLoader(for_val, batch_size=batch_size, shuffle=False, num_workers=2))

    testloaders = DataLoader(testset, batch_size=128)

    return trainloaders, validationloaders, testloaders