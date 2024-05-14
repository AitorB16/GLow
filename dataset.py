import numpy as np
import pandas as pd
import torch
import sys
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision import transforms, utils
import os
from itertools import chain
import torchvision.transforms
import torchvision.datasets as torch_datasets
#import ssl


def get_cifar10(data_path: str = "../datasets"):
    """Downlaod MNIST and apply a simple transform."""
    #ssl._create_default_https_context = ssl._create_unverified_context
    torch_datasets.CIFAR10.url="http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
   
    transform_train = transforms.Compose(
        [transforms.Resize((32,32)),  #resises the image so it can be perfect for our model.
        transforms.RandomHorizontalFlip(), # FLips the image w.r.t horizontal axis
        transforms.RandomRotation(10),     #Rotates the image to a specified angel
        transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)), #Performs actions like zooms, change shear angles.
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Set the color params
        transforms.ToTensor(), # comvert the image to tensor so that it can work with torch
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #Normalize all the images
        ])
    
    transform_test = transforms.Compose(
        [transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    trainset = torch_datasets.CIFAR10(data_path, train=True, download=True, transform=transform_train)
    testset = torch_datasets.CIFAR10(data_path, train=False, download=True, transform=transform_test)

    return trainset, testset


def prepare_dataset(num_clients: int, clients_with_no_data: list[int], last_conneceted_client: int, batch_size: int, seed, val_ratio: float = 0.1):
    """Load CIFAR-10 (training and test set)."""
    trainset, testset = get_cifar10()

    if (num_clients > 1):

        num_images = len(trainset) // num_clients
        partition_len = [num_images] * num_clients
        partition_len[last_conneceted_client] += len(trainset) % num_clients #Last client add remaining samples to avoid splitting error
        
        ####TRICK### JUST GIVA AN INSTANCE TO 2 ISLANDS
        for i in range(num_clients):
            if i in clients_with_no_data:
                partition_len[i] = 10
        partition_len[last_conneceted_client] += num_images*len(clients_with_no_data) - 10*len(clients_with_no_data)
        ##########

        trainsets = random_split(
            trainset, partition_len, torch.Generator().manual_seed(seed)
        )

        trainloaders = []
        validationloaders = []
        
        for trainset_ in trainsets:
            num_total = len(trainset_)
            num_val = int(val_ratio * num_total)
            num_train = num_total - num_val

            for_train, for_val = random_split(
                trainset_, [num_train, num_val], torch.Generator().manual_seed(seed)
            )
            trainloaders.append(
                DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=2)
            )
            validationloaders.append(
                DataLoader(for_val, batch_size=batch_size, shuffle=False, num_workers=2)
            )
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    #centralized
    else:
        num_total = len(trainset)
        num_val = int(val_ratio * num_total)
        num_train = num_total - num_val
        for_train, for_val = random_split(
            trainset, [num_train, num_val], torch.Generator().manual_seed(seed)
        )
        trainloaders = DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=2)
        validationloaders = DataLoader(for_val, batch_size=batch_size, shuffle=True, num_workers=2)    
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=2)

    return trainloaders, validationloaders, testloader
