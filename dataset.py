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


def get_cifar10(data_path: str = "..datasets"):
    """Downlaod CIFAR and apply a simple transform."""
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


def prepare_dataset_iid(num_clients: int, num_classes: int, clients_with_no_data: list[int], batch_size: int, seed: int, val_ratio: float = 0.1):
    """Load CIFAR-10 (training and test set)."""
    trainset, testset = get_cifar10()

    clients_with_data = []
    for i in range(num_clients):
        if i not in clients_with_no_data:
            clients_with_data.append(i)

    # SPLIT DATASET BY CLASSES
    ordered_trainset = []

    for i in range(num_classes):
        tmp_part = []
        for j, data in enumerate(trainset):
            img, label = data
            if label == i:
                tmp_part.append(data)
        ordered_trainset.extend(tmp_part)


    num_images = len(ordered_trainset) // len(clients_with_data)
    num_images_remainder = len(ordered_trainset) % len(clients_with_data)
    
    partition_len = [0] * num_clients
    
    #SPLIT DS ACCORDINGLY
    for i in clients_with_data:
        partition_len[i] = num_images
        if num_images_remainder > 0:
            partition_len[i] += 1
            num_images_remainder -=1
   
    ##########
    trainsets = random_split(
        ordered_trainset, partition_len, torch.Generator().manual_seed(seed)
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
        if num_total > 0:
            trainloaders.append(
                DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=2)
            )
            validationloaders.append(
                DataLoader(for_val, batch_size=batch_size, shuffle=False, num_workers=2)
            )
        else:
            trainloaders.append('')
            validationloaders.append('')

    #TEST SET
    ordered_testset = []

    for i in range(num_classes):
        tmp_part = []
        for j, data in enumerate(testset):
            img, label = data
            if label == i:
                tmp_part.append(data)
        ordered_testset.extend(tmp_part)


    testloader = DataLoader(ordered_testset, batch_size=batch_size, shuffle=True, num_workers=2)
    return trainloaders, validationloaders, testloader, partition_len

def prepare_dataset_niid(num_clients: int, num_classes: int, clients_with_no_data: list[int], batch_size: int, seed: int, val_ratio: float = 0.1):
    """Load CIFAR-10 (training and test set). DIRICHLET"""
    trainset, testset = get_cifar10()

    clients_with_data = []
    for i in range(num_clients):
        if i not in clients_with_no_data:
            clients_with_data.append(i)

    # SPLIT DATASET BY CLASSES
    ordered_trainset = []

    for i in range(num_classes):
        tmp_part = []
        for j, data in enumerate(trainset):
            img, label = data
            if label == i:
                tmp_part.append(data)
        ordered_trainset.extend(tmp_part)

    # SPLIT DIRICHLET DISTRIBUTION
    alpha = 0.4
    np.random.seed(seed=seed)
    dirich = np.random.dirichlet([alpha]*len(clients_with_data))

    #num_images = len(ordered_trainset) // len(clients_with_data)
    #num_images_remainder = len(ordered_trainset) % len(clients_with_data)
    
    partition_len = [0] * num_clients
    total_instances = 0
    j = 0
    
    #SPLIT DS ACCORDINGLY
    for i in clients_with_data:
        partition_len[i] = int(len(ordered_trainset)*dirich[j])
        total_instances += partition_len[i]
        j+=1

    remainder = len(ordered_trainset) - total_instances
    partition_len[clients_with_data[0]] += remainder

    ##########
    trainsets = random_split(
        ordered_trainset, partition_len, torch.Generator().manual_seed(seed)
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
        if num_total > 0:
            trainloaders.append(
                DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=2)
            )
            validationloaders.append(
                DataLoader(for_val, batch_size=batch_size, shuffle=False, num_workers=2)
            )
        else:
            trainloaders.append('')
            validationloaders.append('')

    #TEST SET
    ordered_testset = []

    for i in range(num_classes):
        tmp_part = []
        for j, data in enumerate(testset):
            img, label = data
            if label == i:
                tmp_part.append(data)
        ordered_testset.extend(tmp_part)


    testloader = DataLoader(ordered_testset, batch_size=batch_size, shuffle=True, num_workers=2)
    return trainloaders, validationloaders, testloader, partition_len


def prepare_dataset_niid_class_partition(num_clients: int, num_classes: int, clients_with_no_data: list[int], batch_size: int, seed: int, val_ratio: float = 0.1):
    """Load CIFAR-10 (training and test set)."""
    trainset, testset = get_cifar10()

    #num_images = len(trainset) // num_clients
    clients_with_data = []
    #partition_len = [0] * num_clients
    #num_classes = 10

    for i in range(num_clients):
        if i not in clients_with_no_data:
            clients_with_data.append(i)

    # SPLIT DATASET BY CLASSES
    ordered_trainset = []

    for i in range(num_classes):
        tmp_part = []
        for j, data in enumerate(trainset):
            img, label = data
            if label == i:
                tmp_part.append(data)
        ordered_trainset.append(tmp_part)

    # Smart division
    partition_num_per_agent = num_classes // len(clients_with_data)
    partition_remainder_per_agent = num_classes % len(clients_with_data)

    trainsets = []
    for i in range(num_clients):
        tmp_list = []
        if i in clients_with_data:
            for j in range(partition_num_per_agent):
                tmp_list.extend(ordered_trainset.pop())
            if partition_remainder_per_agent > 0:
                tmp_list.extend(ordered_trainset.pop())
                partition_remainder_per_agent=-1
        trainsets.append(tmp_list)

    
    trainloaders = []
    validationloaders = []
    
    # VALIDATION SET
    for trainset_ in trainsets:
        num_total = len(trainset_)
        num_val = int(val_ratio * num_total)
        num_train = num_total - num_val
        for_train, for_val = random_split(
            trainset_, [num_train, num_val], torch.Generator().manual_seed(seed)
        )
        if num_total > 0:
            trainloaders.append(
                DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=2)
            )
            validationloaders.append(
                DataLoader(for_val, batch_size=batch_size, shuffle=True, num_workers=2)
            )
        else:
            trainloaders.append('')
            validationloaders.append('')

    #Also smart division?
    ordered_testset = []

    for i in range(num_classes):
        tmp_part = []
        for j, data in enumerate(testset):
            img, label = data
            if label == i:
                tmp_part.append(data)
        ordered_testset.extend(tmp_part)


    testloader = DataLoader(ordered_testset, batch_size=batch_size, shuffle=True, num_workers=2)
    return trainloaders, validationloaders, testloader, partition_num_per_agent



def prepare_dataset_cnl(batch_size: int, seed: int, val_ratio: float = 0.1):
    """Load CIFAR-10 (training and test set)."""
    trainset, testset = get_cifar10()
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