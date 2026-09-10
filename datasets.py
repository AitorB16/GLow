import numpy as np
import pandas as pd
import torch
import sys
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils
import os
from itertools import chain
import torchvision.transforms
import torchvision.datasets as torch_datasets


def get_cifar10(data_path: str = ".datasets"):
    """Download CIFAR10 and build three dataset views."""
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
    trainset_eval = torch_datasets.CIFAR10(data_path, train=True, download=False, transform=transform_test)
    testset = torch_datasets.CIFAR10(data_path, train=False, download=True, transform=transform_test)

    return trainset, trainset_eval, testset

def get_mnist(data_path: str = ".datasets"):
    """Downlaod MNIST and apply a simple transform."""
    #ssl._create_default_https_context = ssl._create_unverified_context

    transform = transforms.Compose(
        [transforms.Resize((32,32)),
         transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))
        ])

    trainset = torch_datasets.MNIST(root=data_path, train=True, download=True, transform=transform)
    trainset_eval = torch_datasets.MNIST(data_path, train=True, download=False, transform=transform)
    testset = torch_datasets.MNIST(root=data_path, train=False, download=True, transform=transform)

    return trainset, trainset_eval, testset

def _get_dataset(dataset: str):
    """(trainset, trainset_eval, testset) for `dataset`. Single dispatch point
    for every prepare_* below."""
    if dataset == 'cifar':
        return get_cifar10()
    if dataset == 'mnist':
        return get_mnist()
    raise ValueError(f"Unknown dataset '{dataset}'. Expected 'cifar' or 'mnist'.")


def _clients_with_data(num_clients: int, clients_with_no_data: list[int]) -> list[int]:
    """IDs of clients that receive a data partition (all of them if none are excluded)."""
    if not clients_with_no_data:
        return list(range(num_clients))
    return [i for i in range(num_clients) if i not in clients_with_no_data]


def _partition_indices(index_pool, lengths, seed):
    """Deterministically shuffle `index_pool` (a 1-D np.array) and slice it
    into `len(lengths)` chunks of the given sizes."""
    index_pool = np.asarray(index_pool)
    perm = np.random.default_rng(seed).permutation(len(index_pool))
    shuffled = index_pool[perm]
    parts = []
    start = 0
    for length in lengths:
        parts.append(shuffled[start:start + length])
        start += length
    return parts


def _client_train_val_loaders(indices, trainset, trainset_eval, val_ratio, batch_size, seed):
    """Split `indices` (positions into `trainset`/`trainset_eval`, which
    must be the same underlying data under different transforms) into a
    train/val subset for one client."""
    indices = np.asarray(indices)
    if len(indices) == 0:
        return '', ''
    perm = np.random.default_rng(seed).permutation(len(indices))
    num_val = int(val_ratio * len(indices))
    val_idx = indices[perm[:num_val]]
    train_idx = indices[perm[num_val:]]
    trainloader = DataLoader(
        torch.utils.data.Subset(trainset, train_idx),
        batch_size=batch_size, shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )
    valloader = DataLoader(
        torch.utils.data.Subset(trainset_eval, val_idx),
        batch_size=batch_size, shuffle=False, num_workers=0,
    )
    return trainloader, valloader


def _client_test_loader(indices, testset, batch_size, seed):
    """DataLoader over `testset[indices]`, or `''` if `indices` is empty."""
    indices = np.asarray(indices)
    if len(indices) == 0:
        return ''
    return DataLoader(
        torch.utils.data.Subset(testset, indices),
        batch_size=batch_size, shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )

def prepare_dataset_iid_train_common_test(num_clients: int, num_classes: int, clients_with_no_data: list[int], batch_size: int, seed: int, dataset: str, val_ratio: float = 0.1):
    """IID training data (equal-sized random splits) plus one shared test set
    identical for every client."""
    np.random.seed(seed=seed)
    torch.manual_seed(seed)

    trainset, trainset_eval, testset = _get_dataset(dataset)
    class_client_matrix_train = np.zeros((num_clients, num_classes), dtype=int)
    class_client_matrix_test = np.zeros((num_clients, num_classes), dtype=int)

    clients_with_data = _clients_with_data(num_clients, clients_with_no_data)

    # SPLIT DATASET BY CLASSES
    labels_train = np.array(trainset.targets)
    ordered_train_idx = np.concatenate([np.where(labels_train == i)[0] for i in range(num_classes)])

    num_images = len(ordered_train_idx) // len(clients_with_data)
    num_images_remainder = len(ordered_train_idx) % len(clients_with_data)

    partition_len_train = [0] * num_clients

    #SPLIT DS ACCORDINGLY
    for i in clients_with_data:
        partition_len_train[i] = num_images
        if num_images_remainder > 0:
            partition_len_train[i] += 1
            num_images_remainder -=1

    ##########
    client_train_indices = _partition_indices(ordered_train_idx, partition_len_train, seed)

    trainloaders = []
    validationloaders = []

    for client_id in range(num_clients):
        trainloader, valloader = _client_train_val_loaders(
            client_train_indices[client_id], trainset, trainset_eval, val_ratio, batch_size, seed
        )
        trainloaders.append(trainloader)
        validationloaders.append(valloader)
        class_client_matrix_train[client_id] = np.bincount(
            labels_train[client_train_indices[client_id]], minlength=num_classes
        )

    #TEST SET
    labels_test = np.array(testset.targets)
    ordered_test_idx = np.concatenate([np.where(labels_test == i)[0] for i in range(num_classes)])

    testloader = _client_test_loader(ordered_test_idx, testset, batch_size, seed)
    testloaders = [testloader] * num_clients

    test_counts = np.bincount(labels_test, minlength=num_classes)
    for client_id in range(num_clients):
        class_client_matrix_test[client_id] = test_counts

    return trainloaders, validationloaders, testloaders, class_client_matrix_train, class_client_matrix_test

def prepare_dataset_niid_train_common_test(num_clients: int, num_classes: int, clients_with_no_data: list[int], batch_size: int, seed: int, dataset: str,  val_ratio: float = 0.1):
    """"Coarse" Dirichlet-skewed training data: one Dirichlet draw (fixed
    8-length `alpha`, i.e. sized for an 8-client topology) sizes a contiguous
    slice of the class-sorted train set per client. Test set is shared for
    every client."""
    np.random.seed(seed=seed)
    torch.manual_seed(seed)

    trainset, trainset_eval, testset = _get_dataset(dataset)
    class_client_matrix_train = np.zeros((num_clients, num_classes), dtype=int)
    class_client_matrix_test = np.zeros((num_clients, num_classes), dtype=int)

    clients_with_data = _clients_with_data(num_clients, clients_with_no_data)

    # SPLIT DATASET BY CLASSES
    labels_train = np.array(trainset.targets)
    ordered_train_idx = np.concatenate([np.where(labels_train == i)[0] for i in range(num_classes)])

    # SPLIT DIRICHLET DISTRIBUTION
    alpha = [20., 1., 1., 2., 2., 1., 1., 20. ]
    dirich = np.random.dirichlet(alpha)

    partition_len_train = [0] * num_clients
    total_instances = 0
    j = 0

    #SPLIT DS ACCORDINGLY
    for i in clients_with_data:
        partition_len_train[i] = int(len(ordered_train_idx)*dirich[j])
        total_instances += partition_len_train[i]
        j+=1

    remainder = len(ordered_train_idx) - total_instances
    partition_len_train[clients_with_data[0]] += remainder

    ##########
    client_train_indices = _partition_indices(ordered_train_idx, partition_len_train, seed)

    trainloaders = []
    validationloaders = []

    for client_id in range(num_clients):
        trainloader, valloader = _client_train_val_loaders(
            client_train_indices[client_id], trainset, trainset_eval, val_ratio, batch_size, seed
        )
        trainloaders.append(trainloader)
        validationloaders.append(valloader)
        class_client_matrix_train[client_id] = np.bincount(
            labels_train[client_train_indices[client_id]], minlength=num_classes
        )

    #TEST SET
    labels_test = np.array(testset.targets)
    ordered_test_idx = np.concatenate([np.where(labels_test == i)[0] for i in range(num_classes)])

    testloader = _client_test_loader(ordered_test_idx, testset, batch_size, seed)
    testloaders = [testloader] * num_clients

    test_counts = np.bincount(labels_test, minlength=num_classes)
    for client_id in range(num_clients):
        class_client_matrix_test[client_id] = test_counts

    return trainloaders, validationloaders, testloaders, class_client_matrix_train, class_client_matrix_test

def skew_class_niid_train_common_test(num_clients: int, num_classes: int, clients_with_no_data: list[int], batch_size: int, seed: int, dataset: str,  val_ratio: float = 0.1):
    """"Fine" per-class Dirichlet skew (`alpha<0.1`, strongly skewed): each
    class is independently Dirichlet-split across `clients_with_data`, giving
    direct per-client per-class control -- the standard FL class-skew
    partitioning. Test set is shared for every client."""

    alpha = 0.1
    np.random.seed(seed=seed)
    torch.manual_seed(seed)

    trainset, trainset_eval, testset = _get_dataset(dataset)
    labels_train = np.array(trainset.targets)
    labels_test = np.array(testset.targets)
    client_indices = [[] for _ in range(num_clients)]

    class_client_matrix_train = np.zeros((num_clients, num_classes), dtype=int)
    class_client_matrix_test = np.zeros((num_clients, num_classes), dtype=int)

    clients_with_data = _clients_with_data(num_clients, clients_with_no_data)

    for c in range(num_classes):

        idx = np.where(labels_train == c)[0]
        np.random.shuffle(idx)

        proportions = np.random.dirichlet(alpha * np.ones(len(clients_with_data)))

        split_points = (np.cumsum(proportions) * len(idx)).astype(int)[:-1]
        class_split = np.split(idx, split_points)

        for i, client_id in enumerate(clients_with_data):
            client_indices[client_id].extend(class_split[i])

    trainloaders = []
    validationloaders = []

    for client_id in range(num_clients):
        trainloader, valloader = _client_train_val_loaders(
            client_indices[client_id], trainset, trainset_eval, val_ratio, batch_size, seed
        )
        trainloaders.append(trainloader)
        validationloaders.append(valloader)

    #TEST SET
    testloader = _client_test_loader(np.arange(len(testset)), testset, batch_size, seed)
    testloaders = [testloader] * num_clients

    for client_id in range(num_clients):
        client_labels_train = labels_train[client_indices[client_id]]
        client_labels_test = labels_test
        for c in range(num_classes):
            class_client_matrix_train[client_id, c] = np.sum(client_labels_train == c)
            class_client_matrix_test[client_id, c] = np.sum(client_labels_test == c)

    return trainloaders, validationloaders, testloaders, class_client_matrix_train, class_client_matrix_test


def skew_class_niid_train_niid_test(num_clients: int, num_classes: int, clients_with_no_data: list[int], batch_size: int, seed: int, dataset: str,  val_ratio: float = 0.1):
    """Same per-class Dirichlet skew as `skew_class_niid_train_common_test`,
    but the same per-class proportions (`dirichlet_props`, drawn once from
    the train split) are reapplied to independently partition the test set --
    each client's test set mirrors its own train set's class distribution
    without sharing samples with anyone else."""

    alpha = 0.1
    np.random.seed(seed=seed)
    torch.manual_seed(seed)

    trainset, trainset_eval, testset = _get_dataset(dataset)
    labels_train = np.array(trainset.targets)
    labels_test = np.array(testset.targets)
    client_indices = [[] for _ in range(num_clients)]

    class_client_matrix_train = np.zeros((num_clients, num_classes), dtype=int)
    class_client_matrix_test = np.zeros((num_clients, num_classes), dtype=int)

    clients_with_data = _clients_with_data(num_clients, clients_with_no_data)

    dirichlet_props = np.zeros((num_classes, len(clients_with_data)))
    for c in range(num_classes):

        idx = np.where(labels_train == c)[0]
        np.random.shuffle(idx)

        proportions = np.random.dirichlet(alpha * np.ones(len(clients_with_data)))
        dirichlet_props[c] = proportions

        split_points = (np.cumsum(proportions) * len(idx)).astype(int)[:-1]
        class_split = np.split(idx, split_points)

        for i, client_id in enumerate(clients_with_data):
            client_indices[client_id].extend(class_split[i])

    trainloaders = []
    validationloaders = []

    for client_id in range(num_clients):
        trainloader, valloader = _client_train_val_loaders(
            client_indices[client_id], trainset, trainset_eval, val_ratio, batch_size, seed
        )
        trainloaders.append(trainloader)
        validationloaders.append(valloader)

    # TEST SET
    test_client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        idx = np.where(labels_test == c)[0]
        np.random.shuffle(idx)

        proportions = dirichlet_props[c]

        split_points = (np.cumsum(proportions) * len(idx)).astype(int)[:-1]
        class_split = np.split(idx, split_points)

        for i, client_id in enumerate(clients_with_data):
            test_client_indices[client_id].extend(class_split[i])

    testloaders = []
    for client_id in range(num_clients):
        testloaders.append(_client_test_loader(test_client_indices[client_id], testset, batch_size, seed))

    for client_id in range(num_clients):
        client_labels_train = labels_train[client_indices[client_id]]
        client_labels_test = labels_test[test_client_indices[client_id]]
        for c in range(num_classes):
            class_client_matrix_train[client_id, c] = np.sum(client_labels_train == c)
            class_client_matrix_test[client_id, c] = np.sum(client_labels_test == c)

    return trainloaders, validationloaders, testloaders, class_client_matrix_train, class_client_matrix_test


def prepare_dataset_iid_train_iid_test(num_clients: int, num_classes: int, clients_with_no_data: list[int], batch_size: int, seed: int, dataset: str, val_ratio: float = 0.1):
    """IID training data plus an independently IID-partitioned test set,
    split evenly across all `num_clients`."""
    np.random.seed(seed=seed)
    torch.manual_seed(seed)

    trainset, trainset_eval, testset = _get_dataset(dataset)
    class_client_matrix_train = np.zeros((num_clients, num_classes), dtype=int)
    class_client_matrix_test = np.zeros((num_clients, num_classes), dtype=int)

    clients_with_data = _clients_with_data(num_clients, clients_with_no_data)

    # SPLIT DATASET BY CLASSES
    labels_train = np.array(trainset.targets)
    ordered_train_idx = np.concatenate([np.where(labels_train == i)[0] for i in range(num_classes)])

    num_images = len(ordered_train_idx) // len(clients_with_data)
    num_images_remainder = len(ordered_train_idx) % len(clients_with_data)

    partition_len_train = [0] * num_clients

    #SPLIT DS ACCORDINGLY
    for i in clients_with_data:
        partition_len_train[i] = num_images
        if num_images_remainder > 0:
            partition_len_train[i] += 1
            num_images_remainder -=1

    ##########
    client_train_indices = _partition_indices(ordered_train_idx, partition_len_train, seed)

    trainloaders = []
    validationloaders = []

    for client_id in range(num_clients):
        trainloader, valloader = _client_train_val_loaders(
            client_train_indices[client_id], trainset, trainset_eval, val_ratio, batch_size, seed
        )
        trainloaders.append(trainloader)
        validationloaders.append(valloader)
        class_client_matrix_train[client_id] = np.bincount(
            labels_train[client_train_indices[client_id]], minlength=num_classes
        )

    #TEST SET
    labels_test = np.array(testset.targets)

    partition_len_test = [0] * num_clients

    #SPLIT DS ACCORDINGLY
    len_instances_test = len(testset) // num_clients
    remainder = len(testset) % num_clients

    for i in range(num_clients):
        partition_len_test[i] = len_instances_test

    partition_len_test[0] += remainder

    ##########
    client_test_indices = _partition_indices(np.arange(len(testset)), partition_len_test, seed)
    testloaders = []

    for client_id in range(num_clients):
        testloaders.append(_client_test_loader(client_test_indices[client_id], testset, batch_size, seed))
        class_client_matrix_test[client_id] = np.bincount(
            labels_test[client_test_indices[client_id]], minlength=num_classes
        )

    return trainloaders, validationloaders, testloaders, class_client_matrix_train, class_client_matrix_test


def prepare_dataset_niid_train_iid_test(num_clients: int, num_classes: int, clients_with_no_data: list[int], batch_size: int, seed: int, dataset: str, val_ratio: float = 0.1):
    """"Coarse" Dirichlet-skewed training data plus an independently IID-partitioned
    test set split evenly across all `num_clients`."""
    np.random.seed(seed=seed)
    torch.manual_seed(seed)

    trainset, trainset_eval, testset = _get_dataset(dataset)
    class_client_matrix_train = np.zeros((num_clients, num_classes), dtype=int)
    class_client_matrix_test = np.zeros((num_clients, num_classes), dtype=int)

    clients_with_data = _clients_with_data(num_clients, clients_with_no_data)

    # SPLIT DATASET BY CLASSES
    labels_train = np.array(trainset.targets)
    ordered_train_idx = np.concatenate([np.where(labels_train == i)[0] for i in range(num_classes)])

    # SPLIT DIRICHLET DISTRIBUTION
    alpha = [20., 40., 1., 1., 1., 1., 1., 2., 2., 1., 1., 1., 1., 1., 40., 20. ]
    dirich = np.random.dirichlet(alpha)

    partition_len_train = [0] * num_clients
    total_instances = 0
    j = 0

    #SPLIT DS ACCORDINGLY
    for i in clients_with_data:
        partition_len_train[i] = int(len(ordered_train_idx)*dirich[j])
        total_instances += partition_len_train[i]
        j+=1

    remainder = len(ordered_train_idx) - total_instances
    partition_len_train[clients_with_data[0]] += remainder

    ##########
    client_train_indices = _partition_indices(ordered_train_idx, partition_len_train, seed)

    trainloaders = []
    validationloaders = []

    for client_id in range(num_clients):
        trainloader, valloader = _client_train_val_loaders(
            client_train_indices[client_id], trainset, trainset_eval, val_ratio, batch_size, seed
        )
        trainloaders.append(trainloader)
        validationloaders.append(valloader)
        class_client_matrix_train[client_id] = np.bincount(
            labels_train[client_train_indices[client_id]], minlength=num_classes
        )

    #TEST SET
    labels_test = np.array(testset.targets)

    partition_len_test = [0] * num_clients

    #SPLIT DS ACCORDINGLY
    len_instances_test = len(testset) // num_clients
    remainder = len(testset) % num_clients

    for i in range(num_clients):
        partition_len_test[i] = len_instances_test

    partition_len_test[0] += remainder

    ##########
    client_test_indices = _partition_indices(np.arange(len(testset)), partition_len_test, seed)
    testloaders = []

    for client_id in range(num_clients):
        testloaders.append(_client_test_loader(client_test_indices[client_id], testset, batch_size, seed))
        class_client_matrix_test[client_id] = np.bincount(
            labels_test[client_test_indices[client_id]], minlength=num_classes
        )

    return trainloaders, validationloaders, testloaders, class_client_matrix_train, class_client_matrix_test


def prepare_dataset_niid_train_niid_test(num_clients: int, num_classes: int, clients_with_no_data: list[int], batch_size: int, seed: int, dataset: str,  val_ratio: float = 0.1):
    """"Coarse" Dirichlet-skewed training data plus
    a test set skewed the same way: the identical `dirich` proportions drawn
    for the train split slice the independent, class-sorted test set into
    per-client chunks."""
    np.random.seed(seed=seed)
    torch.manual_seed(seed)

    trainset, trainset_eval, testset = _get_dataset(dataset)
    class_client_matrix_train = np.zeros((num_clients, num_classes), dtype=int)
    class_client_matrix_test = np.zeros((num_clients, num_classes), dtype=int)

    clients_with_data = _clients_with_data(num_clients, clients_with_no_data)

    # SPLIT DATASET BY CLASSES
    labels_train = np.array(trainset.targets)
    ordered_train_idx = np.concatenate([np.where(labels_train == i)[0] for i in range(num_classes)])

    # SPLIT DIRICHLET DISTRIBUTION
    alpha = [20., 40., 1., 1., 1., 1., 1., 2., 2., 1., 1., 1., 1., 1., 40., 20. ]
    dirich = np.random.dirichlet(alpha)

    partition_len_train = [0] * num_clients
    total_instances = 0
    j = 0

    #SPLIT DS ACCORDINGLY
    for i in clients_with_data:
        partition_len_train[i] = int(len(ordered_train_idx)*dirich[j])
        total_instances += partition_len_train[i]
        j+=1

    remainder = len(ordered_train_idx) - total_instances
    partition_len_train[clients_with_data[0]] += remainder

    ##########
    client_train_indices = _partition_indices(ordered_train_idx, partition_len_train, seed)

    trainloaders = []
    validationloaders = []

    for client_id in range(num_clients):
        trainloader, valloader = _client_train_val_loaders(
            client_train_indices[client_id], trainset, trainset_eval, val_ratio, batch_size, seed
        )
        trainloaders.append(trainloader)
        validationloaders.append(valloader)
        class_client_matrix_train[client_id] = np.bincount(
            labels_train[client_train_indices[client_id]], minlength=num_classes
        )

    #TEST SET
    labels_test = np.array(testset.targets)
    ordered_test_idx = np.concatenate([np.where(labels_test == i)[0] for i in range(num_classes)])

    partition_len_test = [0] * num_clients
    total_instances = 0
    j = 0

    #SPLIT DS ACCORDINGLY
    for i in clients_with_data:
        partition_len_test[i] = int(len(ordered_test_idx)*dirich[j])
        total_instances += partition_len_test[i]
        j+=1
    remainder = len(ordered_test_idx) - total_instances
    partition_len_test[clients_with_data[0]] += remainder

    ##########
    client_test_indices = _partition_indices(ordered_test_idx, partition_len_test, seed)
    testloaders = []

    for client_id in range(num_clients):
        testloaders.append(_client_test_loader(client_test_indices[client_id], testset, batch_size, seed))
        class_client_matrix_test[client_id] = np.bincount(
            labels_test[client_test_indices[client_id]], minlength=num_classes
        )

    return trainloaders, validationloaders, testloaders, class_client_matrix_train, class_client_matrix_test
