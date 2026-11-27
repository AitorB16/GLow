"""Models and plain PyTorch train/test routines."""

import io
from collections import OrderedDict

import torch
import torchmetrics
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

#CIFAR
class LeNet(nn.Module):
    def __init__(self, num_classes: int) -> None:
      super().__init__()
      self.conv1 = nn.Conv2d(3, 16, 3, 1, padding=1) # input is color image, hence 3 i/p channels. 16 filters, kernal size is tuned to 3 to avoid overfitting, stride is 1 , padding is 1 extract all edge features.
      self.conv2 = nn.Conv2d(16, 32, 3, 1, padding=1) # We double the feature maps for every conv layer as in pratice it is really good.
      self.conv3 = nn.Conv2d(32, 64, 3, 1, padding=1)
      self.fc1 = nn.Linear(4*4*64, 500) # I/p image size is 32*32, after 3 MaxPooling layers it reduces to 4*4 and 64 because our last conv layer has 64 outputs. Output nodes is 500
      #self.dropout1 = nn.Dropout(0.5)
      self.dropout1 = nn.Dropout(0.2)
      self.fc2 = nn.Linear(500, num_classes) # output nodes are 10 because our dataset have 10 different categories
    def forward(self, x):
      x = F.relu(self.conv1(x)) #Apply relu to each output of conv layer.
      x = F.max_pool2d(x, 2, 2) # Max pooling layer with kernal of 2 and stride of 2
      x = F.relu(self.conv2(x))
      x = F.max_pool2d(x, 2, 2)
      x = F.relu(self.conv3(x))
      x = F.max_pool2d(x, 2, 2)
      x = x.view(-1, 4*4*64) # flatten our images to 1D to input it to the fully connected layers
      x = F.relu(self.fc1(x))
      x = self.dropout1(x) # Applying dropout b/t layers which exchange highest parameters. This is a good practice
      x = self.fc2(x)
      return x

#MNIST
class LeNet5(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size = 5),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size = 2),
            nn.Conv2d(6, 16, kernel_size = 5),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size = 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, num_classes)  
        )
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        logit = self.classifier(x)
        return logit
    
class ScratchNet(nn.Module):
    def __init__(self, num_classes: int) -> None:
      super().__init__()
      self.fc1 = nn.Linear(22, 128)  # 11 numeric + 11 label-encoded features (see get_ton_iot)
      self.fc2 = nn.Linear(128, 64)
      self.fc3 = nn.Linear(64, 64)
      self.dropout = nn.Dropout(p=0.2)
      self.fc4 = nn.Linear(64, 32)
      self.fc5 = nn.Linear(32, 32)
      self.fc6 = nn.Linear(32, num_classes)
    def forward(self, x):
      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))
      x = F.relu(self.fc3(x))
      x = self.dropout(x)
      x = F.relu(self.fc4(x))
      x = F.relu(self.fc5(x))
      return self.fc6(x)

#HAR
class HARModel(nn.Module):
    def __init__(self, num_classes: int = 8) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=6, out_channels=192, kernel_size=16)
        self.pool = nn.MaxPool1d(kernel_size=4)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(192 * 28, 1024)  # (128 - 16 + 1) // 4 = 28
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)  # (batch, seq_len, channels) -> (batch, channels, seq_len)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def build_model(dataset: str, num_classes: int) -> nn.Module:
    if dataset == 'cifar':
        return LeNet(num_classes)
    if dataset == 'mnist':
        return LeNet5(num_classes)
    if dataset == 'tiot':
        return ScratchNet(num_classes)
    if dataset == 'har':
        return HARModel(num_classes)
    raise ValueError(f"Unknown dataset '{dataset}'. Expected 'cifar', 'mnist', 'tiot' or 'har'.")


def corrupt_labels(labels, nature, num_classes):
    """'malicious': class c -> c+1; 'random': random class map, redrawn per
    batch; otherwise unchanged."""
    if nature == 'malicious':
        mapping = torch.roll(torch.arange(num_classes), -1)
    elif nature == 'random':
        mapping = torch.tensor(np.random.randint(num_classes, size=num_classes))
    else:
        return labels
    return mapping[labels]


def train(net, trainloader, validationloader, optimizer, epochs, num_classes, nature, device):
    start_event = time.perf_counter()
    # TRAINING
    criterion = nn.CrossEntropyLoss()
    net.train()
    net.to(device)
    train_loss = []
    for _ in range(epochs):
        loss_sum = 0.
        for inputs, labels in trainloader: #INPUTS ARE TUPLES OF DATABASE
            labels = corrupt_labels(labels, nature, num_classes)
            inputs, labels = inputs.to(device), labels.to(device)
            loss = criterion(net(inputs), labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_sum += loss.item()
        if len(trainloader) > 0:
            train_loss.append(loss_sum/len(trainloader))
        else:
            train_loss.append(float('nan'))
    end_event = time.perf_counter()
    training_time = end_event - start_event

    # VALIDATION
    correct, total_size, valid_loss = 0, 0, 0.0
    centroid = torch.tensor([0.] * num_classes) #COMPUTE CENTROID USING VAL-SET
    instances_per_class = torch.tensor([0.] * num_classes)

    net.eval()     # Optional when not using Model Specific layer
    for inputs, labels in validationloader:
        labels = corrupt_labels(labels, nature, num_classes)
        
        for c in range(num_classes):
            instances_per_class[c]+= (labels == c).sum()

        # Transfer Data to GPU if available
        inputs, labels = inputs.to(device), labels.to(device)
        # Forward Pass
        outputs = net(inputs)
        preds = torch.max(outputs, 1)[1]

        # Find and calculate the Loss
        valid_loss += criterion(outputs, labels).item()
        correct += (preds == labels).sum().item()
        total_size += labels.size(0)

        for c in range(num_classes):
            centroid[c] += ((labels == c) & (preds == labels)).sum().item()

    if total_size > 0:
        val_accuracy = correct / total_size
        mask = instances_per_class > 0
        centroid[mask] /= instances_per_class[mask]
    else: #no local data -- further address?
        val_accuracy = float('nan')
    
    metrics_val_distributed_fit = val_accuracy

    return train_loss, metrics_val_distributed_fit, centroid, training_time

def test(net, testloader, num_classes, nature, device):
    f1 = torchmetrics.classification.MulticlassF1Score(num_classes=num_classes, average='macro')

    criterion = nn.CrossEntropyLoss()
    correct, total_size, loss = 0, 0, 0.
    instances_per_class = torch.tensor([0.] * num_classes)
    centroid = torch.tensor([0.] * num_classes) #COMPUTE CENTROID USING VAL-SET

    net.eval()
    net.to(device)
    with torch.no_grad():
        for inputs, labels in testloader:
            labels = corrupt_labels(labels, nature, num_classes)

            for c in range(num_classes):
                instances_per_class[c]+= (labels == c).sum()
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            preds = torch.max(outputs, 1)[1]

            loss += criterion(outputs, labels).item()
            correct += (preds == labels).sum().item()
            total_size += labels.size(0)

            for c in range(num_classes):
                centroid[c] += ((labels == c) & (preds == labels)).sum().item()
            
            f1.update(preds.cpu(), labels.cpu())

        if total_size > 0:
            accuracy = correct / total_size
            mask = instances_per_class > 0
            centroid[mask] /= instances_per_class[mask]
            macro_f1 = f1.compute().item()
            loss = loss / len(testloader)
        else:
            loss = accuracy = macro_f1 = float('nan')

    return loss, accuracy, centroid, macro_f1

def compute_prob_matrix(net, testloader, num_classes, nature, device):
    """Rows = true class. prob_matrix: mean softmax vector per class;
    preds_matrix: confusion matrix."""

    instances_per_class = torch.zeros(num_classes)
    prob_matrix = torch.zeros((num_classes, num_classes))
    preds_matrix = torch.zeros((num_classes, num_classes), dtype=int)

    net.eval()
    net.to(device)

    with torch.no_grad():
        for inputs, labels in testloader:
            labels = corrupt_labels(labels, nature, num_classes)

            outputs = net(inputs.to(device))
            probs = F.softmax(outputs, dim=1).cpu()
            preds = outputs.argmax(dim=1).cpu()

            preds_matrix += torch.bincount(
                labels * num_classes + preds,
                minlength=num_classes * num_classes,
            ).reshape(num_classes, num_classes)

            # accumulate counts
            instances_per_class += torch.bincount(labels, minlength=num_classes).float()

            # accumulate probability vectors per class
            prob_matrix.index_add_(0, labels, probs)

        # normalize to get means
        mask = instances_per_class > 0
        prob_matrix[mask] /= instances_per_class[mask].unsqueeze(1)

    return prob_matrix, preds_matrix

def softmax_distance(net, ref_net, loader, num_classes, nature, device):
    """Per class L2 distance between the two nets' mean softmax rows on
    `loader` (0 for classes absent from it)."""
    if loader == '':
        return [0.] * num_classes
    prob, _ = compute_prob_matrix(net, loader, num_classes, nature, device)
    ref_prob, _ = compute_prob_matrix(ref_net, loader, num_classes, nature, device)
    dist = torch.linalg.norm(prob - ref_prob, dim=1)
    return [float(dist[c]) if prob[c].sum() > 0 else 0. for c in range(num_classes)]

def pack_ndarrays(ndarrays) -> bytes:
    # Flower configs only take scalars, so send the arrays as bytes
    buf = io.BytesIO()
    np.savez(buf, *ndarrays)
    return buf.getvalue()

def unpack_ndarrays(blob: bytes):
    with np.load(io.BytesIO(blob)) as f:
        return [f[f'arr_{i}'] for i in range(len(f.files))]

def load_weights(net, ndarrays):
    net.load_state_dict(OrderedDict(
        {k: torch.tensor(v) for k, v in zip(net.state_dict().keys(), ndarrays)}), strict=True)
