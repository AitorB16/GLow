import torch
import torch.nn as nn
import torch.nn.functional as F

# Note the model and functions here defined do not have any FL-specific components.

class Net(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super(Net, self).__init__()

        # define the layers
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
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
    
def train(net, trainloader, validationloader, optimizer, epochs, num_classes, device: str):
    """Train the network on the training set.

    This is a fairly simple training loop for PyTorch.
    """
    # TRAIN
    criterion = nn.CrossEntropyLoss()
    net.train()
    net.to(device)
    train_loss = []
    for _ in range(epochs):
        loss_sum = 0.
        for inputs, labels in trainloader: #INPUTS ARE TUPLES OF DATABASE
            inputs, labels = inputs.to(device), labels.to(device)
            loss = criterion(net(inputs), labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_sum += loss.item()
        #print(f"\n** {_+1}/{epochs} mean loss: {loss_sum/len(trainloader)}\n")
        if len(trainloader) > 0:
            train_loss.append(loss_sum/len(trainloader))
        else:
            train_loss.append(1./num_classes)

    # VALIDATION
    correct, total_size, valid_loss = 0, 0, 0.0
    net.eval()     # Optional when not using Model Specific layer
    for inputs, labels in validationloader:
        # Transfer Data to GPU if available
        inputs, labels = inputs.to(device), labels.to(device)
        # Forward Pass
        outputs = net(inputs)
        # Find and calculate the Loss
        valid_loss += criterion(outputs, labels).item()
        correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        total_size += labels.size(0)
    if total_size > 0:
        val_accuracy = correct / total_size
    else:
        val_accuracy = 1./num_classes
    metrics_val_distributed_fit = val_accuracy

    return train_loss, metrics_val_distributed_fit

def test(net, testloader, num_classes, device: str):
    """Validate the network on the entire test set.
    and report loss and accuracy.
    """
    criterion = nn.CrossEntropyLoss()
    correct, total_size, loss = 0, 0, 0.
    net.eval()
    net.to(device)
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
            total_size += labels.size(0)
        if total_size > 0:
            accuracy = correct / total_size
        else:
            accuracy = 1./num_classes
    return loss, accuracy
