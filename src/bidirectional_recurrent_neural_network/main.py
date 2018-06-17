import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import sys
sys.path.append("../")
from utils.utils import *


# Hyper parameters
sequence_length = 28
input_size = 28
hidden_size = 128
num_layers = 2
num_classes = 10
batch_size = 100
num_epochs = 5
learning_rate = 0.003
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# MNIST
train_loader, test_loader = get_mnist_loader(batch_size=batch_size)


class BiRNN(nn.Module):
    """Bidirectional recurrent neural network (many-to-one)"""

    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # *2 is bidirection

    def forward(self, x):
        # Set initial states
        h0 = torch.zeros(self.num_layers*2, x.size(0),
                         self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers*2, x.size(0),
                         self.hidden_size).to(device)

        # Forward
        out, _ = self.lstm(x, (h0, c0))

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


model = BiRNN(input_size, hidden_size, num_layers, num_classes).to(device)
# if torch.cuda.device_count() > 1:
#     print("Lets use {} GPUs".format(torch.cuda.device_count()))
#     model = nn.DataParallel(model)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)

        # Forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            show_learning_status(
                epoch=epoch, num_epochs=num_epochs, i=i, total_step=total_step, loss=loss)

# Test
with torch.no_grad():
    correct, total = 0, 0
    for images, labels in test_loader:
        images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)

        outputs = model(outputs)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print("Accuracy : {} %".format(100 * correct / total))
