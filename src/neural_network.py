import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms

# hyper parameters
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 10
batch_size = 100
learning_rate = 0.001

# MNIST datasets
train_dataset = torchvision.datasets.MNIST(
    root="../data/",
    train=True,
    transform=transforms.ToTensor(),
    download=True
)
test_dataset = torchvision.datasets.MNIST(
    root="../data/",
    train=False,
    transform=transforms.ToTensor()
)

# MNIST data loader
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False
)


class NeuralNet(nn.Module):
    # fully connected neural netrowk
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        return self.main(x)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, num_classes).to(device)
# multiple GPUs
if torch.cuda.device_count() > 1:
    print("Lets use {} GPUs".format(torch.cuda.device_count()))
    model = nn.DataParallel(model)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)

        # Forward Loss
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print("Epoch : [{}/{}], Step : [{}/{}], Loss : {:.4f} ".format(
                epoch+1, num_epochs, i+1, total_step, loss.item()))

# Test
with torch.no_grad():
    correct, total = 0, 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print("Accuracy : {}%".format(100 * correct / total))
# Save
# torch.save(model.state_dict(), 'model.ckpt')
