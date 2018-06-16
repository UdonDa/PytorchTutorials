import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import sys
sys.path.append("../")
from utils.utils import *

# Hyper parameters
num_epochs = 5
num_classes = 10
batch_size = 200
learning_rate = 0.001

train_loader, test_loader = get_mnist_loader(batch_size=batch_size)


class ConvNet(nn.Module):
    """Two convolutional layers"""

    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            # layer1
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            # layer2
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(32),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Linear(7*7*32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ConvNet(num_classes=num_classes).to(device)
if torch.cuda.device_count() > 1:
    print("Lets use {} GPUs".format(torch.cuda.device_count()))
    model = nn.DataParallel(model)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
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
model.eval()
with torch.no_grad():
    correct, total = 0, 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print("Accuracy : {}%".format(100 * correct / total))

# Save
# torch.save(model.state_dict(), 'model.ckpt')
