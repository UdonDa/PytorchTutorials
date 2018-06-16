import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


def get_mnist_dataset():
    train_dataset = torchvision.datasets.MNIST(root='../data/',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)

    test_dataset = torchvision.datasets.MNIST(root='../data/',
                                              train=False,
                                              transform=transforms.ToTensor())
    return train_dataset, test_dataset


def get_mnist_loader(batch_size=16):
    # Data loader
    train_dataset, test_dataset = get_mnist_dataset()
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)
    return train_loader, test_loader


def show_learning_status(epoch, num_epochs, i, total_step, loss):
    print("Epoch L [{}/{}], Step : [{}/{}], Loss : [{:.4f}]".format(
        epoch+1, num_epochs, i+1, total_step, loss.item()))
