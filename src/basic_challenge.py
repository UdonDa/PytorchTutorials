import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms


""" 1. autograd example 2 """
# x = torch.tensor(1, requires_grad=True)
# w = torch.tensor(2, requires_grad=True)
# b = torch.tensor(3, requires_grad=True)
# # print("x : {}\nw : {}\nb : {}".format(x, w, b)) # 1, 2, 3

# y = w * x + b

# # conpute gradient
# y.backward()

# print("x.grad : {}\nw.grad : {}\nb.grad : {}".format(x.grad, w.grad, b.grad))


""" 2. autograd example 2 """
# x = torch.randn(10, 3)
# y = torch.randn(10, 2)

# # fully connected layer
# linear = nn.Linear(3, 2)
# print("w : {}\nb : {}".format(linear.weight, linear.bias))

# criterion = nn.MSELoss()
# optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)

# # forward loss
# pred = linear(x)

# # compute loss
# loss = criterion(pred, y)
# print("loss : {}".format(loss.item()))

# # backward loss
# loss.backward()
# print("dL/dw : {}\ndL/db : {}".format(linear.weight.grad, linear.bias.grad))

# # 1 step gradient descent
# optimizer.step()

# pred = linear(x)
# loss = criterion(pred, y)
# print("loss after 1steps : {}".format(loss.item()))

""" 3. loading value from numpy """
# # create numpy tensor
# x = np.array([[1, 2], [3, 4]])
# # convert numpy array to torch tensor
# y = torch.from_numpy(x)
# # convert torch tensor to numpy array
# z = y.numpy()
# print("type(x) : {}".format(type(x)))
# print("type(y) : {}".format(type(y)))
# print("type(z) : {}".format(type(z)))

""" 4. input pipline """
# train_dataset = torchvision.datasets.CIFAR10(
#     root="../data/",
#     train=True,
#     transform=transforms.ToTensor(),
#     download=True
# )

# image, label = train_dataset[0]
# print("image.size() : {}".format(image.size()))
# print("label : {}".format(label))

# # loader
# train_loader = torch.utils.data.DataLoader(
#     dataset=train_dataset,
#     batch_size=16,
#     shuffle=True
# )

# data_iter = iter(train_loader)

# # mini-batch images and labels
# images, labels = data_iter.next()
# print("image.size() : {}".format(image.size()))
# print("label : {}".format(label))

""" 6. pretrained model """
resnet = torchvision.models.resnet18(pretrained=True)

# when you try to finetune, you should do it
for param in resnet.parameters():
    param.requires_grad = False

resnet.fc = nn.Linear(resnet.fc.in_features, 100)

images = torch.randn(64, 3, 224, 224)
outputs = resnet(images)
print("ouputs.size() : {}".format(outputs.size()))

""" 7. save and load model """
torch.save(resnet.state_dict(), "params.ckpt")
resnet.load_state_dict(torch.load("params.ckpt"))
