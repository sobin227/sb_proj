import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from numpy.testing._private.parameterized import param


trans = transforms.Compose(
    [transforms.Resize((150,150)),transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])


testset = torchvision.datasets.ImageFolder(root="./datasets/test", transform=trans)
testloader = torch.utils.data.DataLoader(testset, batch_size=20, shuffle=True, num_workers=0)


model_ft = torch.load('./datasetsmodel_test1.pt')
#print(model_ft)

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = model_ft(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))