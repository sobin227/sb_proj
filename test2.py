import matplotlib
from PIL import Image
import matplotlib.pyplot as plt
from numpy.testing._private.parameterized import param
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torchvision.models import vgg

net = models.vgg16(pretrained=True)

features = net.features
for params in features.parameters():
    param.requires_grad = False
net.classifier[3].out_features = 3

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

trans = transforms.Compose(
    [transforms.Resize((150,150)),transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

trainset = torchvision.datasets.ImageFolder(root="./datasets/train", transform=trans)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=20, shuffle=True, num_workers=0)

valset = torchvision.datasets.ImageFolder(root="./datasets/evaluation", transform=trans)
validloader = torch.utils.data.DataLoader(trainset, batch_size=20, shuffle=True, num_workers=0)

testset = torchvision.datasets.ImageFolder(root="./datasets/test", transform=trans)
testloader = torch.utils.data.DataLoader(trainset, batch_size=20, shuffle=True, num_workers=0)

print(len(trainset))
classes = trainset.classes
print(classes)

def imshow(img):
    img = img/2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()

dataiter = iter(trainloader)
images, labels = dataiter.next()
imshow(torchvision.utils.make_grid(images))
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

criterion = nn.CrossEntropyLoss()#.to(device)
optimizer = torch.optim.SGD(net.parameters(),lr=0.005,momentum=0.9)

lr_sche = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

print(len(trainloader))
epochs = 3

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

for epoch in range(epochs):
    running_loss = 0.0
    lr_sche.step()
    current_lr = get_lr(optimizer)
    print('Epoch {}/{}'.format(epoch, epochs))
    for i,data in enumerate(trainloader,0): # 0부터 시작
        inputs, labels = data
        inputs = inputs#.to(device)
        labels = labels#.to(device)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        #if i % 30 == 29:
            #print('[%d,%5d] loss: %.3f' %
                  #(epoch+1, i+1, running_loss/30))
            #running_loss = 0.0

print('finished')

dataiter = iter(validloader)
images, labels = dataiter.next()
outputs = net(images.to(device))
correct = 0
total = 0

with torch.no_grad():
    for data in validloader:
        images, labels = data
        images = images#.to(device)
        labels = labels#.to(device)
        outputs = net(images)

        _, predicted = torch.max(outputs.data,1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

print('Acc : %d %%' %(100*correct/total))