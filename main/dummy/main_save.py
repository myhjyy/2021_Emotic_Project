import torch
from torch import nn, optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import (Dataset, DataLoader, TensorDataset)
import numpy as np
import os.path
import os
import sys
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import model
import data_save

'''transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])'''

batch_size = 32
learning_rate = 0.001
num_epoch = 200

# data management ---------------------------------------------------------#

if len(sys.argv) != 2 and sys.argv[2] == 'new':
    if os.path.isfile("/media/unist/29b09bb4-f6d5-47f6-9592-d3e8130e3475/Emotic_Project/CAER-S/dataset/train/X_train_init.npy"):
        os.remove("/media/unist/29b09bb4-f6d5-47f6-9592-d3e8130e3475/Emotic_Project/CAER-S/dataset/train/X_train_init.npy")
    if os.path.isfile("/media/unist/29b09bb4-f6d5-47f6-9592-d3e8130e3475/Emotic_Project/CAER-S/dataset/train/Y_train.npy"):
        os.remove("/media/unist/29b09bb4-f6d5-47f6-9592-d3e8130e3475/Emotic_Project/CAER-S/dataset/train/Y_train.npy")
    if os.path.isfile("/media/unist/29b09bb4-f6d5-47f6-9592-d3e8130e3475/Emotic_Project/CAER-S/dataset/test/X_test_init.npy"):
        os.remove("/media/unist/29b09bb4-f6d5-47f6-9592-d3e8130e3475/Emotic_Project/CAER-S/dataset/test/X_test_init.npy")
    if os.path.isfile("/media/unist/29b09bb4-f6d5-47f6-9592-d3e8130e3475/Emotic_Project/CAER-S/dataset/test/Y_test.npy"):
        os.remove("/media/unist/29b09bb4-f6d5-47f6-9592-d3e8130e3475/Emotic_Project/CAER-S/dataset/test/Y_test.npy")
    if os.path.isfile("/media/unist/29b09bb4-f6d5-47f6-9592-d3e8130e3475/Emotic_Project/CAER-S/dataset/train/X_train.npy"):
        os.remove("/media/unist/29b09bb4-f6d5-47f6-9592-d3e8130e3475/Emotic_Project/CAER-S/dataset/train/X_train.npy")
    if os.path.isfile("/media/unist/29b09bb4-f6d5-47f6-9592-d3e8130e3475/Emotic_Project/CAER-S/dataset/test/X_test.npy"):
        os.remove("/media/unist/29b09bb4-f6d5-47f6-9592-d3e8130e3475/Emotic_Project/CAER-S/dataset/test/X_test.npy")

if not os.path.isfile("/media/unist/29b09bb4-f6d5-47f6-9592-d3e8130e3475/Emotic_Project/CAER-S/dataset/train/X_train_init.npy"):
    train = data_save.train_data()
    train.save()
if not os.path.isfile("/media/unist/29b09bb4-f6d5-47f6-9592-d3e8130e3475/Emotic_Project/CAER-S/dataset/test/X_test_init.npy"):
    test = data_save.test_data()
    test.save()

X_train_init = np.load("/media/unist/29b09bb4-f6d5-47f6-9592-d3e8130e3475/Emotic_Project/CAER-S/dataset/train/X_train_init.npy")
Y_train = np.load("/media/unist/29b09bb4-f6d5-47f6-9592-d3e8130e3475/Emotic_Project/CAER-S/dataset/train/Y_train.npy")
X_test_init = np.load("/media/unist/29b09bb4-f6d5-47f6-9592-d3e8130e3475/Emotic_Project/CAER-S/dataset/test/X_test_init.npy")
Y_test = np.load("/media/unist/29b09bb4-f6d5-47f6-9592-d3e8130e3475/Emotic_Project/CAER-S/dataset/test/Y_test.npy")

X_train = []
X_test = []
if not os.path.isfile("/media/unist/29b09bb4-f6d5-47f6-9592-d3e8130e3475/Emotic_Project/CAER-S/dataset/train/X_train.npy"):
    for i, item in enumerate(X_train_init):
        X_train = np.append(X_train, np.array([[np.reshape(item, (3, 94, 94), order='F')]]))
    X_train = np.reshape(X_train, (-1, 3, 94, 94))
else:
    X_train = np.load("/media/unist/29b09bb4-f6d5-47f6-9592-d3e8130e3475/Emotic_Project/CAER-S/dataset/train/X_train.npy")

if not os.path.isfile("/media/unist/29b09bb4-f6d5-47f6-9592-d3e8130e3475/Emotic_Project/CAER-S/dataset/test/X_test.npy"):
    for item in X_test_init:
        X_test = np.append(X_test, np.array([[np.reshape(item, (3, 94, 94), order='F')]]))
    X_test = np.reshape(X_test, (-1, 3, 94, 94))
else:
    X_test = np.load("/media/unist/29b09bb4-f6d5-47f6-9592-d3e8130e3475/Emotic_Project/CAER-S/dataset/test/X_test.npy")

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
Y_train = torch.tensor(Y_train, dtype=torch.int64)
Y_test = torch.tensor(Y_test, dtype=torch.int64)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

X_test = X_test.to(device)
X_train = X_train.to(device)
Y_test = Y_test.to(device)
Y_train = Y_train.to(device)

train_dataset = TensorDataset(X_train, Y_train)
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
test_dataset = TensorDataset(X_test, Y_test)
test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)
# train and test ------------------------------------------------------------- #
# with face only : python3 main_save.py 0 (new)
# with skel data : python3 main_save.py 1 (new)

if sys.argv == 0:
    net = model.FaceEncoding()
elif sys.argv == 1:
    net = model.SkelEncoding()
else:
    print("error in network selection!!")
    
net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

print('start Training')
for epoch in range(num_epoch):
    
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 1000 == 999:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 1000))
            running_loss = 0.0

print('Finished Training')

correct = 0
total = 0
correct_label = {}
total_label = {}
emotic_labels = {0:'Anger', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Neutral', 5:'Sad', 6:'Surprise'}
for i in range(7):
    correct_label[emotic_labels[i]] = 0
    total_label[emotic_labels[i]] = 0

with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        #total_label[emotic_labels[labels]] += labels.size(0)
        #correct_label[emotic_labels[labels]] += (predicted == labels).sum().item()

print('Accuracy : %d %%' % (100 * correct / total))