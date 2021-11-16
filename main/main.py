import torch
from torch import nn, optim
from torch.utils.data import (Dataset, DataLoader, TensorDataset)
import torch.nn as nn
import torch.optim as optim
import argparse

import model
import data

# (new) : if you want to make new *.npy file, type new. other, just leave blank

# If you execute first : python3 main.py --net [net type, S or F] --data save new True
# execute again : python3 main.py  --net [net type, S or F] --data save

parser = argparse.ArgumentParser(description='main.py argument')
parser.add_argument('--net', type=str, help='model network type. S = skeletal network, F = facial network')
parser.add_argument('--data', type=str, help='where to get img file, save = from saved npy file, load = from original img file')
parser.add_argument('--new', type=bool, default=False, help='when the image data is newly updated, update .npy file newly')
parser.add_argument('--epoch', type=int, default=250, help='number of epochs to train')
parser.add_argument('-batch-size', type=int, default=32, help='number of batch size')
parser.add_argument('--learning-rate', type=float, default=0.001, help='learning rate')

args = parser.parse_args()
# print(args)

batch_size = args.batch_size
learning_rate = args.learning_rate
num_epoch = args.epoch

if args.net == None or args.data == None:
    print("input argument is wrong!!")
    print("executing code : python3 main.py -- net [network type : S or F] --data [load file type : load or save] --new (use updated data : True or False)")
    exit()

# data management ---------------------------------------------------------#

# file type : load or save
if args.data == "save":
    Train = data.train_data()
    Test = data.test_data()
    if args.new == True:
        X_train, Y_train = Train.save("new")
        X_test, Y_test = Test.save("new")
        if args.net == "S":
            X_skel_train = Train.skel("save", "new")
            X_skel_test = Test.skel("save", "new")
    else :
        X_train, Y_train = Train.save()
        X_test, Y_test = Test.save()
        if args.net == "S":
            X_skel_train = Train.skel("save")
            X_skel_test = Test.skel("save")
elif args.data == "load":
    if args.new == True:
        print("Warning :: you cannot choose (new) option when load image files!")
    Train = data.train_data()
    Test = data.test_data()
    X_train, Y_train = Train.load()
    X_test, Y_test = Test.load()
    if args.net == "S":
        X_skel_train = Train.skel()
        X_skel_test = Test.skel()

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
Y_train = torch.tensor(Y_train, dtype=torch.int64)
Y_test = torch.tensor(Y_test, dtype=torch.int64)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(str(device)+" is using!")
X_test = X_test.to(device)
X_train = X_train.to(device)
Y_test = Y_test.to(device)
Y_train = Y_train.to(device)

# train and test ------------------------------------------------------------- #

if args.net == "F":
    net = model.FaceEncoding()
elif args.net == "S":
    net = model.SkelEncoding_twostream()
else:
    print("You input network type "+args.net+"!!Network type must be F or S!!")
    exit()
net.to(device)

# with only face ------------------------------------------------------------- #

if args.net == "F":

    train_dataset = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    test_dataset = TensorDataset(X_test, Y_test)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

    print('Start Face Training')
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
        if epoch % 25 == 24:
            print('epoch [%d/%d] loss: %.4f' %
                (epoch + 1, num_epoch, running_loss / 25))
            running_loss = 0.0

            correct = 0
            total = 0
            with torch.no_grad():
                for data in test_loader:
                    images, labels = data
                    images, labels = data[0].to(device), data[1].to(device)
                    outputs = net(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            print('Accuracy : %.3f %%' % (100 * correct / total))

    print('Finished Face Training')
    print("batch size :", batch_size, ", learning rate :", learning_rate, ", epoch :", num_epoch)

    correct = 0
    total = 0
    emotic_labels = {0:'Anger', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Neutral', 5:'Sad', 6:'Surprise'}

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy : %.3f %%' % (100 * correct / total))
elif args.net == "S":

# with context data ------------------------------------------------------------- #

    X_skel_train = torch.tensor(X_skel_train, dtype=torch.float32)
    X_skel_test = torch.tensor(X_skel_test, dtype=torch.float32)

    X_skel_test = X_skel_test.to(device)
    X_skel_train = X_skel_train.to(device)

    train_dataset = TensorDataset(X_train, X_skel_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    test_dataset = TensorDataset(X_test, X_skel_test, Y_test)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

    print('Start Skeletal Context Training')
    for epoch in range(num_epoch):
        
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, skels, labels = data
            inputs, skels, labels = data[0].to(device), data[1].to(device), data[2].to(device)
            optimizer.zero_grad()

            outputs = net(inputs, skels)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        if epoch % 25 == 24:
            print('epoch [%d/%d] loss: %.4f' %
                (epoch + 1, num_epoch, running_loss / 25))
            running_loss = 0.0
            
            correct = 0
            total = 0
            with torch.no_grad():
                for data in test_loader:
                    images, skels, labels = data
                    images, skels, labels = data[0].to(device), data[1].to(device), data[2].to(device)
                    outputs = net(images, skels)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            print('Accuracy : %.3f %%' % (100 * correct / total))

    print('Finished Skeletal Context Training')
    print("batch size :", batch_size, ", learning rate :", learning_rate, ", epoch :", num_epoch)

    correct = 0
    total = 0
    emotic_labels = {0:'Anger', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Neutral', 5:'Sad', 6:'Surprise'}

    with torch.no_grad():
        for data in test_loader:
            images, skels, labels = data
            images, skels, labels = data[0].to(device), data[1].to(device), data[2].to(device)
            outputs = net(images, skels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy : %.3f %%' % (100 * correct / total))
