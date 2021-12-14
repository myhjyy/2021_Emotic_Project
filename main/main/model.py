import torch
from torch import nn
import torch.nn.functional as F

class FaceEncoding(nn.Module):
    def __init__(self):
        super().__init__()
        # input size is 94 * 94
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # 46 * 46
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2) 

        # 22 * 22
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)

        
        # 10 * 10
        self.conv4 = nn.Conv2d(128, 256, 3)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)

        # 4 * 4
        self.conv5 = nn.Conv2d(256, 256, 3)
        self.bn5 = nn.BatchNorm2d(256)
        self.avgpool = nn.AvgPool2d(2, 2)

        # 1 * 1
        #Adaptive fusion networks
        self.conv6 = nn.Conv2d( 256, 128, 1)
        self.conv7 = nn.Conv2d( 128, 7, 1)
        self.mlp1 = nn.Linear(7, 7)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))
        x = self.avgpool(F.relu(self.conv5(x)))

        #Adaptive fusion networks
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = x.view(-1, 7)
        x = self.mlp1(x)
        x = self.softmax(x)
        return x

# --------------------------------------------------------------------- #

#Now, this network is not used
class SkelEncoding_linear(nn.Module):
    def __init__(self):
        super().__init__()
        # input size is 94 * 94
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # 46 * 46
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        # 22 * 22
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)

        
        # 10 * 10
        self.conv4 = nn.Conv2d(128, 256, 3)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)

        # 4 * 4
        self.conv5 = nn.Conv2d(256, 128, 3)
        self.bn5 = nn.BatchNorm2d(128)
        self.avgpool = nn.AvgPool2d(2, 2)

        # 1 * 1
        self.conv6 = nn.Conv2d(128, 75, 1)


        # batch * 1(channel) * 4 * 25
        self.mlp1 = nn.Linear(75, 128)
        self.mlp2 = nn.Linear(128, 128)
        self.mlp3 = nn.Linear(128, 64)
        self.mlp4 = nn.Linear(64, 32)
        self.mlp5 = nn.Linear(32, 16)
        self.mlp6 = nn.Linear(16, 7)
        self.softmax = nn.Softmax(dim = 1)

    def Face(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))
        x = self.avgpool(F.relu(self.conv5(x)))
        x = self.conv6(x)
        x = x.view(-1, 75)
        return x

    def Skel(self, y):
        y = y.view(-1, 75)
        return y

    def forward(self, x, y):
        x = self.Face(x)
        y = self.Skel(y)
        x = torch.cat((x, y), dim = 1)
        x = F.relu(self.mlp1(y))
        x = F.relu(self.mlp2(x))
        x = F.relu(self.mlp3(x))
        x = F.relu(self.mlp4(x))
        x = F.relu(self.mlp5(x))
        x = self.mlp6(x)
        self.softmax(x)
        return x

# --------------------------------------------------------------------- #

class SkelEncoding_twostream(nn.Module):
    def __init__(self):
        super().__init__()
        # face network
        # input size is 94 * 94
        self.f_conv1 = nn.Conv2d(3, 32, 3)
        self.f_bn1 = nn.BatchNorm2d(32)
        self.f_pool1 = nn.MaxPool2d(2, 2)
        
        # 46 * 46
        self.f_conv2 = nn.Conv2d(32, 64, 3)
        self.f_bn2 = nn.BatchNorm2d(64)
        self.f_pool2 = nn.MaxPool2d(2, 2)

        # 22 * 22
        self.f_conv3 = nn.Conv2d(64, 128, 3)
        self.f_bn3 = nn.BatchNorm2d(128)
        self.f_pool3 = nn.MaxPool2d(2, 2)

        
        # 10 * 10
        self.f_conv4 = nn.Conv2d(128, 256, 3)
        self.f_bn4 = nn.BatchNorm2d(256)
        self.f_pool4 = nn.MaxPool2d(2, 2)

        # 4 * 4
        self.f_conv5 = nn.Conv2d(256, 128, 3)
        self.f_bn5 = nn.BatchNorm2d(128)
        self.f_avgpool = nn.AvgPool2d(2, 2)

        # 1 * 1
        self.f_conv6 = nn.Conv2d(128, 7, 1)
        self.f_softmax = nn.Softmax(dim = 1)

        # skel network
        self.s_mlp1 = nn.Linear(75, 128)
        self.s_mlp2 = nn.Linear(128, 128)
        self.s_mlp3 = nn.Linear(128, 128)
        self.s_mlp4 = nn.Linear(128, 64)
        self.s_mlp5 = nn.Linear(64, 32)
        self.s_mlp6 = nn.Linear(32, 16)
        self.s_mlp7 = nn.Linear(16, 7)
        self.s_softmax = nn.Softmax(dim = 1)

    def Face(self, x):
        x = self.f_pool1(F.relu(self.f_conv1(x)))
        x = self.f_pool2(F.relu(self.f_conv2(x)))
        x = self.f_pool3(F.relu(self.f_conv3(x)))
        x = self.f_pool4(F.relu(self.f_conv4(x)))
        x = self.f_avgpool(F.relu(self.f_conv5(x)))
        x = self.f_conv6(x)
        x = x.view(-1, 7)
        x = self.f_softmax(x)
        return x

    def Skel(self, y):
        y = y.view(-1, 75)
        y = F.relu(self.s_mlp1(y))
        y = F.relu(self.s_mlp2(y))
        y = F.relu(self.s_mlp3(y))
        y = F.relu(self.s_mlp4(y))
        y = F.relu(self.s_mlp5(y))
        y = F.relu(self.s_mlp6(y))
        y = self.s_mlp7(y)
        y = self.s_softmax(y)
        return y

    def forward(self, x, y):
        x = self.Face(x)
        y = self.Skel(y)
        x = 0.9*x + 0.1*y
        return x
    
class MeshEncoding(nn.Module):
    def __init__(self):
        super().__init__()
        # fase network
        # input size is 94 * 94
        self.f_conv1 = nn.Conv2d(3, 32, 3)
        self.f_bn1 = nn.BatchNorm2d(32)
        self.f_pool1 = nn.MaxPool2d(2, 2)
        
        # 46 * 46
        self.f_conv2 = nn.Conv2d(32, 64, 3)
        self.f_bn2 = nn.BatchNorm2d(64)
        self.f_pool2 = nn.MaxPool2d(2, 2)

        # 22 * 22
        self.f_conv3 = nn.Conv2d(64, 128, 3)
        self.f_bn3 = nn.BatchNorm2d(128)
        self.f_pool3 = nn.MaxPool2d(2, 2)

        
        # 10 * 10
        self.f_conv4 = nn.Conv2d(128, 256, 3)
        self.f_bn4 = nn.BatchNorm2d(256)
        self.f_pool4 = nn.MaxPool2d(2, 2)

        # 4 * 4
        self.f_conv5 = nn.Conv2d(256, 128, 3)
        self.f_bn5 = nn.BatchNorm2d(128)
        self.f_avgpool = nn.AvgPool2d(2, 2)

        # 1 * 1
        self.f_conv6 = nn.Conv2d(128, 7, 1)
        self.f_softmax = nn.Softmax(dim = 1)

        # skel network
        self.s_mlp1 = nn.Linear(85, 128)
        self.s_mlp2 = nn.Linear(128, 128)
        self.s_mlp3 = nn.Linear(128, 128)
        self.s_mlp4 = nn.Linear(128, 64)
        self.s_mlp5 = nn.Linear(64, 32)
        self.s_mlp6 = nn.Linear(32, 16)
        self.s_mlp7 = nn.Linear(16, 7)
        self.s_softmax = nn.Softmax(dim = 1)


    def Face(self, x):
        x = self.f_pool1(F.relu(self.f_conv1(x)))
        x = self.f_pool2(F.relu(self.f_conv2(x)))
        x = self.f_pool3(F.relu(self.f_conv3(x)))
        x = self.f_pool4(F.relu(self.f_conv4(x)))
        x = self.f_avgpool(F.relu(self.f_conv5(x)))
        x = self.f_conv6(x)
        x = x.view(-1, 7)
        x = self.f_softmax(x)
        return x

    def Mesh(self, meshs):
        y = meshs
        y = y.view(-1, 85)
        y = F.relu(self.s_mlp1(y))
        y = F.relu(self.s_mlp2(y))
        y = F.relu(self.s_mlp3(y))
        y = F.relu(self.s_mlp4(y))
        y = F.relu(self.s_mlp5(y))
        y = F.relu(self.s_mlp6(y))
        y = self.s_mlp7(y)
        y = self.s_softmax(y)
        return y

    def forward(self, x, meshs):
        x = self.Face(x)
        y = self.Mesh(meshs)
        x = 0.9*x + 0.1*y
        return x