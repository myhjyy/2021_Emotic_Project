import torch
from torch import nn
import torch.nn.functional as F



class FaceEncoding(nn.Module):
    def __init__(self):
        super().__init__()
        #712
        self.conv1 = nn.Conv3d(3, 32, 3)
        self.bn1 = nn.BatchNorm3d(32)
        self.pool1 = nn.MaxPool3d(1, 2, 2)
        
        #355
        self.conv2 = nn.Conv3d(32, 64, 3)
        self.bn2 = nn.BatchNorm3d(64)
        self.pool2 = nn.MaxPool3d(2)

        #176
        self.conv3 = nn.Conv3d(64, 128, 3)
        self.bn3 = nn.BatchNorm3d(128)
        self.pool3 = nn.MaxPool3d(2)

        
        #87
        self.conv4 = nn.Conv3d(128, 256, 3)
        self.bn4 = nn.BatchNorm3d(256)
        self.pool4 = nn.MaxPool3d(2)

        #42
        self.conv5 = nn.Conv3d(256, 256, 3)
        self.bn5 = nn.BatchNorm3d(256)
        self.avgpool = nn.AvgPool3d(2)

        #20
        #Adaptive fusion networks
        self.conv6 = nn.Conv3d( 128, 128, 1)
        self.conv7 = nn.Conv3d( 128, 1, 1)
        self.mlp1 = nn.Linear(20, 7)


    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = self.avgpool(F.relu(self.bn5(self.conv5(x))))

        #Adaptive fusion networks
        x = F.relu(self.conv6(x))
        # x = F.dropout(x)
        x = F.relu(self.conv7(x))
        x = torch.flatten(x,1)
        x = self.mlp1(x)
        x = F.softmax(x)

        
        # x = x.view(-1, 10 * 1 * 1)
        # x = F.relu(self.mlp1(x))
        
        return x



'''
class ContextEncodig(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(3, 32, 3)
        self.bn1 = nn.BatchNorm3d(32)
        self.pool1 = nn.MaxPool3d(1, 2, 2)

        self.conv2 = nn.Conv3d(32, 64, 3)
        self.bn2 = nn.BatchNorm3d(64)
        self.pool2 = nn.MaxPool3d(2)

        self.conv3 = nn.Conv3d(64, 128, 3)
        self.bn3 = nn.BatchNorm3d(128)
        self.pool3 = nn.MaxPool3d(2)

        self.conv4 = nn.Conv3d(128, 256, 3)
        self.bn4 = nn.BatchNorm3d(128)
        self.pool4 = nn.MaxPool3d(2)

        self.conv5 = nn.Conv3d(256, 256, 3)
        self.bn5 = nn.BatchNorm3d(256)
        

        self.conv6 = nn.Conv3d(256,128,3)
        self.bn6 = nn.BatchNorm3d(128)

        self.conv7 = nn.Conv3d(128,1,3)
        self.bn7 = nn.BatchNorm3d(1)

        self.avgpool = nn.AvgPool3d(2)

        #Adaptive fusion networks
        self.conv8 = nn.Conv3d( 128, 128, 1)
        
        self.conv9 = nn.Conv3d( 128, 1, 1)

        
        

        # self.mlp1 = nn.Linear(10, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = F.relu(self.bn5(self.conv5(x)))
        

        final_x = F.relu(self.bn6(self.conv6(x)))
        final_x = F.relu(self.bn7(self.conv7(final_x)))
        final_x = F.softmax(final_x)
        final_x = torch.matmul(final_x, x)
        final_x = self.avgpool(final_x)

        final_x = F.relu(self.conv8(final_x))
        final_x = F.softmax(final_x)
        final_x = self.conv9(final_x)         

        
        # x = x.view(-1, 10 * 1 * 1)
        # x = F.relu(self.mlp1(x))
        
        return final_x

'''