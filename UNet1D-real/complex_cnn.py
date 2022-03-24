# https://github.com/Pytorch-Complex_CNN
""" Full assembly of the parts to form the complete network """

import torch
import torch.nn as nn
import torch.nn.functional as F

# Resnet Basic Block module�C
class Res_BasicBlock(nn.Module):
    def __init__(self, in_ch, kernelsize, stride=1):
        super(Res_BasicBlock, self).__init__()
        padding = int((kernelsize - 1) / 2)
        
        self.bblock = nn.Sequential(
            nn.Conv1d(in_ch, 32, kernel_size=kernelsize, padding=padding),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            nn.Conv1d(32, 16, kernel_size=kernelsize, padding=padding),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),

            nn.Conv1d(16, 32, kernel_size=kernelsize, padding=padding),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )
                              
        self.jump_layer = lambda x:x


    def forward(self, inputs, training=None):

        #Through the convolutional layer
        out = self.bblock(inputs)

        #skip
        identity = self.jump_layer(inputs)

        output = out + identity
    
        return output

class BasicBlockall(nn.Module):
    def __init__(self, in_ch, stride=1):
        super(BasicBlockall, self).__init__()

        self.bblock3 = nn.Sequential(Res_BasicBlock(in_ch, 3),
                              Res_BasicBlock(in_ch, 3)
                              )                      
    
        self.bblock5 = nn.Sequential(Res_BasicBlock(in_ch, 5),
                              Res_BasicBlock(in_ch, 5)
                              )                      

        self.bblock7 = nn.Sequential(Res_BasicBlock(in_ch, 7),
                              Res_BasicBlock(in_ch, 7)
                              )
                              
        self.downsample = lambda x:x


    def forward(self, inputs, training=None):
 
        out3 = self.bblock3(inputs)
        out5 = self.bblock5(inputs)
        out7 = self.bblock7(inputs)

        out = torch.cat((out3,out5,out7), -1)

        return out


class Complex_CNN(nn.Module):
    def __init__(self, in_channels, out_channels, datanum, bilinear=True):
        super(Complex_CNN, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=5, padding=2)
        self.norm1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU(inplace=True)

        self.bblock = BasicBlockall(32, stride=1)

        self.conv2 = nn.Conv1d(32, out_channels, kernel_size=1, padding=2)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.linear1 = nn.Linear(3076, 1024)

    def forward(self, x):
        out = self.relu1(self.norm1(self.conv1(x)))
        out = self.bblock(out)
        out = self.relu2(self.norm2(self.conv2(out)))
        out = self.linear1(out)
        return out

