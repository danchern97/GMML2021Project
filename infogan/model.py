import torch
import torch.nn as nn
from typing import Tuple
import math

class Generator(nn.Module):
    def __init__(self, noise_features : int, parametric_features : int, 
                 before_conv_size : Tuple[int],
                 output_channels: int, dataset: str) -> None:
        super(Generator, self).__init__()
        self.dataset = dataset
        self.img_size_h, self.img_size_w = before_conv_size
        num_hidden_feat = int(math.sqrt(self.img_size_h*self.img_size_w*(noise_features + parametric_features)))
        self.fc1 = nn.Linear(noise_features + parametric_features, num_hidden_feat)
        self.bn1 = nn.BatchNorm1d(num_hidden_feat)
        self.fc2 = nn.Linear(num_hidden_feat, self.img_size_h*self.img_size_w)
        self.bn2 = nn.BatchNorm1d(self.img_size_h*self.img_size_w)

        self.conv1 = nn.ConvTranspose2d(1, 8, kernel_size=(3, 3))
        self.cbn1 = nn.BatchNorm2d(8)
        self.up1 = nn.ConvTranspose2d(8, 8, kernel_size=(2, 2), stride=2)
        if dataset == 'celeba':
            self.conv2 = nn.ConvTranspose2d(8, 8, kernel_size=(3, 3))
            self.cbn2 = nn.BatchNorm2d(8)
            self.up2 = nn.ConvTranspose2d(8, 8, kernel_size=(2, 2), stride=2)
        
        self.conv3 = nn.ConvTranspose2d(8, output_channels, kernel_size=(3, 3))
    
    def forward(self, x):
        x = self.bn1(self.fc1(x)).relu()
        x = self.bn2(self.fc2(x)).relu()
        x = x.view(-1, 1, self.img_size_h, self.img_size_w)
        x = self.up1(self.cbn1(self.conv1(x)).relu())
        if self.dataset == 'celeba':
            x = self.up2(self.cbn2(self.conv2(x)).relu())
        x = self.conv3(x).sigmoid()
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
