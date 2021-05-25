import torch
import torch.nn as nn
from typing import List, Tuple
import math

from torch.tensor import Tensor

class Generator(nn.Module):
    def __init__(self, noise_features : int, parametric_features : int, 
                 before_conv_size : Tuple[int],
                 output_channels: int, dataset: str) -> None:
        super(Generator, self).__init__()
        self.dataset = dataset
        self.img_size_h, self.img_size_w = before_conv_size
        num_hidden_feat = int(math.sqrt(16*self.img_size_h*self.img_size_w*(noise_features + parametric_features)))
        self.fc1 = nn.Linear(noise_features + parametric_features, num_hidden_feat)
        self.bn1 = nn.BatchNorm1d(num_hidden_feat)
        self.fc2 = nn.Linear(num_hidden_feat, 16*self.img_size_h*self.img_size_w)
        self.bn2 = nn.BatchNorm1d(16*self.img_size_h*self.img_size_w)

        self.conv1 = nn.ConvTranspose2d(16, 8, kernel_size=(3, 3))
        self.cbn1 = nn.BatchNorm2d(8)
        self.up1 = nn.ConvTranspose2d(8, 8, kernel_size=(2, 2), stride=2)
        if dataset == 'celeba':
            self.conv2 = nn.ConvTranspose2d(8, 8, kernel_size=(3, 3))
            self.cbn2 = nn.BatchNorm2d(8)
            self.up2 = nn.ConvTranspose2d(8, 8, kernel_size=(2, 2), stride=2)
        self.conv3 = nn.ConvTranspose2d(8, output_channels, kernel_size=(3, 3))
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.bn1(self.fc1(x)).relu()
        x = self.bn2(self.fc2(x)).relu()
        x = x.view(-1, 16, self.img_size_h, self.img_size_w)
        x = self.up1(self.cbn1(self.conv1(x)).relu())
        if self.dataset == 'celeba':
            x = self.up2(self.cbn2(self.conv2(x)).relu())
        x = self.conv3(x).sigmoid()
        return x


class InfoDiscriminator(nn.Module):
    def __init__(self, parametric_features: int, after_conv_size: int, 
                 input_channels: int, dataset: str) -> None:
        super(InfoDiscriminator, self).__init__()
        self.dataset = dataset
        self.conv1 = nn.Conv2d(input_channels, 8, kernel_size=(3, 3))
        self.cbn1 = nn.BatchNorm2d(8)
        self.down1 = nn.MaxPool2d(kernel_size=(2, 2))
        if dataset == 'celeba':
            self.conv2 = nn.Conv2d(8, 8, kernel_size=(3, 3))
            self.cbn2 = nn.BatchNorm2d(8)
            self.down2 = nn.MaxPool2d(kernel_size=(2, 2))        
        self.conv3 = nn.Conv2d(8, 16, kernel_size=(3, 3))
        self.cbn3 = nn.BatchNorm2d(16)

        self.img_size_h, self.img_size_w = after_conv_size
        num_hidden_feat = int(math.sqrt(16*self.img_size_h*self.img_size_w))
        self.fc1 = nn.Linear(16*self.img_size_h*self.img_size_w, num_hidden_feat)
        self.bn1 = nn.BatchNorm1d(num_hidden_feat)
        self.out_d = nn.Linear(num_hidden_feat, 1)
        self.out_q = nn.Linear(num_hidden_feat, parametric_features)
    
    def forward(self, x: Tensor) -> Tuple[Tensor]:
        x = self.down1(self.cbn1(self.conv1(x)).relu())
        if self.dataset == 'celeba':
            x = self.down2(self.cbn2(self.conv2(x)).relu())
        x = self.cbn3(self.conv3(x)).relu()
        x = x.view(-1, 16*self.img_size_h*self.img_size_w)
        x = self.bn1(self.fc1(x)).relu()
        d = self.out_d(x)
        q = self.out_q(x)
        return d, q


class InfoGAN(nn.Module):
    def __init__(self, noise_features: int, categorical_features: List,
                 uniform_features: int, guassian_features: int,
                 inter_img_size: Tuple[int], dataset: str) -> None:

        super(InfoGAN, self).__init__()
        self.feature_spec = {'categorical': categorical_features,
                             'uniform': uniform_features,
                             'gaussian': guassian_features}
        parametric_features = len(categorical_features) + uniform_features + guassian_features
        channels = 3 if dataset == 'celeba' else 1

        self.generator = Generator(noise_features=noise_features, 
                                   parametric_features=parametric_features,
                                   before_conv_size=inter_img_size, 
                                   output_channels=channels, 
                                   dataset=dataset)

        self.info_discriminator = InfoDiscriminator(parametric_features=parametric_features,
                                                    after_conv_size=inter_img_size,
                                                    input_channels=channels,
                                                    dataset=dataset)

        self.load_data()

    def load_data(self, dataset):
        pass

    def train_one_epoch(self, dataloader) -> None:
        pass

    def fit(self, dataloader) -> None:
        pass



