import torch
import torch.nn as nn
from typing import List, Tuple
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data.dataloader import DataLoader
from torch.distributions.uniform import Uniform
from torch.distributions.one_hot_categorical import OneHotCategorical
from torch.optim import Adam
from matplotlib import pyplot as plt
from IPython.display import clear_output
import math
from tqdm.auto import tqdm
import os

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

        self.conv1 = nn.ConvTranspose2d(16, 32, kernel_size=(3, 3))
        self.cbn1 = nn.BatchNorm2d(32)
        self.up1 = nn.ConvTranspose2d(32, 64, kernel_size=(2, 2), stride=2)
        if dataset == 'CelebA':
            self.conv2 = nn.ConvTranspose2d(64, 64, kernel_size=(3, 3))
            self.cbn2 = nn.BatchNorm2d(64)
            self.up2 = nn.ConvTranspose2d(64, 64, kernel_size=(2, 2), stride=2)
        self.conv3 = nn.ConvTranspose2d(64, 16, kernel_size=(3, 3))
        self.cbn3 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16, output_channels, kernel_size=(3, 3), padding=(1, 1))
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.bn1(self.fc1(x)).relu()
        x = self.bn2(self.fc2(x)).relu()
        x = x.view(-1, 16, self.img_size_h, self.img_size_w)
        x = self.up1(self.cbn1(self.conv1(x)).relu())
        if self.dataset == 'CelebA':
            x = self.up2(self.cbn2(self.conv2(x)).relu())
        x = self.cbn3(self.conv3(x)).relu()
        x = self.conv4(x).sigmoid()
        return x


class InfoDiscriminator(nn.Module):
    def __init__(self, categorical_features: int, non_categorical_features:int, 
                 after_conv_size: int, input_channels: int, dataset: str) -> None:
        super(InfoDiscriminator, self).__init__()
        self.dataset = dataset
        self.conv1 = nn.Conv2d(input_channels, 8, kernel_size=(3, 3))
        self.cbn1 = nn.BatchNorm2d(8)
        self.down1 = nn.MaxPool2d(kernel_size=(2, 2))
        if dataset == 'CelebA':
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
        self.out_q_cat = nn.Linear(num_hidden_feat, categorical_features)
        self.out_q_non_cat_mean = nn.Linear(num_hidden_feat, non_categorical_features)
        self.out_q_non_cat_logvar = nn.Linear(num_hidden_feat, non_categorical_features)
    
    def forward(self, x: Tensor) -> Tuple[Tensor]:
        x = self.down1(self.cbn1(self.conv1(x)).relu())
        if self.dataset == 'CelebA':
            x = self.down2(self.cbn2(self.conv2(x)).relu())
        x = self.cbn3(self.conv3(x)).relu()
        x = x.view(-1, 16*self.img_size_h*self.img_size_w)
        x = self.bn1(self.fc1(x)).relu()
        d = self.out_d(x).sigmoid()
        q_cat = self.out_q_cat(x)
        q_non_cat_mean, q_non_cat_logvar = self.out_q_non_cat_mean(x), self.out_q_non_cat_logvar(x)
        return d, q_cat, q_non_cat_mean, q_non_cat_logvar


class InfoGAN(nn.Module):
    def __init__(self, noise_features: int, categorical_features: List,
                 uniform_features: int, guassian_features: int,
                 inter_img_size: Tuple[int], dataset: str, batch_size: int, 
                 gen_lr: float, disc_lr:float, lambda_: float) -> None:

        super(InfoGAN, self).__init__()
        self.feature_spec = {'categorical': categorical_features, #list of integers charechterising number of categories
                             'uniform': uniform_features,
                             'gaussian': guassian_features, 
                             'noise': noise_features}
        parametric_features = sum(categorical_features) + uniform_features + guassian_features
        channels = 3 if dataset == 'CelebA' else 1

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.generator = Generator(noise_features=noise_features, 
                                   parametric_features=parametric_features,
                                   before_conv_size=inter_img_size, 
                                   output_channels=channels, 
                                   dataset=dataset).to(self.device)

        self.info_discriminator = InfoDiscriminator(categorical_features=sum(categorical_features),
                                                    non_categorical_features=uniform_features + guassian_features,
                                                    after_conv_size=inter_img_size,
                                                    input_channels=channels,
                                                    dataset=dataset).to(self.device)

        self.lambda_ = lambda_

        self.train_dataset = None
        self.train_dataloader = None
        self.batch_size = batch_size
        self.gen_optimizer = Adam(self.generator.parameters(), gen_lr)
        self.disc_optimizer = Adam(self.info_discriminator.parameters(), disc_lr)

        self.gen_loss_log = []
        self.disc_loss_log = []
        self.info_loss_log = []

        self.load_data(dataset, self.batch_size)

    def load_data(self, dataset, batch_size) -> None:
        download = False if os.path.exists('../data/'+dataset) else True
        if dataset == 'MNIST':
            self.train_dataset = datasets.MNIST(root='../data', train = True, 
                                                transform=ToTensor(), download=download)
        elif dataset == 'CelebA':
            self.train_dataset = datasets.CelebA(root='../data', split='train', 
                                                 transform=ToTensor(), download=download)

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)

    def generate_input(self) -> torch.Tensor:
        noise_input = torch.randn(self.feature_spec['noise'])

        categorical_input = []
        categorical_labels = []
        for n_cat in self.feature_spec['categorical']:
            categorical_input.append(OneHotCategorical(torch.ones(n_cat)/n_cat).sample())
            categorical_labels.append(torch.argmax(categorical_input[-1]))
        categorical_input = torch.hstack(categorical_input)
        gaussian_input = torch.randn(self.feature_spec['gaussian'])
        uniform_input = Uniform(-1, 1).sample((self.feature_spec['uniform'], ))

        gen_input = torch.hstack([noise_input, categorical_input, gaussian_input, uniform_input])
        gen_input = gen_input.to(self.device)
        return gen_input, torch.tensor(categorical_labels)
    
    def generate_batch(self, batch_size) -> torch.Tensor:
        gen_input_list, gen_cat_list = [], []
        for _ in range(batch_size):
            gen_input, gen_cats = self.generate_input()
            gen_input_list.append(gen_input.unsqueeze(0))
            gen_cat_list.append(gen_cats)
        return torch.vstack(gen_input_list), gen_cat_list

    def train_one_epoch(self) -> None:
        p_bar = tqdm(self.train_dataloader) 
        running_loss = 0
        gaussian_nnl = nn.GaussianNLLLoss()
        ce = nn.CrossEntropyLoss()
        for i, data in enumerate(p_bar):

            self.disc_optimizer.zero_grad()
            self.gen_optimizer.zero_grad()

            # Discriminator optimization step
            #### sampled data
            images, _ = data
            real_images = images.to(self.device)

            gen_input, _ = self.generate_batch(self.batch_size)
            # print(gen_input.shape)
            gen_images = self.generator(gen_input)

            prob_gen, _, _, _ = self.info_discriminator(gen_images)
            prob_real, _, _, _ = self.info_discriminator(real_images)

            dis_loss = -torch.log(prob_real).mean() -torch.log(1 - prob_gen).mean()
            self.disc_loss_log.append(dis_loss.item())
            dis_loss.backward()
            self.disc_optimizer.step()

            # Generator and Posterior optimization steps
            self.disc_optimizer.zero_grad()

            gen_input, gen_categories = self.generate_batch(self.batch_size)
            # print(gen_categories)
            gen_categories = torch.vstack(gen_categories).T.to(self.device)
            # print('gen_categories:', gen_categories)
            gen_images = self.generator(gen_input)

            prob_gen, post_cat, post_ncat_mean, post_ncat_logvar = self.info_discriminator(gen_images)
            post_ncatvar = torch.exp(post_ncat_logvar)

            gen_loss = -torch.log(prob_gen).mean()
            self.gen_loss_log.append(gen_loss.item())
            # print('prob_gen:\n',prob_gen)
            # print('post_cat:\n', post_cat)
            # print('post_ncat (mean + logvar):\n', post_ncat_mean, post_ncat_logvar)
            categorical_distributions = torch.split(post_cat, self.feature_spec['categorical'], dim=1)
            post_dist_gen_categorical = []
            for distribution, category, in zip(categorical_distributions, gen_categories):
                # print('1 cat dist:\n', distribution)
                # print('categories:', category)
                post_dist_gen_categorical.append(ce(distribution, category))

            info_loss = self.lambda_ * torch.sum(torch.hstack(post_dist_gen_categorical))

            info_loss += self.lambda_ * gaussian_nnl(post_ncat_mean, 
                                                     torch.zeros_like(post_ncat_mean), 
                                                     post_ncatvar)
            self.info_loss_log.append(-info_loss.item())
            (info_loss + gen_loss).backward()

            running_loss += dis_loss.item() + gen_loss.item() + info_loss.item()
            p_bar.set_description("Loss: {:.4f}".format(running_loss/(i+1)))
            self.gen_optimizer.step()
            self.info_discriminator.out_d.zero_grad()
            self.disc_optimizer.step()

    def illustrate(self, n: int) -> None:
        batch, _ = self.generate_batch(n**2)
        with torch.no_grad():
            images = self.generator(batch)
        clear_output(wait=True)
        _, axes = plt.subplots(n, n, figsize=(10, 10))
        for i in range(n**2):
            axes[i//n, i%n].imshow(images.cpu().numpy()[i, 0, ...])
        _, axes = plt.subplots(ncols=3, figsize=(18, 6), dpi=100)
        axes[0].plot(self.gen_loss_log)
        axes[1].plot(self.disc_loss_log)
        axes[2].plot(self.info_loss_log)
        axes[0].grid()
        axes[1].grid()
        axes[2].grid()
        axes[0].set_title('Generator loss')
        axes[1].set_title('Discriminator loss')
        axes[2].set_title('Mutual Info loss')
        plt.show()



    def fit(self, n_epochs, k=4) -> None:
        for _ in range(n_epochs):
            self.train()
            self.train_one_epoch()
            self.eval()
            self.illustrate(k)



