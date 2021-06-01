#!/usr/bin/env python

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam, SGD

class CelebA_dataset(Dataset):
    def __init__(self, data):
        """
        Class for CelebA dataset processing.
        
        The __getitem__ method provides a tuple of an image, u (one-hot encoded attributes), and u_star (permuted one-hot encoded atrributes).
        
        Args:
            data (torchvision.datasets.celeba.CelebA): CelebA dataset
        
        """
        super().__init__()
        self.data = data
        self.u_star = data.attr[torch.randperm(data.attr.shape[0])]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image, (attr, identity)  = self.data[idx]
        image = torch.tensor(np.array(image), dtype=torch.float32).moveaxis(-1, 0)
        u = attr
        u_star = self.u_star[idx]
        return image, u, u_star
    
    
class MNIST_dataset(Dataset):
    def __init__(self, data):
        """
        Class for MNIST dataset processing.
        
        The __getitem__ method provides a tuple of an image, u (one-hot encoded attributes), and u_star (permuted one-hot encoded atrributes).
        
        Args:
            data ( torchvision.datasets.mnist.MNIST): MNIST dataset
       
        """
        super().__init__()
        self.data = data
        self.u_star = data.targets[torch.randperm(len(data.targets))]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image, u  = self.data[idx]
        u_star = self.u_star[idx]
        image = torch.tensor(np.array(image), dtype=torch.float32).unsqueeze(0)
        u = nn.functional.one_hot(torch.tensor(u), len(self.data.classes))
        u_star = nn.functional.one_hot(u_star, len(self.data.classes))
        return image, u, u_star



def train_model(model, train_dataloader, test_dataloader, log_dir, epochs=30, lr=0.01, optimizer="SGD", device='cuda:9'):
    writer = SummaryWriter(log_dir)
    
    if optimizer == "SGD":
        optimizer = SGD(model.parameters(), lr=lr)
    elif optimizer == "Adam":
        optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    
    for epoch in tqdm(range(epochs), desc="Training on epoch"):
        model.train()
        for i, batch in enumerate(train_dataloader):
            x, u, u_star = batch
            labels = torch.randint(0, 2, size=(x.shape[0], 1), dtype=x.dtype) # choose random u or u_star -> labels
            u = torch.where(labels.bool(), u, u_star) # get u or u_star depending on label

            x = x.to(device)
            u = u.to(device)
            labels = labels.to(device)

            output = model(x, u)
            loss = criterion(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = (torch.sigmoid(output) > 0.5).float()
            correct = (pred == labels).float().sum()
            writer.add_scalar("Train/loss", loss.cpu().item(), len(train_dataloader)*epoch + i)
            writer.add_scalar("Train/accuracy", correct/output.shape[0], len(train_dataloader)*epoch + i)
        model.eval()
        val_loss = 0.0
        val_correct = 0.0
        with torch.no_grad():
            for batch in test_dataloader:
                x, u, u_star = batch
                labels = torch.randint(0, 2, size=(x.shape[0], 1), dtype=x.dtype) # choose random u or u_star -> labels
                u = torch.where(labels.bool(), u, u_star) # get u or u_star depending on label

                x = x.to(device)
                u = u.to(device)
                labels = labels.to(device)

                output = model(x, u)
                loss = criterion(output, labels)
                val_loss += loss.cpu().item()

                pred = (torch.sigmoid(output) > 0.5).float()
                val_correct += (pred == labels).float().sum()
        writer.add_scalar("Test/loss", val_loss / len(test_dataloader), epoch)
        writer.add_scalar("Test/accuracy", val_correct / (len(test_dataloader)*batch_size), epoch)
    return model