#!/usr/bin/env python


import torch

import torch.nn as nn

class PsiICA(nn.Module):
    def __init__(self, u_dim, dropout=0.25):
        """ """
        super().__init__()
        self.m = nn.Sequential(*[
            nn.ReLU(), 
            nn.Dropout(p=dropout), 
            nn.Linear(u_dim+1, 128),
            nn.ReLU(), 
            nn.Dropout(p=dropout),
            nn.Linear(128, 1)
        ])
        
    def forward(self, x, u):
        x = torch.cat((x, u), dim=-1)
        return self.m(x)

class NonLinearICA(nn.Module):
    def __init__(self, in_channels, n, dropout=0.25, data_type='CelebA'):
        """
        
        
        """
        super().__init__()
        
        if data_type == 'CelebA':
            hidden_dim = 79200
            u_dim = 40
        elif data_type == 'MNIST':
            hidden_dim = 1568
            u_dim = 10
        
        self.h = nn.Sequential(*[
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1), # same size
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # MNIST: 14 x 14, CelebA: 109 X 89
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), # MNIST: 14 x 14, CelebA: 109 X 89
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1), # MNIST: 7 x 7, CelebA: 55 X 45
            nn.Flatten(),
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, n)
        ])
        self.psi = nn.ModuleList([PsiICA(u_dim) for i in range(n)])
        
    def forward(self, x, u):
        x = self.h(x) # batch_size X n
        return sum([psi(x[:, i].unsqueeze(-1), u) for i, psi in enumerate(self.psi)])
    
    def forward_h(self, x):
        return self.h(x)