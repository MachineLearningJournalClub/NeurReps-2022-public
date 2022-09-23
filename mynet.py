import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch import flatten

class deepAutoencoder(nn.Module):
    def __init__(self, N, N1, N2, N3):
        # N = input dim
        super(deepAutoencoder, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(N, N1),
            nn.ReLU(),
            nn.Linear(N1, N2),
            nn.ReLU(),
            nn.Linear(N2, N3),
            nn.ReLU(),
            nn.Linear(N3, N2),
            nn.ReLU(),
            nn.Linear(N2, N)
        )

        self.nonlinearity = nn.Sigmoid()

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        #logits = self.nonlinearity(x)
        return logits

class geomEncoder(nn.Module):
    def __init__(self, N, N1, N2):
        super(geomEncoder, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(N, N1),
            nn.ReLU(),
            nn.Linear(N1, N2),
            nn.ReLU(),
            nn.Linear(N2, 4)
        )

        self.nonlinearity = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        #logits = self.nonlinearity(x)
        return logits


class UnifiedNet(nn.Module):
    def __init__(self, N, N1, N2, N3, M1, M2):
        super(UnifiedNet, self).__init__()
        self.flatten = nn.Flatten()
        
        self.shape_encoder = nn.Sequential(
            nn.Linear(N, N1),
            nn.ReLU(),
            nn.Linear(N1, N2),
            nn.ReLU(),
            nn.Linear(N2, N3),
        )
        
        self.geom_encoder = nn.Sequential(
            nn.Linear(N, M1),
            nn.ReLU(),
            nn.Linear(M1, M2),
            nn.ReLU(),
            nn.Linear(M2, 4)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(N3 + 4, N2),
            nn.ReLU(),
            nn.Linear(N2, N)
        )


    def forward(self, x):
        x = self.flatten(x)
        logits1 = self.shape_encoder(x)
        logits2 = self.geom_encoder(x)
       
        x = torch.cat((logits1, logits2), dim=1)
        
        res = self.decoder(x).reshape(8,1,64,64)
        
        return res