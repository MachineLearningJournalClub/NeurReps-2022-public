import torch
import torch.nn as nn
import torch.nn.functional as F

from importlib import reload

import mynet
reload(mynet)
from mynet import *

torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class GroupEquippedAutoEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.latent_enc = myNeuralNetwork(64*64,512,128,28)#(64*64,512,128,32)
        #self.latent_enc = ConvAutoencoder()
        #self.latent_enc = ConvAutoencoder2()
        #self.latent_enc = myLeNet(16)
        self.group_enc = myNeuralNetwork_group(64*64,132,16)


    def group_representation(self, group):
        # Expected size: group = (batch_size, 4)
        # http://totologic.blogspot.com/2015/02/2d-transformation-matrices-baking.html

        #group = torch.sigmoid(group)
        
        rot, s, t1x, t1y = group[:, 0], group[:,1], group[:, 2], group[:, 3]
        batch_size = group.size()[0]

        s = torch.abs(s)
        cRot = torch.cos(rot*2*torch.pi)
        sRot = torch.sin(rot*2*torch.pi)
        
        matRes = torch.zeros((batch_size, 2, 3))
        matRes[:, 0, 0] = s*cRot
        matRes[:, 0, 1] = s*-sRot
        matRes[:, 0, 2] = t1x*s*cRot + t1y*s*-sRot
        matRes[:, 1, 0] = s*sRot
        matRes[:, 1, 1] = s*cRot
        matRes[:, 1, 2] = t1x*s*sRot + t1y*s*cRot
        #matRes[:, 2, 0] = 0
        #matRes[:, 2, 1] = 0
        #matRes[:, 2, 2] = 1
        return matRes


    def apply_transform(self, free_latent_encoding, group_encoding):
        
        transformed_encoding = 0
        
        return transformed_encoding

    def forward(self, x, evaluate=False):
        self.geometric_encoding = self.group_enc(x)
        self.latent_weights = self.latent_enc(x)
        #print(self.latent_weights.shape)

        self.group_action_matrices = self.group_representation(self.geometric_encoding)
        #print(x.shape)
        #print('lat', self.latent_weights.shape)
        self.affine_grid_flow_field = F.affine_grid(
            self.group_action_matrices,
            self.latent_weights.reshape((8,1,64,64)).shape,
            align_corners=False
        )
        self.affine_grid_flow_field = self.affine_grid_flow_field.to(device)
        if evaluate:
            self.affine_grid_flow_field = self.affine_grid_flow_field.to('cpu')
        
        self.transformed_x = F.grid_sample(self.latent_weights.reshape((8,1,64,64)), self.affine_grid_flow_field, mode='bilinear')
        #self.affine_grid_flow_field = self.affine_grid_flow_field.to('cpu')
        #print('wow',self.transformed_x.shape)
        return self.transformed_x

        # output = group_action_matrices X latent_weights

        #final_image = self.get_image(latent_weights, affine_grid_flow_field)
        #final_image = final_image.unsqueeze(1)
        #return group_action_matrices, latent_weights, final_image

        
import dataset
reload(dataset)
from dataset import dSpritesDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from tqdm import tqdm
torch.pi = torch.acos(torch.zeros(1)).item()

dataset = dSpritesDataset()
l = dataset.__len__()
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [int(0.9*l), int(0.1*l)])
#train_dataset = train_dataset.to(device)
#test_dataset = test_dataset.to(device)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)


model = GroupEquippedAutoEncoder()
#model = UnifiedNet(64*64,512,128,28,132,16)
model.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=8e-4)#, weight_decay=1e-5)

loss_history = []
loss_history_test = []

for epoch in tqdm(range(200)):
    running_loss = 0
    running_loss_test = 0
    for img in train_loader:
        img = img.to(device)
        optimizer.zero_grad()
        #group_action_matrices, latent_weights, final_image = model(img)
        
        final_image = model(img)
        final_image = final_image.to(device)
        loss = criterion(final_image, img)
        loss.backward()
        optimizer.step()
        lossitem = loss.item()
        running_loss += lossitem / 8.

    for img in test_loader:
        img = img.to(device)
        final_image = model(img)
        final_image = final_image.to(device)
        loss = criterion(final_image, img)
        lossitem = loss.item()
        running_loss_test += lossitem / 8.
    
    loss_history.append(running_loss / (125.*8./9.))
    loss_history_test.append(running_loss_test / (125./9.))
    print(epoch, "train", running_loss / (125.*8./9.))
    print(epoch, "test", running_loss_test / (125./9.))
    
    
import numpy as np
import matplotlib.pyplot as plt


testLoss   =  np.array(loss_history_test)*8/9
trainLoss  =  np.array(loss_history)/9

import pickle
with open('geometricModelBatchTraining.pkl', 'wb') as outp:
    geometric = {'testLoss': testLoss, 'trainLoss': trainLoss}
    pickle.dump(geometric, outp, pickle.HIGHEST_PROTOCOL)


