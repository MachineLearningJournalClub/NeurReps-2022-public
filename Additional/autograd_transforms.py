import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
dtype =  torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

tt = torch.tensor
class AffineTransform(nn.Module):
    def __init__(self, scalew=1, scaleh=1, transX=0., transY=0.):
        super().__init__()
        
        def makep(x):
            x = tt(x).float()
            return nn.Parameter(x)
        
        self.scalew = makep(scalew)
        self.scaleh = makep(scaleh)
        self.transX = makep(transX) 
        self.transY = makep(transY)
        
    def forward(self, x):
        theta = tt([
            [self.scalew, 0, self.transX],
            [0, self.scaleh, self.transY]
        ])[None]
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        return F.grid_sample(x, grid, align_corners=False)

def get_rot_mat(theta):
    theta = torch.tensor(theta)
    return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                         [torch.sin(theta), torch.cos(theta), 0]])


def rot_img(x, theta, dtype):
      rot_mat = get_rot_mat(theta)[None, ...].type(dtype).repeat(x.shape[0],1,1)
      grid = F.affine_grid(rot_mat, x.size()).type(dtype)
      x = F.grid_sample(x, grid)
      return x

img = pd.read_csv("./sample_data/mnist_test.csv").to_numpy()[0][1:]
img.shape
img = np.reshape(img, (28,28))
plt.imshow(img)
img = torch.tensor(img)
img = img / img.max()
img = img[None, None, :, :]

img.shape

x = img
target = torch.zeros(1, 1,200,200)
x[0, 0, 75:125,  75:125] = .5
stn = AffineTransform(transX=.2, transY=-.6)
stn2 = AffineTransform()
A = stn(x)[0]
B = stn2(target)[0]

plt.imshow(A.numpy()[0])
plt.imshow(A.numpy()[0])
img = rot_img(img, 10, dtype)
img = rot_img(img, -10, dtype)
plt.imshow(img[0][0])