import torch
import torch.nn as nn
import torch.nn.functional as F
import mynet
from mynet import *


class DisentangledAutoEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.latent_enc = deepAutoencoder(64*64,512,128,28)
        self.group_enc = geomEncoder(64*64,132,16)


    def group_action(self, group):
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
        self.rot, self.s, self.t1x, self.t1y = rot, s, t1x, t1y
        return matRes


    def apply_transform(self, free_latent_encoding, group_encoding):   
        transformed_encoding = 0
        return transformed_encoding

    def forward(self, x, evaluate=False):
        torch.cuda.is_available()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.geometric_encoding = self.group_enc(x)
        self.AE_output = self.latent_enc(x)
        self.group_action_matrices = self.group_action(self.geometric_encoding)       
        self.affine_transformation = F.affine_grid(
            self.group_action_matrices,
            self.AE_output.reshape((8,1,64,64)).shape,
            align_corners=False
        )
        self.affine_transformation = self.affine_transformation.to(device)
        if evaluate:
            self.affine_transformation = self.affine_transformation.to('cpu')
        
        self.transformed_x = F.grid_sample(self.AE_output.reshape((8,1,64,64)), self.affine_transformation, mode='bilinear')
        return self.transformed_x
