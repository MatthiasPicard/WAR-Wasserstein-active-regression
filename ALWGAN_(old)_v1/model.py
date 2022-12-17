import torch.nn as nn
import torch.nn.functional as F
import torch
import pandas as pd

def get_net(name):
        return  ffd_h, ffd_phi

# net_1  for Mnist and Fashion_mnist

class ffd_h(nn.Module):
    """
    Classifier network, also give the latent space and embedding dimension

    """

    def __init__(self):
        super(ffd_h,self).__init__()
        self.fc1 = nn.Linear(in_features=13, out_features=1)

    def forward(self,x):
        #print('X',x.dtype,x.shape,x[0])
        out= F.relu(self.fc1(x.float()))
        '''TESTER VALEUR ABSOLUE'''
        #print(out)
        #print('OUT',out.dtype,out.shape,out[0])
        return out

    
    

    
class ffd_phi(nn.Module):

    """
    Discriminator network, output with [0,1] (sigmoid function)

    """
    def __init__(self):
        super(ffd_phi,self).__init__()
        self.fc1 = nn.Linear(in_features=13, out_features=1)

    def forward(self,x):
        x = F.sigmoid(self.fc1(x))

        return x

