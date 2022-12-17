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
        self.hidden = nn.Linear(13, 100)   # hidden layer
        self.predict = nn.Linear(100, 1)

    def forward(self,x):
        x = F.relu(self.hidden(x.float()))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x

    
    

    
class ffd_phi(nn.Module):

    """
    Discriminator network, output with [0,1] (sigmoid function)

    """
    def __init__(self):
        super(ffd_phi,self).__init__()
        self.hidden = nn.Linear(13, 100)   # hidden layer
        self.predict = nn.Linear(100, 1)
        
    def forward(self,x):
        x = F.sigmoid(self.hidden(x.float()))      # activation function for hidden layer
        x = self.predict(x)   
        return x

