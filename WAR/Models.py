import torch
import torch.nn as nn
import torchvision
from monotonenorm import GroupSort, direct_norm


class NN_phi(nn.Module):
    
    def __init__(self,dim_input):
        
        
        super(NN_phi, self).__init__()
        self.linear1=direct_norm(torch.nn.Linear(dim_input,16),kind="two-inf")
        self.group1=GroupSort(16//2)
        self.linear2=direct_norm(torch.nn.Linear(16,32),kind="inf")
        self.group2=GroupSort(32//2)
        self.linear3=direct_norm(torch.nn.Linear(32,1),kind="inf")
    

    def forward(self, x):
        x=self.linear1(x)
        x=self.group1(x)
        x=self.linear2(x)
        x=self.group2(x)
        x=self.linear3(x)
        
        return x
    


class NN_h_RELU(nn.Module):
    def __init__(self,dim_input):
        
        
        super(NN_h_RELU, self).__init__()
        
        
        self.linear1=torch.nn.Linear(dim_input,16)
        self.RELU=torch.nn.ReLU()
        self.linear2=torch.nn.Linear(16,32)
        self.linear3=torch.nn.Linear(32,1)
            
            
            

    def forward(self, x):
        x=self.linear1(x)
        x=self.RELU(x)
        x=self.linear2(x)
        x=self.RELU(x)
        x=self.linear3(x)
            
        return x
    
    
    