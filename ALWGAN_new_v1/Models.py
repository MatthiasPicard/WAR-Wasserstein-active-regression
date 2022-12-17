import torch
import torch.nn as nn
import torchvision
from monotonenorm import GroupSort, direct_norm,project_norm


class NN_phi(nn.Module):
    def __init__(self,dim_input,dim_output,norm):
        
        dim_hidden= 64
        super(NN_phi, self).__init__()
        self.linear1=direct_norm(torch.nn.Linear(dim_input,dim_hidden ),kind="two-inf")
        self.group1=GroupSort(dim_hidden//2)
        self.linear2=direct_norm(torch.nn.Linear(dim_hidden,dim_hidden),kind=norm)
        self.linear3=direct_norm(torch.nn.Linear(dim_hidden,dim_hidden),kind=norm)
        self.linear4=direct_norm(torch.nn.Linear(dim_hidden,dim_hidden),kind=norm)
        self.linear5=direct_norm(torch.nn.Linear(dim_hidden,dim_hidden),kind=norm)
        self.linear6=direct_norm(torch.nn.Linear(dim_hidden,dim_hidden),kind=norm)
        self.linear7=direct_norm(torch.nn.Linear(dim_hidden,1),kind=norm)
        
        

    def forward(self, x):
        x=self.linear1(x)
        x=self.group1(x)
        x=self.linear2(x)
        x=self.group1(x)
        x=self.linear3(x)
        #x=self.group1(x)
        #x=self.linear4(x)
        #x=self.group1(x)
        #x=self.linear5(x)
        #x=self.group1(x)
        #x=self.linear6(x)
        x=self.group1(x)
        x=self.linear7(x)
        
        return x
    

class NN_h(nn.Module):
    def __init__(self,dim_input,dim_output,norm):
        
        dim_hidden= 64
        super(NN_h, self).__init__()
        self.linear1=direct_norm(torch.nn.Linear(dim_input,dim_hidden),kind="two-inf")
        self.group1=GroupSort(dim_hidden//2)
        self.linear2=direct_norm(torch.nn.Linear(dim_hidden,dim_hidden),kind=norm)
        self.linear3=direct_norm(torch.nn.Linear(dim_hidden,dim_hidden),kind=norm)
        self.linear4=direct_norm(torch.nn.Linear(dim_hidden,dim_hidden),kind=norm)
        self.linear5=direct_norm(torch.nn.Linear(dim_hidden,dim_hidden),kind=norm)
        self.linear6=direct_norm(torch.nn.Linear(dim_hidden,dim_hidden),kind=norm)
        self.linear7=direct_norm(torch.nn.Linear(dim_hidden,1),kind=norm)



    def forward(self, x):
        x=self.linear1(x)
        x=self.group1(x)
        x=self.linear2(x)
        x=self.group1(x)
        x=self.linear3(x)
        #x=self.group1(x)
        #x=self.linear4(x)
        #x=self.group1(x)
        #x=self.linear5(x)
        #x=self.group1(x)
        #x=self.linear6(x)
        x=self.group1(x)
        x=self.linear7(x)

        return x
    