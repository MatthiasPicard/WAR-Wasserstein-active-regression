import torch
import torch.nn as nn
import torchvision
from monotonenorm import GroupSort, direct_norm,project_norm


class NN_phi(nn.Module):
    def __init__(self,dim_input,config_NN_phi,dim_output,norm):
        
        self.config_NN_phi=config_NN_phi
        
        super(NN_phi, self).__init__()
        self.linear1=direct_norm(torch.nn.Linear(dim_input,config_NN_phi[1]),kind="two-inf")
        self.group1=GroupSort(config_NN_phi[1]//2)
        if(config_NN_phi[0]==1):
            self.linear6=direct_norm(torch.nn.Linear(config_NN_phi[1],1),kind=norm)
        else:
            self.linear2=direct_norm(torch.nn.Linear(config_NN_phi[1],config_NN_phi[2]),kind=norm)
            self.group2=GroupSort(config_NN_phi[2]//2)
        if(config_NN_phi[0]==2):
            self.linear6=direct_norm(torch.nn.Linear(config_NN_phi[2],1),kind=norm)
        if(config_NN_phi[0]>2):
            self.linear3=direct_norm(torch.nn.Linear(config_NN_phi[2],config_NN_phi[3]),kind=norm)
            self.group3=GroupSort(config_NN_phi[3]//2)
        if(config_NN_phi[0]==3):
            self.linear6=direct_norm(torch.nn.Linear(config_NN_phi[3],1),kind=norm)
        if(config_NN_phi[0]==4):
            self.linear4=direct_norm(torch.nn.Linear(config_NN_phi[3],config_NN_phi[4]),kind=norm)
            self.group4=GroupSort(config_NN_phi[4]//2)
            self.linear6=direct_norm(torch.nn.Linear(config_NN_phi[4],1),kind=norm)
        
        
        

    def forward(self, x):
        x=self.linear1(x)
        x=self.group1(x)
        if(self.config_NN_phi[0]==1):
            x=self.linear6(x)
        else:
            x=self.linear2(x)
            x=self.group2(x)
        if(self.config_NN_phi[0]==2):
            x=self.linear6(x)
        if(self.config_NN_phi[0]>2):
            x=self.linear3(x)
            x=self.group3(x)
        if(self.config_NN_phi[0]==3):
            x=self.linear6(x)
        if(self.config_NN_phi[0]==4):
            x=self.linear4(x)
            x=self.group4(x)
            x=self.linear6(x)
       
        
        return x
    

class NN_h(nn.Module):
    def __init__(self,dim_input,config_NN_h,dim_output=1,norm="inf",max_norm=2):
        
        self.config_NN_h=config_NN_h
        
        super(NN_h, self).__init__()
        self.linear1=direct_norm(torch.nn.Linear(dim_input,config_NN_h[1]),kind="two-inf",max_norm=max_norm)
        self.group1=GroupSort(config_NN_h[1]//2)
        self.linear2=direct_norm(torch.nn.Linear(config_NN_h[1],config_NN_h[2]),kind=norm,max_norm=max_norm)
        self.group2=GroupSort(config_NN_h[2]//2)
        if(config_NN_h[0]==2):
            self.linear6=direct_norm(torch.nn.Linear(config_NN_h[2],1),kind=norm)
        if(config_NN_h[0]>2):
            self.linear3=direct_norm(torch.nn.Linear(config_NN_h[2],config_NN_h[3]),kind=norm,max_norm=max_norm)
            self.group3=GroupSort(config_NN_h[3]//2)
        if(config_NN_h[0]==3):
            self.linear6=direct_norm(torch.nn.Linear(config_NN_h[3],1),kind=norm)
        if(config_NN_h[0]==4):
            self.linear4=direct_norm(torch.nn.Linear(config_NN_h[3],config_NN_h[4]),kind=norm,max_norm=max_norm)
            self.group4=GroupSort(config_NN_h[4]//2)
            self.linear6=direct_norm(torch.nn.Linear(config_NN_h[4],1),kind=norm,max_norm=max_norm)
        
        
        

    def forward(self, x):
        x=self.linear1(x)
        x=self.group1(x)
        x=self.linear2(x)
        x=self.group2(x)
        if(self.config_NN_h[0]==2):
            x=self.linear6(x)
        if(self.config_NN_h[0]>2):
            x=self.linear3(x)
            x=self.group3(x)
        if(self.config_NN_h[0]==3):
            x=self.linear6(x)
        if(self.config_NN_h[0]==4):
            x=self.linear4(x)
            x=self.group4(x)
            x=self.linear6(x)
       
        
        return x

    
class NN_h_RELU2(nn.Module):
    def __init__(self,dim_input,config_NN_h,dim_output=1):
        super(NN_h_RELU2, self).__init__()
        self.config_NN_h=config_NN_h
        self.net_dropped = torch.nn.Sequential(
        torch.nn.Linear(dim_input, 64),
        torch.nn.Dropout(0.5),  # drop 50% of the neuron
        torch.nn.ReLU(),
        torch.nn.Linear(64, 64),
        torch.nn.Dropout(0.5),  # drop 50% of the neuron
        torch.nn.ReLU(),  
        torch.nn.Linear(64, 1),
        )

    def forward(self, x):
        logits = self.net_dropped(x)
        return logits
    
    


class NN_h_RELU(nn.Module):
    def __init__(self,dim_input,config_NN_h,dim_output=1):
        
        self.config_NN_h=config_NN_h
        
        super(NN_h_RELU, self).__init__()
        
        
        self.linear1=torch.nn.Linear(dim_input,config_NN_h[1])
        self.dropout1=torch.nn.Dropout(p=0)
        self.group1=torch.nn.ReLU()
        if(config_NN_h[0]==1):
            self.linear6=torch.nn.Linear(config_NN_h[1],1)
        else:
            self.linear2=torch.nn.Linear(config_NN_h[1],config_NN_h[2])
            self.dropout2=torch.nn.Dropout(p=0)
            self.group2=torch.nn.ReLU()
        if(config_NN_h[0]==2):
            self.linear6=torch.nn.Linear(config_NN_h[2],1)
        if(config_NN_h[0]>2):
            self.linear3=torch.nn.Linear(config_NN_h[2],config_NN_h[3])
            self.dropout3=torch.nn.Dropout(p=0)
            self.group3=torch.nn.ReLU()
        if(config_NN_h[0]==3):
            self.linear6=torch.nn.Linear(config_NN_h[3],1)
        if(config_NN_h[0]==4):
            self.linear4=torch.nn.Linear(config_NN_h[3],config_NN_h[4])
            self.dropout4=torch.nn.Dropout(p=0)
            self.group4=torch.nn.ReLU()
            self.linear6=torch.nn.Linear(config_NN_h[4],1)
        
        
        

    def forward(self, x):
        x=self.linear1(x)
        x=self.dropout1(x)
        x=self.group1(x)
        if(self.config_NN_h[0]==1):
            x=self.linear6(x)
        else:
            x=self.linear2(x)
            x=self.dropout2(x)
            x=self.group2(x)
        if(self.config_NN_h[0]==2):
            x=self.linear6(x)
        if(self.config_NN_h[0]>2):
            x=self.linear3(x)
            x=self.dropout3(x)
            x=self.group3(x)
        if(self.config_NN_h[0]==3):
            x=self.linear6(x)
        if(self.config_NN_h[0]==4):
            x=self.linear4(x)
            x=self.dropout4(x)
            x=self.group4(x)
            x=self.linear6(x)
       
        
        return x