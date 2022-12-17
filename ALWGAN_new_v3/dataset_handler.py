import torch
from torch.utils.data import Dataset
import random
import numpy as np



class myData(Dataset):
    
    def __init__(self,x,y):
        self.x=x
        self.y=y
        self.shape=x.size(0)
        
    def __getitem__(self,index):
        return self.x[index],self.y[index]
    
    def __len__(self):
        return self.shape
    

class WA_myData(Dataset):

    def __init__(self,X_1, Y_1):
        
        self.X1 = X_1
        self.Y1 = Y_1

    def __len__(self):
        return len(self.X1)

    def __getitem__(self, index):
        
        x_1 = self.X1[index]
        y_1 = self.Y1[index]

        return x_1,y_1

