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
    



