import numpy as np
import torch
from torchvision import datasets
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

def get_dataset(name):
    df = pd.read_csv(name+'.csv').to_numpy()
    np.random.shuffle(df)
    n_sep=round(len(df)*0.8)
    X_tr=np.array([i[1:-1] for i in df[:n_sep]])
    Y_tr=np.array([i[-1] for i in df[:n_sep]])
    X_te=np.array([i[1:-1] for i in df[n_sep:]])
    Y_te=np.array([i[-1] for i in df[n_sep:]])
    return X_tr, Y_tr, X_te, Y_te



def get_handler(name):
    return DataHandler


class DataHandler(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        '''if self.transform is not None:
            x = Image.fromarray(x.numpy(), mode='L')
            x = self.transform(x)'''
        return x, y, index

    def __len__(self):
        return len(self.X)