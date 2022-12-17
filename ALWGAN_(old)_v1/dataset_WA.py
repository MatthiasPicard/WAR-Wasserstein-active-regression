from torch.utils.data import ConcatDataset, Dataset
from torchvision import datasets
import numpy as np
import torch
from PIL import Image
import pandas as pd


"""
This code mainly tests the redundancy trick, different from only using the smaller one 
to make the batch, here instead we used the max len as the data to make the batch

"""

def get_dataset(name):
    df = pd.read_csv(name+'.csv').to_numpy()
    np.random.shuffle(df)
    n_sep=round(len(df)*0.8)
    X_tr=np.array([i[1:-1] for i in df[:n_sep]])
    Y_tr=np.array([i[-1] for i in df[:n_sep]])
    X_te=np.array([i[1:-1] for i in df[n_sep:]])
    Y_te=np.array([i[-1] for i in df[n_sep:]])
    return torch.from_numpy(X_tr).float(), torch.from_numpy(Y_tr).float(), torch.from_numpy(X_te).float(), torch.from_numpy(Y_te).float()


def get_handler(name):
    return Wa_datahandler

class Wa_datahandler(Dataset):

    def __init__(self,X_1, Y_1, X_2, Y_2, transform = None):
        """

        :param X_1: covariate from the first distribution
        :param Y_1: label from the first distribution
        :param X_2:
        :param Y_2:
        :param transform:
        """
        self.X1 = X_1
        self.Y1 = Y_1
        self.X2 = X_2
        self.Y2 = Y_2
        self.transform = transform

    def __len__(self):

        # returning the minimum length of two data-sets

        return max(len(self.X1),len(self.X2))

    def __getitem__(self, index):
        Len1 = len(self.Y1)
        Len2 = len(self.Y2)

        # checking the index in the range or not

        if index < Len1:
            x_1 = self.X1[index]
            y_1 = self.Y1[index]

        else:

            # rescaling the index to the range of Len1
            re_index = index % Len1

            x_1 = self.X1[re_index]
            y_1 = self.Y1[re_index]

        # checking second datasets
        if index < Len2:

            x_2 = self.X2[index]
            y_2 = self.Y2[index]

        else:
            # rescaling the index to the range of Len2
            re_index = index % Len2

            x_2 = self.X2[re_index]
            y_2 = self.Y2[re_index]

        return index,x_1,y_1,x_2,y_2




