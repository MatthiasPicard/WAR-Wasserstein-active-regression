import torch
from torch.utils.data import Dataset
import random
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets


def import_dataset(name):
    if name=="boston":
        boston = datasets.load_boston()
        df = pd.DataFrame(boston.data, columns = boston.feature_names)
        df["target"]=boston.target
        y_boston=df.target
        X_boston=df.drop("target",axis=1)
        y_boston=torch.Tensor(y_boston).view(len(y_boston),1).float()
        X_boston=torch.Tensor(X_boston.values).float()
        return X_boston,y_boston
    if name=="airfoil":
        columns_names=["Frequency","Angle of attack","Chord length","Free-stream velocity","Suction side displacement thickness","sound pressure level"]
        airfoil=pd.read_csv('datasets/airfoil_self_noise.dat',sep='\t',names=columns_names)
        y_airfoil=airfoil["sound pressure level"]
        X_airfoil=airfoil.drop("sound pressure level",axis=1)
        y_airfoil=torch.Tensor(y_airfoil).view(len(y_airfoil),1).float()
        X_airfoil=torch.Tensor(X_airfoil.values).float()
        return X_airfoil,y_airfoil

def get_dataset(proportion=0.2,dataset="boston"):

    scaler = MinMaxScaler()
    X,y=import_dataset(dataset)
    X=torch.Tensor(scaler.fit_transform(X))
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=proportion)
    print(f"Shape of the training set: {X_train.shape}")
    return X_train,X_test,y_train,y_test



class myData(Dataset):
    
    def __init__(self,x,y):
        self.x=x
        self.y=y
        self.shape=x.size(0)
        
    def __getitem__(self,index):
        return self.x[index],self.y[index]
    
    def __len__(self):
        return self.shape
    



