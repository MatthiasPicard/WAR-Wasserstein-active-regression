import matplotlib.pyplot as plt
import seaborn as sns
import time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


#Execution duration

def time_execution(start,end):
    timespan=end-start
    minutes=timespan//60
    secondes=timespan%60
    heures=minutes//60
    minutes=minutes%60
    print(f"{int(heures)}h {int(minutes)} min {secondes} s")
    return(f"{int(heures)}h {int(minutes)} min {secondes} s")


#Graphs

def display_prediction(X_test,h,y_test,rd):
    plt.figure(figsize=[9,6])
        
    plt.scatter(X_test,h(X_test).detach(),label="predicted values")
    plt.scatter(X_test,y_test,label="true_values")
    plt.legend()
    if rd=="final":
        plt.title("true et predicted values at the end")
    else:plt.title(f"true et predicted values after {rd} rounds")
    

def display_chosen_labelled_datas_PCA(X_train,idx_lb,y_train,b_idxs,rd,h):
    
    pca = PCA(n_components=2)
    transformed = pca.fit_transform(X=X_train)
    x_component = transformed[:, 0]
    
    plt.figure(figsize=[9,6])
    plt.scatter(transformed[:, 0][~idx_lb],transformed[:, 1][~idx_lb],label="unlabelled points",c="brown")
    plt.scatter(transformed[:, 0][idx_lb],transformed[:, 1][idx_lb],label="labelled points")
    plt.scatter(transformed[:, 0][b_idxs],transformed[:, 1][b_idxs],label="new points added",c="yellow")
    plt.legend()
    plt.title(f"points selected after {rd} rounds")
    
def display_chosen_labelled_datas(X_train,idx_lb,y_train,b_idxs,rd,h):
    plt.figure(figsize=[9,6])
    
    plt.scatter(X_train[~idx_lb],y_train[~idx_lb].detach(),label="unlabelled points",c="brown")   
    plt.scatter(X_train[idx_lb],y_train[idx_lb].detach(),label="labelled points")
    plt.scatter(X_train[b_idxs],y_train[b_idxs].detach(),label="new points added",c="yellow")
    plt.legend()
    plt.title(f"points selected after {rd} rounds")
        
def display_loss_t1(t1_descend,rd):
    plt.figure(figsize=[9,6])
    plt.plot(t1_descend)
    plt.xlabel("batch")
    plt.title(f"t1 loss evolution each batch after {rd} rounds")
    
def display_loss_val_t1(val_t1_descend,rd):
    plt.figure(figsize=[9,6])
    plt.plot(val_t1_descend)
    plt.xlabel("batch")
    plt.title(f"val_t1 loss evolution each batch after {rd} rounds")

def display_loss_t2(t2_ascend,rd):
    plt.figure(figsize=[9,6])
    plt.plot(t2_ascend)
    plt.xlabel("batch")
    plt.title(f"t2 loss evolution each batch after {rd} rounds")
    
def display_phi(X_train,phi,rd=None):
    plt.figure(figsize=[9,6])
    plt.scatter(X_train,phi(X_train).detach().numpy())
    #print(X_train,phi(X_train).detach().numpy())
    plt.xlabel("X_train")
    plt.title(f"phi function on the full trainset after {rd} rounds")
    

#Metrics        
        
def MAPE(X_test,y_test,h):
    acc_per_i=sum(abs(h(X_test)-y_test)/abs(y_test))
    acc_per_i = acc_per_i/len(y_test)
    return acc_per_i.detach().numpy()    


def MAE(X_test,y_test,h):
    acc_i = sum(abs((h(X_test)-y_test)))
    acc_i = acc_i/len(y_test)
    return acc_i.detach().numpy()  

def RMSE(X_test,y_test,h):
    acc_i = ((h(X_test)-y_test)**2).mean()
    return np.sqrt(acc_i.detach().numpy())  


