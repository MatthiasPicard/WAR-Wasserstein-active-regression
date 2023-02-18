import numpy as np
import pandas as pd
import itertools
import random
import time
import matplotlib.pyplot as plt
import warnings

import torch
import torch.nn as nn
import torch.optim as optim

from WAR.Models import NN_phi,NN_h_RELU
from WAR.training_and_query import WAR
from WAR.dataset_handler import myData,import_dataset,get_dataset
from WAR.Experiment_functions import *
from WAR.EarlyStop import EarlyStopper


def check_num_round(num_round,len_dataset,nb_initial_labelled_datas,num_elem_queried):
    max_round=int(np.ceil((len_dataset-nb_initial_labelled_datas)/num_elem_queried))
    if num_round>max_round:
        warnings.warn(f"when querying {num_elem_queried} data per round, num_rounds={num_round} is exceeding"+
        f" the maximum number of rounds (total data queried superior to number of initial unlabelled data).\nnum_round set to {max_round}")
        num_round=max_round
    return num_round



def full_training(strategy,num_round,X_test,y_test,show_losses,show_chosen_each_round,
                  reset_phi,reset_h,lr_h=None,lr_phi=None,val_proportion=0,reduced=False,cnst_t3phi=1
                 ):
    
    
    t1_descend_list=[]
    val_t1_descend_list=[]
    t2_ascend_list=[]
    acc = []# MAE
    acc_percentage=[] #MAPE
    acc_rmse=[] #RMSE
    
    only_train=False
    for rd in range(1,num_round+1):

        print('\n================Round {:d}==============='.format(rd))
        
        # if not enough unlabelled data to query a full batch, we will query the remaining data
        if len(np.arange(strategy.n_pool)[~strategy.idx_lb])<=strategy.num_elem_queried:
            only_train=True

        if reset_phi==True:
            strategy.phi=NN_phi(dim_input=strategy.X_train.shape[1])
            strategy.opti_phi = optim.Adam(strategy.phi.parameters(), lr=lr_phi,maximize=True)


        if reset_h==True:
            strategy.h=NN_h_RELU(dim_input=strategy.X_train.shape[1])
            strategy.opti_h = optim.Adam(strategy.h.parameters(), lr=lr_h)


        t1_descend,val_t1_descend,t2_ascend,b_idxs=strategy.train(val_proportion,only_train,reduced,cnst_t3phi)

        t1=list(map(lambda x: x.detach(),t1_descend))
        val_t1=list(map(lambda x: x.detach(),val_t1_descend))
        t2=list(map(lambda x: x.detach(),t2_ascend))
        t1_descend_list.append(t1)
        val_t1_descend_list.append(val_t1)
        t2_ascend_list.append(t2)
        if only_train==True:
            strategy.idx_lb[:]= True
        else:
            
            strategy.idx_lb[b_idxs] = True

        if show_losses:
            display_loss_t1(t1,rd)
            display_loss_val_t1(val_t1,rd)
            display_loss_t2(t2,rd)
            
        if show_chosen_each_round:
            if strategy.X_train.shape[1]==1:
                #display_phi(strategy.X_train,strategy.phi,rd)
                display_chosen_labelled_datas(strategy.X_train,strategy.idx_lb,strategy.y_train,b_idxs,rd,strategy.h)
                #display_prediction(X_test,strategy.h,y_test,rd)
            else:
                display_chosen_labelled_datas_PCA(strategy.X_train,strategy.idx_lb,strategy.y_train,b_idxs,rd,strategy.h)
            
       
        acc_rmse.append(RMSE(X_test,y_test,strategy.h))   
        acc.append(MAE(X_test,y_test,strategy.h))
        acc_percentage.append(MAPE(X_test,y_test,strategy.h))
        
            
    print('\n================Final training===============')
    

    
    t1_descend,val_t1_descend,t2_ascend,_=strategy.train(val_proportion,only_train,reduced,cnst_t3phi)
    t1=list(map(lambda x: x.detach(),t1_descend))
    val_t1=list(map(lambda x: x.detach(),val_t1_descend))
    t1_descend_list.append(t1)
    val_t1_descend_list.append(val_t1)
    
    #display_loss_t1(t1,rd)
    #display_prediction(X_test,strategy.h,y_test,"final")
    

    acc.append(MAE(X_test,y_test,strategy.h))
    error_each_round=list(map(lambda x: x[0],acc))

    acc_percentage.append(MAPE(X_test,y_test,strategy.h))
    error_each_round_per=list(map(lambda x: x[0],acc_percentage))
    
    acc_rmse.append(RMSE(X_test,y_test,strategy.h)) 
    error_each_round_rmse=acc_rmse
    
    return error_each_round,error_each_round_per, error_each_round_rmse,t1_descend_list,val_t1_descend_list,t2_ascend_list