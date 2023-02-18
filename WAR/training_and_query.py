import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import grad
import math
import itertools
import math

from WAR.Experiment_functions import display_phi
from WAR.dataset_handler import myData
  
        
class WAR:

    def __init__(self,X_train,y_train, idx_lb,total_epoch_h,total_epoch_phi,batch_size_train,num_elem_queried
                 ,phi,h,opti_phi,opti_h,early_stopper):
        
        """
        X_train: trainset
        Y_train: labels of the trainset
        idx_lb: indices of the trainset that would be considered as labelled
        n_pool: length of the trainset
        total_epoch_h: number of epochs to train h
        total_epoch_phi: number of epochs to train phi
        batch_size_train: size of the batch in the training process
        num_elem_queried: number of elem queried each round
        phi: phi neural network
        h: h neural network
        opti_phi: phi optimizer
        opti_h: h optimizer
        early_stopper: early_stopping strategy (None if no early stopping applied)
        cost: define the cost function for both neural network. "MSE" or MAE"
        """

        self.X_train = X_train
        self.y_train = y_train
        self.idx_lb  = idx_lb
        self.n_pool  = len(y_train)
        self.total_epoch_h=total_epoch_h
        self.total_epoch_phi=total_epoch_phi
        self.batch_size_train=batch_size_train
        self.num_elem_queried=num_elem_queried
        self.phi=phi
        self.h=h
        self.opti_phi=opti_phi
        self.opti_h=opti_h
        self.early_stopper=early_stopper
        self.cost="MSE"
    
    def cost_func(self,predicted,true):
        if self.cost=="MSE":
            return (predicted-true)**2
        elif self.cost=="MAE":
            return abs(predicted-true)
        else:
            raise Exception("invalid cost function")
      
   

    
    def train(self,val_proportion=None,only_train=False,reduced=True,cnst_t3phi=3):
        b_idxs=[]
        
        #recover loss
        t1_descend=[]
        val_t1_descend=[]
        t2_ascend=[]
        
        # separating labelled and unlabelled data respectively
        idx_lb_train = np.arange(self.n_pool)[self.idx_lb]
        idx_ulb_train = np.arange(self.n_pool)[~self.idx_lb]
        
        
        trainset_labelled=myData(self.X_train[idx_lb_train],self.y_train[idx_lb_train])
        trainloader_labelled= DataLoader(trainset_labelled,shuffle=True,batch_size=self.batch_size_train)
        
        self.early_stopper.counter = 0
        self.early_stopper.min_loss = np.inf  
        
        stop_training=False #True when early stop activate (if toggled on)
        #t1_cnst=(1/len(np.arange(self.n_pool)[self.idx_lb]))
        for epoch in range(self.total_epoch_h):
            if(stop_training==True):
                break
                
            for i,data in enumerate(trainloader_labelled,0):
                label_x, label_y=data
                self.opti_h.zero_grad() 
                
                # validationset process (if enabled) 
                threshhold_val=int(val_proportion*len(label_x)) 
                if threshhold_val>=5 and self.early_stopper.early_stop_method==True: 
                    val_label_x,val_label_y ,label_x,label_y =label_x[:threshhold_val],label_y[:threshhold_val],label_x[threshhold_val:],label_y[threshhold_val:]
                    print(len(val_label_x),len(val_label_y) ,len(label_x),len(label_y))
                    val_lb_out = self.h(val_label_x)
                    val_h_descent=torch.mean(self.cost_func(val_lb_out,val_label_y))
                    val_t1_descend.append(val_h_descent)
                    if self.early_stopper.early_stop(val_h_descent.detach().numpy()):
                        stop_training=True
                        break
                
                # T1
                lb_out = self.h(label_x)
                h_descent=torch.mean(self.cost_func(lb_out,label_y))
                t1_descend.append(h_descent)
                h_descent.backward()
                self.opti_h.step()
                
              
                      
        if not only_train: 
            
            #T2 and query
            idxs_temp=self.idx_lb.copy()
            
            for elem_queried in range(self.num_elem_queried):
                
                trainset_total=myData(self.X_train,self.y_train)
                trainloader_total= DataLoader(trainset_total,shuffle=True,batch_size=len(trainset_total))
                trainset_labelled=myData(self.X_train[idx_lb_train],self.y_train[idx_lb_train])
                trainloader_labelled= DataLoader(trainset_labelled,shuffle=True,batch_size=self.batch_size_train)
                for epoch in range(self.total_epoch_phi):  
                    iterator_total_phi=itertools.cycle(trainloader_total)
                    iterator_labelled_phi=itertools.cycle(trainloader_labelled)
                    for i in range(max(len(trainloader_total),len(trainloader_labelled))):
                        label_x,label_y = next(iterator_labelled_phi)
                        total_x,total_y = next(iterator_total_phi)
                        #display_phi(self.X_train,self.phi)
                        self.opti_phi.zero_grad()
                        phi_ascent = torch.mean(self.phi(total_x))-torch.mean(self.phi(label_x))
                        t2_ascend.append(phi_ascent)
                        phi_ascent.backward()
                        self.opti_phi.step()
                        
                 
                b_queried=self.query(reduced,cnst_t3phi,idx_ulb_train)
                idxs_temp[b_queried]=True
                idx_ulb_train = np.arange(self.n_pool)[~idxs_temp]
                idx_lb_train = np.arange(self.n_pool)[idxs_temp]
                b_idxs.append(b_queried)
            self.idx_lb=idxs_temp
        return t1_descend,val_t1_descend,t2_ascend,b_idxs

    
    
    def query(self,reduced,cnst_t3phi,idx_ulb_train):


        idxs_unlabeled = idx_ulb_train
        losses = self.predict_loss(self.X_train[idxs_unlabeled])

        with torch.no_grad():
            phi_scores =  self.phi(self.X_train[idxs_unlabeled]).view(-1)
            
        t3_cnst=(1/(1+len(np.arange(self.n_pool)[self.idx_lb])))
        if reduced:
            phi_scores_reduced=phi_scores/torch.std(phi_scores)
            losses_reduced=losses/np.std(losses)
            #print(np.std(losses_reduced),torch.std(phi_scores_reduced))
            total_scores =-t3_cnst*(cnst_t3phi*phi_scores_reduced.detach().numpy()+losses_reduced )
         
        else:
            total_scores =-t3_cnst*(cnst_t3phi*phi_scores.detach().numpy())+losses 
        b=np.argmin(total_scores)
        # sort the score with minimal query_number examples
        # expected value outputs from smaller to large
        
        return idxs_unlabeled[b]
    
    
    
    def Idx_NearestP(self,Xu,idxs_lb):
        
        distances=[]
        for i in idxs_lb:
            distances.append(torch.norm(Xu-self.X_train[i]))
            
        return idxs_lb[distances.index(min(distances))],float(min(distances))
    

    
    def Max_cost_B(self,idx_Xk,distance,i):
       
    
        est_h_unl_X=self.h(i)
        true_value_labelled_X=self.y_train[idx_Xk]
        bound_min= true_value_labelled_X-distance
        bound_max= true_value_labelled_X+distance
        return max(self.cost_func(est_h_unl_X,bound_min),self.cost_func(est_h_unl_X,bound_max)).detach().numpy()[0]
    
        

    
    def predict_loss(self,X):
        

        idxs_lb=np.arange(self.n_pool)[self.idx_lb]
        losses=[]
        with torch.no_grad():
            for i in X:
                idx_nearest_Xk,dist=self.Idx_NearestP(i,idxs_lb) 
                losses.append(self.Max_cost_B(idx_nearest_Xk,dist,i)) 
        
        return np.array(losses)
            

    
    
    
