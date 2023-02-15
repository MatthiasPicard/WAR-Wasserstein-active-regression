# NEW V7

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import grad
import math
import itertools
import math

from ALWGAN_new_v7_diversity.Experiment_functions import display_phi
from ALWGAN_new_v7_diversity.dataset_handler import myData
  
        
class WAAL:

    def __init__(self,X_train,y_train, idx_lb,total_epoch_h,total_epoch_phi,batch_size_train,num_elem_queried,phi,h,opti_phi,opti_h):
        
        """
        param X: trainset
        param Y: labels of the trainset
        param idx_lb: indices of the trainset that would be considered as labelled
        n_pool: length of the trainset
        total_epoch_train: number of epochs to train h and phi each round
        batch_size_train: size of the batch in the training process
        num_elem_queried: number of elem queried each round
        phi: 
        h: 
        opti_phi: 
        opti_h:
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
        #use_cuda     = torch.cuda.is_available()
        self.device  = torch.device("cpu")#torch.device("cuda" if use_cuda else "cpu")
        self.cost="MSE"
    
    def cost_func(self,predicted,true):
        if self.cost=="MSE":
            return (predicted-true)**2
        elif self.cost=="MAE":
            return abs(predicted-true)
        else:
            raise Exception("invalid cost function")
      
    
    def optimized_batch_size(self,len_labelled,len_unlabelled,batch_size):
        maxx=max(len_labelled,len_unlabelled)
        minn=min(len_labelled,len_unlabelled)
        nb_batch=math.ceil(maxx/batch_size)
        if minn<nb_batch:
            #print(minn)
            if maxx==len_labelled:
                return math.ceil(maxx/minn),1,minn
            else: return 1,math.ceil(maxx/minn),minn
        else:
            #print(nb_batch)
            if maxx==len_labelled:
                return min(batch_size,maxx),math.ceil(minn/nb_batch),nb_batch
            else: return math.ceil(minn/nb_batch),min(batch_size,maxx),nb_batch

    
    def train(self,show_losses,early_stopper,val_proportion,only_train=False,reduced=False,cnst_t3phi=1):
        b_idxs=[]
        t1_descend=[]
        val_t1_descend=[]
        t2_ascend=[]
        # get labelled and unlabelled data respectively
        idx_lb_train = np.arange(self.n_pool)[self.idx_lb]
        idx_ulb_train = np.arange(self.n_pool)[~self.idx_lb]
        
        #create  trainsets adapted to our algorithm ( with unlabelled and labelled datas)
        trainset_labelled=myData(self.X_train[idx_lb_train],self.y_train[idx_lb_train])
        #print(len(trainset_labelled),len(trainset_unlabelled))
        #print(self.X_train[idx_lb_train],self.y_train[idx_lb_train])
        #print(batch_labelled,batch_unlabelled)
        trainloader_labelled= DataLoader(trainset_labelled,shuffle=True,batch_size=self.batch_size_train)
        #print(len(trainloader_labelled),len(trainloader_unlabelled))
        
        
        stop_training=False
        t1_cnst=(1/(self.num_elem_queried+len(np.arange(self.n_pool)[self.idx_lb])))
        for epoch in range(self.total_epoch_h):
            if(stop_training==True):
                break
            for i,data in enumerate(trainloader_labelled,0):
                label_x, label_y=data
                self.opti_h.zero_grad() 
                #print(len(label_x.unique()))
                #print(len(label_x),len(label_y))
                #print(label_x, label_y)
                threshhold_val=int(val_proportion*len(label_x))
                if threshhold_val>=15 and early_stopper.early_stop_method=="v":
                    val_label_x,val_label_y ,label_x,label_y =label_x[:threshhold_val],label_y[:threshhold_val],label_x[threshhold_val:],label_y[threshhold_val:]
                    #print(len(val_label_x),len(val_label_y) ,len(label_x),len(label_y))
                    #print(val_label_x,val_label_y ,label_x,label_y)
                    val_lb_out = self.h(val_label_x)
                    val_h_descent=t1_cnst*torch.sum(self.cost_func(val_lb_out,val_label_y))
                    val_t1_descend.append(val_h_descent)
                    #print(early_stopper.min_validation_loss,val_h_descent.detach().numpy())
                    if early_stopper.early_stop(val_h_descent.detach().numpy()):
                        stop_training=True
                        break
                
                #print(len(label_x),len(label_y))
                #print(label_x, label_y)
                lb_out = self.h(label_x)
                h_descent=t1_cnst*torch.sum(self.cost_func(lb_out,label_y))
                t1_descend.append(h_descent)
                h_descent.backward()
                self.opti_h.step()
                if early_stopper.early_stop_method=="training":
                    if early_stopper.early_stop(h_descent.detach().numpy()):
                        stop_training=True
                        break
                if show_losses==True: 
                    print(f"\n(h)EPOCH {epoch+1}, batch {i+1}\n")
                    print(f"T1:{h_descent}")
                
        #cnst_T2 = (batch_unlabelled - self.num_elem_queried  ) / ( self.num_elem_queried + batch_labelled )        
        if not only_train: 
            
            idxs_temp=self.idx_lb.copy()
            
            for elem_queried in range(self.num_elem_queried):
                
                trainset_total=myData(self.X_train,self.y_train)
                trainloader_total= DataLoader(trainset_total,shuffle=True,batch_size=len(trainset_total))
                trainset_labelled=myData(self.X_train[idx_lb_train],self.y_train[idx_lb_train])
                trainloader_labelled= DataLoader(trainset_labelled,shuffle=True,batch_size=self.batch_size_train)
                for epoch in range(self.total_epoch_phi):  
                    iterator_total_phi=itertools.cycle(trainloader_total)
                    iterator_labelled_phi=itertools.cycle(trainloader_labelled)
                    #print(len(trainloader_unlabelled),len(trainloader_labelled))
                    for i in range(max(len(trainloader_total),len(trainloader_labelled))):
                        label_x,label_y = next(iterator_labelled_phi)
                        total_x,total_y = next(iterator_total_phi)
                        #print(len(label_x),len(total_x))
                        #print(label_x,total_y)
                        #display_phi(self.X_train,self.phi)
                        self.opti_phi.zero_grad()
                        phi_ascent = torch.mean(self.phi(total_x))-torch.mean(self.phi(label_x))
                        #phi_ascent = torch.mean(self.phi(unlabel_x))-torch.mean(self.phi(label_x))
                        #(1/self.n_pool)*(self.phi(unlabel_x).sum() - cnst_T2 * self.phi(label_x).sum()-regularization_t2*self.phi(self.X_train).abs().sum())
                        #print(phi_ascent.detach(),self.phi(unlabel_x).sum().detach(),0*self.phi(self.X_train).abs().sum(),cnst_T2 * self.phi(label_x).sum().detach(),self.phi(label_x).sum().detach(),cnst_T2)
                        t2_ascend.append(phi_ascent)
                        phi_ascent.backward()
                        self.opti_phi.step()
                        if show_losses==True: 
                            print(f"\n(phi)EPOCH {epoch+1}, batch {i+1}\n")
                            print(f"T2:{phi_ascent}")
                 
                b_queried=self.query(reduced,cnst_t3phi,idx_ulb_train)
                idxs_temp[b_queried]=True
                idx_ulb_train = np.arange(self.n_pool)[~idxs_temp]
                idx_lb_train = np.arange(self.n_pool)[idxs_temp]
                b_idxs.append(b_queried)
                #strategy.idx_lb[b_idxs]
            self.idx_lb=idxs_temp
        return t1_descend,val_t1_descend,t2_ascend,b_idxs

    
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
        #print(max(self.cost_func(est_h_unl_X,bound_min),self.cost_func(est_h_unl_X,bound_max)).detach().numpy())
        #print(type(max(self.cost_func(est_h_unl_X,bound_min),self.cost_func(est_h_unl_X,bound_max)).detach().numpy()[0]))
        #print(est_h_unl_X,bound_min,self.cost_func(est_h_unl_X,bound_min),self.cost_func(est_h_unl_X,bound_max))
        return max(self.cost_func(est_h_unl_X,bound_min),self.cost_func(est_h_unl_X,bound_max)).detach().numpy()[0]
    
        

    
    def predict_loss(self,X):
        

        idxs_lb=np.arange(self.n_pool)[self.idx_lb]
        losses=[]
        with torch.no_grad():
            for i in X:
                idx_nearest_Xk,dist=self.Idx_NearestP(i,idxs_lb) #minimum (distance with xk)
                #print('idx',idx_nearest_Xk,'distance',dist)
                losses.append(self.Max_cost_B(idx_nearest_Xk,dist,i)) #maximum (loss value)
        
        #print(losses)
        return np.array(losses)
            

    
    
    def query(self,reduced,cnst_t3phi,idx_ulb_train):


        idxs_unlabeled = idx_ulb_train
        # prediction output probability
        losses = self.predict_loss(self.X_train[idxs_unlabeled])

        # prediction output discriminative score
        #phi_score = self.pred_phi_score(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
        with torch.no_grad():
            phi_scores =  self.phi(self.X_train[idxs_unlabeled]).view(-1)
            
        
        # print("phi_score:")
        # print(phi_score)
        
        # computing the decision score
        t3_cnst=(1/(1+len(np.arange(self.n_pool)[self.idx_lb])))
        
        
        if reduced:
            phi_scores_reduced=phi_scores/torch.std(phi_scores)
            losses_reduced=losses/np.std(losses)
            print(np.std(losses_reduced),torch.std(phi_scores_reduced))
            total_scores =t3_cnst*( - cnst_t3phi*phi_scores_reduced.detach().numpy() -losses_reduced )
         
        else:
            total_scores =t3_cnst*( - cnst_t3phi*phi_scores.detach().numpy()) -losses 
            #print(np.std(losses),torch.std(phi_scores))
        #print(losses_reduced,phi_scores_reduced.detach().numpy())
        #print(np.argpartition(total_scores,1))
        b=np.argpartition(total_scores,1)[:1]
        # sort the score with minimal query_number examples
        # expected value outputs from smaller to large
        
        return idxs_unlabeled[b]
