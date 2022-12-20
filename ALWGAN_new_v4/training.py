# NEW V4

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import grad
import math
import itertools

from ALWGAN_new_v4.dataset_handler import myData
  
        
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

    
    def abs_cost_func(self,predicted,true):
        return abs(predicted-true)
        
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

    
    def train(self,show_losses,early_stopper,val_proportion):
        t1_descend=[]
        val_t1_descend=[]
        t2_ascend=[]
        # get labelled and unlabelled data respectively
        idx_lb_train = np.arange(self.n_pool)[self.idx_lb]
        idx_ulb_train = np.arange(self.n_pool)[~self.idx_lb]
        
        #create  trainsets adapted to our algorithm ( with unlabelled and labelled datas)
        trainset_labelled=myData(self.X_train[idx_lb_train],self.y_train[idx_lb_train])
        trainset_unlabelled=myData(self.X_train[idx_ulb_train],self.y_train[idx_ulb_train])
        #print(len(trainset_labelled),len(trainset_unlabelled))
        #print(self.X_train[idx_lb_train],self.y_train[idx_lb_train])
        batch_labelled,batch_unlabelled,nb_batch=self.optimized_batch_size(len(trainset_labelled),len(trainset_unlabelled),self.batch_size_train)
        #print(batch_labelled,batch_unlabelled)
        trainloader_labelled_h= DataLoader(trainset_labelled,shuffle=True,batch_size=self.batch_size_train)
        trainloader_labelled_phi= DataLoader(trainset_labelled,shuffle=True,batch_size=batch_labelled,drop_last=True)
        trainloader_unlabelled= DataLoader(trainset_unlabelled,shuffle=True,batch_size=batch_unlabelled,drop_last=True)
        #print(len(trainloader_labelled_h),len(trainloader_labelled_phi),len(trainloader_unlabelled))
        
        
        stop_training=False
        
        for epoch in range(self.total_epoch_h):
            if(stop_training==True):
                break
            for i,data in enumerate(trainloader_labelled_h,0):
                label_x, label_y=data
                self.opti_h.zero_grad() 
                #print(len(label_x.unique()))
                #print(len(label_x),len(label_y))
                #print(label_x, label_y)
                threshhold_val=int(val_proportion*len(label_x))
                if threshhold_val>=5 and early_stopper.early_stop_method=="validation":
                    val_label_x,val_label_y ,label_x,label_y =label_x[:threshhold_val],label_y[:threshhold_val],label_x[threshhold_val:],label_y[threshhold_val:]
                    #print(len(val_label_x),len(val_label_y) ,len(label_x),len(label_y))
                    #print(val_label_x,val_label_y ,label_x,label_y)
                    val_lb_out = self.h(val_label_x)
                    val_h_descent=torch.mean(self.abs_cost_func(val_lb_out,val_label_y))
                    val_t1_descend.append(val_h_descent)
                    #print(early_stopper.min_validation_loss,val_h_descent.detach().numpy())
                    if early_stopper.early_stop(val_h_descent.detach().numpy()):
                        stop_training=True
                        break
                
                #print(len(label_x),len(label_y))
                #print(label_x, label_y)
                lb_out = self.h(label_x)
                t1_cnst=(1/(self.num_elem_queried+len(np.arange(self.n_pool)[self.idx_lb])))
                h_descent=t1_cnst*torch.sum(self.abs_cost_func(lb_out,label_y))
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
                
        cnst_T2 = (batch_unlabelled - self.num_elem_queried  ) / ( self.num_elem_queried + batch_labelled )        
                
        for epoch in range(self.total_epoch_phi):  
            iterator_unlabelled_phi=itertools.cycle(trainloader_unlabelled)
            iterator_labelled_phi=itertools.cycle(trainloader_labelled_phi)
            for i in range(nb_batch):
                label_x,label_y = next(iterator_labelled_phi)
                unlabel_x,unlabel_y = next(iterator_unlabelled_phi)
                self.opti_phi.zero_grad()
                phi_ascent = (1/self.n_pool)*(self.phi(unlabel_x).sum() - cnst_T2 * self.phi(label_x).sum())  # removed abs  
                print(phi_ascent.detach(),self.phi(unlabel_x).sum().detach(),cnst_T2 * self.phi(label_x).sum().detach(),self.phi(label_x).sum().detach(),cnst_T2)
                t2_ascend.append(phi_ascent)
                phi_ascent.backward()
                self.opti_phi.step()
                if show_losses==True: 
                    print(f"\n(phi)EPOCH {epoch+1}, batch {i+1}\n")
                    print(f"T2:{phi_ascent}")
               
        return t1_descend,val_t1_descend,t2_ascend

    
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
        #print(max(self.abs_cost_func(est_h_unl_X,bound_min),self.abs_cost_func(est_h_unl_X,bound_max)).detach().numpy())
        #print(type(max(self.abs_cost_func(est_h_unl_X,bound_min),self.abs_cost_func(est_h_unl_X,bound_max)).detach().numpy()[0]))
        return max(self.abs_cost_func(est_h_unl_X,bound_min),self.abs_cost_func(est_h_unl_X,bound_max)).detach().numpy()[0]
    
        

    
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
            

    
    
    def query(self):


        idxs_unlabeled = np.arange(self.n_pool)[~self.idx_lb]
        # prediction output probability
        losses = self.predict_loss(self.X_train[idxs_unlabeled])

        # prediction output discriminative score
        #phi_score = self.pred_phi_score(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
        with torch.no_grad():
            phi_scores =  self.phi(self.X_train[idxs_unlabeled]).view(-1)
        
        # print("phi_score:")
        # print(phi_score)
        
        # computing the decision score
        t3_cnst=(1/(self.num_elem_queried+len(np.arange(self.n_pool)[self.idx_lb])))
        total_scores =t3_cnst*( - phi_scores.detach().numpy()) #+losses     # removed csts
        #print(losses,phi_scores.detach().numpy())
        #print(total_score,np.argpartition(total_score,batch_size)[:batch_size])
        b=np.argpartition(total_scores,self.num_elem_queried)[:self.num_elem_queried]
        # sort the score with minimal query_number examples
        # expected value outputs from smaller to large
        
        return idxs_unlabeled[b]
