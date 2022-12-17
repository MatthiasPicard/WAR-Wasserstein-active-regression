# NEW V1


import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import grad


from ALWGAN_new_v1.dataset_handler import myData,WA_myData
  
        
class WAAL:

    def __init__(self,X_train,y_train, idx_lb,total_epoch_train,batch_size_train,num_elem_queried,lr,phi,h,opti_phi,opti_h):
        
        """
        param X: trainset
        param Y: labels of the trainset
        param idx_lb: indices of the trainset that would be considered as labelled
        n_pool: length of the trainset
        total_epoch_train: number of epochs to train h and phi each round
        batch_size_train: size of the batch in the training process
        num_elem_queried: number of elem queried each round
        lr: learning rate
        phi: 
        h: 
        opti_phi: 
        opti_h:
        """

        self.X_train = X_train
        self.y_train = y_train
        self.idx_lb  = idx_lb
        self.n_pool  = len(y_train)
        self.total_epoch_train=total_epoch_train
        self.batch_size_train=batch_size_train
        self.num_elem_queried=num_elem_queried
        self.lr=lr
        self.phi=phi
        self.h=h
        self.opti_phi=opti_phi
        self.opti_h=opti_h
        #use_cuda     = torch.cuda.is_available()
        self.device  = torch.device("cpu")#torch.device("cuda" if use_cuda else "cpu")

    
    def abs_cost_func(self,predicted,true):
        return abs(predicted-true)
        
    
    
    def train(self,show_losses=False):
        t1_descend=[]
        t2_ascend=[]
        # get labelled and unlabelled data respectively
        idx_lb_train = np.arange(self.n_pool)[self.idx_lb]
        idx_ulb_train = np.arange(self.n_pool)[~self.idx_lb]
        
        #create a trainset adapted to our algorithm ( with unlabelled and labelled datas)
        trainset=WA_myData(self.X_train[idx_lb_train],self.y_train[idx_lb_train],self.X_train[idx_ulb_train],self.y_train[idx_ulb_train])
        
        #print(len(trainset))
        trainloader= DataLoader(trainset,shuffle=True,batch_size=self.batch_size_train)
        cnst_T2 = (len(idx_ulb_train) - self.num_elem_queried  ) / ( self.num_elem_queried + len(idx_lb_train) )
        for epoch in range(self.total_epoch_train):  
            for index, label_x, label_y, unlabel_x, _ in trainloader:
                #print(label_x)
                #print(unlabel_x)
                #print(label_x)
                self.opti_h.zero_grad() 
                lb_out = self.h(label_x)
                h_descent=torch.mean(self.abs_cost_func(lb_out,label_y))
                t1_descend.append(h_descent)                                        
                h_descent.backward()
                self.opti_h.step()# added

                self.opti_phi.zero_grad()
                phi_ascent = -(self.phi(unlabel_x).mean() - cnst_T2 * self.phi(label_x).mean())  # removed abs , put minus because we maximize 
                t2_ascend.append(phi_ascent)
                phi_ascent.backward()
                self.opti_phi.step()
                
                if show_losses==True: 
                    print(f"\nEPOCH {epoch+1}\n")
                    print(f"T1:{h_descent}")
                    print(f"T2:{phi_ascent}")
               
        return t1_descend,t2_ascend
        
       
    def f_cost(self,x,b):
        f= -abs(x-b)
        return f
        
    def df_cost(self,x,b):
        if(x<b):
            return 1
        else:
            return -1

    
    def Max_cost_B(self,idx_Xk,distance):
        
        delta = 1
        epsilon=0.1
        y_n = self.h(self.X_train[idx_Xk]) #h_pred(xk plus proche)
        y_true = self.y_train[idx_Xk]
        iterations=1
        #print ('y_true',y_true)
        #print('distance',distance)
        while(delta > epsilon and iterations<100):
            #print('y_n',y_n)
            y_i = y_n - self.f_cost(y_n,y_true)/self.df_cost(y_n,y_true)
            #print('y_i',y_i)
            if abs(y_true-y_i)>distance:
                if(y_i<y_true):
                    y_n=y_true-distance
                    break
                else:
                    y_n=y_true+distance
                    break
            delta = abs(y_n-y_i)
            #print('delta',delta)
            y_n = y_i
            iterations+=1
        #print('iter',iterations) 
        #print('final label',y_n)
        #print(-self.f_cost(y_n,y_true)) # 0 is the only value
        return -self.f_cost(y_n,y_true) # - because f_cost is inversed to do gradient descent
    
    
    def Idx_NearestP(self,Xu,idxs_lb):
        
        distances=[]
        for i in idxs_lb:
            distances.append(torch.norm(Xu-self.X_train[i]))
            
        return idxs_lb[distances.index(min(distances))],float(min(distances))
 
    
    def predict_loss(self,X):
        

        idxs_lb=np.arange(self.n_pool)[self.idx_lb]
        losses=[]
        with torch.no_grad():
            for i in X:
                idx_nearest_Xk,dist=self.Idx_NearestP(i,idxs_lb) #minimum (distance with xk)
                #print('idx',idx_nearest_Xk,'distance',dist)
                losses.append(self.Max_cost_B(idx_nearest_Xk,dist)) #maximum (loss value)
        
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
        total_scores = (1/(self.num_elem_queried+len(np.arange(self.n_pool)[self.idx_lb]))) * (losses - phi_scores.detach().numpy())   
        
        #print(total_score,np.argpartition(total_score,batch_size)[:batch_size])
        b=np.argpartition(total_scores,self.num_elem_queried)[:self.num_elem_queried]
        # sort the score with minimal query_number examples
        # expected value outputs from smaller to large
        
        return idxs_unlabeled[b]
