import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import grad







# setting gradient values
def set_requires_grad(model, requires_grad=True):
    
    """
    Used in training adversarial approach
    :param model:
    :param requires_grad:
    :return:
    """

    for param in model.parameters():
        param.requires_grad = requires_grad



        
        
class WAAL:

    def __init__(self,X,Y, idx_lb, ffd_h, ffd_phi, train_handler, test_handler, args):
        
        """
        :param X:
        :param Y:
        :param idx_lb:
        :param ffd_h:
        :param ffd_phi:
        :param train_handler: generate a dataset in the training procedure, since training requires two datasets, the returning value
                                looks like a (index, x_dis1, y_dis1, x_dis2, y_dis2)
        :param test_handler: generate a dataset for the prediction, only requires one dataset
        :param args:
        """

        self.X = X
        self.Y = Y
        self.idx_lb  = idx_lb
        self.ffd_h = ffd_h
        self.ffd_phi = ffd_phi
        self.train_handler = train_handler
        self.test_handler  = test_handler
        self.args    = args

        self.n_pool  = len(Y)
        #use_cuda     = torch.cuda.is_available()
        self.device  = torch.device("cpu")#torch.device("cuda" if use_cuda else "cpu")

        self.selection = 2
        # for cifar 10 or svhn or fashion mnist  self.selection = 10
        
        
        
    def update(self, idx_lb):

        self.idx_lb = idx_lb
        
        
        
        
    def L2Lip(self, w):
        max_k=1
        norm=float(torch.norm(w[0]))
        return w*(1.0 / max(1.0, norm / max_k))


        
        
        
    def train(self, alpha, total_epoch):

        """
        Only training samples with labeled and unlabeled data-set
        alpha is the trade-off between the empirical loss and error, the more interaction, the smaller \alpha
        :return:
        """

        #print("[Training] labeled and unlabeled data")
        # n_epoch = self.args['n_epoch']
        n_epoch = total_epoch
        
        self.h = self.ffd_h().to(self.device)
        self.phi = self.ffd_phi().to(self.device)
        
        lr= self.args['optimizer_args']['lr']
        
        #print('_______________BEFORE TRAIN___________________')
        #print('H_WEIGHTS\n',self.h.fc1.weight)
        #print('PHI_WEIGHTS\n',self.phi.fc1.weight)
        
        # setting three optimizers
        opt_h = optim.SGD(self.h.parameters(),**self.args['optimizer_args'])
        opt_phi = optim.SGD(self.phi.parameters(),**self.args['optimizer_args'])
        
        # setting idx_lb and idx_ulb
        idx_lb_train = np.arange(self.n_pool)[self.idx_lb]
        idx_ulb_train = np.arange(self.n_pool)[~self.idx_lb]
        
        # coefficient of T2 (page 4)
        gamma_ratio = ( len(idx_ulb_train) - self.args['loader_tr_args']['batch_size']  ) / ( self.args['loader_tr_args']['batch_size'] + len(idx_lb_train) ) / len(self.Y)
        
        # Data-loading (Redundant Trick)        
        loader_tr = DataLoader(self.train_handler(self.X[idx_lb_train],self.Y[idx_lb_train],self.X[idx_ulb_train],self.Y[idx_ulb_train],transform = self.args['transform_tr']), shuffle= True, **self.args['loader_tr_args'])

        for epoch in range(n_epoch):
            '''for name,param in self.h.named_parameters():
                print(name,param.data)'''

            # setting the training mode in the beginning of EACH epoch
            # (since we need to compute the training accuracy during the epoch, optional)

            self.h.train()
            self.phi.train()

            # Total_loss = 0
            # n_batch    = 0
            # acc        = 0
            
            for index, label_x, label_y, unlabel_x, _ in loader_tr:
                # n_batch += 1

                #label_x, label_y = label_x.cuda(), label_y.cuda()
                #unlabel_x = unlabel_x.cuda()

                #NOW: training predictor / BEFORE: training feature extractor and predictor
                set_requires_grad(self.h,requires_grad=True)
                set_requires_grad(self.phi,requires_grad=False)            
                
                opt_h.zero_grad()
                
                lb_out = self.h(label_x)
                h_descent=torch.mean(abs(lb_out-label_y)) * (1/(self.args['loader_tr_args']['batch_size']+len(idx_lb_train)))
                                                             
                #print('h_loss',h_descent)
                h_descent.backward()
                #opt_h.step()
                dw_h_hid=self.h.hidden.weight.grad
                dw_h_pred=self.h.predict.weight.grad
                with torch.no_grad():
                    #print('w_h_hid+grad',self.h.hidden.weight.data,dw_h_hid)
                    self.h.hidden.weight.data = self.phi.hidden.weight.data - lr*dw_h_hid
                    self.h.hidden.weight.data = self.L2Lip(self.h.hidden.weight)
                    
                    self.h.predict.weight.data = self.h.predict.weight.data - lr*dw_h_pred
                    self.h.predict.weight.data = self.L2Lip(self.h.predict.weight)
                
                
                # Then the second step, training discriminator
                set_requires_grad(self.h, requires_grad=False)
                set_requires_grad(self.phi, requires_grad=True)
                
                opt_phi.zero_grad()
                
                phi_ascent = (-1/len(self.Y)) * abs(self.phi(unlabel_x).mean() - gamma_ratio * self.phi(label_x).mean())
                
                phi_ascent.backward()
                #opt_phi.step()
                dw_phi_hid=self.phi.hidden.weight.grad
                dw_phi_pred=self.phi.predict.weight.grad
                with torch.no_grad():
                    self.phi.hidden.weight.data = self.phi.hidden.weight.data - lr*dw_phi_hid
                    self.phi.hidden.weight.data = self.L2Lip(self.phi.hidden.weight)
                    
                    self.phi.predict.weight.data = self.phi.predict.weight.data - lr*dw_phi_pred
                    self.phi.predict.weight.data = self.L2Lip(self.phi.predict.weight)
             
            print('==========Inner epoch {:d} ========'.format(epoch))
            #product of norm is superior to the norm of the product
            h_norm=torch.norm(self.h.hidden.weight.grad)+torch.norm(self.h.predict.weight.grad)
            phi_norm=torch.norm(self.phi.hidden.weight.grad)+torch.norm(self.phi.predict.weight.grad)
            if(max(h_norm,phi_norm)<3):
                print('STOP')
                print('==========Inner epoch {:d} ========'.format(epoch))
                break

            
     
    
    
    def Idx_NearestP(self,Xu,idxs_lb):
        
        distances=[]
        for i in idxs_lb:
            distances.append(torch.norm(abs(Xu-self.X[i])))
            
        return idxs_lb[distances.index(min(distances))],float(min(distances))
    
    
    
    
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
        y_n = self.h(self.X[idx_Xk]) #h_pred(xk plus proche)
        y_true = self.Y[idx_Xk]
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
        return -self.f_cost(y_n,y_true) # - because f_cost is inversed to do gradient descent
    
    

    
    def predict_loss(self,X,Y):
        
        '''loader_te = DataLoader(self.test_handler(X, Y, transform=self.args['transform_te']), shuffle=False, **self.args['loader_te_args'])
        
        losses = torch.zeros(len(Y))'''
        idxs_lb=np.arange(self.n_pool)[self.idx_lb]
        losses=[]
        with torch.no_grad():
            for i in X:
                idx_nearest_Xk,dist=self.Idx_NearestP(i,idxs_lb) #minimum (distance with xk)
                #print('idx',idx_nearest_Xk,'distance',dist)
                losses.append(self.Max_cost_B(idx_nearest_Xk,dist)) #maximum (loss value)
        #print(losses)
        return np.array(losses)
            
        
        
            
            
    def predict(self,X,Y):

        return self.h(X)


    
    
    
    '''def pred_phi_score(self,X,Y):

        
        prediction discrimnator score
        :param X:
        :param Y:  FOR numerical simplification, NEVER USE Y information for prediction
        :return:

        loader_te = DataLoader(self.test_handler(X, Y, transform=self.args['transform_te']), shuffle=False, **self.args['loader_te_args'])
        self.phi.eval()
        scores = torch.zeros(len(Y))

        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                out = self.phi(x).cpu()
                scores[idxs] = out.view(-1)

        return scores'''


    
    
    
    def query(self,batch_size):
        
        """
        adversarial query strategy

        :param n:
        :return:

        """

        idxs_unlabeled = np.arange(self.n_pool)[~self.idx_lb]
        # prediction output probability
        losses = self.predict_loss(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])

        # prediction output discriminative score
        #phi_score = self.pred_phi_score(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
        with torch.no_grad():
            phi_scores =  self.phi(self.X[idxs_unlabeled]).view(-1)
        
        # print(phi_score)
        
        # computing the decision score
        total_scores = (1/(batch_size+len(np.arange(self.n_pool)[self.idx_lb]))) * (losses - self.selection * phi_scores.detach().numpy())
        
        #print(total_score,np.argpartition(total_score,batch_size)[:batch_size])
        b=np.argpartition(total_scores,batch_size)[:batch_size]
        # sort the score with minimal query_number examples
        # expected value outputs from smaller to large

        return idxs_unlabeled[b]
