import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import grad

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression





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
        gamma_ratio = ( len(idx_ulb_train) - self.args['loader_tr_args']['batch_size']  ) / ( self.args['loader_tr_args']['batch_size'] + len(idx_lb_train) )
        
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
                
                '''lb_z   = self.fea(label_x)
                unlb_z = self.fea(unlabel_x)

                opt_fea.zero_grad()'''
                opt_h.zero_grad()
                lb_out = self.h(label_x)
                
                # prediction loss (deafult we use F.cross_entropy)
                pred_loss=torch.mean(abs(lb_out-label_y))
                #BEFORE (classification) : pred_loss = torch.mean(F.cross_entropy(lb_out,label_y))

                # Wasserstein loss (here is the unbalanced loss, because we used the redundant trick)
                wassertein_distance = abs(self.phi(unlabel_x).mean() - gamma_ratio * self.phi(label_x).mean())
                #BEFORE: wassertein_distance = self.phi(unlb_z).mean() - gamma_ratio * self.phi(lb_z).mean()
                
                '''with torch.no_grad():
                    lb_z = self.fea(label_x)
                    unlb_z = self.fea(unlabel_x)
                gp = gradient_penalty(self.phi, unlb_z, lb_z)'''
                
                
                
                alpha=1
                h_loss = pred_loss + alpha * wassertein_distance #+ alpha * gp * 5 / for CIFAR10 the gradient penality is 5 / for SVHN the gradient penality is 2
                h_loss.backward()
                '''opt_fea.step()'''                
                print("grad H", torch.norm(self.h.fc1.weight.grad))
                opt_h.step()
                #print(self.h.fc1.weight)
                
                self.h.fc1.weight.data = self.L2Lip(self.h.fc1.weight)
                
                # Then the second step, training discriminator
                set_requires_grad(self.h, requires_grad=False)
                set_requires_grad(self.phi, requires_grad=True)

                '''with torch.no_grad():
                    lb_z = self.fea(label_x)
                    unlb_z = self.fea(unlabel_x)
                for _ in range(1):
                    # gradient ascent for multiple times like GANS training
                    gp = gradient_penalty(self.phi, unlb_z, lb_z)'''
                
                #BEFORE (change because unlb_z & lb_z change with the feature extractor): wassertein_distance = self.phi(unlb_z).mean() - gamma_ratio * self.phi(lb_z).mean()
                #NB: obliged to do the calculation because now phi is used with required_grad= TRUE, else can't do phi_loss.bacward()
                wassertein_distance = abs(self.phi(unlabel_x).mean() - gamma_ratio * self.phi(label_x).mean())  
                
                phi_loss = -1 * alpha * wassertein_distance
                #phi_loss = -1 * alpha * wassertein_distance - alpha * gp * 2
                
                opt_phi.zero_grad()
                phi_loss.backward()
                print("grad PHI", torch.norm(self.phi.fc1.weight.grad))
                opt_phi.step()

                self.phi.fc1.weight.data = self.L2Lip(self.phi.fc1.weight)
                
                #print('h_weights',self.h.fc1.weight)
                #print('phi_weights,self.phi.fc1.weight)
             
            print('==========Inner epoch {:d} ========'.format(epoch))
            ''''if(max(abs(torch.norm(self.h.fc1.weight.grad).numpy()),abs(torch.norm(self.phi.fc1.weight.grad).numpy()))<0.1):
                #print('BREAK',max(abs(torch.norm(self.h.fc1.weight.grad).numpy()),abs(torch.norm(self.phi.fc1.weight.grad).numpy())))
                #print('_______________AFTER TRAIN___________________')
                #print('H_WEIGHTS\n',self.h.fc1.weight)
                #print('PHI_WEIGHTS\n',self.phi.fc1.weight)
                break'''
        #print('_______________AFTER TRAIN___________________')
        #print('H_WEIGHTS\n',self.h.fc1.weight)
        #print('PHI_WEIGHTS\n',self.phi.fc1.weight)


            
     
    
    
    def Idx_NearestP(self,Xu,idxs_lb):
        
        distances=[]
        for i in idxs_lb:
            distances.append(torch.norm(abs(Xu-self.X[i])))
            
        return idxs_lb[distances.index(min(distances))],float(min(distances))
    
    
    
    
    def f_cost(self,x,b):
        f=-abs(b-x)
        return f
        
    def df_cost(self,x):
        
        df=-1
        return df

    
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
            y_i = y_n - self.f_cost(y_n,y_true)/self.df_cost(y_n)
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
        return -self.f_cost(y_n,y_true) # - because f_cost is inversed to do gradienr descent
    
    

    
    def predict_loss(self,X,Y):
        
        print('len(Y)',len(Y))
        loader_te = DataLoader(self.test_handler(X, Y, transform=self.args['transform_te']), shuffle=False, **self.args['loader_te_args'])
        self.h.eval()
        losses = torch.zeros(len(Y))
        idxs_lb=np.arange(self.n_pool)[self.idx_lb]
        
        '''Known_losses=[]
        Known_labels=[]
        for i in idxs_lb:
            Known_losses.append(abs(self.h(self.X[i])-self.Y[i])) 
            Known_labels.append(self.Y[i])'''
        
        
        #print('-----------------------PREDICTIONS----------------------')
        #print('H_WEIGHTS\n',self.h.fc1.weight)
        #print('PHI_WEIGHTS\n',self.phi.fc1.weight)
        
        with torch.no_grad():
            for x, y, idxs in loader_te: #in batch
                x, y = x.to(self.device), y.to(self.device)
                sub_losses=[]
                for i in x:
                    idx_nearest_Xk,dist=self.Idx_NearestP(i,idxs_lb)
                    #print('idx',idx_nearest_Xk,'distance',dist)
                    sub_losses.append(self.Max_cost_B(idx_nearest_Xk,dist))
                idxs=torch.reshape(idxs, (len(idxs),1))
                #print('idxs',idxs,'loss',sub_losses)
                #print(idxs.shape)
                loss=max(sub_losses)
                losses[idxs] = loss.cpu()
        
        return losses
            
        
        
            
            
    def predict(self,X,Y):

        
        loader_te = DataLoader(self.test_handler(X, Y, transform=self.args['transform_te']), shuffle=False, **self.args['loader_te_args'])
        self.h.eval()  
        P = torch.zeros(len(Y), dtype=Y.dtype).long()
        
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                out  = self.h(x)
                #print('pred',pred)
                P[idxs] = pred.cpu()

        return P


    
    
    
    def pred_phi_score(self,X,Y):

        '''
        prediction discrimnator score
        :param X:
        :param Y:  FOR numerical simplification, NEVER USE Y information for prediction
        :return:
        '''

        loader_te = DataLoader(self.test_handler(X, Y, transform=self.args['transform_te']), shuffle=False, **self.args['loader_te_args'])
        self.phi.eval()
        scores = torch.zeros(len(Y))

        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                out = self.phi(x).cpu()
                scores[idxs] = out.view(-1)

        return scores


    
    
    
    def query(self,query_num):
        
        """
        adversarial query strategy

        :param n:
        :return:

        """

        idxs_unlabeled = np.arange(self.n_pool)[~self.idx_lb]
        # prediction output probability
        losses = self.predict_loss(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])

        # uncertainly score (three options, single_worst, L2_upper, L1_upper)
        # uncertainly_score = self.single_worst(probs)
        '''uncertainly_score = 0.5* self.L2_upper(probs) + 0.5* self.L1_upper(probs)'''

        # print(uncertainly_score)

        # prediction output discriminative score
        phi_score = self.pred_phi_score(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
        
        # print(phi_score)


        # computing the decision score
        total_score = losses - self.selection * phi_score
        # print(total_score)
        b = total_score.sort()[1][:query_num]
        # print(total_score[b])

        # sort the score with minimal query_number examples
        # expected value outputs from smaller to large

        return idxs_unlabeled[total_score.sort()[1][:query_num]]
