import numpy as np
from dataset_WA import get_dataset,get_handler
import dataset
from model import get_net
from torchvision import transforms
import torch
from query_strategies import WAAL
import pandas as pd



''' FINIR OPTIMISATION l(h,y)
rajouter L2Lips sur H et PHI
Tester loss abs/abs.sq/log/log.sq
Vérifier décroissance gradients (h et phi)
'''







NUM_INIT_LB = 10
NUM_QUERY   = 2
NUM_ROUND   = 1
DATA_NAME   = 'Boston2'


args_pool = {
            'Boston2':
                {'transform_tr': {transforms.ToTensor()},
                 'transform_te': {transforms.ToTensor()},
                 'loader_tr_args':{'batch_size': 5, 'num_workers': 1},
                 'loader_te_args':{'batch_size': 2, 'num_workers': 1},
                 'optimizer_args':{'lr': 0.01, 'momentum': 0.5}
                }
            }

args = args_pool[DATA_NAME]



# load dataset (Only using the first 50 K)
X_tr, Y_tr, X_te, Y_te = get_dataset(DATA_NAME)

X_tr=X_tr[:50]
Y_tr=Y_tr[:50]

# start experiment
n_pool = len(Y_tr)
n_test = len(Y_te)
print('number of labeled pool: {}'.format(NUM_INIT_LB))
print('number of unlabeled pool: {}'.format(n_pool - NUM_INIT_LB))
print('number of testing pool: {}'.format(n_test))

# setting training parameters
alpha = 2e-3
epoch = 10

# Generate the initial labeled pool
idxs_lb = np.zeros(n_pool, dtype=bool)
idxs_tmp = np.arange(n_pool)
np.random.shuffle(idxs_tmp)
idxs_lb[idxs_tmp[:NUM_INIT_LB]] = True

# loading neural network
ffd_h, ffd_phi = get_net(DATA_NAME)

# here the training handlers and testing handlers are different
train_handler = get_handler(DATA_NAME)
test_handler  = dataset.get_handler(DATA_NAME)

strategy = WAAL(X_tr,Y_tr,idxs_lb,ffd_h,ffd_phi,train_handler,test_handler,args)


# print information
print(DATA_NAME)
#print('SEED {}'.format(SEED))
print(type(strategy).__name__)

# round 0 accuracy
strategy.train(alpha=alpha, total_epoch=epoch)
P = strategy.predict(X_te,Y_te)
acc = np.zeros(NUM_ROUND+1)
for i in range(len(P)):
    acc[0] += (P[i]-Y_te[i])*(P[i]-Y_te[i])
acc[0] = acc[0]/len(Y_te)
print('Round 0\ntesting accuracy {:.3f}'.format(acc[0]))

for rd in range(1,NUM_ROUND+1):

    print('================Round {:d}==============='.format(rd))

    #epoch += 5
    q_idxs = strategy.query(NUM_QUERY)
    idxs_lb[q_idxs] = True

    # update
    strategy.update(idxs_lb)
    strategy.train(alpha=alpha,total_epoch=epoch)

    # compute accuracy at each round
    print(X_te[0])
    P = strategy.predict(X_te,Y_te)
    print(P)
    print(Y_te)
    print(Y_te.mean(),P.float().mean())
    for i in range(len(P)):
        acc[rd] += (P[i]-Y_te[i])*(P[i]-Y_te[i])
    acc[rd] = acc[rd]/len(Y_te)
    print('Accuracy (MSE) {:.3f}'.format(acc[rd]))


# print final results for each round
# print('SEED {}'.format(SEED))
print(type(strategy).__name__)
print(acc)

