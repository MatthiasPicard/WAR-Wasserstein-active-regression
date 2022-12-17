import numpy as np
from dataset_WA import get_dataset,get_handler
import dataset
from model import get_net
from torchvision import transforms
import torch
from query_strategies import WAAL
import pandas as pd


DATA_NAME   = 'Boston2'


inner_epochs=[20+i*5 for i in range(7)]
accs=[[i for i in range(1,40)]]
t_i=0
for max_inner in inner_epochs:
    NUM_INIT_LB = 4
    NUM_QUERY   = 4
    NUM_ROUND   = 39
    t_i+=1
    print('TEST n°',t_i,'/',len(inner_epochs))
    args_pool = {
                'Boston2':
                    {'transform_tr': {transforms.ToTensor()},
                     'transform_te': {transforms.ToTensor()},
                     'loader_tr_args':{'batch_size': NUM_QUERY, 'num_workers': 1},
                     'loader_te_args':{'batch_size': NUM_QUERY, 'num_workers': 1},
                     'optimizer_args':{'lr': 0.001, 'momentum': 0.5}
                    }
                }

    args = args_pool[DATA_NAME]


    # load dataset (Only using the first 50K)
    X_tr, Y_tr, X_te, Y_te = get_dataset(DATA_NAME)


    # start experiment
    n_pool = len(Y_tr)
    n_test = len(Y_te)
    print('number of labeled pool: {}'.format(NUM_INIT_LB))
    print('number of unlabeled pool: {}'.format(n_pool - NUM_INIT_LB))
    print('number of testing pool: {}'.format(n_test))

    # setting training parameters
    alpha = 1
    epoch = max_inner

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
    strategy.train(alpha=alpha, total_epoch= epoch)
    P = strategy.predict(X_te,Y_te)
    acc = []
    acc_i=0
    for i in range(len(P)):
        acc_i += abs((P[i]-Y_te[i])/Y_te[i])
    acc_i = ( 1-acc_i/len(Y_te) )
    acc_i=acc_i.numpy()[0]

    print('Round 0\ntesting accuracy {:.3f}'.format(acc_i))
    acc.append(acc_i)

    for rd in range(1,NUM_ROUND+1):

        print('================Round {:d}==============='.format(rd))

        acc_i=0
        #epoch += 5
        q_idxs = strategy.query(NUM_QUERY)
        idxs_lb[q_idxs] = True

        # update
        strategy.update(idxs_lb)
        strategy.train(alpha=alpha,total_epoch=epoch)

        # compute accuracy at each round
        P = strategy.predict(X_te,Y_te)
        #print(P)
        #print(Y_te)
        #print('Moyenne vraies valeurs',Y_te.mean())
        #print('Moyenne valeurs prédites',P.float().mean())
        for i in range(len(P)):
            acc_i += abs((P[i]-Y_te[i])/Y_te[i])
        acc_i = ( 1-acc_i/len(Y_te) )
        acc_i=acc_i.numpy()[0]
        #print('Accuracy (ME) {:.3f}'.format(acc_i))
        acc.append(acc_i)

    # print final results for each round
    # print('SEED {}'.format(SEED))
    print(type(strategy).__name__)
    print(acc)
    accs.append(acc)

import csv

with open("2MaxEpoch_test_lr0001.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(accs)