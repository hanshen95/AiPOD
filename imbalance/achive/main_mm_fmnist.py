#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import yaml
import time
from core.test import test_img
from utils.Fed import FedAvg, FedAvgGradient
from models.SvrgUpdate import LocalUpdate
from utils.options import args_parser
from utils.dataset_normal import load_data
from models.ModelBuilder import build_model
from core.mm_fmnist.ClientManage_fm import ClientManageMM
from utils.my_logging import Logger
from core.function import assign_hyper_gradient
from torch.optim import SGD
import torch
import sys
import numpy as np
import copy
import os
from pathlib import Path
import random
start_time = int(time.time())

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.model="fmnist_cnn"
    args.dataset = "fmnist"

    dataset_train, dataset_test, dict_users, args.img_size, dataset_train_real = load_data(args)

    torch.manual_seed(1)
    net_glob = build_model(args)
    
    # copy weights
    w_glob = net_glob.state_dict()
    if args.output == None:
        os.makedirs("./save",exist_ok=True)
        logs = Logger(f'./save/minmax_fmnist_{args.optim}_{args.dataset}\
_{args.model}_{args.epochs}_C{args.frac}_iid{args.iid}_\
{args.lr}_blo{not args.no_blo}_\
IE{args.inner_ep}_N{args.neumann}_HLR{args.hlr}_{args.hvp_method}_{start_time}.yaml')  
    else:
        p=Path(args.output)
        if not os.path.exists(p.parent):
            os.makedirs(p.parent,exist_ok=True)
        logs = Logger(args.output)                                                           
    
    hyper_param= [k for n,k in net_glob.named_parameters() if "outer" in n]
    param= [k for n,k in net_glob.named_parameters() if "inner" in n]

    #print(hyper_param)

    comm_round=0
    hyper_optimizer=SGD(hyper_param, lr=1)
    
    
    for iter in range(args.epochs):
        m = max(int(args.frac * args.num_users), 1)
        
        for _ in range(args.inner_ep):
            client_idx = np.random.choice(range(args.num_users), m, replace=False)
            client_manage=ClientManageMM(args,net_glob,client_idx, dataset_train, dict_users,hyper_param)
            if args.hvp_method=='joint':
                w_glob,loss_avg, hg_glob, r = client_manage.fed_joint()
                assign_hyper_gradient(hyper_param, hg_glob)
                hyper_optimizer.step()
                args.no_blo=True
            else:
                w_glob, loss_avg = client_manage.fed_in()
            loss_avg=-loss_avg
            if args.optim == 'svrg':
                comm_round+=2
            else:
                comm_round+=1
        net_glob.load_state_dict(w_glob)
        
        if args.no_blo== False:
            client_idx = np.random.choice(range(args.num_users), m, replace=False)
            client_manage=ClientManageMM(args,net_glob,client_idx, dataset_train, dict_users,hyper_param)
            hg_glob, r = client_manage.fed_out()
            assign_hyper_gradient(hyper_param, hg_glob)
            hyper_optimizer.step()
            comm_round+=r
        

        # print loss
        print('Round {:3d}, Average loss {:.3f}, t-values {}'.format(iter, loss_avg, np.round(net_glob.t_inner.detach().cpu().numpy(),3)))

        # testing
        net_glob.eval()
        acc_train, loss_train = test_img(net_glob, dataset_train_real, args)
        acc_test, loss_test = test_img(net_glob, dataset_test, args)
        print("Test acc/loss: {:.2f} {:.2f}".format(acc_test, loss_test),
              "Train acc/loss: {:.2f} {:.2f}".format(acc_train, loss_train),
              f"Comm round: {comm_round}")

        logs.logging(client_idx, acc_test, acc_train, loss_test, loss_train, comm_round)
        logs.save()
        if args.round>0 and comm_round>args.round:
            break
        
    logs.save()