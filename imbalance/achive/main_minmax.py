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
from core.minmax.ClientManage_mm import ClientManageMM
from utils.my_logging import Logger
from core.function import assign_hyper_gradient
from torch.optim import SGD
import torch
import sys
import numpy as np
import copy
import os
from pathlib import Path

start_time = int(time.time())

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.model="linear"
    args.dataset = "minmax_synthetic"
    args.d=10
    # args.frac=1.
    # args.local_bs=100
    # args.num_users=1
    args.n=args.local_bs
    
    dataset_train, dataset_test, dict_users, args.img_size, dataset_train_real = load_data(args)
    net_glob = build_model(args)
    #print(net_glob.x, net_glob.y_header)
    
    # copy weights
    w_glob = net_glob.state_dict()
    if args.output == None:
        os.makedirs("./save",exist_ok=True)
        logs = Logger(f'./save/minmax_{args.optim}_{args.dataset}\
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

    comm_round=0
    hyper_optimizer=SGD(hyper_param, lr=1)
    
    
    for iter in range(args.epochs):
        m = max(int(args.frac * args.num_users), 1)
        #print("x and y", net_glob.x_outer,net_glob.y_inner)
        
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
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))

        # In Min-max synthetic experiments, we measure the error as x-x* where x*=0
        loss_x = torch.norm(net_glob.x_outer).detach().cpu().numpy()**2
        loss_y = torch.norm(net_glob.y_inner).detach().cpu().numpy()
        print("Loss: {:.2f} {:.2f}".format(loss_x,loss_y))
        # In min-max experiments, we store the error into accuracy position in logs for easier plotting
        logs.logging(client_idx, loss_x , loss_y , 0, 0, comm_round)
    
        if args.round>0 and comm_round>args.round:
            break
        
    logs.save()