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
from core.ClientManage_hr import ClientManageHR
from utils.my_logging import Logger
from core.function import assign_hyper_gradient
from torch.optim import SGD
import torch

import numpy as np
import copy

start_time = int(time.time())

if __name__ == '__main__':
    # parse args
    args = args_parser()
    dataset_train, dataset_test, dict_users, args.img_size, dataset_train_real = load_data(args)
    net_glob = build_model(args)

    # copy weights
    w_glob = net_glob.state_dict()
    if args.output == None:
        logs = Logger(f'./save/hrfed_IO{args.optim}_{args.dataset}_sz{args.size}_iid{args.iid}_\
K{args.num_users}_C{args.frac}_{args.model}_E{args.epochs}_\
lr{args.lr}_hlr{args.hlr}_B{args.local_bs}_tau{args.outer_tau}_blo{not args.no_blo}_\
IE{args.inner_ep}_p{args.p}_N{args.neumann}_{args.hvp_method}_s{args.seed}_{start_time}.yaml')  
    else:
        logs = Logger(args.output)                                                           
    
    # Set inner and outer parameters, and outer optimizer
    hyper_param= [k for n,k in net_glob.named_parameters() if not "header" in n]
    param= [k for n,k in net_glob.named_parameters() if "header" in n]
    # for p in net_glob.parameters(): print(p,p.name)
    # for n,p in net_glob.named_parameters(): print(n,p,p.name)
    comm_round=0
    hyper_optimizer=SGD(hyper_param, lr=1)
    
    # Initialize correction term, cs[user index][header index]
    cs = []
    for i in range(args.num_users):
        c=[]
        for n,p in net_glob.named_parameters():
            if not 'header' in n:
                continue
            c.append(torch.zeros_like(p.detach()))
        cs.append(c)
    # print('init correction term',cs,'c for user 0',cs[0])
    
    # Initialize local nets, unfortuantely this is needed due to FedSkip
    nets_loc = [copy.deepcopy(net_glob) for i in range(args.num_users)]
    # print('Local net list',nets_loc, 'net user 0', nets_loc[0])

    # Global epoch (Fedskip+Fedout)
    for iter in range(args.epochs):
        m = max(int(args.frac * args.num_users), 1)
        
        # FedSkip
        for _ in range(args.inner_ep):
            client_idx = np.random.choice(range(args.num_users), m, replace=False)
            client_manage=ClientManageHR(args,net_glob,client_idx, dataset_train, dict_users,hyper_param, nets_loc)
            w_glob, loss_avg, cs, comm = client_manage.fed_skip(cs)
            comm_round+=int(comm)    
            #print(comm_round)
        net_glob.load_state_dict(w_glob)
        
        # FedOut
        if args.no_blo== False:
            client_idx = np.random.choice(range(args.num_users), m, replace=False)
            client_manage=ClientManageHR(args,net_glob,client_idx, dataset_train, dict_users,hyper_param, nets_loc)
            hg_glob, r = client_manage.fed_out()
            assign_hyper_gradient(hyper_param, hg_glob)
            hyper_optimizer.step()
            # sync hyper param, the comm round is counted in fed_out
            # this is sadly needed due to FedSKip
            for net_loc in nets_loc:
                hp_loc = [p for n,p in net_loc.named_parameters() if not "header" in n]
                for i, p in enumerate(hp_loc):
                    d = hyper_param[i].detach() - p.detach()
                    p.data.add_(d)
            # for net_loc in nets_loc:
            #     net_loc.load_state_dict(net_glob.state_dict())
            comm_round+=r
        
        # print loss
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))

        # testing
        net_glob.eval()
        acc_train, loss_train = test_img(net_glob, dataset_train_real, args)
        acc_test, loss_test = test_img(net_glob, dataset_test, args)
        print("Test acc/loss: {:.2f} {:.2f}".format(acc_test, loss_test),
              "Train acc/loss: {:.2f} {:.2f}".format(acc_train, loss_train),
              f"Comm round: {comm_round}")

        logs.logging(np.array([-1]), acc_test, acc_train, loss_test, loss_train, comm_round)
        logs.save()

        if args.round>0 and comm_round>args.round:
            break
        
