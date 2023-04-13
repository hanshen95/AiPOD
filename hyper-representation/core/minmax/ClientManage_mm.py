import copy
from cv2 import log
import numpy as np

import torch


from utils.Fed import FedAvg,FedAvgGradient, FedAvgP
from core.minmax.SGDClient_mm import SGDClient
from core.minmax.SVRGClient_mm import SVRGClient
from core.minmax.Client_mm import Client
from core.ClientManage import ClientManage

class ClientManageMM(ClientManage):
    def __init__(self,args, net_glob, client_idx, dataset, dict_users, hyper_param) -> None:
        super().__init__(args, net_glob, client_idx, dataset, dict_users, hyper_param)
            
        self.client_idx=client_idx
        self.args=args
        self.dataset=dataset
        self.dict_users=dict_users
        
        self.hyper_param = copy.deepcopy(hyper_param)

    def fed_in(self):
        #print(self.client_idx)
        w_glob = self.net_glob.state_dict()
        if self.args.all_clients:
            print("Aggregation over all clients")
            w_locals = [w_glob for i in range(self.args.num_users)]
        else:
            w_locals=[]

        loss_locals = []
        grad_locals = []
        client_locals = []

        temp_net=copy.deepcopy(self.net_glob)
        for name, w in temp_net.named_parameters():
            if "outer" in name:
                w.requires_grad= False

        for idx in self.client_idx:
            if self.args.optim == 'sgd':
                client = SGDClient(self.args, idx, copy.deepcopy(temp_net),self.dataset, self.dict_users, self.hyper_param)
            elif self.args.optim == 'svrg':
                client = SVRGClient(self.args, idx, copy.deepcopy(temp_net),self.dataset, self.dict_users, self.hyper_param)
                grad = client.batch_grad()
                grad_locals.append(grad)
            else:
                raise NotImplementedError
            client_locals.append(client)
        if self.args.optim == 'svrg':
            avg_grad = FedAvgGradient(grad_locals)
            #print(avg_grad)
            for client in client_locals:
                client.set_avg_q(avg_grad)
        for client in client_locals:
            w, loss = client.train_epoch()
            if self.args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        # update global weights
        w_glob = FedAvg(w_locals)

        # copy weight to net_glob
        self.net_glob.load_state_dict(w_glob)
        loss_avg = sum(loss_locals) / len(loss_locals)
        return w_glob, loss_avg

    def lfed_out(self,client_locals):
        hg_locals =[]
        for client in client_locals:
            for _ in range(self.args.outer_tau):
                hg_client = client.hyper_grad(None)
                hg = client.hyper_update(hg_client)
            hg_locals.append(hg)
        hg_glob=FedAvgP(hg_locals, self.args)
        return hg_glob, 1


    def fed_out(self):
        client_locals=[]
        for idx in self.client_idx:
            client= Client(self.args, idx, copy.deepcopy(self.net_glob),self.dataset, self.dict_users, self.hyper_param)
            client_locals.append(client)

        if self.args.hvp_method == 'seperate':
            return self.lfed_out(client_locals)

        comm_round = 0

        hg_locals =[]
        for client in client_locals:
            hg= client.hyper_grad(None)
            hg_locals.append(hg)
        hg_glob=FedAvgP(hg_locals, self.args)
        #print("hg_glob",hg_glob)
        comm_round+=1
        #print(hg_glob)
        hg_locals =[]
        for client in client_locals:
            for _ in range(self.args.outer_tau):
                h = client.hyper_svrg_update(hg_glob)
            hg_locals.append(h)
        hg_glob=FedAvgP(hg_locals, self.args)
        comm_round+=1


        return hg_glob, comm_round

    def fed_joint(self):
        #print(self.client_idx)
        w_glob = self.net_glob.state_dict()
        if self.args.all_clients:
            print("Aggregation over all clients")
            w_locals = [w_glob for i in range(self.args.num_users)]
        else:
            w_locals=[]

        loss_locals = []
        grad_locals = []
        client_locals = []

        temp_net=copy.deepcopy(self.net_glob)
        

        for idx in self.client_idx:
            if self.args.optim == 'sgd':
                client = SGDClient(self.args, idx, copy.deepcopy(temp_net),self.dataset, self.dict_users, self.hyper_param)
            elif self.args.optim == 'svrg':
                client = SVRGClient(self.args, idx, copy.deepcopy(temp_net),self.dataset, self.dict_users, self.hyper_param)
                grad = client.batch_grad()
                grad_locals.append(grad)
            else:
                raise NotImplementedError
            client_locals.append(client)
        if self.args.optim == 'svrg':
            avg_grad = FedAvgGradient(grad_locals)
            for client in client_locals:
                client.set_avg_q(avg_grad)
        for client in client_locals:
            w, loss = client.train_epoch()
            if self.args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        # update global weights
        w_glob = FedAvg(w_locals)

        # copy weight to net_glob
        self.net_glob.load_state_dict(w_glob)
        loss_avg = sum(loss_locals) / len(loss_locals)
        #return w_glob, loss_avg

        hg_glob, comm_round = self.lfed_out(client_locals)
        return w_glob, loss_avg, hg_glob, comm_round



    
