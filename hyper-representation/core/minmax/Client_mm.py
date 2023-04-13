import copy
from math import ceil
from warnings import catch_warnings
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from core.function import gather_flat_grad, get_trainable_hyper_params, loss_adjust_cross_entropy, gather_flat_hyper_params
from utils.svrg import SVRG_Snapshot
from numpy import random
from torch.autograd import grad
import torch.nn.functional as F
from torch.optim import SGD
from core.function import assign_hyper_gradient

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class Client():
    def __init__(self, args, client_id, net, dataset=None, idxs=None, hyper_param = None) -> None:
        self.client_id = client_id
        self.args = args
        self.net = copy.deepcopy(net)
        self.init_net = copy.deepcopy(net)
        self.net.zero_grad()
        self.init_net.zero_grad()
        self.beta = 0.1

        self.ldr_train = DataLoader(DatasetSplit(
            dataset, idxs[client_id]), batch_size=self.args.local_bs, shuffle=True)
        self.ldr_val = DataLoader(DatasetSplit(
            dataset, idxs[-client_id-1]), batch_size=self.args.local_bs, shuffle=True)
        
        # self.hyper_param = self.net.x
        # self.hyper_param_init = self.init_net.x
        self.hyper_param= [k for n,k in self.net.named_parameters() if "outer" in n]
        self.hyper_param_init = [k for n,k in self.init_net.named_parameters() if "outer" in n]
        self.hyper_optimizer= SGD(self.hyper_param,
                                lr=args.hlr)
        self.val_loss = self.minmax_outer
        self.loss_func = self.minmax_inner #nn.CrossEntropyLoss()
        self.hyper_iter = 0

    def train_epoch(self):
        pass

    def batch_grad(self):
        self.net0 = copy.deepcopy(self.net)
        self.net0.train()
        optimizer = SVRG_Snapshot([k for k in self.net0.parameters() if k.requires_grad==True])
        self.net0.zero_grad()
        for batch_idx, (images, labels) in enumerate(self.ldr_train):
            images, labels = images.to(
                self.args.device), labels.to(self.args.device)
            #print(images.shape,labels.shape)
            log_probs = self.net0(images)
            #print(log_probs.shape)
            loss = self.loss_func(log_probs, labels, self.net0)
            
            #print('loss',loss)
            loss.backward()
        return optimizer.get_param_groups(batch_idx+1)

    
    def grad_d_out_d_x(self, net = None):
        if net == None:
            net = copy.deepcopy(self.net)
        else:
            net = copy.deepcopy(net)
        net.train()
        hyper_param = [k for n,k in net.named_parameters() if "outer" in n]
        num_weights = sum(p.numel() for p in hyper_param)

        d_out_d_x = torch.zeros(num_weights, device=self.args.device)
        for batch_idx, (images, labels) in enumerate(self.ldr_val):
            images, labels = images.to(
                self.args.device), labels.to(self.args.device)
            net.zero_grad()
            log_probs = net(images)
            loss = self.val_loss(log_probs, labels,net)
            d_out_d_x += gather_flat_grad(grad(loss,
                                         get_trainable_hyper_params(hyper_param), create_graph=True))
        
        d_out_d_x /= (batch_idx+1.)
        return d_out_d_x

    def hyper_grad(self, p):
        direct_grad= self.grad_d_out_d_x()
        return direct_grad
        
    def hyper_update(self, hg):
        assign_hyper_gradient(self.hyper_param, hg.detach())
        self.hyper_optimizer.step()
        return -gather_flat_hyper_params(self.hyper_param)+gather_flat_hyper_params(self.hyper_param_init)
    
    def hyper_svrg_update(self, hg):
        direct_grad = self.grad_d_out_d_x()
        direct_grad_0 = self.grad_d_out_d_x(net=self.init_net)
        h = direct_grad - direct_grad_0 + hg
        h=h.detach()
        assign_hyper_gradient(self.hyper_param, h)
        self.hyper_optimizer.step()
        return -gather_flat_hyper_params(self.hyper_param)+gather_flat_hyper_params(self.hyper_param_init)
    
    def minmax_inner(self, Ax, b, model):
        lmbda=0.1
        #loss=torch.matmul(model.y_inner.view(-1),Ax)
        # loss =(1./self.args.local_bs)*(-0.5*torch.norm(model.y_inner)**2-torch.matmul(b.view(-1),model.y_inner)\
        #      +torch.matmul(model.y_inner.view(-1),Ax))+lmbda/2.*torch.norm(model.x_outer)**2
        loss = -model.y_inner.pow(2).sum()-torch.matmul(b.view(-1),model.y_inner)+torch.matmul(model.y_inner.view(-1),Ax)
        loss *=0.5
        loss += lmbda/2.*model.x_outer.pow(2).sum()
        loss = (1./self.args.local_bs)*loss
        return -loss
    def minmax_outer(self, Ax, b, model):
        lmbda=0.1
        #loss=torch.matmul(model.y_inner.view(-1),Ax)
        # loss =(1./self.args.local_bs)*(-0.5*torch.norm(model.y_inner)**2-torch.matmul(b.view(-1),model.y_inner)\
        #     +torch.matmul(model.y_inner.view(-1),Ax))+lmbda/2.*torch.norm(model.x_outer)**2
        loss = -model.y_inner.pow(2).sum()-torch.matmul(b.view(-1),model.y_inner)+torch.matmul(model.y_inner.view(-1),Ax)
        loss *=0.5
        loss += lmbda/2.*model.x_outer.pow(2).sum()
        loss = (1./self.args.local_bs)*loss
        return loss



