from torch.optim import Optimizer
import copy
import torch

class SGDC(Optimizer):
    r"""Optimizer class: SGD with corrected direction
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
    """
    def __init__(self, params, lr):
        
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr)
        super(SGDC, self).__init__(params, defaults)
    
    def get_param_groups(self):
            return self.param_groups


    def step(self, c):
        """Performs a single optimization step.
        Args:
            c: gradient correction tensor
        """
        for group in self.param_groups:
            i = 0
            for p in group['params']:
                if p.grad is None:
                    continue
                new_d = p.grad.data - c[i].data
                # print('i is',i,'p.grad is',p.grad.data,'c[i] is', c[i].data)
                i+=1
                p.data.add_(new_d, alpha=-group['lr'])