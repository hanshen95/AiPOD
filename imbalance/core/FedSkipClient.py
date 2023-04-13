import torch
from core.Client import Client
# from utils.SGDC import SGDC
# from function import assign_correction_grad

class FedSkipClient(Client):
    def __init__(self, args, client_id, net, dataset=None, idxs=None, hyper_param= None, if_inner=False) -> None:
        super().__init__(args, client_id, net, dataset, idxs, hyper_param, if_inner)
        self.batch_num = len(self.ldr_train)
    def train_epoch(self, c):
        self.net.train()
        # train and update
        optimizer = torch.optim.SGD(self.net.parameters(), lr=self.args.lr)
    
        epoch_loss = []
        batch_loss = []
        # rand_idx = torch.randint(0,self.batch_num,())
        self.net.zero_grad()
        for batch_idx, (images, labels) in enumerate(self.ldr_train):
            # if batch_idx != rand_idx.item():
            #     continue
            images, labels = images.to(self.args.device), labels.to(self.args.device)
            # print('now label is', labels,'length is',len(labels))
            log_probs = self.net(images)
            loss = self.loss_func(log_probs, labels)
            batch_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            break
            
            
        self.net.zero_grad()
        i = 0
        for k in self.net.parameters():
            k.data.add_(c[i],alpha=self.args.lr)
            i+=1
        
        epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return self.net.state_dict(), sum(epoch_loss) / len(epoch_loss)