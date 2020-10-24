import sys,time
import numpy as np
import torch
# from copy import deepcopy

import utils
from tqdm import tqdm, trange

rnn_weights = [
    'mcl.lstm.rnn.weight_ih_l0',
    'mcl.lstm.rnn.weight_hh_l0',
    'mcl.lstm.rnn.bias_ih_l0',
    'mcl.lstm.rnn.bias_hh_l0',
    'mcl.gru.rnn.weight_ih_l0',
    'mcl.gru.rnn.weight_hh_l0',
    'mcl.gru.rnn.bias_ih_l0',
    'mcl.gru.rnn.bias_hh_l0']

class Appr(object):
    def __init__(self,model,nepochs=100,sbatch=64,lr=0.05,lr_min=1e-4,lr_factor=3,lr_patience=3,clipgrad=10000,args=None,logger=None):
    # def __init__(self,model,nepochs=100,sbatch=64,lr=0.001,lr_min=1e-5,lr_factor=2,lr_patience=3,clipgrad=10000,args=None,logger=None):
        self.model=model
        # self.initial_model=deepcopy(model)

        self.nepochs=nepochs
        self.sbatch=sbatch
        self.lr=lr
        self.lr_min=lr_min
        self.lr_factor=lr_factor
        self.lr_patience=lr_patience
        self.clipgrad=clipgrad

        self.criterion=torch.nn.CrossEntropyLoss()
        self.optimizer=self._get_optimizer()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_gpu = torch.cuda.device_count()

        self.smax = 400
        self.thres_cosh=50
        self.thres_emb=6
        self.lamb=0.75

        print('CONTEXTUAL + RNN NCL')

        return

    def _get_optimizer(self,lr=None,which_type=None):

        if which_type=='mcl':
            if lr is None: lr=self.lr
            return torch.optim.SGD(
                [p for p in self.model.mcl.parameters()]+[p for p in self.model.last.parameters()],lr=lr)
        elif which_type=='ac':
            if lr is None: lr=self.lr
            return torch.optim.SGD(
                [p for p in self.model.ac.parameters()]+[p for p in self.model.last.parameters()],lr=lr)

    def train(self,t,train,valid,args):
        # self.model=deepcopy(self.initial_model) # Restart model: isolate


        if t == 0: which_types = ['mcl']
        else: which_types = ['ac','mcl']

        for which_type in which_types:

            print('Training Type: ',which_type)

            best_loss=np.inf
            best_model=utils.get_model(self.model)
            lr=self.lr
            patience=self.lr_patience
            self.optimizer=self._get_optimizer(lr,which_type)

            # Loop epochs
            for e in range(self.nepochs):
                # Train
                clock0=time.time()
                iter_bar = tqdm(train, desc='Train Iter (loss=X.XXX)')
                self.train_epoch(t,train,iter_bar,which_type)
                clock1=time.time()
                train_loss,train_acc=self.eval(t,train,which_type)
                clock2=time.time()
                print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.1f}% |'.format(e+1,
                    1000*self.sbatch*(clock1-clock0)/len(train),1000*self.sbatch*(clock2-clock1)/len(train),train_loss,100*train_acc),end='')
                # Valid
                valid_loss,valid_acc=self.eval(t,valid,which_type)
                print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss,100*valid_acc),end='')
                # Adapt lr
                if valid_loss<best_loss:
                    best_loss=valid_loss
                    best_model=utils.get_model(self.model)
                    patience=self.lr_patience
                    print(' *',end='')
                else:
                    patience-=1
                    if patience<=0:
                        lr/=self.lr_factor
                        print(' lr={:.1e}'.format(lr),end='')
                        if lr<self.lr_min:
                            print()
                            break
                        patience=self.lr_patience
                        self.optimizer=self._get_optimizer(lr,which_type)
                print()

            # Restore best
            utils.set_model_(self.model,best_model)

        return



    def train_epoch(self,t,data,iter_bar,which_type):
        self.model.train()
        # Loop batches
        for step, batch in enumerate(iter_bar):
            batch = [
                bat.to(self.device) if bat is not None else None for bat in batch]
            input_ids, segment_ids, input_mask, targets,_= batch
            s=(self.smax-1/self.smax)*step/len(data)+1/self.smax
            task=torch.autograd.Variable(torch.LongTensor([t]).cuda(),volatile=True)

            # Forward
            outputs=self.model.forward(task,input_ids, segment_ids, input_mask,which_type,s)
            output=outputs[t]
            loss=self.criterion(output,targets)
            iter_bar.set_description('Train Iter (loss=%5.3f)' % loss.item())

            # Backward
            self.optimizer.zero_grad()
            loss.backward()

            if t>0 and which_type=='mcl':
                task=torch.autograd.Variable(torch.LongTensor([t]).cuda(),volatile=False)
                mask=self.model.ac.mask(task,s=self.smax)
                mask = torch.autograd.Variable(mask.data.clone(),requires_grad=False)
                for n,p in self.model.named_parameters():
                    if n in rnn_weights:
                        # print('n: ',n)
                        # print('p: ',p.grad.size())
                        p.grad.data*=self.model.get_view_for(n,mask)

            # Compensate embedding gradients
            for n,p in self.model.ac.named_parameters():
                if 'ac.e' in n:
                    num=torch.cosh(torch.clamp(s*p.data,-self.thres_cosh,self.thres_cosh))+1
                    den=torch.cosh(p.data)+1
                    p.grad.data*=self.smax/s*num/den


            torch.nn.utils.clip_grad_norm(self.model.parameters(),self.clipgrad)
            self.optimizer.step()

            # Constrain embeddings
            for n,p in self.model.ac.named_parameters():
                if 'ac.e' in n:
                    p.data=torch.clamp(p.data,-self.thres_emb,self.thres_emb)

        return

    def eval(self,t,data,which_type):
        total_loss=0
        total_acc=0
        total_num=0
        self.model.eval()


        for step, batch in enumerate(data):
            batch = [
                bat.to(self.device) if bat is not None else None for bat in batch]
            input_ids, segment_ids, input_mask, targets,_= batch
            real_b=input_ids.size(0)
            task=torch.autograd.Variable(torch.LongTensor([t]).cuda(),volatile=True)
            outputs = self.model.forward(task,input_ids, segment_ids, input_mask,which_type,s=self.smax)
            output=outputs[t]
            loss=self.criterion(output,targets)

            _,pred=output.max(1)
            hits=(pred==targets).float()

            # Log
            total_loss+=loss.data.cpu().numpy().item()*real_b
            total_acc+=hits.sum().data.cpu().numpy().item()
            total_num+=real_b

        return total_loss/total_num,total_acc/total_num
