import sys, time
import numpy as np
import torch
import utils
import numpy as np

import torch
from public.data import *
import public.config
import random

# initialize seeds
torch.backends.cudnn.enabled = False
torch.manual_seed(config.args.seed)
np.random.seed(config.args.seed)
random.seed(config.args.seed)
if config.args.cuda:
    torch.cuda.manual_seed_all(config.args.seed)
dtype = torch.cuda.FloatTensor  # run on GPU
import utils
from sklearn.metrics import precision_recall_fscore_support


########################################################################################################################

class Appr(object):

    def __init__(self, model, nepochs=0, sbatch=config.args.batch_size, lr=0,  clipgrad=10, lr_min=1e-4,lr_factor=3,lr_patience=5,args=None):
        self.model = model

        self.nepochs = nepochs
        self.sbatch = sbatch
        self.lr = lr
        self.clipgrad = clipgrad

        self.ce = torch.nn.CrossEntropyLoss()
        self.optimizer = self._get_optimizer()
        self.lr_min=lr_min
        self.lr_factor=lr_factor
        self.lr_patience=lr_patience

        self.Pc1 = torch.autograd.Variable(torch.eye(1 * 2 * 2).type(dtype), volatile=True)
        self.Pc2 = torch.autograd.Variable(torch.eye(64 * 2 * 2).type(dtype), volatile=True)
        self.Pc3 = torch.autograd.Variable(torch.eye(128 * 2 * 2).type(dtype), volatile=True)
        self.P1 = torch.autograd.Variable(torch.eye(256 * 4 * 4).type(dtype), volatile=True)
        self.P2 = torch.autograd.Variable(torch.eye(1000).type(dtype), volatile=True)
        self.P3 = torch.autograd.Variable(torch.eye(1000).type(dtype), volatile=True)

        self.test_max = 0

        return

    def _get_optimizer(self, t=0, lr=None):
        # if lr is None:
        #     lr = self.lr
        lr = self.lr
        lr_owm = self.lr
        fc1_params = list(map(id, self.model.fc1.parameters()))
        fc2_params = list(map(id, self.model.fc2.parameters()))
        fc3_params = list(map(id, self.model.fc3.parameters()))
        base_params = filter(lambda p: id(p) not in fc1_params + fc2_params + fc3_params,
                             self.model.parameters())
        optimizer = torch.optim.SGD([{'params': base_params},
                                     {'params': self.model.fc1.parameters(), 'lr': lr_owm},
                                     {'params': self.model.fc2.parameters(), 'lr': lr_owm},
                                     {'params': self.model.fc3.parameters(), 'lr': lr_owm} #fix the other, so that they cannot effect the forgetting
                                     ], lr=lr, momentum=0.9)

        return optimizer

    def train(self, t, xtrain, ytrain, xvalid, yvalid, data):
        best_loss = np.inf
        best_acc = 0
        best_model = utils.get_model(self.model)
        lr = self.lr
        # patience = self.lr_patience
        self.optimizer = self._get_optimizer(t, lr)
        nepochs = self.nepochs
        test_max = 0
        # Loop epochs
        try:
            for e in range(nepochs):
                # Train

                self.train_epoch(xtrain, ytrain, t, cur_epoch=e, nepoch=nepochs)
                train_loss, train_acc, package = self.eval(xtrain, ytrain,t)
                print('| [{:d}/24], Epoch {:d}/{:d}, | Train: loss={:.3f}, acc={:2.2f}% |'.format(t + 1, e + 1,
                                                                                                 nepochs, train_loss,
                                                                                                 100 * train_acc),
                      end='')
                # # Valid
                valid_loss, valid_acc, package = self.eval(xvalid, yvalid,t)
                print(' Valid: loss={:.3f}, acc={:5.2f}% |'.format(valid_loss, 100 * valid_acc), end='')
                print()

                xtest = data[config.args.n_tasks-1]['test']['x'].cuda()
                ytest = data[config.args.n_tasks-1]['test']['y'].cuda()

                _, test_acc,package = self.eval(xtest, ytest, t)

                # # Adapt lr
                # if valid_loss < best_loss:
                #     best_loss = min(best_loss,valid_loss)

                # if valid_acc > best_acc:
                #     best_acc = max(best_acc, valid_acc)
                if test_acc>self.test_max:
                    self.test_max = max(self.test_max, test_acc)
                    best_model = utils.get_model(self.model)

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
                        self.optimizer=self._get_optimizer(lr)

                print('>>> Test on All Task:->>> Max_acc : {:2.2f}%  Curr_acc : {:2.2f}%<<<'.format(100 * self.test_max, 100 * test_acc))

        except KeyboardInterrupt:
            print()

        # Restore best validation model
        utils.set_model_(self.model, best_model)
        return

    def train_epoch(self, x, y, task_index, cur_epoch=0, nepoch=0):
        self.model.train()

        r_len = np.arange(x.size(0))
        np.random.shuffle(r_len)
        r_len = torch.LongTensor(r_len).cuda()

        # Loop batches
        for i_batch in range(0, len(r_len), self.sbatch):
            b = r_len[i_batch:min(i_batch + self.sbatch, len(r_len))]
            images = torch.autograd.Variable(x[b], volatile=False)
            targets = torch.autograd.Variable(y[b], volatile=False)


            # Forward
            output, h_list, x_list = self.model.forward(images)
            output=output[task_index]
            loss = self.ce(output, targets)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()

            lamda = i_batch / len(r_len)/nepoch + cur_epoch/nepoch

            alpha_array = [1.0 * 0.00001 ** lamda, 1.0 * 0.0001 ** lamda, 1.0 * 0.01 ** lamda, 1.0 * 0.1 ** lamda]

            def pro_weight(p, x, w, alpha=1.0, cnn=True, stride=1):
                x=x.detach()
                p=p.detach()

                if cnn:
                    _, _, H, W = x.shape
                    F, _, HH, WW = w.shape
                    S = stride  # stride
                    Ho = int(1 + (H - HH) / S)
                    Wo = int(1 + (W - WW) / S)
                    for i in range(Ho):
                        for j in range(Wo):
                            # N*C*HH*WW, C*HH*WW = N*C*HH*WW, sum -> N*1
                            r = x[:, :, i * S: i * S + HH, j * S: j * S + WW].contiguous().view(1, -1)
                            # r = r[:, range(r.shape[1] - 1, -1, -1)]
                            k = torch.mm(p, torch.t(r))
                            p.sub_(torch.mm(k, torch.t(k)) / (alpha + torch.mm(r, k)))
                    w.grad.data = torch.mm(w.grad.data.view(F, -1), torch.t(p.data)).view_as(w)
                else:
                    r = x
                    k = torch.mm(p, torch.t(r))
                    p.sub_(torch.mm(k, torch.t(k)) / (alpha + torch.mm(r, k)))
                    w.grad.data = torch.mm(w.grad.data, torch.t(p.data))
            # Compensate embedding gradients
            for n, w in self.model.named_parameters():
                if n == 'c1.weight':
                    pro_weight(self.Pc1, x_list[0], w, alpha=alpha_array[0], stride=2)

                if n == 'c2.weight':
                    pro_weight(self.Pc2, x_list[1], w, alpha=alpha_array[0], stride=2)

                if n == 'c3.weight':
                    pro_weight(self.Pc3, x_list[2], w, alpha=alpha_array[0], stride=2)

                if n == 'fc1.weight':
                    pro_weight(self.P1,  h_list[0], w, alpha=alpha_array[1], cnn=False)

                if n == 'fc2.weight':
                    pro_weight(self.P2,  h_list[1], w, alpha=alpha_array[2], cnn=False)

                if n == 'fc3.weight':
                    pro_weight(self.P3,  h_list[2], w, alpha=alpha_array[3], cnn=False)

            # Apply step
            torch.nn.utils.clip_grad_norm(self.model.parameters(), self.clipgrad)
            self.optimizer.step()
        return



    def eval(self, x, y, task_index):
        total_loss = 0
        total_acc = 0
        total_num = 0
        self.model.eval()

        accs = []
        prec_pos = []
        prec_neg = []
        recal_pos = []
        recal_neg = []
        f1_pos = []
        f1_neg = []
        f1_macro = []


        r = np.arange(x.size(0))
        r = torch.LongTensor(r).cuda()

        # Loop batches
        for i in range(0, len(r), self.sbatch):
            b = r[i:min(i + self.sbatch, len(r))]
            images = torch.autograd.Variable(x[b], volatile=True)
            targets = torch.autograd.Variable(y[b], volatile=True)


            output, h_list, x_list = self.model.forward(images)
            output=output[task_index]
            loss = self.ce(output, targets)

            _, pred = output.max(1)
            hits = (pred == targets).float()

            # Log
            total_loss += loss.data.cpu().numpy().item() *len(b)
            total_acc += hits.sum().data.cpu().numpy().item()
            total_num += len(b)

            pred = np.array(pred.data.cpu().numpy())
            targets = np.array(targets.data.cpu().numpy())

            num_correct = sum(pred == targets)
            acc =  float(num_correct)/ len(pred)
            p_class, r_class, f_class, support_micro = precision_recall_fscore_support(y_true=targets, y_pred=pred,
                                                                                       labels=[0, 1], average=None)

            accs.append(acc)
            prec_neg.append(p_class[0])
            prec_pos.append(p_class[1])

            recal_neg.append(r_class[0])
            recal_pos.append(r_class[1])

            f1_neg.append(f_class[0])
            f1_pos.append(f_class[1])

            f1_macro.append(f_class.mean())


        package = (np.array(accs).mean(),np.array(prec_neg).mean(),np.array(prec_pos).mean(),\
                        np.array(recal_neg).mean(),np.array(recal_pos).mean(),np.array(f1_neg).mean(),\
                        np.array(f1_pos).mean(),np.array(f1_macro).mean())

        return total_loss / total_num, total_acc / total_num, package

