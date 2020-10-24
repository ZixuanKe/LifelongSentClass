import sys
import torch
from transformers import BertModel, BertConfig
import utils
from torch import nn
import torch.nn.functional as F

class Net(torch.nn.Module):

    def __init__(self,taskcla,args):

        super(Net,self).__init__()

        self.taskcla=taskcla
        self.args=args

        config = BertConfig.from_pretrained(args.bert_model)
        self.bert = BertModel.from_pretrained(args.bert_model,config=config)

        #BERT fixed, i.e. BERT as feature extractor===========
        for param in self.bert.parameters():
            param.requires_grad = False

        self.relu=torch.nn.ReLU()
        self.mcl = MCL(args,taskcla)
        self.ac = AC(args,taskcla)

        self.last=torch.nn.ModuleList()
        for t,n in self.taskcla:
            self.last.append(torch.nn.Linear(args.bert_hidden_size,n))


        print('BERT (Fixed) + LSTM + KAN')


        return

    def forward(self,t, input_ids, segment_ids, input_mask, which_type,s):
        sequence_output, pooled_output = \
            self.bert(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)

        gfc=self.ac.mask(t=t,s=s)

        if which_type == 'mcl':
            mcl_output,mcl_hidden = self.mcl.lstm(sequence_output)
            if t == 0: mcl_hidden = mcl_hidden*torch.ones_like(gfc.expand_as(mcl_hidden)) # everyone open
            else: mcl_hidden=mcl_hidden*gfc.expand_as(mcl_hidden)

            h=self.relu(mcl_hidden)

        elif which_type == 'ac':
            mcl_output,mcl_hidden = self.mcl.lstm(sequence_output)
            mcl_output=self.relu(mcl_output)
            mcl_output=mcl_output*gfc.expand_as(mcl_output)
            ac_output,ac_hidden = self.ac.lstm(mcl_output)
            h=self.relu(ac_hidden)

        #loss ==============
        y=[]
        for t,i in self.taskcla:
            y.append(self.last[t](h))
        return y

    def get_view_for(self,n,mask):
        if n=='mcl.lstm.rnn.weight_ih_l0':
            # print('not none')
            return mask.data.view(1,-1).expand_as(self.mcl.lstm.rnn.weight_ih_l0)
        elif n=='mcl.lstm.rnn.weight_hh_l0':
            return mask.data.view(1,-1).expand_as(self.mcl.lstm.rnn.weight_hh_l0)
        elif n=='mcl.lstm.rnn.bias_ih_l0':
            return mask.data.view(-1).repeat(4)
        elif n=='mcl.lstm.rnn.bias_hh_l0':
            return mask.data.view(-1).repeat(4)
        return None



class AC(nn.Module):
    def __init__(self,args,taskcla):
        super().__init__()

        self.lstm = LSTM(
                    embedding_dim = args.bert_hidden_size,
                    hidden_dim = args.bert_hidden_size,
                    n_layers=1,
                    bidirectional=False,
                    dropout=0.5,
                    args=args)

        self.efc=torch.nn.Embedding(args.num_task,args.bert_hidden_size)
        self.gate=torch.nn.Sigmoid()


    def mask(self,t,s=1):
        gfc=self.gate(s*self.efc(torch.LongTensor([t]).cuda()))
        return gfc


class MCL(nn.Module):
    def __init__(self,args,taskcla):
        super().__init__()

        self.lstm = LSTM(
                    embedding_dim = args.bert_hidden_size,
                    hidden_dim = args.bert_hidden_size,
                    n_layers=1,
                    bidirectional=False,
                    dropout=0.5,
                    args=args)

class LSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, n_layers,
                 bidirectional, dropout, args):
        super().__init__()

        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           dropout=dropout,
                           batch_first=True)
        self.args = args

    def forward(self, x):
        output, (hidden, cell) = self.rnn(x)
        hidden = hidden.view(-1,self.args.bert_hidden_size)
        output = output.view(-1,self.args.max_seq_length,self.args.bert_hidden_size)

        return output,hidden