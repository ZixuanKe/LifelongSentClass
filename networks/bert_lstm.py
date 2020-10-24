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

        self.lstm = LSTM(
                    embedding_dim = args.bert_hidden_size,
                    hidden_dim = args.bert_hidden_size,
                    n_layers=1,
                    bidirectional=False,
                    dropout=0.5,
                    args=args)

        self.last=torch.nn.ModuleList()
        for t,n in self.taskcla:
            self.last.append(torch.nn.Linear(args.bert_hidden_size,n))
        print('BERT (Fixed) + LSTM')

        return

    def forward(self,input_ids, segment_ids, input_mask):
        sequence_output, pooled_output = \
            self.bert(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)

        h = self.lstm(sequence_output)

        #loss ==============
        y=[]
        for t,i in self.taskcla:
            y.append(self.last[t](h))
        return y

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
        return hidden