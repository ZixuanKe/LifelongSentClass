import sys, argparse
import numpy as np
import torch
import utils
import datetime
import sys,os,argparse,time
import numpy as np
import torch

import utils
import numpy as np

import torch
from public.data import *
import public.config
import random

from public.pytorch.util2d import *

# initialize seeds
torch.backends.cudnn.enabled = False
torch.manual_seed(config.args.seed)
np.random.seed(config.args.seed)
random.seed(config.args.seed)
if config.args.cuda:
    torch.cuda.manual_seed_all(config.args.seed)

args=config.args

print('='*100)
print('Arguments =')
for arg in vars(args):
    print('\t'+arg+':', getattr(args,arg))
print('='*100)
print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'GPU  '+os.environ["CUDA_VISIBLE_DEVICES"])
print('='*100)
########################################################################################################################

import owm as approach
import cnn_owm as network

########################################################################################################################
file_name = config.args.experiment+'_'+config.args.approach+'_'+str(config.args.sequence_id)+'Performance'

# Load
print('Load data...')

f = open('RandomSequence','r')
domain_list = f.readlines()[config.args.sequence_id].replace('\n','').split()

# domain_list = domains()
print('domain_list: ',domain_list)


word2id, weights_matrix, voc_size= compute_embedding(domain_list)
inputsize = [config.args.n_channel,config.args.max_doc_len,config.args.embedding_dim]
taskcla = [(tid,2) for tid in range(24)] # for embedding


data = {}
for current_domain_index,current_domain in enumerate(domain_list):
    print('current_domain: ',current_domain)
    data[current_domain_index] = {}
    data[current_domain_index]['train'] = {}
    data[current_domain_index]['valid'] = {}
    data[current_domain_index]['test'] = {}
    data[current_domain_index]['name'] = current_domain



    train_x, train_doc_len, train_doc_y = load_inputs_document_mongo2D(
        [current_domain], 'train', word2id, config.args.max_doc_len)
    val_x, val_doc_len, val_doc_y = load_inputs_document_mongo2D(
        [current_domain], 'dev', word2id, config.args.max_doc_len)
    test_x, test_doc_len, test_doc_y = load_inputs_document_mongo2D(
        [current_domain], 'test', word2id,config.args.max_doc_len)

    data[current_domain_index]['train']['x'] = torch.from_numpy(train_x).long().cuda()
    data[current_domain_index]['train']['y'] = torch.argmax(torch.from_numpy(train_doc_y).long(),dim=1).cuda()

    data[current_domain_index]['valid']['x'] = torch.from_numpy(val_x).long().cuda()
    data[current_domain_index]['valid']['y'] = torch.argmax(torch.from_numpy(val_doc_y).long(),dim=1).cuda()

    data[current_domain_index]['test']['x'] = torch.from_numpy(test_x).long().cuda()
    data[current_domain_index]['test']['y'] = torch.argmax(torch.from_numpy(test_doc_y).long(),dim=1).cuda()



print('Input size =', inputsize, '\nTask info =', taskcla)

# Inits
print('Inits...')
net = network.Net(inputsize,taskcla,weights_matrix,voc_size).cuda()
utils.print_model_report(net)

appr = approach.Appr(net, nepochs=args.nepochs, lr=args.lr, args=args)
utils.print_optimizer_config(appr.optimizer)
print('-'*100)

# Loop tasks
acc = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)
lss = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)
for t, ncla in taskcla:
    print('*'*100)
    print('Task {:2d} ({:s})'.format(t+1, data[t]['name']))
    print('*'*100)

    xtrain = data[t]['train']['x'].cuda()
    ytrain = data[t]['train']['y'].cuda()
    xvalid = data[t]['test']['x'].cuda()
    yvalid = data[t]['test']['y'].cuda()

    # Train
    appr.train(t, xtrain, ytrain, xvalid, yvalid, data)
    print('-'*100)

    # Test
    for u in range(t+1):
        xtest = data[u]['test']['x'].cuda()
        ytest = data[u]['test']['y'].cuda()
        test_loss, test_acc,package = appr.eval(xtest, ytest,t)
        print('>>> Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.2f}% <<<'.format(u, data[u]['name'], test_loss, 100*test_acc))
        acc[t, u] = test_acc
        lss[t, u] = test_loss

    if t == config.args.n_tasks-1: #last one
        for u in range(config.args.n_tasks):
            xtest=data[u]['test']['x'].cuda()
            ytest=data[u]['test']['y'].cuda()
            test_loss,test_acc,package=appr.eval(xtest,ytest,t)

            with open(file_name,'a+') as fn:
                fn.writelines(
                    str(package[0]) + '\t' + \
                    str(package[1]) + '\t' + str(package[2]) + \
                    '\t' + str(package[3]) + '\t' + str(package[4]) + \
                    '\t' + str(package[5]) + '\t' + str(package[6]) + \
                    '\t' + str(package[7]) + '\t' + '\n')


    # Save
    print('Save at '+file_name + ' Accuracy')
    np.savetxt(file_name + ' Accuracy' ,acc,'%.4f')

# Done
print('*'*100)
print('Accuracies =')
for i in range(acc.shape[0]):
    print('\t',end='')
    for j in range(acc.shape[1]):
        print('{:5.2f}% '.format(100*acc[i, j]),end='')
    print()
print('*'*100)
print('Done!')
print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'GPU  '+os.environ["CUDA_VISIBLE_DEVICES"])
print('='*100)




