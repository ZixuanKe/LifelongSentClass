import sys,os,argparse,time
import numpy as np
import torch
from config import set_args
import utils
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import RandomSampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import logging
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, ConcatDataset
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
tstart=time.time()

# Arguments


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

args = set_args()

if args.output=='':
    args.output='./res/'+args.experiment+'_'+args.approach+'_'+str(args.note)+'.txt'

performance_output=args.output+'_performance'
performance_output_forward=args.output+'_forward_performance'

# print('='*100)
# print('Arguments =')
# for arg in vars(args):
#     print('\t'+arg+':',getattr(args,arg))
# print('='*100)

########################################################################################################################

# Seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available(): torch.cuda.manual_seed(args.seed)
else: print('[CUDA unavailable]'); sys.exit()

# Args -- Experiment
if args.experiment=='w2v':
    from dataloaders import w2v as dataloader
elif args.experiment=='bert':
    from dataloaders import bert as dataloader
elif  args.experiment=='bert_gen_hat':
    from dataloaders import bert_gen_hat as dataloader
elif  args.experiment=='bert_gen' or args.experiment=='bert_gen_single':
    from dataloaders import bert_gen as dataloader
elif args.experiment=='bert_sep':
    from dataloaders import bert_sep as dataloader


# Args -- Approach
if args.approach=='bert_lstm_ncl' or args.approach=='bert_gru_ncl':
    from approaches import bert_rnn_ncl as approach
elif args.approach=='bert_lstm_kan_ncl' or args.approach=='bert_gru_kan_ncl':
    from approaches import bert_rnn_kan_ncl as approach

# # Args -- Network
if 'bert_lstm_kan' in args.approach:
    from networks import bert_lstm_kan as network
elif 'bert_lstm' in args.approach:
    from networks import bert_lstm as network
if 'bert_gru_kan' in args.approach:
    from networks import bert_gru_kan as network
elif 'bert_gru' in args.approach:
    from networks import bert_gru as network
#
# else:
#     raise NotImplementedError
#

########################################################################################################################

# Load
print('Load data...')
data,taskcla=dataloader.get(logger=logger,args=args)

print('\nTask info =',taskcla)

# Inits
print('Inits...')
net=network.Net(taskcla,args=args).cuda()


appr=approach.Appr(net,logger=logger,args=args)

# Loop tasks
acc=np.zeros((len(taskcla),len(taskcla)),dtype=np.float32)
lss=np.zeros((len(taskcla),len(taskcla)),dtype=np.float32)

for t,ncla in taskcla:
    print('*'*100)
    print('Task {:2d} ({:s})'.format(t,data[t]['name']))
    print('*'*100)

    # if t>1: exit()

    if 'mtl' in args.approach:
        # Get data. We do not put it to GPU
        if t==0:
            train=data[t]['train']
            valid=data[t]['valid']
            num_train_steps=data[t]['num_train_steps']

        else:
            train = ConcatDataset([train,data[t]['train']])
            valid = ConcatDataset([valid,data[t]['valid']])
            num_train_steps+=data[t]['num_train_steps']
        task=t

        if t < len(taskcla)-1: continue #only want the last one

    else:
        # Get data
        train=data[t]['train']
        valid=data[t]['valid']
        num_train_steps=data[t]['num_train_steps']
        task=t

    train_sampler = RandomSampler(train)
    train_dataloader = DataLoader(train, sampler=train_sampler, batch_size=args.train_batch_size)

    valid_sampler = SequentialSampler(valid)
    valid_dataloader = DataLoader(valid, sampler=valid_sampler, batch_size=args.eval_batch_size)


    appr.train(task,train_dataloader,valid_dataloader,args)
    print('-'*100)
    #

    # Test
    for u in range(t+1):
        test=data[u]['test']
        test_sampler = SequentialSampler(test)
        test_dataloader = DataLoader(test, sampler=test_sampler, batch_size=args.eval_batch_size)

        if 'kan' in args.approach:
            test_loss,test_acc=appr.eval(u,test_dataloader,'mcl')
        else:
            test_loss,test_acc=appr.eval(u,test_dataloader)
        print('>>> Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.1f}% <<<'.format(u,data[u]['name'],test_loss,100*test_acc))
        acc[t,u]=test_acc
        lss[t,u]=test_loss

    # Save
    print('Save at '+args.output)
    np.savetxt(args.output,acc,'%.4f',delimiter='\t')

    # appr.decode(train_dataloader)
    # break

# Done
print('*'*100)
print('Accuracies =')
for i in range(acc.shape[0]):
    print('\t',end='')
    for j in range(acc.shape[1]):
        print('{:5.1f}% '.format(100*acc[i,j]),end='')
    print()
print('*'*100)
print('Done!')

print('[Elapsed time = {:.1f} h]'.format((time.time()-tstart)/(60*60)))


with open(performance_output,'w') as file:
    if 'ncl' in args.approach  or 'mtl' in args.approach:
        for j in range(acc.shape[1]):
            file.writelines(str(acc[-1][j]) + '\n')

    elif 'one' in args.approach:
        for j in range(acc.shape[1]):
            file.writelines(str(acc[j][j]) + '\n')


with open(performance_output_forward,'w') as file:
    if 'ncl' in args.approach  or 'mtl' in args.approach:
        for j in range(acc.shape[1]):
            file.writelines(str(acc[j][j]) + '\n')


########################################################################################################################
