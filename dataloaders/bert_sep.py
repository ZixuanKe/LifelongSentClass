#Coding: UTF-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from transformers import BertTokenizer as BertTokenizer
import os
import torch
import numpy as np
import random
import absa_data_utils as data_utils
from absa_data_utils import ABSATokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

datasets = [
            './dat/XuSemEval/14/rest',
            './dat/XuSemEval/14/laptop',

            './dat/Bing3Domains/Speaker',
            './dat/Bing3Domains/Router',
            './dat/Bing3Domains/Computer',

            './dat/Bing5Domains/Nokia6610',
            './dat/Bing5Domains/NikonCoolpix4300',
            './dat/Bing5Domains/CreativeLabsNomadJukeboxZenXtra40GB',
            './dat/Bing5Domains/CanonG3',
            './dat/Bing5Domains/ApexAD2600Progressive',

            './dat/Bing9Domains/CanonPowerShotSD500',
            './dat/Bing9Domains/CanonS100',
            './dat/Bing9Domains/DiaperChamp',
            './dat/Bing9Domains/HitachiRouter',
            './dat/Bing9Domains/ipod',
            './dat/Bing9Domains/LinksysRouter',
            './dat/Bing9Domains/MicroMP3',
            './dat/Bing9Domains/Nokia6600',
            './dat/Bing9Domains/Norton',
            ]


domains = [
     'XuSemEval14_rest',
     'XuSemEval14_laptop',

     'Bing3domains_Speaker',
     'Bing3domains_Router',
     'Bing3domains_Computer',

     'Bing5domains_Nokia6610',
     'Bing5domains_NikonCoolpix4300',
     'Bing5domains_CreativeLabsNomadJukeboxZenXtra40GB',
     'Bing5domains_CanonG3',
     'Bing5domains_ApexAD2600Progressive',

     'Bing9domains_CanonPowerShotSD500',
     'Bing9domains_CanonS100',
     'Bing9domains_DiaperChamp',
     'Bing9domains_HitachiRouter',
     'Bing9domains_ipod',
     'Bing9domains_LinksysRouter',
     'Bing9domains_MicroMP3',
     'Bing9domains_Nokia6600',
     'Bing9domains_Norton']



def get(logger=None,args=None):
    data={}
    taskcla=[]
    # Others
    f_name = 'asc_random'

    with open(f_name,'r') as f_random_seq:
        random_sep = f_random_seq.readlines()[args.idrandom].split()

    print('random_sep: ',random_sep)
    print('domains: ',domains)

    print('random_sep: ',len(random_sep))
    print('domains: ',len(domains))

    for t in range(args.ntasks):
        dataset = datasets[domains.index(random_sep[t])]

        data[t]={}
        if 'Bing' in dataset:
            data[t]['name']=dataset
            data[t]['ncla']=2
        elif 'XuSemEval' in dataset:
            data[t]['name']=dataset
            data[t]['ncla']=3


        processor = data_utils.AscProcessor()
        label_list = processor.get_labels()
        tokenizer = ABSATokenizer.from_pretrained(args.bert_model)
        train_examples = processor.get_train_examples(dataset)
        num_train_steps = int(len(train_examples) / args.train_batch_size) * args.num_train_epochs

        train_features = data_utils.convert_examples_to_features_bert_sep(
            train_examples, label_list, args.max_term_length, args.max_sentence_length, tokenizer, "asc")
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)

        all_term_input_ids = torch.tensor([f.term_input_ids for f in train_features], dtype=torch.long)
        all_term_segment_ids = torch.tensor([f.term_segment_ids for f in train_features], dtype=torch.long)
        all_term_input_mask = torch.tensor([f.term_input_mask for f in train_features], dtype=torch.long)
        
        all_sentence_input_ids = torch.tensor([f.sentence_input_ids for f in train_features], dtype=torch.long)
        all_sentence_segment_ids = torch.tensor([f.sentence_segment_ids for f in train_features], dtype=torch.long)
        all_sentence_input_mask = torch.tensor([f.sentence_input_mask for f in train_features], dtype=torch.long)        
        
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        all_tasks = torch.tensor([t for f in train_features], dtype=torch.long)

        train_data = TensorDataset(all_term_input_ids, all_term_segment_ids, all_term_input_mask,
                                   all_sentence_input_ids,all_sentence_segment_ids,all_sentence_input_mask,
                                   all_label_ids, all_tasks)

        data[t]['train'] = train_data
        data[t]['num_train_steps']=num_train_steps

        valid_examples = processor.get_dev_examples(dataset)
        valid_features=data_utils.convert_examples_to_features_bert_sep(
            valid_examples, label_list, args.max_term_length, args.max_sentence_length, tokenizer, "asc")
        valid_term_all_input_ids = torch.tensor([f.term_input_ids for f in valid_features], dtype=torch.long)
        valid_term_all_segment_ids = torch.tensor([f.term_segment_ids for f in valid_features], dtype=torch.long)
        valid_term_all_input_mask = torch.tensor([f.term_input_mask for f in valid_features], dtype=torch.long)
        
        valid_sentence_all_input_ids = torch.tensor([f.sentence_input_ids for f in valid_features], dtype=torch.long)
        valid_sentence_all_segment_ids = torch.tensor([f.sentence_segment_ids for f in valid_features], dtype=torch.long)
        valid_sentence_all_input_mask = torch.tensor([f.sentence_input_mask for f in valid_features], dtype=torch.long)        
        
        valid_all_label_ids = torch.tensor([f.label_id for f in valid_features], dtype=torch.long)
        valid_all_tasks = torch.tensor([t for f in valid_features], dtype=torch.long)

        valid_data = TensorDataset(
            valid_term_all_input_ids, valid_term_all_segment_ids, valid_term_all_input_mask,
            valid_sentence_all_input_ids, valid_sentence_all_segment_ids, valid_sentence_all_input_mask,
            valid_all_label_ids, valid_all_tasks)

        logger.info("***** Running validations *****")
        logger.info("  Num orig examples = %d", len(valid_examples))
        logger.info("  Num split examples = %d", len(valid_features))
        logger.info("  Batch size = %d", args.train_batch_size)

        data[t]['valid']=valid_data


        processor = data_utils.AscProcessor()
        label_list = processor.get_labels()
        tokenizer = BertTokenizer.from_pretrained(args.bert_model)
        eval_examples = processor.get_test_examples(dataset)
        eval_features = data_utils.convert_examples_to_features_bert_sep(
            eval_examples, label_list, args.max_term_length, args.max_sentence_length, tokenizer, "asc")

        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        
        all_term_input_ids = torch.tensor([f.term_input_ids for f in eval_features], dtype=torch.long)
        all_term_segment_ids = torch.tensor([f.term_segment_ids for f in eval_features], dtype=torch.long)
        all_term_input_mask = torch.tensor([f.term_input_mask for f in eval_features], dtype=torch.long)
        
        all_sentence_input_ids = torch.tensor([f.sentence_input_ids for f in eval_features], dtype=torch.long)
        all_sentence_segment_ids = torch.tensor([f.sentence_segment_ids for f in eval_features], dtype=torch.long)
        all_sentence_input_mask = torch.tensor([f.sentence_input_mask for f in eval_features], dtype=torch.long)

        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        all_tasks = torch.tensor([t for f in eval_features], dtype=torch.long)

        eval_data = TensorDataset(
            all_term_input_ids, all_term_segment_ids, all_term_input_mask,
            all_sentence_input_ids, all_sentence_segment_ids, all_sentence_input_mask,
            all_label_ids,all_tasks)
        # Run prediction for full data

        data[t]['test']=eval_data
        taskcla.append((t,int(data[t]['ncla'])))

    # Others
    n=0
    for t in data.keys():
        n+=data[t]['ncla']
    data['ncla']=n


    return data,taskcla


