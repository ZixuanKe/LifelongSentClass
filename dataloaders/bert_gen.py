#Coding: UTF-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from pytorch_pretrained_bert import BertTokenizer
import torch
import absa_data_utils as data_utils
from absa_data_utils import ABSATokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import math
asc_datasets = [
            './dat/XuSemEval/asc/14/rest',
            './dat/XuSemEval/asc/14/laptop',

            './dat/Bing3Domains/asc/Speaker',
            './dat/Bing3Domains/asc/Router',
            './dat/Bing3Domains/asc/Computer',

            './dat/Bing5Domains/asc/Nokia6610',
            './dat/Bing5Domains/asc/NikonCoolpix4300',
            './dat/Bing5Domains/asc/CreativeLabsNomadJukeboxZenXtra40GB',
            './dat/Bing5Domains/asc/CanonG3',
            './dat/Bing5Domains/asc/ApexAD2600Progressive',

            './dat/Bing9Domains/asc/CanonPowerShotSD500',
            './dat/Bing9Domains/asc/CanonS100',
            './dat/Bing9Domains/asc/DiaperChamp',
            './dat/Bing9Domains/asc/HitachiRouter',
            './dat/Bing9Domains/asc/ipod',
            './dat/Bing9Domains/asc/LinksysRouter',
            './dat/Bing9Domains/asc/MicroMP3',
            './dat/Bing9Domains/asc/Nokia6600',
            './dat/Bing9Domains/asc/Norton',
            ]

ae_datasets = [
            './dat/XuSemEval/ae/14/rest',
            './dat/XuSemEval/ae/14/laptop',

            './dat/Bing3Domains/ae/Speaker',
            './dat/Bing3Domains/ae/Router',
            './dat/Bing3Domains/ae/Computer',

            './dat/Bing5Domains/ae/Nokia6610',
            './dat/Bing5Domains/ae/NikonCoolpix4300',
            './dat/Bing5Domains/ae/CreativeLabsNomadJukeboxZenXtra40GB',
            './dat/Bing5Domains/ae/CanonG3',
            './dat/Bing5Domains/ae/ApexAD2600Progressive',

            './dat/Bing9Domains/ae/CanonPowerShotSD500',
            './dat/Bing9Domains/ae/CanonS100',
            './dat/Bing9Domains/ae/DiaperChamp',
            './dat/Bing9Domains/ae/HitachiRouter',
            './dat/Bing9Domains/ae/ipod',
            './dat/Bing9Domains/ae/LinksysRouter',
            './dat/Bing9Domains/ae/MicroMP3',
            './dat/Bing9Domains/ae/Nokia6600',
            './dat/Bing9Domains/ae/Norton',
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

    #TODO: 另外生成多一个mask for generation

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
        asc_dataset = asc_datasets[domains.index(random_sep[t])]
        ae_dataset = ae_datasets[domains.index(random_sep[t])]

        data[t]={}
        if 'Bing' in asc_dataset:
            data[t]['name']=asc_dataset
            data[t]['ncla']=2
        elif 'XuSemEval' in asc_dataset:
            data[t]['name']=asc_dataset
            data[t]['ncla']=3


        print('ae_dataset: ',ae_dataset)


        logger.info("***** Running training *****")

        #ASC for encoder ====================
        processor = data_utils.AscProcessor()
        label_list = processor.get_labels()
        tokenizer = ABSATokenizer.from_pretrained(args.bert_model)
        train_examples = processor.get_train_examples(asc_dataset)
        train_features = data_utils.convert_examples_to_features_gen(
            train_examples, label_list, args.max_seq_length, tokenizer, "asc")

        all_asc_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_asc_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_asc_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_asc_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        all_tasks = torch.tensor([t for f in train_features], dtype=torch.long)

        #AE for decoder ====================
        processor = data_utils.AeProcessor()
        label_list = processor.get_labels()
        tokenizer = ABSATokenizer.from_pretrained(args.bert_model)
        train_examples = processor.get_train_examples(ae_dataset)
        train_features = data_utils.convert_examples_to_features_gen(
            train_examples, label_list, args.max_seq_length, tokenizer, "ae")

        all_ae_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_ae_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_ae_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_ae_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)

        #SG (sentence generation) for decoder ====================
        processor = data_utils.SgProcessor()
        label_list = None
        tokenizer = ABSATokenizer.from_pretrained(args.bert_model)
        train_examples = processor.get_train_examples(asc_dataset)

        mask_source_words=args.mask_source_words
        max_pred=args.max_pred
        mask_prob=args.mask_prob
        skipgram_prb=args.skipgram_prb
        skipgram_size=args.skipgram_size
        mask_whole_word=args.mask_whole_word
        vocab_words=list(tokenizer.vocab.keys())
        indexer=tokenizer.convert_tokens_to_ids

        train_features = data_utils.convert_examples_to_features_gen(
            train_examples, label_list, args.max_seq_length*2, tokenizer, "sg",
            mask_source_words=mask_source_words,max_pred=max_pred,mask_prob=mask_prob,
            skipgram_prb=skipgram_prb,skipgram_size=skipgram_size,
            mask_whole_word=mask_whole_word,vocab_words=vocab_words,indexer=indexer) #seq2seq task


        all_sg_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_sg_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_sg_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_sg_masked_lm_labels = torch.tensor([f.masked_lm_labels for f in train_features], dtype=torch.long).squeeze(1)
        all_sg_masked_pos = torch.tensor([f.masked_pos for f in train_features], dtype=torch.long).squeeze(1)
        all_sg_masked_weights = torch.tensor([f.masked_weights for f in train_features], dtype=torch.long)


        ae_length = all_ae_input_ids.size(0)
        while all_ae_input_ids.size(0) < all_sg_input_ids.size(0):
            rand_id = torch.randint(low=0,high=ae_length,size=(1,))
            all_ae_input_ids=torch.cat([all_ae_input_ids,all_ae_input_ids[rand_id]],0)
            all_ae_segment_ids=torch.cat([all_ae_segment_ids,all_ae_segment_ids[rand_id]],0)
            all_ae_input_mask=torch.cat([all_ae_input_mask,all_ae_input_mask[rand_id]],0)
            all_ae_label_ids=torch.cat([all_ae_label_ids,all_ae_label_ids[rand_id]],0)

        #some have sentiment conflict, the ae can be larger than asc
        asc_length = all_asc_input_ids.size(0)
        while all_asc_input_ids.size(0) < all_ae_input_ids.size(0):
            rand_id = torch.randint(low=0,high=asc_length,size=(1,))
            all_asc_input_ids=torch.cat([all_asc_input_ids,all_asc_input_ids[rand_id]],0)
            all_asc_segment_ids=torch.cat([all_asc_segment_ids,all_asc_segment_ids[rand_id]],0)
            all_asc_input_mask=torch.cat([all_asc_input_mask,all_asc_input_mask[rand_id]],0)
            all_asc_label_ids=torch.cat([all_asc_label_ids,all_asc_label_ids[rand_id]],0)
            all_sg_input_ids=torch.cat([all_sg_input_ids,all_sg_input_ids[rand_id]],0)
            all_sg_segment_ids=torch.cat([all_sg_segment_ids,all_sg_segment_ids[rand_id]],0)
            all_sg_input_mask=torch.cat([all_sg_input_mask,all_sg_input_mask[rand_id]],0)
            all_sg_masked_lm_labels=torch.cat([all_sg_masked_lm_labels,all_sg_masked_lm_labels[rand_id]],0)
            all_sg_masked_pos=torch.cat([all_sg_masked_pos,all_sg_masked_pos[rand_id]],0)
            all_sg_masked_weights=torch.cat([all_sg_masked_weights,all_sg_masked_weights[rand_id]],0)
            all_tasks=torch.cat([all_tasks,all_tasks[rand_id]],0)

            # ae is smaller in size than others. beacuase a sentence can have multiple terms


        num_train_steps = int(math.ceil(all_asc_input_ids.size(0) / args.train_batch_size)) * args.num_train_epochs
        # num_train_steps = int(len(train_examples) / args.train_batch_size) * args.num_train_epochs

        logger.info("  Num asc examples = %d", all_asc_input_ids.size(0))
        logger.info("  Num sg examples = %d", all_sg_input_ids.size(0))
        logger.info("  Num ae examples = %d", all_ae_input_ids.size(0))

        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)



        train_data = \
            TensorDataset(all_asc_input_ids,all_asc_segment_ids, all_asc_input_mask,\
            all_sg_input_ids, all_sg_segment_ids, all_sg_input_mask,\
            all_sg_masked_lm_labels,all_sg_masked_pos,all_sg_masked_weights,\
            all_ae_input_ids, all_ae_segment_ids, all_ae_input_mask,all_ae_label_ids,all_asc_label_ids,all_tasks)

        data[t]['train'] = train_data
        data[t]['num_train_steps']=num_train_steps


        logger.info("***** Running validations *****")

        processor = data_utils.AscProcessor()
        label_list = processor.get_labels()
        tokenizer = BertTokenizer.from_pretrained(args.bert_model)
        dev_examples = processor.get_dev_examples(asc_dataset)
        dev_features = data_utils.convert_examples_to_features_gen(dev_examples, label_list, args.max_seq_length, tokenizer, "asc")

        all_asc_input_ids = torch.tensor([f.input_ids for f in dev_features], dtype=torch.long)
        all_asc_segment_ids = torch.tensor([f.segment_ids for f in dev_features], dtype=torch.long)
        all_asc_input_mask = torch.tensor([f.input_mask for f in dev_features], dtype=torch.long)
        all_asc_label_ids = torch.tensor([f.label_id for f in dev_features], dtype=torch.long)
        all_tasks = torch.tensor([t for f in dev_features], dtype=torch.long)

        #AE for decoder ====================
        processor = data_utils.AeProcessor()
        label_list = processor.get_labels()
        tokenizer = ABSATokenizer.from_pretrained(args.bert_model)
        dev_examples = processor.get_dev_examples(ae_dataset)
        dev_features = data_utils.convert_examples_to_features_gen(
            dev_examples, label_list, args.max_seq_length, tokenizer, "ae")

        all_ae_input_ids = torch.tensor([f.input_ids for f in dev_features], dtype=torch.long)
        all_ae_segment_ids = torch.tensor([f.segment_ids for f in dev_features], dtype=torch.long)
        all_ae_input_mask = torch.tensor([f.input_mask for f in dev_features], dtype=torch.long)
        all_ae_label_ids = torch.tensor([f.label_id for f in dev_features], dtype=torch.long)




        #SG (sentence generation) for decoder ====================
        processor = data_utils.SgProcessor()
        label_list = None
        tokenizer = ABSATokenizer.from_pretrained(args.bert_model)
        dev_examples = processor.get_dev_examples(asc_dataset)
        mask_source_words=args.mask_source_words
        max_pred=args.max_pred
        mask_prob=args.mask_prob
        skipgram_prb=args.skipgram_prb
        skipgram_size=args.skipgram_size
        mask_whole_word=args.mask_whole_word
        vocab_words=list(tokenizer.vocab.keys())
        indexer=tokenizer.convert_tokens_to_ids


        dev_features = data_utils.convert_examples_to_features_gen(
            dev_examples, label_list, args.max_seq_length*2, tokenizer, "sg",
            mask_source_words=mask_source_words,max_pred=max_pred,mask_prob=mask_prob,
            skipgram_prb=skipgram_prb,skipgram_size=skipgram_size,
            mask_whole_word=mask_whole_word,vocab_words=vocab_words,indexer=indexer) #seq2seq task

        all_sg_input_ids = torch.tensor([f.input_ids for f in dev_features], dtype=torch.long)
        all_sg_segment_ids = torch.tensor([f.segment_ids for f in dev_features], dtype=torch.long)
        all_sg_input_mask = torch.tensor([f.input_mask for f in dev_features], dtype=torch.long)
        all_sg_masked_lm_labels = torch.tensor([f.masked_lm_labels for f in dev_features], dtype=torch.long).squeeze(1)
        all_sg_masked_pos = torch.tensor([f.masked_pos for f in dev_features], dtype=torch.long).squeeze(1)
        all_sg_masked_weights = torch.tensor([f.masked_weights for f in dev_features], dtype=torch.long)

        ae_length = all_ae_input_ids.size(0)
        while all_ae_input_ids.size(0) < all_sg_input_ids.size(0):
            rand_id = torch.randint(low=0,high=ae_length,size=(1,))
            all_ae_input_ids=torch.cat([all_ae_input_ids,all_ae_input_ids[rand_id]],0)
            all_ae_segment_ids=torch.cat([all_ae_segment_ids,all_ae_segment_ids[rand_id]],0)
            all_ae_input_mask=torch.cat([all_ae_input_mask,all_ae_input_mask[rand_id]],0)
            all_ae_label_ids=torch.cat([all_ae_label_ids,all_ae_label_ids[rand_id]],0)

        #some have sentiment conflict, the ae can be larger than asc
        asc_length = all_asc_input_ids.size(0)
        while all_asc_input_ids.size(0) < all_ae_input_ids.size(0):
            rand_id = torch.randint(low=0,high=asc_length,size=(1,))
            all_asc_input_ids=torch.cat([all_asc_input_ids,all_asc_input_ids[rand_id]],0)
            all_asc_segment_ids=torch.cat([all_asc_segment_ids,all_asc_segment_ids[rand_id]],0)
            all_asc_input_mask=torch.cat([all_asc_input_mask,all_asc_input_mask[rand_id]],0)
            all_asc_label_ids=torch.cat([all_asc_label_ids,all_asc_label_ids[rand_id]],0)
            all_sg_input_ids=torch.cat([all_sg_input_ids,all_sg_input_ids[rand_id]],0)
            all_sg_segment_ids=torch.cat([all_sg_segment_ids,all_sg_segment_ids[rand_id]],0)
            all_sg_input_mask=torch.cat([all_sg_input_mask,all_sg_input_mask[rand_id]],0)
            all_sg_masked_lm_labels=torch.cat([all_sg_masked_lm_labels,all_sg_masked_lm_labels[rand_id]],0)
            all_sg_masked_pos=torch.cat([all_sg_masked_pos,all_sg_masked_pos[rand_id]],0)
            all_sg_masked_weights=torch.cat([all_sg_masked_weights,all_sg_masked_weights[rand_id]],0)
            all_tasks=torch.cat([all_tasks,all_tasks[rand_id]],0)

        logger.info("  Num asc examples = %d", all_asc_input_ids.size(0))
        logger.info("  Num sg examples = %d", all_sg_input_ids.size(0))
        logger.info("  Num ae examples = %d", all_ae_input_ids.size(0))


        valid_data = \
            TensorDataset(all_asc_input_ids,all_asc_segment_ids, all_asc_input_mask,\
            all_sg_input_ids, all_sg_segment_ids, all_sg_input_mask,\
            all_sg_masked_lm_labels,all_sg_masked_pos,all_sg_masked_weights,\
            all_ae_input_ids, all_ae_segment_ids, all_ae_input_mask,all_ae_label_ids,all_asc_label_ids,all_tasks)


        data[t]['valid']=valid_data

        logger.info("***** Running evaluation *****")

        processor = data_utils.AscProcessor()
        label_list = processor.get_labels()
        tokenizer = BertTokenizer.from_pretrained(args.bert_model)
        eval_examples = processor.get_test_examples(asc_dataset)
        eval_features = data_utils.convert_examples_to_features_gen(eval_examples, label_list, args.max_seq_length, tokenizer, "asc")



        all_asc_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_asc_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_asc_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_asc_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        all_tasks = torch.tensor([t for f in eval_features], dtype=torch.long)

        #AE for decoder ====================
        processor = data_utils.AeProcessor()
        label_list = processor.get_labels()
        tokenizer = ABSATokenizer.from_pretrained(args.bert_model)
        eval_examples = processor.get_test_examples(ae_dataset)

        eval_features = data_utils.convert_examples_to_features_gen(
            eval_examples, label_list, args.max_seq_length, tokenizer, "ae")


        all_ae_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_ae_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_ae_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_ae_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)

        #SG (sentence generation) for decoder ====================
        processor = data_utils.SgProcessor()
        label_list = None
        tokenizer = ABSATokenizer.from_pretrained(args.bert_model)
        eval_examples = processor.get_test_examples(asc_dataset)

        mask_source_words=args.mask_source_words
        max_pred=args.max_pred
        mask_prob=args.mask_prob
        skipgram_prb=args.skipgram_prb
        skipgram_size=args.skipgram_size
        mask_whole_word=args.mask_whole_word
        vocab_words=list(tokenizer.vocab.keys())
        indexer=tokenizer.convert_tokens_to_ids

        eval_features = data_utils.convert_examples_to_features_gen(
            eval_examples, label_list, args.max_seq_length*2, tokenizer, "sg",
            mask_source_words=mask_source_words,max_pred=max_pred,mask_prob=mask_prob,
            skipgram_prb=skipgram_prb,skipgram_size=skipgram_size,
            mask_whole_word=mask_whole_word,vocab_words=vocab_words,indexer=indexer) #seq2seq task


        all_sg_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_sg_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_sg_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_sg_masked_lm_labels = torch.tensor([f.masked_lm_labels for f in eval_features], dtype=torch.long).squeeze(1)
        all_sg_masked_pos = torch.tensor([f.masked_pos for f in eval_features], dtype=torch.long).squeeze(1)
        all_sg_masked_weights = torch.tensor([f.masked_weights for f in eval_features], dtype=torch.long)

        ae_length = all_ae_input_ids.size(0)
        while all_ae_input_ids.size(0) < all_sg_input_ids.size(0):
            rand_id = torch.randint(low=0,high=ae_length,size=(1,))
            all_ae_input_ids=torch.cat([all_ae_input_ids,all_ae_input_ids[rand_id]],0)
            all_ae_segment_ids=torch.cat([all_ae_segment_ids,all_ae_segment_ids[rand_id]],0)
            all_ae_input_mask=torch.cat([all_ae_input_mask,all_ae_input_mask[rand_id]],0)
            all_ae_label_ids=torch.cat([all_ae_label_ids,all_ae_label_ids[rand_id]],0)

        #some have sentiment conflict, the ae can be larger than asc
        asc_length = all_asc_input_ids.size(0)
        while all_asc_input_ids.size(0) < all_ae_input_ids.size(0):
            rand_id = torch.randint(low=0,high=asc_length,size=(1,))
            all_asc_input_ids=torch.cat([all_asc_input_ids,all_asc_input_ids[rand_id]],0)
            all_asc_segment_ids=torch.cat([all_asc_segment_ids,all_asc_segment_ids[rand_id]],0)
            all_asc_input_mask=torch.cat([all_asc_input_mask,all_asc_input_mask[rand_id]],0)
            all_asc_label_ids=torch.cat([all_asc_label_ids,all_asc_label_ids[rand_id]],0)
            all_sg_input_ids=torch.cat([all_sg_input_ids,all_sg_input_ids[rand_id]],0)
            all_sg_segment_ids=torch.cat([all_sg_segment_ids,all_sg_segment_ids[rand_id]],0)
            all_sg_input_mask=torch.cat([all_sg_input_mask,all_sg_input_mask[rand_id]],0)
            all_sg_masked_lm_labels=torch.cat([all_sg_masked_lm_labels,all_sg_masked_lm_labels[rand_id]],0)
            all_sg_masked_pos=torch.cat([all_sg_masked_pos,all_sg_masked_pos[rand_id]],0)
            all_sg_masked_weights=torch.cat([all_sg_masked_weights,all_sg_masked_weights[rand_id]],0)
            all_tasks=torch.cat([all_tasks,all_tasks[rand_id]],0)

        logger.info("  Num asc examples = %d", all_asc_input_ids.size(0))
        logger.info("  Num sg examples = %d", all_sg_input_ids.size(0))
        logger.info("  Num ae examples = %d", all_ae_input_ids.size(0))



        eval_data = \
            TensorDataset(all_asc_input_ids,all_asc_segment_ids, all_asc_input_mask,\
            all_sg_input_ids, all_sg_segment_ids, all_sg_input_mask,\
            all_sg_masked_lm_labels,all_sg_masked_pos,all_sg_masked_weights,\
            all_ae_input_ids, all_ae_segment_ids, all_ae_input_mask,all_ae_label_ids,all_asc_label_ids,all_tasks)

        # Run prediction for full data

        data[t]['test']=eval_data

        taskcla.append((t,int(data[t]['ncla'])))



    # Others
    n=0
    for t in data.keys():
        n+=data[t]['ncla']
    data['ncla']=n


    return data,taskcla


