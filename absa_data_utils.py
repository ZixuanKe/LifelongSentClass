# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team and authors from University of Illinois at Chicago.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import json
import os
import numpy as np
from transformers import BertTokenizer
import string
import torch
from random import randint, shuffle, choice
from random import random as rand

class ABSATokenizer(BertTokenizer):     
    def subword_tokenize(self, tokens, labels): # for AE
        split_tokens, split_labels= [], []
        idx_map=[]
        for ix, token in enumerate(tokens):
            sub_tokens=self.wordpiece_tokenizer.tokenize(token)
            for jx, sub_token in enumerate(sub_tokens):
                split_tokens.append(sub_token)
                if labels[ix]=="B" and jx>0:
                    split_labels.append("I")
                else:
                    split_labels.append(labels[ix])
                idx_map.append(ix)
        return split_tokens, split_labels, idx_map

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid=None, text_a=None, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                    input_ids=None,
                    input_mask=None,
                    segment_ids=None,

                    tokens_term_ids=None,
                    tokens_sentence_ids=None,

                    term_input_ids=None,
                    term_input_mask=None,
                    term_segment_ids=None,

                    sentence_input_ids=None,
                    sentence_input_mask=None,
                    sentence_segment_ids=None,

                    label_id=None,

                    masked_lm_labels = None,
                    masked_pos = None,
                    masked_weights = None,

                    position_ids=None

                    ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids

        self.label_id = label_id

        self.masked_lm_labels = masked_lm_labels,
        self.masked_pos = masked_pos,
        self.masked_weights = masked_weights

        self.tokens_term_ids = tokens_term_ids
        self.tokens_sentence_ids = tokens_sentence_ids

        self.term_input_ids = term_input_ids
        self.term_input_mask = term_input_mask
        self.term_segment_ids = term_segment_ids

        self.sentence_input_ids = sentence_input_ids
        self.sentence_input_mask = sentence_input_mask
        self.sentence_segment_ids = sentence_segment_ids

        self.position_ids = position_ids

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()
        
    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_json(cls, input_file):
        """Reads a json file for tasks in sentiment analysis."""
        with open(input_file) as f:
            return json.load(f)
        
        
class AeProcessor(DataProcessor):
    """Processor for the SemEval Aspect Extraction ."""

    def get_train_examples(self, data_dir, fn="train.json"):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, fn)), "train")

    def get_dev_examples(self, data_dir, fn="dev.json"):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, fn)), "dev")
    
    def get_test_examples(self, data_dir, fn="test.json"):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, fn)), "test")

    def get_labels(self):
        """See base class."""
        return ["O", "B", "I"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, ids) in enumerate(lines):
            guid = "%s-%s" % (set_type, ids )
            text_a = lines[ids]['tokens'] #no text b appearently
            label = lines[ids]['labels']
            examples.append(
                InputExample(guid=guid, text_a=text_a, label=label) )
        return examples        
        

class AscProcessor(DataProcessor):
    """Processor for the SemEval Aspect Sentiment Classification."""

    def get_train_examples(self, data_dir, fn="train.json"):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, fn)), "train")

    def get_dev_examples(self, data_dir, fn="dev.json"):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, fn)), "dev")
    
    def get_test_examples(self, data_dir, fn="test.json"):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, fn)), "test")

    def get_labels(self):
        """See base class."""
        return ["positive", "negative", "neutral"]
    
    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, ids) in enumerate(lines):
            guid = "%s-%s" % (set_type, ids )
            text_a = lines[ids]['term']
            text_b = lines[ids]['sentence']
            label = lines[ids]['polarity']
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples     



class SgProcessor(DataProcessor):
    """Processor for the Sentence Generation Task."""

    def get_train_examples(self, data_dir, fn="train.json"):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, fn)), "train")

    def get_dev_examples(self, data_dir, fn="dev.json"):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, fn)), "dev")

    def get_test_examples(self, data_dir, fn="test.json"):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, fn)), "test")

    def _create_examples(self, lines, set_type):
        #no label, or, label is the sentence itself
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, ids) in enumerate(lines):
            guid = "%s-%s" % (set_type, ids )
            text_a = lines[ids]['sentence']
            text_b = lines[ids]['sentence']
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b))
        return examples



class StringProcessor(DataProcessor):
    """Processor for the SemEval Aspect Sentiment Classification."""

    def get_examples(self, lines):
        """See base class."""
        return self._create_examples(lines)

    def _create_examples(self, lines):
        """Creates examples for the training and dev sets."""
        examples = []
        for line in lines:
            text_a = line
            examples.append(
                InputExample(text_a=text_a))
        return examples

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, mode):
    """Loads a data file into a list of `InputBatch`s.""" #check later if we can merge this function with the SQuAD preprocessing 
    # label_map = {}
    # for (i, label) in enumerate(label_list):
    #     label_map[label] = i

    label_map={'+': 0,'positive': 0, '-': 1, 'negative': 1, 'neutral': 2}

    features = []
    for (ex_index, example) in enumerate(examples):
        if mode!="ae":
            tokens_a = tokenizer.tokenize(example.text_a)
        else: #only do subword tokenization.
            tokens_a, labels_a, example.idx_map= tokenizer.subword_tokenize([token.lower() for token in example.text_a], example.label )

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if mode!="ae":
            label_id = label_map[example.label]
        else:
            label_id = [-1] * len(input_ids) #-1 is the index to ignore
            #truncate the label length if it exceeds the limit.
            lb=[label_map[label] for label in labels_a]
            if len(lb) > max_seq_length - 2:
                lb = lb[0:(max_seq_length - 2)]
            label_id[1:len(lb)+1] = lb

        features.append(
                InputFeatures(
                        input_ids=input_ids,
                        input_mask=input_mask,
                        segment_ids=segment_ids,
                        label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_examples_to_features_bert_sep(examples, label_list, max_term_length, max_sentence_length, tokenizer, mode):
    # 'sep' means separate representation for sentence and term"""
    """Loads a data file into a list of `InputBatch`s.""" #check later if we can merge this function with the SQuAD preprocessing
    # label_map = {}
    # for (i, label) in enumerate(label_list):
    #     label_map[label] = i

    '''seperate the sentence and term, instead of merging everything together'''
    label_map={'+': 0,'positive': 0, '-': 1, 'negative': 1, 'neutral': 2}

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)


        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_b) > max_term_length - 2:
            tokens_b = tokens_b[0:(max_sentence_length - 2)]
            
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_term_length - 2:
            tokens_a = tokens_a[0:(max_term_length - 2)]

        term_tokens = []
        term_segment_ids = []
        term_tokens.append("[CLS]")
        term_segment_ids.append(0)
        for token in tokens_a:
            term_tokens.append(token)
            term_segment_ids.append(0)
        term_tokens.append("[SEP]")
        term_segment_ids.append(0)

        sentence_tokens = []
        sentence_segment_ids = []
        sentence_tokens.append("[CLS]")
        sentence_segment_ids.append(0)
        for token in tokens_b:
            sentence_tokens.append(token)
            sentence_segment_ids.append(0)
        sentence_tokens.append("[SEP]")
        sentence_segment_ids.append(0)

        term_input_ids = tokenizer.convert_tokens_to_ids(term_tokens)
        sentence_input_ids = tokenizer.convert_tokens_to_ids(sentence_tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        term_input_mask = [1] * len(term_input_ids)
        # Zero-pad up to the sequence length.
        while len(term_input_ids) < max_term_length:
            term_input_ids.append(0)
            term_input_mask.append(0)
            term_segment_ids.append(0)
            
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        sentence_input_mask = [1] * len(sentence_input_ids)
        # Zero-pad up to the sequence length.
        while len(sentence_input_ids) < max_sentence_length:
            sentence_input_ids.append(0)
            sentence_input_mask.append(0)
            sentence_segment_ids.append(0)            
            

        assert len(term_input_ids) == max_term_length
        assert len(term_input_mask) == max_term_length
        assert len(term_segment_ids) == max_term_length

        label_id = label_map[example.label]


        features.append(
                InputFeatures(
                        term_input_ids=term_input_ids,
                        term_input_mask=term_input_mask,
                        term_segment_ids=term_segment_ids,
                        sentence_input_ids=sentence_input_ids,
                        sentence_input_mask=sentence_input_mask,
                        sentence_segment_ids=sentence_segment_ids,
                        label_id=label_id))
    return features




def convert_examples_to_features_w2v(examples, label_list, max_term_length, max_sentence_length, tokenizer, word_index_pre_trained,vocab_size):

    # prepare for word2vector experiments

    """Loads a data file into a list of `InputBatch`s.""" #check later if we can merge this function with the SQuAD preprocessing
    # label_map = {}
    # for (i, label) in enumerate(label_list):
    #     label_map[label] = i

    label_map={'+': 0,'positive': 0, '-': 1, 'negative': 1, 'neutral': 2}

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = whitespace_tokenize(example.text_a.translate(str.maketrans('', '', string.punctuation)).lower())
        tokens_b = whitespace_tokenize(example.text_b.translate(str.maketrans('', '', string.punctuation)).lower())

        while True:
            if len(tokens_a) <= max_term_length:
                break
            else:
                tokens_a.pop()

        while True:
            if len(tokens_b) <= max_sentence_length:
                break
            else:
                tokens_b.pop()

        tokens_a_ids = []
        tokens_b_ids = []

        for word in tokens_a:
            embedding_index = word_index_pre_trained.get(word)
            if embedding_index is not None:
                tokens_a_ids.append(embedding_index)
            else:
                print('miss word: ',str(word))
                tokens_a_ids.append(vocab_size)

        for word in tokens_b:
            embedding_index = word_index_pre_trained.get(word)
            if embedding_index is not None:
                tokens_b_ids.append(embedding_index)
            else:
                print('miss word: ',str(word))
                tokens_b_ids.append(vocab_size)

        # Zero-pad up to the sequence length.
        while len(tokens_a_ids) < max_term_length:
            tokens_a_ids.append(vocab_size)
        while len(tokens_b_ids) < max_sentence_length:
            tokens_b_ids.append(vocab_size)

        label_id = label_map[example.label]

        features.append(
                InputFeatures(
                        tokens_term_ids=tokens_a_ids,
                        tokens_sentence_ids=tokens_b_ids,
                        label_id=label_id))




    return features

def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a peice of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens



def get_random_word(vocab_words):
    i = randint(0, len(vocab_words)-1)
    return vocab_words[i]


def convert_examples_to_features_gen(
        examples, label_list, max_seq_length, tokenizer, mode,
        mask_source_words=None,max_pred=None,mask_prob=None,skipgram_prb=None,skipgram_size=None,
        mask_whole_word=None,vocab_words=None,indexer=None


):
    # A simpler version
    # input sentence, term, z and y (t excluded)

    """Loads a data file into a list of `InputBatch`s.""" #check later if we can merge this function with the SQuAD preprocessing
    # label_map = {}
    # for (i, label) in enumerate(label_list):
    #     label_map[label] = i


    #2 sets of mask for decoder: input_mask: all open; imput_mask_gen: seq2seq LM

    vars = ['z','y']


    masked_lm_labels = None
    masked_pos = None
    masked_weights = None



    features = []
    for (ex_index, example) in enumerate(examples):
        if mode=="asc" or mode=="sg":
            tokens_a = tokenizer.tokenize(example.text_a)
        elif mode=="ae": #only do subword tokenization.
            tokens_a, labels_a, example.idx_map= tokenizer.subword_tokenize([token.lower() for token in example.text_a], example.label )

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        if mode=="sg":
            # For masked Language Models
            # the number of prediction is sometimes less than max_pred when sequence is short

            tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']
            effective_length = len(tokens_b)
            if mask_source_words:
                effective_length += len(tokens_a)
            n_pred = min(max_pred, max(
                1, int(round(effective_length*mask_prob))))
            # candidate positions of masked tokens
            cand_pos = []
            special_pos = set()
            for i, tk in enumerate(tokens):
                # only mask tokens_b (target sequence)
                # we will mask [SEP] as an ending symbol
                if (i >= len(tokens_a)+2) and (tk != '[CLS]'):
                    cand_pos.append(i)
                elif mask_source_words and (i < len(tokens_a)+2) and (tk != '[CLS]') and (not tk.startswith('[SEP')):
                    cand_pos.append(i)
                else:
                    special_pos.add(i)
            shuffle(cand_pos)

            masked_pos = set()
            max_cand_pos = max(cand_pos)
            for pos in cand_pos:
                if len(masked_pos) >= n_pred:
                    break
                if pos in masked_pos:
                    continue

                def _expand_whole_word(st, end):
                    new_st, new_end = st, end
                    while (new_st >= 0) and tokens[new_st].startswith('##'):
                        new_st -= 1
                    while (new_end < len(tokens)) and tokens[new_end].startswith('##'):
                        new_end += 1
                    return new_st, new_end

                if (skipgram_prb > 0) and (skipgram_size >= 2) and (rand() < skipgram_prb):
                    # ngram
                    cur_skipgram_size = randint(2, skipgram_size)
                    if mask_whole_word:
                        st_pos, end_pos = _expand_whole_word(
                            pos, pos + cur_skipgram_size)
                    else:
                        st_pos, end_pos = pos, pos + cur_skipgram_size
                else:
                    # directly mask
                    if mask_whole_word:
                        st_pos, end_pos = _expand_whole_word(pos, pos + 1)
                    else:
                        st_pos, end_pos = pos, pos + 1

                for mp in range(st_pos, end_pos):
                    if (0 < mp <= max_cand_pos) and (mp not in special_pos):
                        masked_pos.add(mp)
                    else:
                        break

            masked_pos = list(masked_pos)
            if len(masked_pos) > n_pred:
                shuffle(masked_pos)
                masked_pos = masked_pos[:n_pred]

            masked_tokens = [tokens[pos] for pos in masked_pos]
            for pos in masked_pos:
                if rand() < 0.8:  # 80%
                    tokens[pos] = '[MASK]'
                elif rand() < 0.5:  # 10%
                    tokens[pos] = get_random_word(vocab_words)
            # when n_pred < max_pred, we only calculate loss within n_pred
            masked_weights = [1]*len(masked_tokens) #masking will be 1, otherwise 0

            # Token Indexing
            masked_ids = indexer(masked_tokens)


            _tril_matrix = torch.tril(torch.ones(
                (max_seq_length, max_seq_length), dtype=torch.long))

            sg_input_mask = torch.zeros(max_seq_length+len(vars), max_seq_length+len(vars), dtype=torch.long)
            sg_input_mask[:, :len(tokens_a)+2+len(vars)].fill_(1)
            second_st, second_end = len(
                tokens_a)+2+len(vars), len(tokens_a)+len(vars)+len(tokens_b)+3
            sg_input_mask[second_st:second_end, second_st:second_end].copy_(
                _tril_matrix[:second_end-second_st, :second_end-second_st])

            # Zero Padding for masked target
            if max_pred > n_pred:
                n_pad = max_pred - n_pred
                if masked_ids is not None:
                    masked_ids.extend([0]*n_pad)
                if masked_pos is not None:
                    masked_pos.extend([0]*n_pad)
                if masked_weights is not None:
                    masked_weights.extend([0]*n_pad)

            masked_lm_labels = [-1]*len(vars) + masked_ids
            masked_pos = [0]*len(vars) + masked_pos
            masked_weights = [0]*len(vars) + masked_weights

        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if mode=="asc":
            label_map={'+': 0,'positive': 0, '-': 1, 'negative': 1, 'neutral': 2}
            label_id = label_map[example.label]
        elif mode=="ae":
            label_map={'B': 0,'I': 1, 'O': 2}
            label_id = [-1] * len(input_ids) #-1 is the index to ignore
            #truncate the label length if it exceeds the limit.
            lb=[label_map[label] for label in labels_a]
            if len(lb) > max_seq_length - 2:
                lb = lb[0:(max_seq_length - 2)]
            label_id[1:len(lb)+1] = lb

            #ae is for decoder task
            segment_ids = [0]*len(vars) + segment_ids#will be replaced
            input_ids = [0]*len(vars) + input_ids #will be replaced
            label_id = [-1]*len(vars) + label_id #ignore the latent variables
            input_mask = [1]*len(vars) + input_mask # all opened

        elif mode=="sg":
            segment_ids = [0]*len(vars) + segment_ids#will be replaced
            input_ids = [0]*len(vars) + input_ids#will be replaced
            label_id=None
            input_mask = sg_input_mask.clone().numpy()
        features.append(
                InputFeatures(
                        input_ids=input_ids,
                        input_mask=input_mask,
                        segment_ids=segment_ids,
                        label_id=label_id,
                        masked_lm_labels=masked_lm_labels,
                        masked_pos=masked_pos,
                        masked_weights=masked_weights))
    return features


def convert_examples_to_features_gen_decoder(
        examples, max_len, tokenizer,
        max_tgt_length,indexer

):
    vars = ['z','y']

    max_a_len = max([len(tokenizer.tokenize(example.text_a)) for (ex_index, example) in enumerate(examples)])

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        padded_tokens_a = ['[CLS]'] + tokens_a + ['[SEP]']

        assert len(padded_tokens_a) <= max_a_len + 2
        if max_a_len + 2 > len(padded_tokens_a):
            padded_tokens_a += ['[PAD]'] * \
                (max_a_len + 2 - len(padded_tokens_a))
        assert len(padded_tokens_a) == max_a_len + 2
        max_len_in_batch = min(max_tgt_length +
                               max_a_len + 2, max_len)
        tokens = padded_tokens_a
        segment_ids = [0]*(len(padded_tokens_a)+len(vars)) \
            + [1]*(max_len_in_batch - len(padded_tokens_a))

        position_ids = []
        for i in range(len(tokens_a) + 2 + len(vars)):
            position_ids.append(i)
        for i in range(len(tokens_a) + 2, max_a_len + 2):
            position_ids.append(0)
        for i in range(max_a_len + 2, max_len_in_batch):
            position_ids.append(i - (max_a_len + 2) + len(tokens_a) + 2)

        # Token Indexing
        input_ids = [0]*len(vars)+indexer(tokens)
        _tril_matrix = torch.tril(torch.ones(
            (max_len, max_len), dtype=torch.long))
        # Zero Padding
        input_mask = torch.zeros(
            max_len_in_batch+len(vars), max_len_in_batch+len(vars), dtype=torch.long)
        input_mask[:, :len(tokens_a)+2+len(vars)].fill_(1)

        second_st, second_end = len(padded_tokens_a), max_len_in_batch

        input_mask[second_st:second_end, second_st:second_end].copy_(
            _tril_matrix[:second_end-second_st, :second_end-second_st])


        input_mask = input_mask.numpy()

        features.append(
                InputFeatures(
                        input_ids=input_ids,
                        input_mask=input_mask,
                        segment_ids=segment_ids,
                        position_ids=position_ids))
    return features


