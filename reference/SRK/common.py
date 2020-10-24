#!/usr/bin/env python3

"""
@author: xi
@since: 2018-02-02
"""

import os
import shutil
import uuid

import photinia as ph

import sys
import time


class Logger(object):

    def __init__(self, log_file):
        self._terminal = sys.stdout
        self._log_file = open(log_file, 'wt')

    def __del__(self):
        self._log_file.close()

    def write(self, message):
        self._terminal.write(message)
        self._log_file.write(message)
        self._terminal.flush()
        self._log_file.flush()

    def flush(self):
        pass


class Experiment(object):

    def __init__(self, log_file, model_dir):
        self._log_file = log_file
        self._model_dir = model_dir
        #
        sys.stdout = Logger(log_file)
        if os.path.exists(model_dir):
            if input(model_dir + ' exists, delete?\n').lower() in {'y', 'yes'}:
                shutil.rmtree(model_dir)
                print(model_dir + ' deleted.')
                os.mkdir(model_dir)
        else:
            os.mkdir(model_dir)
        self._dump_count = 0

    def dump_model(self, model, msg=None):
        t = int(time.time())
        c = self._dump_count
        self._dump_count += 1
        dir_ = '%d_%d' % (t, c) if msg is None else '%d_%d_%s' % (t, c, msg)
        ph.utils.dump_model_as_tree(model, os.path.join(self._model_dir, dir_))
        print('Model %s dumped.' % dir_)

    def load_model(self, model, dir_=None):
        if dir_ is None:
            dirs = os.listdir(self._model_dir)
            if len(dirs) == 0:
                print('This is a new model.')
                return
            dirs.sort(key=lambda a: a)
            dir_ = dirs[-1]
        ph.utils.load_model_from_tree(model, os.path.join(self._model_dir, dir_))
        print('Model %s loaded.' % dir_)


class TrainSource(ph.DataSource):

    def __init__(self, coll, we, domain):
        pos_ds = ph.MongoSource(
            coll,
            {'domain': domain, 'label': 1, 'rnd': {'$gte': 0.1}},
            ['lemma', 'label']
        )
        self._pos_ds = pos_ds
        pos_ds.add_field_mappers('lemma', lambda a: we.words_to_vectors(a))
        pos_ds.add_field_mappers('label', lambda a: ph.utils.one_hot(a, 2))
        neg_ds = ph.MongoSource(
            coll,
            {'domain': domain, 'label': 0, 'rnd': {'$gte': 0.1}},
            ['lemma', 'label']
        )
        self._neg_ds = neg_ds
        neg_ds.add_field_mappers('lemma', lambda a: we.words_to_vectors(a))
        neg_ds.add_field_mappers('label', lambda a: ph.utils.one_hot(a, 2))

    def next_batch(self, size=0):
        size = size // 2
        pos_tokens, pos_labels = self._pos_ds.next_batch(size)
        neg_tokens, neg_labels = self._neg_ds.next_batch(size)
        tokens = ph.utils.pad_sequences(pos_tokens + neg_tokens)
        labels = pos_labels + neg_labels
        return tokens, labels


class DevSource(ph.DataSource):

    def __init__(self, coll, we, domain):
        pos_ds = ph.MongoSource(
            coll,
            {'domain': domain, 'label': 1, 'rnd': {'$lt': 0.1}},
            ['lemma', 'label']
        )
        self._pos_ds = pos_ds
        pos_ds.add_field_mappers('lemma', lambda a: we.words_to_vectors(a))
        pos_ds.add_field_mappers('label', lambda a: ph.utils.one_hot(a, 2))
        neg_ds = ph.MongoSource(
            coll,
            {'domain': domain, 'label': 0, 'rnd': {'$lt': 0.1}},
            ['lemma', 'label']
        )
        self._neg_ds = neg_ds
        neg_ds.add_field_mappers('lemma', lambda a: we.words_to_vectors(a))
        neg_ds.add_field_mappers('label', lambda a: ph.utils.one_hot(a, 2))

    def next_batch(self, size=0):
        size = size // 2
        pos_tokens, pos_labels = self._pos_ds.next_batch(size)
        neg_tokens, neg_labels = self._neg_ds.next_batch(size)
        tokens = ph.utils.pad_sequences(pos_tokens + neg_tokens)
        labels = pos_labels + neg_labels
        return tokens, labels


class TestSource(ph.MongoSource):

    def __init__(self, coll, we, domain):
        super(TestSource, self).__init__(
            coll,
            {'domain': domain},
            ['lemma', 'label']
        )
        self.add_field_mappers('lemma', lambda a: we.words_to_vectors(a))
        self.add_field_mappers('label', lambda a: ph.utils.one_hot(a, 2))
        self.add_batch_mappers('lemma', ph.utils.pad_sequences)


class DevFitter(ph.Fitter):

    def __init__(self, dev_ds, batch_size, interval):
        self._dev_ds = dev_ds
        self._batch_size = batch_size
        super(DevFitter, self).__init__(interval=interval)
        self._models = []

    def _fit(self, i, max_loop, context):
        tmp_dir = '/tmp'
        tmp_model_dir = os.path.join(tmp_dir, 'model_' + str(uuid.uuid4()))
        trainer = context[ph.CONTEXT_TRAINER]
        fn = trainer.get_slot(ph.CONTEXT_VALID)
        error = 0.0
        for _ in range(20):
            batch = self._dev_ds.next_batch(self._batch_size)
            error += fn(*batch)['Error']
        error /= (self._batch_size * 20)
        print(error)
        ph.utils.dump_model_as_tree(trainer, tmp_model_dir)
        self._models.append((tmp_model_dir, error))
        if i == max_loop:
            self._models.sort(key=lambda a: a[1])
            tmp_model_dir, error = self._models[0]
            ph.utils.load_model_from_tree(trainer, tmp_model_dir)
            for tmp_model_dir, _ in self._models:
                shutil.rmtree(tmp_model_dir)
