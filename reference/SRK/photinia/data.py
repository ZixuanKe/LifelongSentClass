#!/usr/bin/env python3

"""
@author: xi
@since: 2017-12-24
"""

import collections
import queue
import random
import threading

import numpy as np
import pymongo


class DataSource(object):
    """DataSource
    """

    def next_batch(self, size=0):
        """Get a batch of data.

        :param size: Batch size. Default is zero, which means extract all data.
        :return: Tuple of np.array.
        """
        raise NotImplementedError()


class Dataset(DataSource):
    """Dataset
    """

    def __init__(self,
                 *data,
                 dtype=None):
        """Construct a dataset.

        :param data: Tuple of list, np.array or any iterable objects.
        :param dtype: Data type.
        """
        self._num_comp = len(data)
        if self._num_comp == 0:
            raise ValueError('At least 1 data object should be given.')
        self._data = [np.array(mat, dtype=dtype) for mat in data]
        size = None
        for mat in self._data:
            if size is None:
                size = len(mat)
                continue
            if len(mat) != size:
                raise ValueError('All data components must have the same size.')
        self._size = size
        self._start = 0
        self._loop = 0

    @property
    def size(self):
        return self._size

    @property
    def start(self):
        return self._start

    @property
    def loop(self):
        return self._loop

    def next_batch(self, size=0):
        batch = self._next_batch(size)
        if size == 0:
            return batch
        real_size = len(batch[0])
        while real_size < size:
            batch1 = self._next_batch(size - real_size)
            batch = tuple(np.concatenate((batch[i], batch1[i]), 0) for i in range(self._num_comp))
            real_size = len(batch[0])
        return batch

    def _next_batch(self, size=0):
        if size <= 0:
            return self.all()
        if self._start == 0 and self._loop != 0:
            self.shuffle()
        end = self._start + size
        if end < self._size:
            batch = tuple(self._data[i][self._start:end].copy() for i in range(self._num_comp))
            self._start += size
        else:
            batch = tuple(self._data[i][self._start:].copy() for i in range(self._num_comp))
            self._start = 0
            self._loop += 1
        return batch

    def shuffle(self, num=3):
        perm = np.arange(self._size)
        for _ in range(num):
            np.random.shuffle(perm)
        for i in range(self._num_comp):
            self._data[i] = self._data[i][perm]
        return self

    def all(self):
        return self._data


class MongoSource(DataSource):
    """MongoDB data source
    """

    def __init__(self,
                 coll,
                 match=None,
                 fields=(),
                 buffer_size=10000):
        """Construct from a mongodb collection instance.

        :param coll: pymongo.collection.Collection, mongodb collection instance.
        :param match: dict, e.g., {'domain': 'AlarmClock', 'rnd': {'$lt': 200}}.
        :param fields: list, e.g., ['tokens', 'label'].
        :param buffer_size: Positive integer. Default is 10000.
        """
        super(MongoSource, self).__init__()
        #
        # MongoDB Collection
        if isinstance(coll, pymongo.collection.Collection):
            self._coll = coll
        else:
            raise ValueError(
                'Argument coll should be an object of '
                'pymongo.collection.Collection.'
            )
        #
        # Match and Project
        self._match = match if match is not None else {}
        self._fields = fields if fields is not None else ()
        self._project = {field: 1 for field in fields}
        #
        # Buffer Size
        if isinstance(buffer_size, int) and buffer_size > 0:
            self._buffer_size = buffer_size
        else:
            raise ValueError('Argument buffer_size should be a positive integer.')
        self._buffer_size = buffer_size if buffer_size > 0 else 10000
        #
        # Converters
        self._field_converters = collections.defaultdict(collections.deque)
        self._batch_converters = collections.defaultdict(collections.deque)
        #
        # Async Loading
        self._main_thread = threading.current_thread()
        self._queue = queue.Queue(self._buffer_size)
        self._thread = None
        #
        # One Pass Loading
        self._one_pass_buffer = None
        self._start = 0

    def set_match(self, match):
        self._match = match

    def set_fields(self, fields):
        self._fields = fields
        self._project = {field: 1 for field in fields}

    def add_field_mappers(self, field, fns):
        if callable(fns):
            fns = [fns]
        elif not isinstance(fns, (list, tuple)):
            raise ValueError('fns should be callable or list(tuple) of callables.')
        self._field_converters[field] += fns

    def add_batch_mappers(self, field, fns):
        if callable(fns):
            fns = [fns]
        elif not isinstance(fns, (list, tuple)):
            raise ValueError('fns should be callable or list(tuple) of callables.')
        self._batch_converters[field] += fns

    def next_batch(self, size=0):
        if size > 0:
            batch = tuple([] for _ in self._fields)
            for _ in range(size):
                if self._queue.qsize() < self._buffer_size / 3 \
                        and (self._thread is None or not self._thread.is_alive()):
                    self._thread = threading.Thread(target=self._load)
                    self._thread.start()
                doc = self._queue.get()
                if isinstance(doc, Exception):
                    raise doc
                for i, value in enumerate(doc):
                    batch[i].append(value)
        else:
            batch = self._get_one_pass_buffer()
        batch = tuple(self.__apply_batch_converters(field, column) for field, column in zip(self._fields, batch))
        return batch

    def next_batch_one_pass(self, size):
        buffer = self._get_one_pass_buffer()
        buffer_size = len(buffer[0])
        if self._start >= buffer_size:
            self._start = 0
            return None
        end = self._start + size
        batch = tuple(
            self.__apply_batch_converters(
                field,
                column[self._start: end] if end <= buffer_size else column[self._start:]
            )
            for field, column in zip(self._fields, buffer)
        )
        self._start = end
        return batch

    def __apply_batch_converters(self, field, batch_column):
        if field in self._batch_converters:
            for fn in self._batch_converters[field]:
                batch_column = fn(batch_column)
        return batch_column

    def _get_one_pass_buffer(self):
        if self._one_pass_buffer is None:
            batch = tuple([] for _ in self._fields)
            cur = self._coll.find(self._match, self._project, cursor_type=pymongo.CursorType.EXHAUST)
            for doc in cur:
                doc = tuple(self._get_value(doc, field) for field in self._fields)
                for i, value in enumerate(doc):
                    batch[i].append(value)
            self._one_pass_buffer = batch
        return self._one_pass_buffer

    def _load(self):
        """This method is executed in another thread!
        """
        try:
            count = self._coll.count()
            if count < 2 * self._buffer_size:
                cur = self._coll.aggregate([
                    {'$match': self._match},
                    {'$project': self._project},
                    {'$sample': {'size': self._buffer_size}}
                ])
            else:
                cur = self._coll.find(
                    self._match,
                    self._project
                )
            try:
                for doc in cur:
                    doc = tuple(self._get_value(doc, field) for field in self._fields)
                    if random.uniform(0.0, 1.0) < 0.1:
                        continue
                    self._queue.put(doc)
                    if not self._main_thread.is_alive():
                        break
            except:
                pass
        except Exception as e:
            self._queue.put(e)

    def _get_value(self, doc, field):
        value = doc[field]
        if field in self._field_converters:
            for fn in self._field_converters[field]:
                value = fn(value)
        return value
