#!/usr/bin/env python3

import tensorflow as tf

D_TYPE = tf.float32

NAME_TRAIN_SLOT = 'train'
NAME_VALID_SLOT = 'validate'
NAME_PREDICT_SLOT = 'predict'

CONTEXT_TRAINER = 'trainable'
CONTEXT_LOOP = 'loop'
CONTEXT_MAX_LOOP = 'max_loop'
CONTEXT_TRAIN = NAME_TRAIN_SLOT
CONTEXT_VALID = NAME_VALID_SLOT
CONTEXT_PREDICT = NAME_PREDICT_SLOT


class __GlobalContext(object):

    def __init__(self):
        self._session_config = tf.ConfigProto()
        self._session_config.gpu_options.allow_growth = True
        self._session = None

    @property
    def session_config(self):
        return self._session_config

    @property
    def session(self):
        if self._session is None:
            self._session = tf.Session(config=self._session_config)
        return self._session


__GLOBAL = __GlobalContext()


def get_session_config():
    return __GLOBAL.session_config


def get_session():
    return __GLOBAL.session


def initialize_global_variables():
    __GLOBAL.session.run(tf.global_variables_initializer())
