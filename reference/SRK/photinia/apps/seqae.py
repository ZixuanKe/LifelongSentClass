#!/usr/bin/env python3

"""
@author: xi
@since: 2018-03-19
"""

import tensorflow as tf

import photinia as ph


class Encoder(ph.Widget):

    def __init__(self,
                 name,
                 voc_size,
                 emb_size,
                 state_size):
        self._voc_size = voc_size
        self._emb_size = emb_size
        self._state_size = state_size
        super(Encoder, self).__init__(name)

    @property
    def voc_size(self):
        return self._voc_size

    @property
    def emb_size(self):
        return self._emb_size

    @property
    def state_size(self):
        return self._state_size

    def _build(self):
        self._emb_layer = ph.Linear('emb_layer', self._voc_size, self._emb_size)
        self._cell = ph.GRUCell('cell', self._emb_size, self._state_size)

    @property
    def emb_layer(self):
        return self._emb_layer

    def _setup(self, seq, activation=ph.lrelu):
        seq_len = ph.sequence_length(seq)
        states = self._cell.setup_sequence(seq, [self._emb_layer, activation])
        y = ph.last_elements(states, seq_len)
        return y


class Decoder(ph.Widget):

    def __init__(self,
                 name,
                 voc_size,
                 emb_size,
                 state_size,
                 emb_layer=None):
        self._voc_size = voc_size
        self._emb_size = emb_size
        self._state_size = state_size
        self._emb_layer = emb_layer
        super(Decoder, self).__init__(name)

    @property
    def voc_size(self):
        return self._voc_size

    @property
    def emb_size(self):
        return self._emb_size

    @property
    def state_size(self):
        return self._state_size

    @property
    def emb_layer(self):
        return self._emb_layer

    def _build(self):
        if self._emb_layer is None:
            self._emb_layer = ph.Linear('emb_layer', self._voc_size, self._emb_size)
        else:
            self._emb_size = self._emb_layer.emb_size
        self._cell = ph.GRUCell('cell', self._emb_size, self._state_size)
        self._out_layer = ph.Linear('out_layer', self._state_size, self._voc_size)

    def _setup(self, h, max_len, activation=ph.lrelu):
        _, outputs = self._cell.setup_recursive(
            max_len,
            input_widgets=[self._emb_layer, activation],
            output_widgets=[self._out_layer, activation, tf.nn.softmax],
            init_state=h
        )
        return outputs


class Trainer(ph.Trainer):

    def __init__(self,
                 name,
                 voc_size,
                 emb_size,
                 state_size,
                 optimizer=tf.train.RMSPropOptimizer(1e-4, 0.9, 0.9),
                 reg=1e-6):
        self._voc_size = voc_size
        self._emb_size = emb_size
        self._state_size = state_size
        self._optimizer = optimizer
        self._reg = reg
        super(Trainer, self).__init__(name)

    @property
    def voc_size(self):
        return self._voc_size

    @property
    def emb_size(self):
        return self._emb_size

    @property
    def state_size(self):
        return self._state_size

    def _build(self):
        encoder = Encoder(
            'seq_encoder',
            self._voc_size,
            self._emb_size,
            self._state_size
        )
        decoder = Decoder(
            'seq_decoder',
            self._voc_size,
            self._emb_size,
            self._state_size,
            self._encoder.emb_layer
        )
        self._encoder = encoder
        self._decoder = decoder

        seq = ph.placeholder('seq', (None, None, self._voc_size))
        max_len = tf.shape(seq)[1]
        h = encoder.setup(seq)
        seq_ = decoder.setup(h, max_len)
        self._seq = seq
        self._h = h
        self._seq_ = seq_

        loss = -(seq * ph.log(1 - seq_) + (1 - seq) * ph.log(seq_))
        loss = tf.reduce_mean(loss)
        self._loss = loss

        reg = ph.Regularizer()
        reg.add_l1_l2(self.get_trainable_variables())

        update = self._optimizer.minimize(loss + reg.get_loss(self._reg) if self._reg > 0 else loss)
        self._add_slot(
            ph.TRAIN,
            inputs=seq,
            outputs={'seq_': seq_, 'loss': loss},
            updates=update
        )
        self._add_slot(
            ph.PREDICT,
            inputs=seq,
            outputs={'h': h, 'seq_': seq_, 'loss': loss}
        )

    @property
    def encoder(self):
        return self._encoder

    @property
    def decoder(self):
        return self._decoder

    @property
    def seq(self):
        return self._seq

    @property
    def h(self):
        return self._h

    @property
    def seq_(self):
        return self._seq_

    @property
    def loss(self):
        return self._loss
