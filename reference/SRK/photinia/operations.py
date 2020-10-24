#!/usr/bin/env python3

"""
@author: xi
@since: 2017-03
"""

import tensorflow as tf

from . import settings


def log(x, eps=1e-7, name=None):
    return tf.log(x + eps, name=name)


def lrelu(x,
          leak=1e-3,
          name=None):
    """Leaky ReLU activation function.

    f(x) =        x     , x >= 0,
           leak * x     , x < 0

    :param x: Input tensor.
    :param leak: Leak. Default is 1e-3.
    :param name: Operation name.
    :return: Activated tensor.
    """
    return tf.maximum(x, leak * x, name=name)


def swish(x,
          name=None):
    """Swish activation function.

    f(x) = x * sigmoid(x)

    :param x: Input tensor.
    :param name: Operation name.
    :return: Activated tensor.
    """
    return tf.multiply(tf.nn.sigmoid(x), x, name=name)


def random_gumbel(shape,
                  mu=0.0,
                  beta=1.0,
                  dtype=settings.D_TYPE,
                  seed=None,
                  name=None):
    """Outputs random values from a Gumbel distribution.
    
    :param shape: Output shape.
    :param mu: mu.
    :param beta: beta.
    :param dtype: Data type.
    :param seed: Random seed.
    :param name: Operation name.
    :return: A tensor of the specified shape filled with random Gumbel values.
    """
    u = tf.random_uniform(
        shape=shape,
        minval=0,
        maxval=1,
        dtype=dtype,
        seed=seed,
        name=name
    )
    g = -tf.log(-tf.log(u))
    g = mu + g * beta
    return g


def kl_normal(mu0, var0,
              mu1=0.0, var1=1.0,
              name=None):
    """KL divergence for normal distribution.
    Note that this is a simple version. We don't use covariance matrix (∑) here. Instead, 
    var is the vector that indicates the elements in ∑'s main diagonal (diag(∑)).

    :param mu0: μ0.
    :param var0: diag(∑0).
    :param mu1: μ1.
    :param var1: diag(∑1).
    :param name: Operation name.
    :return: The KL divergence.
    """
    e = 1e-4
    var0 += e
    if mu1 == 0.0 and var1 == 1.0:
        kl = var0 + mu0 ** 2 - 1 - tf.log(var0)
    else:
        var1 += e
        kl = var0 / var1 + (mu0 - mu1) ** 2 / var1 - 1 - tf.log(var0 / var1)
    kl = tf.multiply(0.5, tf.reduce_sum(kl, 1), name=name)
    return kl


def clip_gradient(pair_list,
                  max_norm):
    """Perform gradient clipping.
    If the gradients' global norm exceed 'max_norm', then shrink it to 'max_norm'.
    
    :param pair_list: (grad, var) pair list.
    :param max_norm: The max global norm.
    :return: (grad, var) pair list, the original gradients' norm, the clipped gradients' norm
    """
    grad_list = [grad for grad, _ in pair_list]
    grad_list, raw_grad = tf.clip_by_global_norm(grad_list, max_norm)
    grad = tf.global_norm(grad_list)
    pair_list = [(grad, pair[1]) for grad, pair in zip(grad_list, pair_list)]
    return pair_list, raw_grad, grad


def setup(x,
          widget_list):
    """Setup a series of widgets/ops with the given input "x".

    :param x: The input tensor.
    :param widget_list: List of widgets/ops.
    :return: The output form the last widget/op.
    """
    if widget_list is None:
        return x
    if not isinstance(widget_list, (list, tuple)):
        widget_list = [widget_list]
    y = x
    for w in widget_list:
        if callable(w):
            #
            # Note that Widget is also callable.
            y = w(y)
        elif isinstance(w, (tuple, list)):
            if len(w) != 2:
                raise ValueError('The tuple must have two elements.')
            fn = w[0]
            if not callable(fn):
                raise ValueError('%s is not callable.' % str(fn))
            if isinstance(w[1], dict):
                kwargs = w[1]
                y = fn(y, **kwargs)
            elif isinstance(w[1], str):
                y = fn(y, name=w[1])
            else:
                raise ValueError('The second term of the tuple must be str or dict.')
        elif w is None:
            continue
        else:
            raise ValueError('%s is not callable.' % str(w))
    return y


def transpose_sequence(seq,
                       seq_axis=1,
                       name=None):
    """Transpose a batch of sequence, i.e., exchange the batch axis and the sequence axis.
    By default, the sequence axis is 1.

    :param seq: Tensor shaped (batch_size, seq_length, ...).
    :param seq_axis: The sequence axis. Default is 1.
    :param name: Operation anme.
    :return: Tensor shaped (seq_length, batch_size, ...).
    """
    perm = [i for i in range(len(seq.shape))]
    perm[0], perm[seq_axis] = seq_axis, 0
    return tf.transpose(seq, perm, name=name)


def setup_sequence(seq, widget_list):
    """Setup a series of widgets/ops with the given sequence "seq".

    :param seq: Tensor represents a sequence.
    :param widget_list: List of widgets/ops.
    :return: The output sequence.
    """
    seq = transpose_sequence(seq)
    y = tf.map_fn(
        fn=lambda elem: setup(elem, widget_list),
        elems=seq
    )
    y = transpose_sequence(y)
    return y


def flatten(x):
    batch_size = tf.shape(x)[0]
    return tf.reshape(x, (batch_size, -1))


def sequence_length(seq):
    used = tf.sign(tf.reduce_max(tf.abs(seq), 2))
    length = tf.reduce_sum(used, 1)
    length = tf.cast(length, tf.int32)
    return length


def last_elements(seq, seq_len):
    h, _ = tf.map_fn(
        fn=lambda elem: (elem[0][elem[1] - 1], elem[1]),
        elems=(seq, seq_len)
    )
    return h


def variance(x, axis=-1):
    mu = tf.reduce_mean(x, axis=axis)
    return tf.reduce_mean(x ** 2) - mu ** 2


def skewness(x, axis=-1, epsilon=1e-5):
    mu = tf.reduce_mean(x, axis=axis, keep_dims=True)
    up = tf.reduce_mean((x - mu) ** 3, axis=axis)
    down = tf.reduce_mean((x - mu) ** 2, axis=axis)
    down = tf.sqrt(down) ** 3 + epsilon
    return up / down
