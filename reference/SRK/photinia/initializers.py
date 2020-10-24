#!/usr/bin/env python3

"""
@author: winton, xi
@since: 2017-11-06
"""

import numpy as np
import tensorflow as tf

from . import settings


class Initializer(object):
    """Initializer base class: all initializers inherit from this class.
    """

    def _build(self, shape, name, seed):
        raise NotImplementedError()

    def build(self, shape, name=None, seed=None):
        if name is None:
            return self._build(shape, name, seed)
        else:
            if not isinstance(name, str):
                raise ValueError('Name of initializer must be specified with string.')
            if len(name.strip()) != len(name) or name == '':
                raise ValueError('Name of initializer cannot be empty or contain space characters.')
            return self._build(shape, name, seed)


class Zeros(Initializer):
    """Initializer that generates tensors initialized to 0.
    """

    def _build(self, shape, name, seed):
        return tf.zeros(shape, dtype=settings.D_TYPE, name=name)


class Ones(Initializer):
    """Initializer that generates tensors initialized to 1.
    """

    def _build(self, shape, name, seed):
        return tf.ones(shape, dtype=settings.D_TYPE, name=name)


class Constant(Initializer):
    """Initializer that generates tensors initialized to a constant value.
    # Arguments
        value: float; the value of the generator tensors.
    """

    def __init__(self, value=0.):
        self._value = value

    @property
    def value(self):
        return self._value

    def _build(self, shape, name, seed):
        return tf.constant(self._value, dtype=settings.D_TYPE, shape=shape, name=name)


class RandomNormal(Initializer):
    """Initializer that generates tensors with a normal distribution.
    # Arguments
        mean: a python scalar or a scalar tensor. Mean of the random values
          to generate.
        stddev: a python scalar or a scalar tensor. Standard deviation of the
          random values to generate.
        seed: A Python integer. Used to seed the random generator.
    """

    def __init__(self,
                 mean=0.,
                 stddev=0.05):
        self._mean = mean
        self._stddev = stddev

    @property
    def mean(self):
        return self._mean

    @property
    def stddev(self):
        return self._stddev

    def _build(self, shape, name, seed):
        return tf.random_normal(
            shape=shape,
            mean=self._mean,
            stddev=self._stddev,
            dtype=settings.D_TYPE,
            seed=seed,
            name=name
        )


class RandomUniform(Initializer):
    """Initializer that generates tensors with a uniform distribution.
    # Arguments
        minval: A python scalar or a scalar tensor. Lower bound of the range
          of random values to generate.
        maxval: A python scalar or a scalar tensor. Upper bound of the range
          of random values to generate.  Defaults to 1 for float types.
        seed: A Python integer. Used to seed the random generator.
    """

    def __init__(self,
                 minval=-0.05,
                 maxval=0.05):
        self._minval = minval
        self._maxval = maxval

    @property
    def minval(self):
        return self._minval

    @property
    def maxval(self):
        return self._maxval

    def _build(self, shape, name, seed):
        return tf.random_uniform(
            shape=shape,
            minval=self._minval,
            maxval=self._maxval,
            dtype=settings.D_TYPE,
            seed=seed,
            name=name
        )


class TruncatedNormal(Initializer):
    """Initializer that generates a truncated normal distribution.
    These values are similar to values from a `RandomNormal`
    except that values more than two standard deviations from the mean
    are discarded and re-drawn. This is the recommended initializer for
    neural network weights and filters.
    # Arguments
        mean: a python scalar or a scalar tensor. Mean of the random values
          to generate.
        stddev: a python scalar or a scalar tensor. Standard deviation of the
          random values to generate.
        seed: A Python integer. Used to seed the random generator.
    """

    def __init__(self,
                 mean=0.,
                 stddev=0.05):
        self._mean = mean
        self._stddev = stddev

    @property
    def mean(self):
        return self._mean

    @property
    def stddev(self):
        return self._stddev

    def _build(self, shape, name, seed):
        return tf.truncated_normal(
            shape=shape,
            mean=self._mean,
            stddev=self._stddev,
            dtype=settings.D_TYPE,
            seed=seed,
            name=name
        )


class Orthogonal(Initializer):
    """Initializer that generates a random orthogonal matrix.
    # Arguments
        gain: Multiplicative factor to apply to the orthogonal matrix.
        seed: A Python integer. Used to seed the random generator.
    # References
        Saxe et al., Exact solutions to the nonlinear dynamics of learning in deep linear neural networks,
        http://arxiv.org/abs/1312.6120
    """

    def __init__(self,
                 gain=1.):
        self._gain = gain

    @property
    def gain(self):
        return self._gain

    def _build(self, shape, name, seed):
        num_rows = 1
        for dim in shape[:-1]:
            num_rows *= dim
        num_cols = shape[-1]
        flat_shape = (num_rows, num_cols)
        if seed is not None:
            np.random.seed(seed)
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        # Pick the one with the correct shape.
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape)
        return tf.constant(
            value=self._gain * q[:shape[0], :shape[1]],
            dtype=settings.D_TYPE,
            shape=shape,
            name=name
        )


class Identity(Initializer):
    """Initializer that generates the identity matrix.
    Only use for square 2D matrices.
    # Arguments
        gain: Multiplicative factor to apply to the identity matrix.
    """

    def __init__(self,
                 gain=1.):
        self._gain = gain

    @property
    def gain(self):
        return self._gain

    def _build(self, shape, name, seed):
        if len(shape) != 2 or shape[0] != shape[1]:
            raise ValueError('Identity matrix initializer can only be used '
                             'for 2D square matrices.')
        else:
            return tf.constant(
                value=self._gain * np.identity(shape[0]),
                dtype=settings.D_TYPE,
                shape=shape,
                name=name
            )


class VarianceScaling(Initializer):
    """Initializer capable of adapting its scale to the shape of weights.
    With `distribution="normal"`, samples are drawn from a truncated normal
    distribution centered on zero, with `stddev = sqrt(scale / n)` where n is:
        - number of input units in the weight tensor, if mode = "fan_in"
        - number of output units, if mode = "fan_out"
        - average of the numbers of input and output units, if mode = "fan_avg"
    With `distribution="uniform"`,
    samples are drawn from a uniform distribution
    within [-limit, limit], with `limit = sqrt(3 * scale / n)`.
    # Arguments
        scale: Scaling factor (positive float).
        mode: One of "fan_in", "fan_out", "fan_avg".
        distribution: Random distribution to use. One of "normal", "uniform".
        seed: A Python integer. Used to seed the random generator.
    # Raises
        ValueError: In case of an invalid value for the "scale", mode" or
          "distribution" arguments.
    """

    def __init__(self,
                 scale=1.0,
                 mode='fan_in',
                 distribution='normal'):
        if scale <= 0.:
            raise ValueError('`scale` must be a positive float.')
        mode = mode.lower()
        if mode not in {'fan_in', 'fan_out', 'fan_avg'}:
            raise ValueError('Invalid `mode` argument: '
                             'expected on of {"fan_in", "fan_out", "fan_avg"} '
                             'but got', mode)
        distribution = distribution.lower()
        if distribution not in {'normal', 'uniform'}:
            raise ValueError('Invalid `distribution` argument: '
                             'expected one of {"normal", "uniform"} '
                             'but got', distribution)
        self._scale = scale
        self._mode = mode
        self._distribution = distribution

    @property
    def scale(self):
        return self._scale

    @property
    def mode(self):
        return self._mode

    @property
    def distribution(self):
        return self._distribution

    def _build(self, shape, name, seed):
        fan_in, fan_out = self._compute_fans(shape)
        scale = self._scale
        if self._mode == 'fan_in':
            scale /= max(1., fan_in)
        elif self._mode == 'fan_out':
            scale /= max(1., fan_out)
        else:
            scale /= max(1., float(fan_in + fan_out) / 2)
        if self._distribution == 'normal':
            stddev = np.sqrt(scale)
            return tf.truncated_normal(
                shape=shape,
                mean=0.,
                stddev=stddev,
                dtype=settings.D_TYPE,
                seed=seed,
                name=name
            )
        else:
            limit = np.sqrt(3. * scale)
            return tf.random_uniform(
                shape=shape,
                minval=-limit,
                maxval=limit,
                dtype=settings.D_TYPE,
                seed=seed,
                name=name
            )

    @staticmethod
    def _compute_fans(shape):
        """Computes the number of input and output units for a weight shape.
        # Arguments
            shape: Integer shape tuple.
        # Returns
            A tuple of scalars, `(fan_in, fan_out)`.
        # Raises
            ValueError: in case of invalid shape size.
        """
        if len(shape) >= 2:
            fan_in = shape[-2]
            fan_out = shape[-1]
        else:
            raise ValueError('Invalid shape size.')
        return fan_in, fan_out


class LecunNormal(VarianceScaling):
    """LeCun normal initializer.
    It draws samples from a truncated normal distribution centered on 0
    with `stddev = sqrt(1 / fan_in)`
    where `fan_in` is the number of input units in the weight tensor.
    # Arguments
        seed: A Python integer. Used to seed the random generator.
    # Returns
        An initializer.
    # References
        - [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
        - [Efficient Backprop](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)
    """

    def __init__(self,
                 scale=1.,
                 mode='fan_in',
                 distribution='normal'):
        super(LecunNormal, self).__init__(scale, mode, distribution)


class LecunUniform(VarianceScaling):
    """LeCun uniform initializer.
    It draws samples from a uniform distribution within [-limit, limit]
    where `limit` is `sqrt(3 / fan_in)`
    where `fan_in` is the number of input units in the weight tensor.
    # Arguments
        seed: A Python integer. Used to seed the random generator.
    # Returns
        An initializer.
    # References
        LeCun 98, Efficient Backprop,
        http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
    """

    def __init__(self,
                 scale=1.,
                 mode='fan_in',
                 distribution='uniform'):
        super(LecunUniform, self).__init__(scale, mode, distribution)


class GlorotNormal(VarianceScaling):
    """Glorot normal initializer, also called Xavier normal initializer.
    It draws samples from a truncated normal distribution centered on 0
    with `stddev = sqrt(2 / (fan_in + fan_out))`
    where `fan_in` is the number of input units in the weight tensor
    and `fan_out` is the number of output units in the weight tensor.
    # Arguments
        seed: A Python integer. Used to seed the random generator.
    # Returns
        An initializer.
    # References
        Glorot & Bengio, AISTATS 2010
        http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
    """

    def __init__(self,
                 scale=1.,
                 mode='fan_avg',
                 distribution='normal'):
        super(GlorotNormal, self).__init__(scale, mode, distribution)


class GlorotUniform(VarianceScaling):
    """Glorot uniform initializer, also called Xavier uniform initializer.
    It draws samples from a uniform distribution within [-limit, limit]
    where `limit` is `sqrt(6 / (fan_in + fan_out))`
    where `fan_in` is the number of input units in the weight tensor
    and `fan_out` is the number of output units in the weight tensor.
    # Arguments
        seed: A Python integer. Used to seed the random generator.
    # Returns
        An initializer.
    # References
        Glorot & Bengio, AISTATS 2010
        http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
    """

    def __init__(self,
                 scale=1.,
                 mode='fan_avg',
                 distribution='uniform'):
        super(GlorotUniform, self).__init__(scale, mode, distribution)


class HeNormal(VarianceScaling):
    """He normal initializer.
    It draws samples from a truncated normal distribution centered on 0
    with `stddev = sqrt(2 / fan_in)`
    where `fan_in` is the number of input units in the weight tensor.
    # Arguments
        seed: A Python integer. Used to seed the random generator.
    # Returns
        An initializer.
    # References
        He et al., http://arxiv.org/abs/1502.01852
    """

    def __init__(self,
                 scale=2.,
                 mode='fan_in',
                 distribution='normal'):
        super(HeNormal, self).__init__(scale, mode, distribution)


class HeUniform(VarianceScaling):
    """He uniform variance scaling initializer.
    It draws samples from a uniform distribution within [-limit, limit]
    where `limit` is `sqrt(6 / fan_in)`
    where `fan_in` is the number of input units in the weight tensor.
    # Arguments
        seed: A Python integer. Used to seed the random generator.
    # Returns
        An initializer.
    # References
        He et al., http://arxiv.org/abs/1502.01852
    """

    def __init__(self,
                 scale=2.,
                 mode='fan_in',
                 distribution='uniform'):
        super(HeUniform, self).__init__(scale, mode, distribution)
