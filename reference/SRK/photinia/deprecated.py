#!/usr/bin/env python3

import warnings

import tensorflow as tf

from . import settings
from . import initializers
from . import widgets
from . import operations
from . import training


class Convolutional(widgets.Widget):
    """Convolutional layer.
    """

    def __init__(self,
                 name,
                 input_depth,
                 output_depth,
                 filter_height=5,
                 filter_width=5,
                 stride_height=2,
                 stride_width=2,
                 kernel_initializer=initializers.TruncatedNormal(),
                 bias_initializer=initializers.Zeros()):
        warnings.warn('Please use Conv2D instead.', DeprecationWarning)
        self._input_depth = input_depth
        self._output_depth = output_depth
        self._filter_height = filter_height
        self._filter_width = filter_width
        self._stride_height = stride_height
        self._stride_width = stride_width
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        super(Convolutional, self).__init__(name)

    @property
    def filter_width(self):
        return self._filter_width

    @property
    def filter_height(self):
        return self._filter_height

    @property
    def input_depth(self):
        return self._input_depth

    @property
    def output_depth(self):
        return self._output_depth

    @property
    def stride_width(self):
        return self._stride_width

    @property
    def stride_height(self):
        return self._stride_height

    def _build(self):
        self._w = tf.Variable(
            self._kernel_initializer.build(
                shape=(
                    self._filter_height,
                    self._filter_width,
                    self._input_depth,
                    self._output_depth
                )
            ),
            dtype=settings.D_TYPE,
            name='w'
        )
        self._b = tf.Variable(
            self._bias_initializer.build(
                shape=(self._output_depth,)
            ),
            dtype=settings.D_TYPE,
            name='b'
        )

    def _setup(self, x):
        y = tf.nn.conv2d(
            input=x,
            filter=self._w,
            strides=[1, self._stride_height, self._stride_width, 1],
            padding='SAME',
            data_format='NHWC'
        ) + self._b
        return y

    @property
    def w(self):
        return self._w

    @property
    def b(self):
        return self._b


class ConvPool(widgets.Widget):
    """Convolution-Pooling layer
    This layer consists of a convolution layer and a pooling layer.
    """

    def __init__(self,
                 name,
                 input_depth,
                 output_depth,
                 filter_height=5,
                 filter_width=5,
                 stride_height=2,
                 stride_width=2,
                 pool_type='max',
                 kernel_initializer=initializers.TruncatedNormal(),
                 bias_initializer=initializers.Zeros()):
        """Construct a convolutional pooling layer.

        :param name: Name.
        :param input_depth: Input depth (channel).
        :param output_depth: Output depth (channel, number of feature map).
        :param filter_height: Filter height (rows).
        :param filter_width: Filter width (columns).
        :param stride_height: Pooling height (sub-sampling rows).
        :param stride_width: Pooling width (sub-sampling columns).
        :param pool_type: Pooling (sub-sampling) type. Must be one of "max" or "avg".
        """
        warnings.warn('Please use Conv2D instead.', DeprecationWarning)
        self._input_depth = input_depth
        self._output_depth = output_depth
        self._filter_height = filter_height
        self._filter_width = filter_width
        self._pool_height = stride_height
        self._pool_width = stride_width
        if pool_type not in {'max', 'avg'}:
            raise ValueError('Pool type must be one of "max" or "avg".')
        self._pool_type = pool_type
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        super(ConvPool, self).__init__(name)

    @property
    def filter_width(self):
        return self._filter_width

    @property
    def filter_height(self):
        return self._filter_height

    @property
    def input_depth(self):
        return self._input_depth

    @property
    def output_depth(self):
        return self._output_depth

    @property
    def pool_width(self):
        return self._pool_width

    @property
    def pool_height(self):
        return self._pool_height

    def _build(self):
        """Build the layer.
        Two parameters: filter (weight) and bias.

        :return: None.
        """
        self._w = tf.Variable(
            self._kernel_initializer.build(
                shape=(
                    self._filter_height,
                    self._filter_width,
                    self._input_depth,
                    self._output_depth
                )
            ),
            dtype=settings.D_TYPE,
            name='w'
        )
        self._b = tf.Variable(
            self._bias_initializer.build(
                shape=(self._output_depth,)
            ),
            dtype=settings.D_TYPE,
            name='b'
        )

    def _setup(self, x):
        """Setup the layer.

        :param x: Input tensor with "NHWC" format.
        :return: Output tensor with "NHWC" format.
        """
        y = tf.nn.conv2d(
            input=x,
            filter=self._w,
            strides=[1, 1, 1, 1],
            padding='SAME',
            data_format='NHWC'
        ) + self._b
        if self._pool_type == 'max':
            y = tf.nn.max_pool(
                value=y,
                ksize=[1, self._pool_height, self._pool_width, 1],
                strides=[1, self._pool_height, self._pool_width, 1],
                padding='SAME',
                data_format='NHWC'
            )
        elif self._pool_type == 'avg':
            tf.nn.avg_pool(
                value=y,
                ksize=[1, self._pool_height, self._pool_width, 1],
                strides=[1, self._pool_height, self._pool_width, 1],
                padding='SAME',
                data_format='NHWC'
            )
        return y

    @property
    def w(self):
        return self._w

    @property
    def b(self):
        return self._b


class CNN(widgets.Widget):
    """Convolution-Pooling layers
    Stacked Convolution-Pooling layers.
    """

    def __init__(self,
                 name,
                 input_height,
                 input_width,
                 input_depth,
                 layer_shapes,
                 activation=operations.lrelu,
                 kernel_initializer=initializers.LecunNormal(),
                 bias_initializer=initializers.Zeros(),
                 with_batch_norm=True,
                 flat_output=True,
                 layer_type=Convolutional):
        """
        Each layer is described as a tuple:
        (filter_height, filter_width,
         output_depth,
         pool_height, pool_width)
        """
        self._input_height = input_height
        self._input_width = input_width
        self._input_depth = input_depth
        assert isinstance(layer_shapes, (tuple, list))
        self._layer_shapes = layer_shapes.copy()
        self._activation = activation
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        self._with_batch_norm = with_batch_norm
        self.flat_output = flat_output
        self._layer_type = layer_type
        #
        self._layers = []
        #
        # The constructor doesn't do any build operations.
        # It just need to compute the output info.
        last_height, last_width, last_depth = input_height, input_width, input_depth
        for layer_shape in layer_shapes:
            (filter_height, filter_width,
             output_depth,
             stride_height, stride_width) = layer_shape
            last_height = -(-last_height // stride_height)
            last_width = -(-last_width // stride_width)
            last_depth = output_depth
        self._output_height = last_height
        self._output_width = last_width
        self._output_depth = last_depth
        self._flat_size = self._output_height * self._output_width * self._output_depth
        super(CNN, self).__init__(name)

    def _build(self):
        last_depth = self._input_depth
        layer_type = self._layer_type
        for index, layer_shape in enumerate(self._layer_shapes):
            #
            # Get layer parameters.
            (filter_height, filter_width,
             output_depth,
             stride_height, stride_width) = layer_shape
            #
            # Create layer.
            layer = layer_type(
                name='C{}'.format(index),
                input_depth=last_depth,
                output_depth=output_depth,
                filter_height=filter_height,
                filter_width=filter_width,
                stride_height=stride_height,
                stride_width=stride_width,
                kernel_initializer=self._kernel_initializer,
                bias_initializer=self._bias_initializer
            )
            if self._with_batch_norm:
                bn_layer = widgets.BatchNorm(
                    name='BN{}'.format(index),
                    size=output_depth
                )
                layer = (layer, bn_layer)
            self._layers.append(layer)
            #
            # Update output.
            last_depth = output_depth

    def _setup(self, x):
        y = x
        for layer in self._layers:
            if isinstance(layer, tuple):
                y = layer[0].setup(y)
                y = layer[1].setup(y)
            else:
                y = layer.setup(y)
            y = self._activation(y) if self._activation is not None else y
        if self.flat_output:
            y = tf.reshape(y, (-1, self._flat_size))
        return y

    @property
    def input_height(self):
        return self._input_height

    @property
    def input_width(self):
        return self._input_width

    @property
    def input_depth(self):
        return self._input_depth

    @property
    def output_height(self):
        return self._output_height

    @property
    def output_width(self):
        return self._output_width

    @property
    def output_depth(self):
        return self._output_depth

    @property
    def flat_size(self):
        return self._flat_size


class ConvTrans(widgets.Widget):
    """ConvTransposeLayer
    """

    def __init__(self,
                 name,
                 input_depth,
                 output_depth,
                 filter_height=5,
                 filter_width=5,
                 stride_height=2,
                 stride_width=2,
                 kernel_initializer=initializers.TruncatedNormal(),
                 bias_initializer=initializers.Zeros()):
        """Construct a convolutional transpose layer.

        :param name: Name.
        :param input_depth: Input depth (channel).
        :param output_depth: Output depth (channel, number of feature map).
        :param filter_height: Filter height (rows).
        :param filter_width: Filter width (columns).
        :param stride_height: Stride height (up-sampling rows).
        :param stride_width: Stride width (up-sampling columns).
        """
        self._input_depth = input_depth
        self._output_depth = output_depth
        self._filter_height = filter_height
        self._filter_width = filter_width
        self._stride_height = stride_height
        self._stride_width = stride_width
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        super(ConvTrans, self).__init__(name)

    @property
    def filter_width(self):
        return self._filter_width

    @property
    def filter_height(self):
        return self._filter_height

    @property
    def input_depth(self):
        return self._input_depth

    @property
    def output_depth(self):
        return self._output_depth

    def _build(self):
        """Build the layer.
        Two parameters: filter (weight) and bias.

        :return: None.
        """
        self._w = tf.Variable(
            self._kernel_initializer.build(
                shape=(
                    self._filter_height,
                    self._filter_width,
                    self._output_depth,
                    self._input_depth
                )
            ),
            dtype=settings.D_TYPE,
            name='w'
        )
        self._b = tf.Variable(
            self._bias_initializer.build(
                shape=(self._output_depth,)
            ),
            dtype=settings.D_TYPE,
            name='b'
        )

    def _setup(self, x):
        """Setup the layer.

        :param x: Input tensor with "NHWC" format.
        :return: Output tensor with "NHWC" format.
        """
        input_shape = tf.shape(x)
        batch_size, input_height, input_width = input_shape[0], input_shape[1], input_shape[2]
        output_shape = (
            batch_size,
            input_height * self._stride_height,
            input_width * self._stride_width,
            self._output_depth
        )
        y = tf.nn.conv2d_transpose(
            value=x,
            filter=self._w,
            output_shape=output_shape,
            strides=[1, self._stride_height, self._stride_width, 1],
            padding='SAME',
            data_format='NHWC'
        ) + self._b
        return y

    @property
    def w(self):
        return self._w

    @property
    def b(self):
        return self._b


class TransCNN(widgets.Widget):
    """Convolution transpose layers
    Stacked Convolution transpose layers.
    """

    def __init__(self,
                 name,
                 init_height,
                 init_width,
                 init_depth,
                 layer_shapes,
                 activation=operations.lrelu,
                 kernel_initializer=initializers.LecunNormal(),
                 bias_initializer=initializers.Zeros(),
                 with_batch_norm=True):
        """
        Each layer is described as a tuple:
        (filter_height, filter_width,
         output_depth,
         stride_height, stride_height)
        """
        self._init_height = init_height
        self._init_width = init_width
        self._init_depth = init_depth
        assert isinstance(layer_shapes, (tuple, list))
        self._layer_shapes = layer_shapes.copy()
        self._activation = activation
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        self._with_batch_norm = with_batch_norm
        #
        self._layers = []
        #
        # The constructor doesn't do any build operations.
        # It just need to compute the output info.
        self._init_flat_size = self._init_height * self._init_width * self._init_depth
        super(TransCNN, self).__init__(name)

    def _build(self):
        last_depth = self._init_depth
        for index, layer_shape in enumerate(self._layer_shapes):
            #
            # Get layer parameters.
            (filter_height, filter_width,
             output_depth,
             stride_height, stride_width) = layer_shape
            #
            # Create layer.
            layer = ConvTrans(
                name='CT{}'.format(index),
                input_depth=last_depth,
                output_depth=output_depth,
                filter_height=filter_height,
                filter_width=filter_width,
                stride_height=stride_height,
                stride_width=stride_width,
                kernel_initializer=self._kernel_initializer,
                bias_initializer=self._bias_initializer
            )
            if self._with_batch_norm and index != len(self._layer_shapes) - 1:
                bn_layer = widgets.BatchNorm(
                    name='BN{}'.format(index),
                    size=output_depth
                )
                layer = (layer, bn_layer)
            self._layers.append(layer)
            #
            # Update output.
            last_depth = output_depth

    def _setup(self, x):
        maps = tf.reshape(x, (-1, self._init_height, self._init_width, self._init_depth))
        for index, layer in enumerate(self._layers):
            if isinstance(layer, tuple):
                maps = layer[0].setup(maps)
                maps = layer[1].setup(maps)
            else:
                maps = layer.setup(maps)
            if self._activation is not None and index != len(self._layer_shapes) - 1:
                maps = self._activation(maps)
        return maps

    @property
    def init_height(self):
        return self._init_height

    @property
    def init_width(self):
        return self._init_width

    @property
    def init_depth(self):
        return self._init_depth

    @property
    def init_flat_size(self):
        return self._init_flat_size


class Trainable(training.Trainer):

    def __init__(self, name, session=None, build=True):
        warnings.warn('Trainable will be deleted in the future. Please use Trainer instead.', DeprecationWarning)
        super(Trainable, self).__init__(name, session, build)

    def _build(self):
        NotImplementedError()
