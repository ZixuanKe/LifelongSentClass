#!/usr/bin/env python3

"""
@author: xi
@since: 2017-12-25
"""

import os
import random

import numpy as np
import scipy.ndimage as ndi
from PIL import Image

import photinia as ph


def load_as_array(fn_or_fp, height, width):
    """Load an image from file and convert it into array.
    The data type of the array is np.uint8.

    :param fn_or_fp: File name or file object.
    :param height: Height.
    :param width: Width.
    :return: An array represents the image.
    """
    image = Image.open(fn_or_fp)
    image = image.resize((width, height), Image.LANCZOS)
    return np.asarray(image, dtype=np.uint8)


def save_array(fn_or_fp, array):
    """Save the array into file.
    The image format is specified be the suffix of the file name.

    :param fn_or_fp: FIle name or file object.
    :param array: The array.
    :return: None.
    """
    image = Image.fromarray(array)
    image.save(fn_or_fp)


def array_to_mat(array, height=None, width=None):
    """Convert an image array into a matrix.
    The data type of the matrix is np.float32.
    Elements in the matrix are valued in range -1 ~ 1.

    :param array: The array.
    :param height: Height.
    :param width: Width.
    :return: The matrix.
    """
    if height is not None and width is not None:
        image = Image.fromarray(array)
        image = image.resize((width, height), Image.LANCZOS)
        return (np.asarray(image, dtype=np.float32) - 128.0) / 130.0
    return (array.astype(np.float32) - 128.0) / 130.0


def mat_to_array(mat):
    """Convert a matrix into an array.
    Elements in the matrix must be valued in range -1 ~ 1.
    The data type of the array is np.uint8.

    :param mat: The matrix.
    :return: The array.
    """
    return (mat * 128.0 + 127.75).astype(np.uint8)


def load_as_mat(fn_or_fp, height, width):
    """Load an image from file and convert it into matrix.
    The data type of the array is np.float32.
    Elements in the matrix must be valued in range -1 ~ 1.

    :param fn_or_fp: File name or file object.
    :param height: Height.
    :param width: Width.
    :return: An matrix represents the image.
    """
    image = Image.open(fn_or_fp)
    image = image.resize((width, height))
    return (np.asarray(image, dtype=np.float32) - 128.0) / 128.0


def save_mat(fn_or_fp, mat):
    """Save the array into file.
    The image format is specified be the suffix of the file name.

    :param fn_or_fp: FIle name or file object.
    :param mat: The matrix.
    :return: None.
    """
    array = mat_to_array(mat)
    save_array(fn_or_fp, array)


class BufferedImageSource(ph.DataSource):
    """Image data source.
    """

    def __init__(self,
                 image_dir,
                 height,
                 width,
                 depth=3,
                 buffer_size=0):
        self._height = height
        self._width = width
        self._depth = depth
        self._buffer_size = buffer_size
        #
        file_list = [os.path.join(image_dir, img_file) for img_file in os.listdir(image_dir)]
        self._dataset = ph.Dataset(file_list).shuffle()
        self._buffer = {}

    def next_batch(self, size=0):
        mat_list = []
        image_files, = self._dataset.next_batch(size)
        for i, image_file in enumerate(image_files):
            if image_file in self._buffer:
                mat = self._buffer[image_file]
            else:
                mat = load_as_mat(image_file, self._width, self._height)
                mat = mat.reshape((self._width, self._height, self._depth))
                self._buffer[image_file] = mat
                if 0 < self._buffer_size < len(self._buffer):
                    self._buffer.popitem()
            mat_list.append(mat)
        return mat_list,


class ImageSourceWithLabels(ph.DataSource):
    """Image data source.
    """

    def __init__(self,
                 image_dir,
                 height,
                 width,
                 depth,
                 num_classes,
                 buffer_size=0):
        self._height = height
        self._width = width
        self._depth = depth
        self._num_classes = num_classes
        self._buffer_size = buffer_size
        #
        file_list = [
            os.path.join(image_dir, img_file)
            for img_file in os.listdir(image_dir)
            if not img_file.endswith('.txt')
        ]
        self._dataset = ph.Dataset(file_list).shuffle()
        self._buffer = {}

    def next_batch(self, size=0):
        mat_list = []
        onehot_list = []
        files, = self._dataset.next_batch(size)
        for i, file_ in enumerate(files):
            if file_ in self._buffer:
                mat, onehot = self._buffer[file_]
            else:
                mat = load_as_mat(file_, self._height, self._width)
                mat = mat.reshape((self._height, self._width, self._depth))
                with open(os.path.splitext(file_)[0] + '.txt', 'rt') as f:
                    label = int(f.readline())
                onehot = np.zeros((self._num_classes,), dtype=np.float32)
                onehot[label % self._num_classes] = 1.0
                self._buffer[file_] = (mat, onehot)
                if 0 < self._buffer_size < len(self._buffer):
                    self._buffer.popitem()
            mat_list.append(mat)
            onehot_list.append(onehot)
        return mat_list, onehot_list


class ImageSource(ph.DataSource):
    """Image data source.
    """

    def __init__(self,
                 image_dir,
                 height,
                 width,
                 channels):
        self._height = height
        self._width = width
        self._channels = channels
        #
        file_list = [
            os.path.join(image_dir, file_) for file_ in os.listdir(image_dir)
        ]
        file_list = [file_ for file_ in file_list if os.path.isfile(file_)]
        self._dataset = ph.Dataset(file_list).shuffle()

    def next_batch(self, size=0):
        mat_list = np.zeros((size, self._height, self._width, self._channels), dtype=np.float32)
        image_files, = self._dataset.next_batch(size)
        for i, image_file in enumerate(image_files):
            try:
                mat_list[i] = load_as_mat(image_file, self._width, self._height)
            except Exception as e:
                print(e)
                continue
        return mat_list,


def _trans_mat_offset_center(mat, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, mat), reset_matrix)
    return transform_matrix


def _apply_transform(mat,
                     trans_mat,
                     channel_axis=0,
                     fill_mode='nearest',
                     const_value=0.):
    """Apply the image transformation specified by a matrix.

    :param mat: 2D numpy array, single image.
    :param trans_mat: Numpy array specifying the geometric transformation.
    :param channel_axis: Index of axis for channels in the input tensor.
    :param fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
    :param const_value: Value used for points outside the boundaries
            of the input if `mode='constant'`.
    :return: The transformed version of the input.
    """
    mat = np.rollaxis(mat, channel_axis, 0)
    final_affine_matrix = trans_mat[:2, :2]
    final_offset = trans_mat[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(
        x_channel,
        final_affine_matrix,
        final_offset,
        order=0,
        mode=fill_mode,
        cval=const_value) for x_channel in mat]
    mat = np.stack(channel_images, axis=0)
    mat = np.rollaxis(mat, 0, channel_axis + 1)
    return mat


def random_rotate(mat,
                  rg,
                  row_axis=0,
                  col_axis=1,
                  channel_axis=2,
                  fill_mode='nearest',
                  const_value=0.0):
    """Performs a random rotation of a Numpy image tensor.

    :param mat: Input tensor. Must be 3D.
    :param rg:
    :param row_axis: Index of axis for rows in the input tensor.
    :param col_axis: Index of axis for columns in the input tensor.
    :param channel_axis: Index of axis for channels in the input tensor.
    :param fill_mode: Points outside the boundaries of the input
             are filled according to the given mode
             (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
    :param const_value: Value used for points outside the boundaries
             of the input if `mode='constant'`.
    :return: Rotated Numpy image tensor.
    """
    theta = np.pi / 180 * np.random.uniform(-rg, rg)
    trans_mat = np.array(
        [[np.cos(theta), -np.sin(theta), 0],
         [np.sin(theta), np.cos(theta), 0],
         [0, 0, 1]]
    )
    h, w = mat.shape[row_axis], mat.shape[col_axis]
    trans_mat = _trans_mat_offset_center(trans_mat, h, w)
    mat = _apply_transform(mat, trans_mat, channel_axis, fill_mode, const_value)
    return mat


def random_shift(mat,
                 wrg,
                 hrg,
                 row_axis=0,
                 col_axis=1,
                 channel_axis=2,
                 fill_mode='nearest',
                 const_value=0.0):
    """Performs a random spatial shift of a Numpy image tensor.

    :param mat: Input tensor. Must be 3D.
    :param wrg: Width shift range, as a float fraction of the width.
    :param hrg: Height shift range, as a float fraction of the height.
    :param row_axis: Index of axis for rows in the input tensor.
    :param col_axis: Index of axis for columns in the input tensor.
    :param channel_axis: Index of axis for channels in the input tensor.
    :param fill_mode: Points outside the boundaries of the input
             are filled according to the given mode
             (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
    :param const_value: Value used for points outside the boundaries
             of the input if `mode='constant'`.
    :return: Shifted Numpy image tensor.
    """
    h, w = mat.shape[row_axis], mat.shape[col_axis]
    tx = np.random.uniform(-hrg, hrg) * h
    ty = np.random.uniform(-wrg, wrg) * w
    trans_mat = np.array([[1, 0, tx],
                          [0, 1, ty],
                          [0, 0, 1]])
    mat = _apply_transform(mat, trans_mat, channel_axis, fill_mode, const_value)
    return mat


def random_shear(mat,
                 intensity,
                 row_axis=0,
                 col_axis=1,
                 channel_axis=2,
                 fill_mode='nearest',
                 const_value=0.0):
    """Performs a random spatial shear of a Numpy image tensor.

    :param mat: Input tensor. Must be 3D.
    :param intensity: Transformation intensity.
    :param row_axis: Index of axis for rows in the input tensor.
    :param col_axis: Index of axis for columns in the input tensor.
    :param channel_axis: Index of axis for channels in the input tensor.
    :param fill_mode: Points outside the boundaries of the input
             are filled according to the given mode
             (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
    :param const_value: Value used for points outside the boundaries
             of the input if `mode='constant'`.
    :return: Sheared Numpy image tensor.
    """
    shear = np.random.uniform(-intensity, intensity)
    trans_mat = np.array([[1, -np.sin(shear), 0],
                          [0, np.cos(shear), 0],
                          [0, 0, 1]])
    h, w = mat.shape[row_axis], mat.shape[col_axis]
    transform_matrix = _trans_mat_offset_center(trans_mat, h, w)
    mat = _apply_transform(mat, transform_matrix, channel_axis, fill_mode, const_value)
    return mat


def random_zoom(mat,
                zoom_range,
                row_axis=0,
                col_axis=1,
                channel_axis=2,
                fill_mode='nearest',
                const_value=0.0):
    """Performs a random spatial zoom of a Numpy image tensor.

    :param mat: Input tensor. Must be 3D.
    :param zoom_range: Tuple of floats; zoom range for width and height.
    :param row_axis: Index of axis for rows in the input tensor.
    :param col_axis: Index of axis for columns in the input tensor.
    :param channel_axis: Index of axis for channels in the input tensor.
    :param fill_mode: Points outside the boundaries of the input
             are filled according to the given mode
             (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
    :param const_value: Value used for points outside the boundaries
             of the input if `mode='constant'`.
    :return: Zoomed Numpy image tensor.
    """
    zoom_range = zoom_range
    if len(zoom_range) != 2:
        raise ValueError('`zoom_range` should be a tuple or list of two floats. '
                         'Received arg: ', zoom_range)
    if zoom_range[0] == 1 and zoom_range[1] == 1:
        zx, zy = 1, 1
    else:
        zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
    trans_mat = np.array([[zx, 0, 0],
                          [0, zy, 0],
                          [0, 0, 1]])
    h, w = mat.shape[row_axis], mat.shape[col_axis]
    transform_matrix = _trans_mat_offset_center(trans_mat, h, w)
    mat = _apply_transform(mat, transform_matrix, channel_axis, fill_mode, const_value)
    return mat


def random_channel(mat,
                   intensity,
                   channel_axis=2):
    mat = np.rollaxis(mat, channel_axis, 0)
    min_x, max_x = np.min(mat), np.max(mat)
    channel_images = [np.clip(x_channel + np.random.uniform(-intensity, intensity), min_x, max_x)
                      for x_channel in mat]
    mat = np.stack(channel_images, axis=0)
    mat = np.rollaxis(mat, 0, channel_axis + 1)
    return mat


class MatFilter(object):
    """Matrix filter
    """

    def __init__(self,
                 row_axis=0,
                 col_axis=1,
                 channel_axis=2):
        self._row_axis = row_axis
        self._col_axis = col_axis
        self._channel_axis = channel_axis

    def __call__(self, mat):
        raise NotImplementedError()


class RandomRotationFilter(MatFilter):

    def __init__(self,
                 rg,
                 row_axis=0,
                 col_axis=1,
                 channel_axis=2,
                 fill_mode='nearest',
                 const_value=0.0):
        super(RandomRotationFilter, self).__init__(row_axis, col_axis, channel_axis)
        self._rg = rg
        self._fill_mode = fill_mode
        self._const_value = const_value

    def __call__(self, mat):
        mat = random_rotate(
            mat,
            self._rg,
            self._row_axis,
            self._col_axis,
            self._channel_axis,
            self._fill_mode,
            self._const_value
        )
        return mat


class RandomShiftFilter(MatFilter):

    def __init__(self,
                 wrg,
                 hrg,
                 row_axis=0,
                 col_axis=1,
                 channel_axis=2,
                 fill_mode='nearest',
                 const_value=0.0):
        super(RandomShiftFilter, self).__init__(row_axis, col_axis, channel_axis)
        self._wrg = wrg
        self._hrg = hrg
        self._fill_mode = fill_mode
        self._const_value = const_value

    def __call__(self, mat):
        mat = random_shift(
            mat,
            self._wrg,
            self._hrg,
            self._row_axis,
            self._col_axis,
            self._channel_axis,
            self._fill_mode,
            self._const_value
        )
        return mat


class RandomShearFilter(MatFilter):

    def __init__(self,
                 intensity,
                 row_axis=0,
                 col_axis=1,
                 channel_axis=2,
                 fill_mode='nearest',
                 const_value=0.0):
        super(RandomShearFilter, self).__init__(row_axis, col_axis, channel_axis)
        self._intensity = intensity
        self._fill_mode = fill_mode
        self._const_value = const_value

    def __call__(self, mat):
        mat = random_shear(
            mat,
            self._intensity,
            self._row_axis,
            self._col_axis,
            self._channel_axis,
            self._fill_mode,
            self._const_value
        )
        return mat


class RandomZoomFilter(MatFilter):

    def __init__(self,
                 zoom_range,
                 row_axis=0,
                 col_axis=1,
                 channel_axis=2,
                 fill_mode='nearest',
                 const_value=0.0):
        super(RandomZoomFilter, self).__init__(row_axis, col_axis, channel_axis)
        self._zoom_range = zoom_range
        self._fill_mode = fill_mode
        self._const_value = const_value

    def __call__(self, mat):
        mat = random_zoom(
            mat,
            self._zoom_range,
            self._row_axis,
            self._col_axis,
            self._channel_axis,
            self._fill_mode,
            self._const_value
        )
        return mat


class RandomChannelFilter(MatFilter):
    def __init__(self,
                 intensity,
                 row_axis=0,
                 col_axis=1,
                 channel_axis=2):
        super(RandomChannelFilter, self).__init__(row_axis, col_axis, channel_axis)
        self._intensity = intensity

    def __call__(self, mat):
        mat = random_channel(mat, self._intensity, self._channel_axis)
        return mat


class RandomComboFilter(MatFilter):

    def __init__(self,
                 row_axis=0,
                 col_axis=1,
                 channel_axis=2):
        super(RandomComboFilter, self).__init__(row_axis, col_axis, channel_axis)
        self._filter_list = []

    def add_filter(self, filter_):
        if not isinstance(filter_, MatFilter):
            raise ValueError('filter_ should be an instance of MatFilter.')
        self._filter_list.append(filter_)

    def __call__(self, mat):
        filter_ = random.choice(self._filter_list)
        mat = filter_.__call__(mat)
        return mat


def default_augmentation_filter():
    filter_ = RandomComboFilter()
    filter_.add_filter(RandomRotationFilter(30))
    filter_.add_filter(RandomShearFilter(0.5))
    filter_.add_filter(RandomZoomFilter((0.8, 1.5)))
    filter_.add_filter(RandomChannelFilter(0.4))
    return filter_


class AugmentedImageSource(ph.DataSource):

    def __init__(self,
                 data_source,
                 image_col=0):
        self._data_source = data_source
        self._image_col = image_col
        filter_ = RandomComboFilter()
        self._filter = filter_
        filter_.add_filter(RandomRotationFilter(30))
        filter_.add_filter(RandomShearFilter(0.5))
        filter_.add_filter(RandomZoomFilter((0.8, 1.5)))
        filter_.add_filter(RandomChannelFilter(0.4))

    def next_batch(self, size=0):
        data_batch = self._data_source.next_batch(size)
        mat_batch = data_batch[self._image_col]
        mat_batch = np.array(
            [self._filter(mat) for mat in mat_batch],
            dtype=np.float32
        )
        return tuple(col if i != self._image_col else mat_batch for i, col in enumerate(data_batch))
