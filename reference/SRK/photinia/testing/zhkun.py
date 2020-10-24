#!/usr/bin/env python3

"""
@author: zhkun
@since: 2017-12
"""

import tensorflow as tf

import photinia as ph


class SeqCoAttention(ph.Widget):
    """
    sequential co-attention模块，
    H_i = tanh(W_x*x_i + W_g1*g1 + W_g2*g2)
    alpha_i = softmax(w^T*H_i)
    \hat(x) = sum(alpha_i*x_i)
    输入：处理好的premise, hypothesis, image的表示
          premise: [seq_len, batch_size, emb_size]
          hypothesis: [seq_len, batch_size, emb_size]
          image: [img_len, batch_size, img_size]
         图像序列长度：img_len，图像展平后的尺寸：img_size, 中间隐层的尺寸：state_size
         要求输出的shape都应该为：[len, batch_size, state_size]
    输出：三个attention表示的结果，为三个向量[batch_size, state_size]
    """

    def __init__(self, name, vec_size, img_size, state_size, keep_prob, is_train=False):
        self._vec_size = vec_size
        self._img_size = img_size
        self._state_size = state_size
        self._keep_prob = keep_prob
        self._is_train = is_train
        super(SeqCoAttention, self).__init__(name)

    def _build(self):
        """
        简单的办法是保证三个输入向量的维度是一致的，
        这里我们不设成一致的参数，这样灵活性更大
        w：seq的参数，v：image的参数
        :return:
        """
        # stage 1:
        w_init_value = tf.random_normal(
            dtype=ph.D_TYPE,
            shape=(self._vec_size, self._state_size),
            stddev=1.0 / (self._vec_size + self._state_size)
        )
        v_init_value = tf.random_normal(
            dtype=ph.D_TYPE,
            shape=(self._img_size, self._state_size),
            stddev=1.0 / (self._img_size + self._state_size)
        )
        u_init_value = tf.random_normal(
            dtype=ph.D_TYPE,
            shape=(self._state_size, 1),
            stddev=1.0 / (self._state_size + 1)
        )

        # stage 1:
        # input: img
        # output: attentioned img
        self._wx1 = tf.Variable(name='wx1', dtype=ph.D_TYPE, initial_value=v_init_value)
        self._ww1 = tf.Variable(name='ww1', dtype=ph.D_TYPE, initial_value=u_init_value)

        # stage 2:
        # input: attentioned img, seqA
        # output: attentioned seqA
        self._wx2 = tf.Variable(name='wx2', dtype=ph.D_TYPE, initial_value=w_init_value)
        self._wg21 = tf.Variable(name='wg21', dtype=ph.D_TYPE, initial_value=v_init_value)
        self._ww2 = tf.Variable(name='ww2', dtype=ph.D_TYPE, initial_value=u_init_value)

        # stage 3:
        # input: attentioned seq A, attentioned image
        # output: attentioned seq B
        self._wx3 = tf.Variable(name='wx3', dtype=ph.D_TYPE, initial_value=w_init_value)
        self._wg31 = tf.Variable(name='wg31', dtype=ph.D_TYPE, initial_value=w_init_value)
        self._wg32 = tf.Variable(name='wg32', dtype=ph.D_TYPE, initial_value=v_init_value)
        self._ww3 = tf.Variable(name='ww3', dtype=ph.D_TYPE, initial_value=u_init_value)

        # stage 4:
        # input: attentioned seq B, attentioned seq A
        # output: attentioned img
        self._wx4 = tf.Variable(name='wx4', dtype=ph.D_TYPE, initial_value=v_init_value)
        self._wg41 = tf.Variable(name='wg41', dtype=ph.D_TYPE, initial_value=w_init_value)
        self._wg42 = tf.Variable(name='wg42', dtype=ph.D_TYPE, initial_value=w_init_value)
        self._ww4 = tf.Variable(name='ww4', dtype=ph.D_TYPE, initial_value=u_init_value)

        # stage 5:
        # input: attentioned seq B, attentioned img
        # output: attentioned seqA
        self._wx5 = tf.Variable(name='wx5', dtype=ph.D_TYPE, initial_value=w_init_value)
        self._wg51 = tf.Variable(name='wg51', dtype=ph.D_TYPE, initial_value=w_init_value)
        self._wg52 = tf.Variable(name='wg52', dtype=ph.D_TYPE, initial_value=v_init_value)
        self._ww5 = tf.Variable(name='ww5', dtype=ph.D_TYPE, initial_value=u_init_value)

    def _setup(self, seqa, img, seqb):
        """
        模型在这块做了修改，首先处理的是image的信息
        :param seqA: [seq_len, batch_size, vec_size]
        :param img:  [img_len, batch_size, img_size]
        :param seqB: [seq_len, batch_size, vec_size]
        :return: attentioned seqA, attentioned img, attentioned seqB, shape=[batch_size, vec_size/img_size]
        """
        img_drop = dropout(img, self._keep_prob, self._is_train)
        seqa_drop = dropout(seqa, self._keep_prob, self._is_train)
        seqb_drop = dropout(seqb, self._keep_prob, self._is_train)

        # stage 1
        img_ = self._calAttention([self._wx1, self._ww1], img_drop, 1)
        # stage 2
        seqA_ = self._calAttention([self._wx2, self._wg21, self._ww2], [seqa_drop, img_], 2)
        # stage 3
        atten_seqB = self._calAttention([self._wx3, self._wg31, self._wg32, self._ww3],
                                        [seqb_drop, seqA_, img_], 3)
        # stage 4
        atten_img = self._calAttention([self._wx4, self._wg41, self._wg42, self._ww4],
                                       [img_drop, atten_seqB, seqA_], 4)
        # stage 5
        atten_seqA = self._calAttention([self._wx5, self._wg51, self._wg52, self._ww5],
                                        [seqa_drop, atten_seqB, atten_img], 5)

        result = {'atten_seqA': atten_seqA, 'atten_img': atten_img, 'atten_seqB': atten_seqB}
        return result

    def _calAttention(self, param, infors, stage):
        # assert len(param) == len(infors)+1

        # stage 1:
        if stage == 1:
            target = infors
            # img = tf.reshape(infors, [-1, self._img_size])
            wx, ww = param
            # shape = [img_len, batch_size, state_size]
            tmp1 = tf.nn.tanh(tf.tensordot(target, wx, [[-1], [0]]))

            attentioned = self._comm(target, tmp1, ww)
            return attentioned

        # stage 2:
        elif stage == 2:
            target, img = infors
            wx, wg1, ww = param

            tmp1 = tf.tensordot(target, wx, axes=[[2], [0]])

            mid_m = tf.nn.tanh(tmp1 + tf.matmul(img, wg1))

            attentioned = self._comm(target, mid_m, ww)
            return attentioned

        # stage 3, 4, 5
        else:  # stage == 4:
            target, seq1, seq2 = infors
            wx, wg1, wg2, ww = param

            tmp1 = tf.tensordot(target, wx, [[-1], [0]])
            mid_m = tf.nn.tanh(tmp1 + tf.matmul(seq1, wg1) + tf.matmul(seq2, wg2))

            attentioned = self._comm(target, mid_m, ww)
            return attentioned

    def _supported(self, x_input, param, length, state_size):
        """
        对attention计算阶段的一个抽取
        :param x_input: 输入的向量
        :param param: 对应的参数
        :param length: 要扩展的长度
        :param state_size: 每个的state_size
        :return: 计算好的结果，shape=[length*batch_size, state_size]
        """

        tmp = tf.matmul(x_input, param)
        tmp = tf.tile(tmp, multiples=[length, 1])

        return tmp

    def _comm(self, target, x_input, param):
        """
        attention的后两个计算步骤，每次都一样的，这里做了求和的操作，如果不需要求和的话，这里应该把最后的reduce_sum删掉
        :param target: 要计算的attentioned目标，对应公式中的x
        :param x_input: attention计算过程第一步计算出来的结果
        :param param: 对应的softmax函数的参数
        :param length: 对应的序列长度
        :return: target的attentioned之后的结果
        """
        # shape = [seq_len, batch_size, 1]
        tmp = tf.tensordot(x_input, param, [[-1], [0]])
        # shape = [length, batch_size, 1]
        # tmp = tf.reshape(tmp, [length, -1, 1])
        # 在seq_len维度上进行softmax，shape=[seq_len, batch_size, 1]
        alpha = tf.nn.softmax(tmp, dim=0)

        attentioned = tf.multiply(target, alpha)
        # 在seq_len维度上进行求和，shape=[batch_size, target_size]
        attentioned = tf.reduce_sum(attentioned, axis=0)

        return attentioned

    @property
    def wx1(self):
        return self._wx1

    @property
    def ww1(self):
        return self._ww1

    @property
    def wx2(self):
        return self._wx2

    @property
    def wg21(self):
        return self._wg21

    @property
    def ww2(self):
        return self._ww2

    @property
    def wx3(self):
        return self._wx3

    @property
    def wg31(self):
        return self._wg31

    @property
    def wg32(self):
        return self._wg32

    @property
    def ww3(self):
        return self._ww3

    @property
    def wx4(self):
        return self._wx4

    @property
    def wg41(self):
        return self._wg41

    @property
    def wg42(self):
        return self._wg42

    @property
    def ww4(self):
        return self._ww4

    @property
    def wx5(self):
        return self._wx5

    @property
    def wg51(self):
        return self._wg51

    @property
    def wg52(self):
        return self._wg52

    @property
    def ww5(self):
        return self._ww5



# 计算两个向量之间的相似度，有三种，分别是cosin，bilinear，tensor
class Similar(ph.Widget):
    """
    需要完成的事情：计算三种不同的相似度，并将计算出结果的拼起来
    输入：需要计算相似度的两个向量，以及需要的相似度计算方法
    输出：默认情况下计算出来两个向量之间的相似度向量拼接结果，或者选择需要的相似度计算方法
    """

    def __init__(self, name, batch_size=None, state_size=None, num_tensor=None):
        self._state_size = state_size
        self._batch_size = batch_size
        self._num_tensor = num_tensor
        super(Similar, self).__init__(name)

    def _build(self):
        self._xav_init = tf.contrib.layers.xavier_initializer

        # Bilinear_similarity的参数
        self._bilinear_W = tf.get_variable('Bilinear_w', shape=[self._state_size, self._state_size],
                                           initializer=self._xav_init())
        self._bilinear_b = tf.get_variable('Bilinear_b', shape=[], initializer=tf.constant_initializer(0.))

        # tensor_similarity的参数
        self._tensor_W = tf.get_variable('tensor_w', shape=[self._num_tensor, self._state_size, self._state_size],
                                         initializer=self._xav_init())
        self._tensor_V = tf.get_variable('tensor_v', shape=[self._state_size * 2, self._num_tensor],
                                         initializer=self._xav_init())
        self._tensor_b = tf.get_variable('tensor_b', shape=[self._num_tensor],
                                         initializer=tf.constant_initializer(0.))

    def _setup(self, x, y):
        simi1 = self._cosin_similarity(x, y)
        simi2 = self._Bilinear_similarity(x, y)
        simi3 = self._tl_similarity(x, y)

        # 经过增加维度之后，得到的shape=[batch_size, 1]
        simi1 = tf.expand_dims(simi1, axis=1)
        simi2 = tf.expand_dims(simi2, axis=1)
        result = tf.concat([simi1, simi2, simi3], axis=1)

        return result

    def _cosin_similarity(self, x, y):
        # shape = x.shape
        # # batch_size = shape[0]
        # print(x.shape)

        tmp = tf.matmul(x, tf.transpose(y, [1, 0]))
        eye = tf.eye(self._batch_size)
        tmp = tf.multiply(tmp, eye)
        tmp = tf.reduce_sum(tmp, axis=0)
        zx = tf.sqrt(tf.reduce_sum(tf.multiply(x, x), axis=1))
        zy = tf.sqrt(tf.reduce_sum(tf.multiply(x, x), axis=1))

        # result = tf.div(tmp, tf.add(zx, zy))
        result = tmp / (zx + zy)

        return result

    def _Bilinear_similarity(self, x, y):
        tmp = tf.matmul(x, self._bilinear_W)
        tmp = tf.matmul(tmp, tf.transpose(y, [1, 0]))
        eye = tf.eye(self._batch_size)
        tmp = tf.multiply(tmp, eye)
        result = tf.reduce_sum(tmp, axis=0)

        result = result + self._bilinear_b

        return result

    def _tl_similarity(self, x, y):
        # x * w 得到的shape=[batch_size, num_tensor, state_size]
        tmp = tf.tensordot(x, self._tensor_W, axes=[[1], [1]])
        # x*w*y，得到的shape=[batch_size, num_tensor, batch_size]
        tmp = tf.tensordot(tmp, y, axes=[[2], [1]])
        # 为了方便计算，shape=[num_tensor, batch_size, batch_size]
        tmp = tf.transpose(tmp, [1, 0, 2])

        # 创建需要的单位阵，shape=[num_tensor, batch_size, batch_size]
        eye = tf.eye(self._batch_size, batch_shape=[self._num_tensor])
        # 获得需要的结果, shape=[num_tensor, batch_size, batch_size]
        tmp = tf.multiply(tmp, eye)
        # 求和，得到需要的第一部分结果, shape=[num_tensor, batch_size]
        tmp = tf.reduce_sum(tmp, axis=1)
        # 转置，方便下边计算shape=[batch_size, num_tensor]
        tmp = tf.transpose(tmp, [1, 0])

        # tensor_similarity公式第二部分V*[x, y]^T
        cont = tf.concat([x, y], axis=1)
        v = tf.matmul(cont, self._tensor_V)

        result = tmp + v + self._tensor_b
        result = ph.lrelu(result)

        return result

    @property
    def tl_similarity(self):
        return self._tl_similarity

    @property
    def Bilinear_similarity(self):
        return self._Bilinear_similarity

    @property
    def cosin_similarity(self):
        return self._cosin_similarity


# highway network cell
class HighWayCell(ph.Widget):
    """
    highway network的实现
    要求输入的x_data, 经过线性变换的结果必须都得有相同的维度，这点感觉有点坑
    """

    def __init__(self, name, inputs_size, output_size):
        self._inputs_size = inputs_size
        self._output_size = output_size
        super(HighWayCell, self).__init__(name)

    def _build(self):
        self._linear = ph.Linear('linear', input_size=self._inputs_size, output_size=self._output_size)
        self._gate = ph.Linear('gate', input_size=self._inputs_size, output_size=self._output_size)

    def _setup(self, x_data):
        result = self._linear.setup(x_data)
        result = tf.nn.relu(result)

        gate = self._gate.setup(x_data)
        gate = tf.nn.sigmoid(gate)

        y = tf.multiply(result, gate) + tf.multiply(x_data, (1 - gate))

        return y


class HighWayLayer(ph.Widget):
    """
    实现多层的highway network
    """

    def __init__(self, name, input_size, output_size, num_layer):
        self._input_size = input_size
        self._output_size = output_size
        self._num_layer = num_layer
        super(HighWayLayer, self).__init__(name)

    def _build(self):
        self._layers = []
        for idx in range(self._num_layer):
            self._layers.append(HighWayCell('highway' + str(idx), inputs_size=self._input_size,
                                            output_size=self._output_size))

    def _setup(self, x_data):
        pre = x_data
        cur = None

        for idx in range(self._num_layer):
            cur = self._layers[idx].setup(pre)
            pre = cur

        return cur


# Residual network cell
class ResidualCell(ph.Widget):
    """
    residual network的实现
    """

    def __init__(self, name, inputs_size, output_size):
        self._inputs_size = inputs_size
        self._output_size = output_size
        super(ResidualCell, self).__init__(name)

    def _build(self):
        self._linear = ph.Linear('linear', input_size=self._inputs_size, output_size=self._output_size)

    def _setup(self, x_data):
        result = self._linear.setup(x_data)
        result = tf.nn.relu(result)

        y = result + x_data

        return y


# memory network
class MemoryNetwork(ph.Widget):
    """
    end-to-end memory network的实现
    做一个简单的修改，这里我们全都给写成一个线性变换，
    待完成，1：输入是两个句子，针对普通的nli，2：输入时多个premise和单个hypothesis
    """

    def __init__(self, name, inputs_size, state_size, output_size, num_layers, batch_size, is_bilinear=False):
        self._inputs_size = inputs_size
        self._output_size = output_size
        self._state_size = state_size
        self._batch_size = batch_size
        self._num_layers = num_layers
        self._is_bilinear = is_bilinear
        super(MemoryNetwork, self).__init__(name)

    def _build(self):
        self._xav_init = tf.contrib.layers.xavier_initializer

        self._projA = []
        for idx in range(self._num_layers):
            self._projA[idx] = ph.Linear('proA' + str(idx), input_size=self._inputs_size, output_size=self._state_size)
        self._projC = []
        for idx in range(self._num_layers):
            self._projC[idx] = ph.Linear('proB' + str(idx), input_size=self._inputs_size, output_size=self._state_size)

        self._projB = ph.Linear('projC', input_size=self._inputs_size, output_size=self._state_size)

        self._w = tf.get_variable('w', shape=[self._state_size, self._state_size], initializer=self._xav_init())

        self._ow = tf.get_variable('ow', shape=[self._state_size, self._output_size], initializer=self._xav_init())

    def _setup(self, x_data, y_data):
        # 这里并没有对最后的输出层做激活或者做softmax
        # hypothesis
        embB = self._projB.setup(y_data)
        # different layer premise
        embA = self._projA[0].setup(x_data)
        embC = self._projC[0].setup(y_data)

        alpha = self._calcu(embA, embB)
        out = tf.reduce_sum(tf.multiply(embC, alpha), axis=0)

        for idx in range(1, self._num_layers):
            embB = embB + out
            embA = self._projA[idx].setup(x_data)
            embC = self._projC[idx].setup(x_data)

            alpha = self._calcu(embA, embB)
            out = tf.reduce_sum(tf.multiply(embC, alpha), axis=0)

        result = tf.matmul(out, self._ow)

        return result

    def _calcu(self, embA, embB):
        # 这里可以直接让两个向量相乘，或者乘以一个矩阵，就是Bilinear similarity，然后在seq_len上做reduce_mean或者max_pooling,
        # 假设我们的输入为A=[seq_len, batch_size, emb_size], B = [batch_size, emb_size]
        # 或者A为[num_seq, seq_len, batch_size, emb_size], B为[batch_size, emb_size]
        if not self._is_bilinear:
            # shape变换为[seq_len*batch_size, _state_size]
            # tmpA = tf.reshape(embA, shape=[-1, self._state_size])
            # shape = [seq_len*batch_size, seq_len*batch_size]
            # tmp = tf.matmul(tmpA, tf.transpose(tmpB, [1, 0]))

            # 直接tensordot乘，得到的结果为[seq_len, batch_size, batch_size]
            tmp = tf.tensordot(embA, embB, axes=[[0], [1]])

            # 创建相应的对角阵，shape = [seq_len, batch_size, batch_size]
            eye = tf.eye(self._batch_size, batch_shape=[tmp.shape[0]])

            # 获得需要的结果, shape=[seq_len, batch_size, batch_size]
            tmp = tf.multiply(tmp, eye)
            # 求和，得到需要的第一部分结果, shape=[seq_len, batch_size]
            tmp = tf.reduce_sum(tmp, axis=1)
            # shape = [seq_len, batch_size, 1]
            tmp = tf.reshape(tmp, shape=[-1, self._batch_size, 1])

            return tmp

        else:
            # 直接tensordot乘，得到的结果为[seq_len, batch_size, batch_size]
            tmp = tf.tensordot(embA, self._w, axes=[[0], [0]])
            tmp = tf.tensordot(tmp, embB, axes=[[0], [1]])

            # 创建相应的对角阵，shape = [seq_len, batch_size, batch_size]
            eye = tf.eye(self._batch_size, batch_shape=[tmp.shape[0]])

            # 获得需要的结果, shape=[seq_len, batch_size, batch_size]
            tmp = tf.multiply(tmp, eye)
            # 求和，得到需要的第一部分结果, shape=[seq_len, batch_size]
            tmp = tf.reduce_sum(tmp, axis=1)
            # shape = [seq_len, batch_size, 1]
            tmp = tf.reshape(tmp, shape=[-1, self._batch_size, 1])

            return tmp


def dropout(x, keep_prob, is_train, noise_shape=None, seed=None, name=None):
    with tf.name_scope(name or "dropout"):
        # if keep_prob < 1.0:
        d = tf.nn.dropout(x, keep_prob, noise_shape=noise_shape, seed=seed)
        if is_train:
            return d
        else:
            return x
