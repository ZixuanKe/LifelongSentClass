#!/usr/bin/env python3

"""
@author: xi
@since: 2017-12-25
"""

import photinia as ph


class VGG16(ph.Widget):

    def __init__(self,
                 name,
                 input_size,
                 output_size):
        self._input_size = input_size
        self._output_size = output_size
        self._input_height = input_size[0]
        self._input_width = input_size[1]
        self._input_channels = input_size[2]
        super(VGG16, self).__init__(name)

    @property
    def input_size(self):
        return self._input_size

    @property
    def input_height(self):
        return self._input_height

    @property
    def input_width(self):
        return self._input_width

    @property
    def input_channels(self):
        return self._input_channels

    @property
    def output_size(self):
        return self._output_size

    def _build(self):
        self._c1 = ph.Conv2D('c1', self._input_size, 64, 3, 3)
        self._c2 = ph.Conv2D('c2', self._c1.output_size, 64, 3, 3)
        self._p1 = ph.Pool2D('p1', self._c2.output_size, 2, 2)
        #
        self._c3 = ph.Conv2D('c3', self._p1.output_size, 128, 3, 3)
        self._c4 = ph.Conv2D('c4', self._c3.output_size, 128, 3, 3)
        self._p2 = ph.Pool2D('p2', self._c4.output_size, 2, 2)
        #
        self._c5 = ph.Conv2D('c5', self._p2.output_size, 256, 3, 3)
        self._c6 = ph.Conv2D('c6', self._c5.output_size, 256, 3, 3)
        self._c7 = ph.Conv2D('c7', self._c6.output_size, 256, 3, 3)
        self._p3 = ph.Pool2D('p3', self._c7.output_size, 2, 2)
        #
        self._c8 = ph.Conv2D('c8', self._p3.output_size, 512, 3, 3)
        self._c9 = ph.Conv2D('c9', self._c8.output_size, 512, 3, 3)
        self._c10 = ph.Conv2D('c10', self._c9.output_size, 512, 3, 3)
        self._p4 = ph.Pool2D('p4', self._c10.output_size, 2, 2)
        #
        self._c11 = ph.Conv2D('c11', self._p4.output_size, 512, 3, 3)
        self._c12 = ph.Conv2D('c12', self._c11.output_size, 512, 3, 3)
        self._c13 = ph.Conv2D('c13', self._c12.output_size, 512, 3, 3)
        self._p5 = ph.Pool2D('p5', self._c13.output_size, 2, 2)
        #
        self._h1 = ph.Linear(
            'h1', self._p5.flat_size, 4096,
            w_init=ph.RandomNormal(stddev=1e-4)
        )
        self._h2 = ph.Linear(
            'h2', self._h1.output_size, 4096,
            w_init=ph.RandomNormal(stddev=1e-4)
        )
        self._h3 = ph.Linear(
            'h3', self._h2.output_size, self._output_size,
            w_init=ph.RandomNormal(stddev=1e-4)
        )

    def _setup(self, x, dropout=None, activation=ph.swish):
        s = activation
        y = ph.setup(
            x,
            [self._c1, s, dropout,
             self._c2, s, self._p1,
             self._c3, s, dropout,
             self._c4, s, self._p2,
             self._c5, s, dropout,
             self._c6, s, dropout,
             self._c7, s, self._p3,
             self._c8, s, dropout,
             self._c9, s, dropout,
             self._c10, s, self._p4,
             self._c11, s, dropout,
             self._c12, s, dropout,
             self._c13, s, self._p5, dropout,
             ph.flatten,
             self._h1, s, dropout,
             self._h2, s, dropout,
             self._h3]
        )
        return y
