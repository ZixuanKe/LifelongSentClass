#!/usr/bin/env python3

"""
@author: xi
@since: 2018-02-04
"""

import argparse

import pymongo
import re
import pickle
import numpy as np
from matplotlib import pyplot as plt


def main(args):
    with open('flat_states', 'rb') as f:
        v = pickle.load(f)
    y, r = np.histogram(v, bins=100)
    x = r[:-1] * 1e-2
    plt.figure(figsize=(7, 3))
    plt.bar(x, y * 2, 100)
    plt.xlabel('Activation Degree')
    plt.ylabel('Number of Neurons')
    plt.grid()
    plt.show()
    print(r.shape)
    return 0


if __name__ == '__main__':
    _parser = argparse.ArgumentParser()
    _args = _parser.parse_args()
    exit(main(_args))
