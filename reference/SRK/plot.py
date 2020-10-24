#!/usr/bin/env python3

"""
@author: xi
@since: 2018-02-03
"""

import argparse
import os

from matplotlib import pyplot as plt

names = [
    'Kitchen',
    'Software',
    'Sports',
    'Music',
    'Baby',
    'Home',
    'Books',
    'Shoes',
    'Automotive',
    'Bed',
    'PC',
    'Players',
    'Camera',
    'Tools',
    'Audio',
    'Phone',
    'Laptop',
    'TV',
    'Network',
    'Office'
]


def main(args):
    # _, ax1 = plt.subplots()
    plt.ylabel('Accuracy')
    symbols = 'xsodp*'
    for i, file_ in enumerate(args.files):
        with open(file_, 'rt') as f:
            y_list = [
                100.0 - float(item)
                for item in (line.strip() for line in f.readlines())
                if len(item) > 0
            ]
        x_list = [i for i in range(len(y_list))]
        name = os.path.basename(file_)
        plt.plot(x_list, y_list, '-', marker=symbols[i], label=name, ms=5, markerfacecolor='None')
    domains = [17, 5, 0, 2, 15, 13, 12, 7, 8, 9, 16, 19, 1, 4, 3, 6, 10, 18, 11, 14]
    domains = [names[domain] for domain in domains]
    plt.xticks(range(20), domains, rotation=60)
    plt.legend()
    plt.grid()
    plt.show()
    return 0


if __name__ == '__main__':
    _parser = argparse.ArgumentParser()
    _parser.add_argument('files', nargs='+')
    _args = _parser.parse_args()
    exit(main(_args))
