import os, sys
import numpy as np
import torch
from sklearn.utils import shuffle
import json
import xml.etree.ElementTree as ET

import os, sys
import numpy as np
import torch
from torchvision import datasets, transforms
from sklearn.utils import shuffle
from torchvision import datasets,transforms
import json
from torch.utils.data import Dataset
from sklearn.utils import shuffle
from PIL import Image
import random
def get(seed=0):

    ninteen_domains = \
    [
        'Bing3domains_Speaker',
        'Bing3domains_Router',
        'Bing3domains_Computer',

        'Bing5domains_Nokia6610',
        'Bing5domains_NikonCoolpix4300',
        'Bing5domains_CreativeLabsNomadJukeboxZenXtra40GB',
        'Bing5domains_CanonG3',
        'Bing5domains_ApexAD2600Progressive',

        'Bing9domains_CanonPowerShotSD500',
        'Bing9domains_CanonS100',
        'Bing9domains_DiaperChamp',
        'Bing9domains_HitachiRouter',
        'Bing9domains_ipod',
        'Bing9domains_LinksysRouter',
        'Bing9domains_MicroMP3',
        'Bing9domains_Nokia6600',
        'Bing9domains_Norton',

        'XuSemEval14_rest',
        'XuSemEval14_laptop',

    ]



    ten_domains = \
    [
        'Bing3domains_Speaker',
        'Bing3domains_Router',

        'Bing5domains_Nokia6610',
        'Bing5domains_NikonCoolpix4300',
        'Bing5domains_CreativeLabsNomadJukeboxZenXtra40GB',

        'Bing9domains_CanonPowerShotSD500',
        'Bing9domains_CanonS100',
        'Bing9domains_DiaperChamp',

        'XuSemEval14_rest',
        'XuSemEval14_laptop',

    ]

    # with open('asc_random_19','w') as f_random_seq:
    #     for repeat_num in range(20):
    #         random.shuffle(ninteen_domains)
    #         f_random_seq.writelines('\t'.join(ninteen_domains) + '\n')

    with open('asc_random_10','w') as f_random_seq:
        for repeat_num in range(20):
            random.shuffle(ten_domains)
            f_random_seq.writelines('\t'.join(ten_domains) + '\n')


if __name__ == "__main__":
    get()