from nilmtk.api import API
import warnings

warnings.filterwarnings("ignore")
from nilmtk.disaggregate import SGN

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import pytorch_lightning as pl

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

# from src import *

USE_GPU = True
device = torch.device("cuda:0" if USE_GPU and torch.cuda.is_available() else "cpu")
print(torch.__version__, pl.__version__, device)

e = {
    # Specify power type, sample rate and disaggregated appliance
    'power': {
        'mains': ['active'],
        'appliance': ['active']
    },
    'sample_rate': 6,
    'appliances': ['microwave'],
    # Universally no pre-training
    'pre_trained': False,
    # Specify algorithm hyper-parameters
    'methods': {"SGN": SGN({'n_epochs': 3, 'batch_size': 256})},
    # Specify train and test data
    'train': {
        'datasets': {
            'ukdale': {
                'path': 'mnt/ukdale.h5',
                'buildings': {
                    1: {
                        'start_time': '2013-03-18',
                        'end_time': '2014-12-31'
                    },
                    5: {
                        'start_time': '2014-06-30',
                        'end_time': '2014-09-06'
                    }
                }
            },
            # 'refit': {
            #     'path': 'mnt/refit.h5',
            #     'buildings': {
            #         7: {
            #             'start_time': '2013-11-02',
            #             'end_time': '2014-12-31'
            #         },
            #         10: {
            #             'start_time': '2013-11-21',
            #             'end_time': '2014-12-31'
            #         },
            #         # 14: {
            #         #     'start_time': '2013-12-18',
            #         #     'end_time': '2014-12-31'
            #         # },
            #         # 15: {
            #         #     'start_time': '2014-01-11',
            #         #     'end_time': '2014-12-31'
            #         # },
            #         # 17: {
            #         #     'start_time': '2014-03-08',
            #         #     'end_time': '2014-12-31'
            #         # },
            #     }
            # },
        }
    },
    'test': {
        'datasets': {
            'ukdale': {
                'path': 'mnt/ukdale.h5',
                'buildings': {
                    2: {
                        'start_time': '2013-5-21',
                        'end_time': '2013-7-22'
                    }
                }
            },
            # 'refit': {
            #     'path': 'mnt/refit.h5',
            #     'buildings': {
            #         1: {
            #             'start_time': '2013-10-10',
            #             'end_time': '2013-12-10'
            #         },
            #     }
            # },
        },
        # Specify evaluation metrics
        'metrics': ['mae', 'f1score', 'recall', 'precision', 'nep', 'omae', 'MCC']
    }
}

if __name__ == '__main__':
    API(e)
