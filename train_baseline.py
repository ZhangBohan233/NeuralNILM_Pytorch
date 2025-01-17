from nilmtk.api import API
import warnings

warnings.filterwarnings("ignore")
from nilmtk.disaggregate import SGN, DM, DMCoral, GatedDM, DM_SDA

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

REDD_AVAIL = {
    'fridge': [1, 2, 3, 5, 6],
    'washing machine': [1, 2, 3, 4, 5, 6],
    'microwave': [1, 2, 3, 5]
}

REDD_TRAIN_STD = {
    'path': 'mnt/redd.h5',
    'buildings': {
        2: {
            'start_time': '2011-04-18',
            'end_time': '2011-05-21'
        },
        3: {
            'start_time': '2011-04-17',
            'end_time': '2011-05-29'
        },
        # 4: {
        #     'start_time': '2011-04-17',
        #     'end_time': '2011-06-02'
        # },
        5: {
            'start_time': '2011-04-19',
            'end_time': '2011-05-30'
        },
        # 6: {
        #     'start_time': '2011-05-22',
        #     'end_time': '2011-06-13'
        # },
    }

}

REDD_TEST_STD = {
    'path': 'mnt/redd.h5',
    # 'buildings': {
    #     2: {
    #         'start_time': '2011-04-26',
    #         'end_time': '2011-04-30'
    #     }
    # }
    'buildings': {
        1: {
            'start_time': '2011-04-19',
            'end_time': '2011-05-23'
        }
        # 1: {
        #     'start_time': '2011-04-28 05:57',
        #     'end_time': '2011-05-01'
        # }
    }
}

UKDALE_TRAIN_STD = {
    'path': 'mnt/ukdale.h5',
    'buildings': {
        1: {
            'start_time': '2013-05-31',
            'end_time': '2014-12-31'
        },
        # 2: {
        #     'start_time': '2013-05-22',
        #     'end_time': '2013-08-01'
        # },
        5: {
            'start_time': '2014-07-01',
            'end_time': '2014-09-05'
        },

    },
}

e = {
    # Specify power type, sample rate and disaggregated appliance
    'power': {
        # 'mains': ['active'],
        # 'appliance': ['active']
        'mains': ['apparent'],  # problem: ukdale active, redd apparent
        'appliance': ['active']
    },
    'sample_rate': 6,
    # 'appliances': ['fridge'],
    'appliances': ['microwave'],
    # Universally no pre-training
    'pre_trained': False,
    # Specify algorithm hyper-parameters
    'methods': {
        # "GaterSGN": GaterSGN(
        # {'n_epochs': 10, 'batch_size': 256, 'test_only': False}),
        "SGN": SGN(
            {'n_epochs': 100, 'batch_size': 256, 'test_only': False, 'gate_only': False,
             'patience': 5, 'note': 'redd'})
    },

    # Specify train and test data
    'train': {
        'datasets': {
            'redd': REDD_TRAIN_STD
        }
    },
    # 'transfer': {
    #     'datasets': {
    #         'redd': {
    #             'path': 'mnt/redd.h5',
    #             'buildings': {
    #                 # 1: {
    #                 #     'start_time': '2011-04-19',
    #                 #     'end_time': '2011-05-04'
    #                 # }
    #                 2: {
    #                     'start_time': '2011-04-18',
    #                     'end_time': '2011-04-25'
    #                 }
    #             }
    #         },
    #         # 'ukdale': {
    #         #   'path': 'mnt/ukdale.h5',
    #         #   'buildings': {
    #         #         1: {
    #         #               'start_time': '2013-05-01 00:00',
    #         #               'end_time': '2013-05-14 00:00'
    #         #         }
    #         #     }
    #         #   },
    #     },
    # },
    'test': {
        'datasets': {
            # 'ukdale': {
            #   'path': 'mnt/ukdale.h5',
            #   'buildings': {
            #         2: {
            #               'start_time': '2013-05-22 00:00',
            #               'end_time': '2013-08-01 00:00'
            #         }
            #     }
            #   },
            'redd': REDD_TEST_STD
        },
        # Specify evaluation metrics
        'metrics': ['accuracy', 'mae', 'f1score', 'recall', 'precision', 'nep', 'MCC']
    }
}

if __name__ == '__main__':
    API(e)
