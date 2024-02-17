from nilmtk.api import API
import warnings

warnings.filterwarnings("ignore")
from nilmtk.disaggregate import SGN, DM, DMCoral, GatedDM, DM_SDA, DM_GATE2
import nilmtk.utils as utils

import torch

USE_GPU = True
device = torch.device("cuda:0" if USE_GPU and torch.cuda.is_available() else "cpu")
print(torch.__version__, device)

torch.set_float32_matmul_precision('high')

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
        6: {
            'start_time': '2011-05-22',
            'end_time': '2011-06-13'
        },
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

UKDALE_TEST_STD = {
    'path': 'mnt/ukdale.h5',
    'buildings': {
        2: {
            'start_time': '2013-05-25',
            'end_time': '2013-09-30'
        },
    },
}

USING_DATASET = 'redd'

e = {
    # Specify power type, sample rate and disaggregated appliance
    'power': {
        # 'mains_train': ['active'],  # problem: ukdale active, redd apparent
        # 'mains_test': ['active'],
        'mains_train': ['apparent'],  # problem: ukdale active, redd apparent
        'mains_test': ['apparent'],
        'appliance': ['active']
    },
    'sample_rate': 3,
    # 'appliances': ['washing machine'],
    'app_meta': utils.APP_META[USING_DATASET],
    'appliances': ['fridge'],
    # Universally no pre-training
    'pre_trained': False,
    # Specify algorithm hyper-parameters
    'save_note': 'gated',
    'gater': SGN(
        {'n_epochs': 0, 'batch_size': 256,
         'sequence_length': 400 if USING_DATASET == 'redd' else 200,
         'appliance_length': 64 if USING_DATASET == 'redd' else 32,
         'test_only': True, 'gate_only': True,
         'note': USING_DATASET}),
    'methods': {
        "DM_GATE2": DM_GATE2(
            {'n_epochs': 400 if USING_DATASET == "redd" else 100,
             'batch_size': 64, 'sequence_length': 720, 'overlapping_step': 10,
             'note': USING_DATASET,
             'test_only': True, 'fine_tune': False, 'src_rate': 0.5, 'lr': 3e-5,
             "sampler": "ddim",
             'patience': 5 if USING_DATASET == "redd" else 3,
             "app_meta": utils.APP_META[USING_DATASET], 'filter_train': True
             }
        )
    },
    # Specify train and test data
    'train': {
        'datasets': {
            'redd': REDD_TRAIN_STD
            # 'ukdale': UKDALE_TRAIN_STD
            # 'ukdale': UKDALE_TEST_STD
        }
    },
    'test': {
        'datasets': {
            'redd': REDD_TEST_STD,
            # 'ukdale': UKDALE_TEST_STD
        },
        # Specify evaluation metrics
        'metrics': ['accuracy', 'f1score', 'mae', 'sae', 'precision', 'recall']
    }
}

if __name__ == '__main__':
    API(e)
