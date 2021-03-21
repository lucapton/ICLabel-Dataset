from time import (time, gmtime, strftime)
import h5py
import os
from shutil import rmtree
from os.path import isdir, isfile, join, basename
import cPickle as pkl
import sqlite3
from collections import OrderedDict
from copy import copy

import numpy as np
from sklearn.decomposition import PCA
import joblib
from matplotlib import pyplot as plt
import webbrowser as wb
import requests
import tqdm


class ICLabelDataset:

    """

    This class provides an easy interface to downloading, loading, organizing, and processing the ICLabel dataset.
    The ICLabel dataset is intended for training and validating electroencephalographic (EEG) independent component
    (IC) classifiers.

    It contains an unlabled training dataset, several collections of labels for small subset of the training dataset,
    and a test dataset 130 ICs where each IC was labeled by 6 experts.

    Features included:
    * Scalp topography images (32x32 pixel flattened to 740 elements after removing white-space)
    * Power spectral densities (1-100 Hz)
    * Autocorrelation functions (1 second)
    * Equivalent current dipole fits (1 and 2 dipole)
    * Hand crafted features (some new and some from previously published classifiers)

    :Example:

        icl = ICLabelDataset();
        icldata = icl.load_semi_supervised()

    """

    def __init__(self,
                 features='all',
                 label_type='all',
                 datapath='',
                 n_test_datasets=50,
                 n_val_ics=200,
                 transform='none',
                 unique=True,
                 do_pca=False,
                 combine_output=False,
                 seed=np.random.randint(0, int(1e5))):
        """
        Initialize an ICLabelDataset object.
        :param features: The types of features to return.
        :param label_type: Which ICLabels to use.
        :param datapath: Where the dataset and cache is stored.
        :param n_test_datasets: How many unlabeled datasets to include in the test set.
        :param n_val_ics: How many labeled components to transfer to the validation set.
        :param transform: The inverse log-ratio transform to use for labels and their covariances.
        :param unique: Whether or not to use ICs with the same scalp topography. Non-unique is not implemented.
        :param combine_output: determines whether output features are dictionaries or an array of combined features.
        :param seed: The seed for the pseudo random shuffle of data points.
        :return: Initialized ICLabelDataset object.
        """
        # data parameters
        self.datapath = datapath
        self.features = features
        self.n_test_datasets = n_test_datasets
        self.n_val_ics = n_val_ics
        self.transform = transform
        self.unique = unique
        if not self.unique:
            raise NotImplementedError
        self.do_pca = do_pca
        self.combine_output = combine_output
        self.label_type = label_type
        assert(label_type in ('all', 'luca', 'database'))
        self.seed = seed
        self.psd_mean = None
        self.psd_mean_var = None
        self.psd_mean_kurt = None
        self.psd_limits = None
        self.psd_var_limits = None
        self.psd_kurt_limits = None
        self.pscorr_mean = None
        self.pscorr_std = None
        self.pscorr_limits = None
        self.psd_freqs = 100

        # training feature-sets
        self.train_feature_indices = OrderedDict([
            ('ids', np.arange(2)),
            ('topo', np.arange(2, 742)),
            ('handcrafted', np.arange(742, 760)),  # one lost due to removal in load_data
            ('dipole', np.arange(760, 780)),
            ('psd', np.arange(780, 880)),
            ('psd_var', np.arange(880, 980)),
            ('psd_kurt', np.arange(980, 1080)),
            ('autocorr', np.arange(1080, 1180)),
        ])
        self.test_feature_indices = OrderedDict([
            ('ids', np.arange(3)),
            ('topo', np.arange(3, 743)),
            ('handcrafted', np.arange(743, 761)),  # one lost due to removal in load_data
            ('dipole', np.arange(761, 781)),
            ('psd', np.arange(781, 881)),
            ('psd_var', np.arange(881, 981)),
            ('psd_kurt', np.arange(981, 1081)),
            ('autocorr', np.arange(1081, 1181)),
        ])

        # reorganize features
        if self.features == 'all' or 'all' in self.features:
            self.features = self.train_feature_indices.keys()
        if isinstance(self.features, str):
            self.features = [self.features]
        if 'ids' not in self.features:
            self.features = ['ids'] + self.features

        # visualization parameters
        self.topo_ind = np.array([
                            43,
                            44,
                            45,
                            46,
                            47,
                            48,
                            49,
                            50,
                            51,
                            52,
                            72,
                            73,
                            74,
                            75,
                            76,
                            77,
                            78,
                            79,
                            80,
                            81,
                            82,
                            83,
                            84,
                            85,
                            86,
                            87,
                            103,
                            104,
                            105,
                            106,
                            107,
                            108,
                            109,
                            110,
                            111,
                            112,
                            113,
                            114,
                            115,
                            116,
                            117,
                            118,
                            119,
                            120,
                            134,
                            135,
                            136,
                            137,
                            138,
                            139,
                            140,
                            141,
                            142,
                            143,
                            144,
                            145,
                            146,
                            147,
                            148,
                            149,
                            150,
                            151,
                            152,
                            153,
                            165,
                            166,
                            167,
                            168,
                            169,
                            170,
                            171,
                            172,
                            173,
                            174,
                            175,
                            176,
                            177,
                            178,
                            179,
                            180,
                            181,
                            182,
                            183,
                            184,
                            185,
                            186,
                            196,
                            197,
                            198,
                            199,
                            200,
                            201,
                            202,
                            203,
                            204,
                            205,
                            206,
                            207,
                            208,
                            209,
                            210,
                            211,
                            212,
                            213,
                            214,
                            215,
                            216,
                            217,
                            218,
                            219,
                            227,
                            228,
                            229,
                            230,
                            231,
                            232,
                            233,
                            234,
                            235,
                            236,
                            237,
                            238,
                            239,
                            240,
                            241,
                            242,
                            243,
                            244,
                            245,
                            246,
                            247,
                            248,
                            249,
                            250,
                            251,
                            252,
                            258,
                            259,
                            260,
                            261,
                            262,
                            263,
                            264,
                            265,
                            266,
                            267,
                            268,
                            269,
                            270,
                            271,
                            272,
                            273,
                            274,
                            275,
                            276,
                            277,
                            278,
                            279,
                            280,
                            281,
                            282,
                            283,
                            284,
                            285,
                            290,
                            291,
                            292,
                            293,
                            294,
                            295,
                            296,
                            297,
                            298,
                            299,
                            300,
                            301,
                            302,
                            303,
                            304,
                            305,
                            306,
                            307,
                            308,
                            309,
                            310,
                            311,
                            312,
                            313,
                            314,
                            315,
                            316,
                            317,
                            322,
                            323,
                            324,
                            325,
                            326,
                            327,
                            328,
                            329,
                            330,
                            331,
                            332,
                            333,
                            334,
                            335,
                            336,
                            337,
                            338,
                            339,
                            340,
                            341,
                            342,
                            343,
                            344,
                            345,
                            346,
                            347,
                            348,
                            349,
                            353,
                            354,
                            355,
                            356,
                            357,
                            358,
                            359,
                            360,
                            361,
                            362,
                            363,
                            364,
                            365,
                            366,
                            367,
                            368,
                            369,
                            370,
                            371,
                            372,
                            373,
                            374,
                            375,
                            376,
                            377,
                            378,
                            379,
                            380,
                            381,
                            382,
                            385,
                            386,
                            387,
                            388,
                            389,
                            390,
                            391,
                            392,
                            393,
                            394,
                            395,
                            396,
                            397,
                            398,
                            399,
                            400,
                            401,
                            402,
                            403,
                            404,
                            405,
                            406,
                            407,
                            408,
                            409,
                            410,
                            411,
                            412,
                            413,
                            414,
                            417,
                            418,
                            419,
                            420,
                            421,
                            422,
                            423,
                            424,
                            425,
                            426,
                            427,
                            428,
                            429,
                            430,
                            431,
                            432,
                            433,
                            434,
                            435,
                            436,
                            437,
                            438,
                            439,
                            440,
                            441,
                            442,
                            443,
                            444,
                            445,
                            446,
                            449,
                            450,
                            451,
                            452,
                            453,
                            454,
                            455,
                            456,
                            457,
                            458,
                            459,
                            460,
                            461,
                            462,
                            463,
                            464,
                            465,
                            466,
                            467,
                            468,
                            469,
                            470,
                            471,
                            472,
                            473,
                            474,
                            475,
                            476,
                            477,
                            478,
                            481,
                            482,
                            483,
                            484,
                            485,
                            486,
                            487,
                            488,
                            489,
                            490,
                            491,
                            492,
                            493,
                            494,
                            495,
                            496,
                            497,
                            498,
                            499,
                            500,
                            501,
                            502,
                            503,
                            504,
                            505,
                            506,
                            507,
                            508,
                            509,
                            510,
                            513,
                            514,
                            515,
                            516,
                            517,
                            518,
                            519,
                            520,
                            521,
                            522,
                            523,
                            524,
                            525,
                            526,
                            527,
                            528,
                            529,
                            530,
                            531,
                            532,
                            533,
                            534,
                            535,
                            536,
                            537,
                            538,
                            539,
                            540,
                            541,
                            542,
                            545,
                            546,
                            547,
                            548,
                            549,
                            550,
                            551,
                            552,
                            553,
                            554,
                            555,
                            556,
                            557,
                            558,
                            559,
                            560,
                            561,
                            562,
                            563,
                            564,
                            565,
                            566,
                            567,
                            568,
                            569,
                            570,
                            571,
                            572,
                            573,
                            574,
                            577,
                            578,
                            579,
                            580,
                            581,
                            582,
                            583,
                            584,
                            585,
                            586,
                            587,
                            588,
                            589,
                            590,
                            591,
                            592,
                            593,
                            594,
                            595,
                            596,
                            597,
                            598,
                            599,
                            600,
                            601,
                            602,
                            603,
                            604,
                            605,
                            606,
                            609,
                            610,
                            611,
                            612,
                            613,
                            614,
                            615,
                            616,
                            617,
                            618,
                            619,
                            620,
                            621,
                            622,
                            623,
                            624,
                            625,
                            626,
                            627,
                            628,
                            629,
                            630,
                            631,
                            632,
                            633,
                            634,
                            635,
                            636,
                            637,
                            638,
                            641,
                            642,
                            643,
                            644,
                            645,
                            646,
                            647,
                            648,
                            649,
                            650,
                            651,
                            652,
                            653,
                            654,
                            655,
                            656,
                            657,
                            658,
                            659,
                            660,
                            661,
                            662,
                            663,
                            664,
                            665,
                            666,
                            667,
                            668,
                            669,
                            670,
                            674,
                            675,
                            676,
                            677,
                            678,
                            679,
                            680,
                            681,
                            682,
                            683,
                            684,
                            685,
                            686,
                            687,
                            688,
                            689,
                            690,
                            691,
                            692,
                            693,
                            694,
                            695,
                            696,
                            697,
                            698,
                            699,
                            700,
                            701,
                            706,
                            707,
                            708,
                            709,
                            710,
                            711,
                            712,
                            713,
                            714,
                            715,
                            716,
                            717,
                            718,
                            719,
                            720,
                            721,
                            722,
                            723,
                            724,
                            725,
                            726,
                            727,
                            728,
                            729,
                            730,
                            731,
                            732,
                            733,
                            738,
                            739,
                            740,
                            741,
                            742,
                            743,
                            744,
                            745,
                            746,
                            747,
                            748,
                            749,
                            750,
                            751,
                            752,
                            753,
                            754,
                            755,
                            756,
                            757,
                            758,
                            759,
                            760,
                            761,
                            762,
                            763,
                            764,
                            765,
                            771,
                            772,
                            773,
                            774,
                            775,
                            776,
                            777,
                            778,
                            779,
                            780,
                            781,
                            782,
                            783,
                            784,
                            785,
                            786,
                            787,
                            788,
                            789,
                            790,
                            791,
                            792,
                            793,
                            794,
                            795,
                            796,
                            804,
                            805,
                            806,
                            807,
                            808,
                            809,
                            810,
                            811,
                            812,
                            813,
                            814,
                            815,
                            816,
                            817,
                            818,
                            819,
                            820,
                            821,
                            822,
                            823,
                            824,
                            825,
                            826,
                            827,
                            837,
                            838,
                            839,
                            840,
                            841,
                            842,
                            843,
                            844,
                            845,
                            846,
                            847,
                            848,
                            849,
                            850,
                            851,
                            852,
                            853,
                            854,
                            855,
                            856,
                            857,
                            858,
                            870,
                            871,
                            872,
                            873,
                            874,
                            875,
                            876,
                            877,
                            878,
                            879,
                            880,
                            881,
                            882,
                            883,
                            884,
                            885,
                            886,
                            887,
                            888,
                            889,
                            903,
                            904,
                            905,
                            906,
                            907,
                            908,
                            909,
                            910,
                            911,
                            912,
                            913,
                            914,
                            915,
                            916,
                            917,
                            918,
                            919,
                            920,
                            936,
                            937,
                            938,
                            939,
                            940,
                            941,
                            942,
                            943,
                            944,
                            945,
                            946,
                            947,
                            948,
                            949,
                            950,
                            951,
                            971,
                            972,
                            973,
                            974,
                            975,
                            976,
                            977,
                            978,
                            979,
                            980,
                        ])
        self.psd_ind = np.arange(1, 101)
        self.max_grid_plot = 144
        self.base_url_image = 'https://labeling.ucsd.edu/images/'

        # data url
        self.base_url_download = 'https://labeling.ucsd.edu/download/'
        self.feature_train_zip_url = self.base_url_download + 'features.zip'
        self.feature_train_zip_parts_url = self.base_url_download + 'features{:02d}.zip'
        self.num_feature_train_files = 25
        self.feature_train_urls = [
            self.base_url_download + 'features_0D1D2D.mat',
            self.base_url_download + 'features_PSD_med_var_kurt.mat',
            self.base_url_download + 'features_AutoCorr.mat',
            self.base_url_download + 'features_ICAChanlocs.mat',
            self.base_url_download + 'features_MI.mat',
        ]
        self.label_train_urls = [
            self.base_url_download + 'ICLabels_expert.pkl',
            self.base_url_download + 'ICLabels_onlyluca.pkl',
        ]
        self.feature_test_url = self.base_url_download + 'features_testset_full.mat'
        self.label_test_url = self.base_url_download + 'ICLabels_test.pkl'
        self.db_url = self.base_url_download + 'anonymized_database.sqlite'
        self.cls_url = self.base_url_download + 'other_classifiers.mat'

    # util

    @staticmethod
    def __load_matlab_cellstr(f, var_name=''):
        var = []
        if var_name:
            for column in f[var_name]:
                row_data = []
                for row_number in range(len(column)):
                    row_data.append(''.join(map(unichr, f[column[row_number]][:])))
                var.append(row_data)
        return [str(x)[3:-2] for x in var]

    @staticmethod
    def __match_indices(*indices):
        """ Match sets of multidimensional ids/indices when there is a 1-1 relationtionship """

        # find matching indices
        index = np.concatenate(indices)  # array of values
        _, duplicates, counts = np.unique(index, return_inverse=True, return_counts=True, axis=0)
        duplicates = np.split(duplicates, np.cumsum([x.shape[0] for x in indices[:-1]]), 0)  # list of vectors of ints
        sufficient_counts = np.where(counts == len(indices))[0]  # vector of ints
        matching_indices = [np.where(np.in1d(x, sufficient_counts))[0] for x in duplicates]  # list of vectors of ints
        indices = [y[x] for x, y in zip(matching_indices, indices)]  # list of arrays of values

        # organize to match first index array
        try:
            sort_inds = [np.lexsort(np.fliplr(x).T) for x in indices]
        except ValueError:
            sort_inds = [np.argsort(x) for x in indices]
        out = np.array([x[y[sort_inds[0]]] for x, y in zip(matching_indices, sort_inds)])

        return out

    # data access

    def load_data(self):
        """
        Load the ICL dataset in an unprocessed form.
        Follows the settings provided during initializations
        :return: Dictionary of unprocessed but matched feature-sets and labels.
        """
        start = time()

        # organize info
        if self.transform in (None, 'none'):
            if self.label_type == 'all':
                file_name = 'ICLabels_expert.pkl'
            elif self.label_type == 'luca':
                file_name = 'ICLabels_onlyluca.pkl'
            processed_file_name = 'processed_dataset'
        if self.unique:
            processed_file_name += '_unique'
        if self.label_type == 'all':
            processed_file_name += '_all'
            self.check_for_download('train_labels')
        elif self.label_type == 'luca':
            processed_file_name += '_luca'
            self.check_for_download('train_labels')
        elif self.label_type == 'database':
            processed_file_name += '_database'
            self.check_for_download('database')
        processed_file_name += '.pkl'

        # load processed data file if it exists
        if isfile(join(self.datapath, 'cache', processed_file_name)):
            dataset = joblib.load(join(self.datapath, 'cache', processed_file_name))

        # if not, create it
        else:
            # load features
            features = []
            feature_labels = []
            print('Loading full dataset...')

            self.check_for_download('train_features')
            # topo maps, old psd, dipole, and handcrafted
            with h5py.File(join(self.datapath, 'features', 'features_0D1D2D.mat'), 'r') as f:
                print('Loading 0D1D2D features...')
                features.append(np.asarray(f['features']).T)
                feature_labels.append(self.__load_matlab_cellstr(f, 'labels'))
            # new psd
            with h5py.File(join(self.datapath, 'features', 'features_PSD_med_var_kurt.mat'), 'r') as f:
                print('Loading PSD features...')
                features.append(list())
                for element in f['features_out'][0]:
                    data = np.array(f[element]).T
                    # if no data, skip
                    if data.ndim == 1 or data.dtype != np.float64:
                        continue
                    nyquist = (data.shape[1] - 2) / 3
                    nfreq = 100
                    # if more than nfreqs, remove extra
                    if nyquist > nfreq:
                        data = data[:, np.concatenate((range(2 + nfreq),
                                                      range(2 + nyquist, 2 + nyquist + nfreq),
                                                      range(2 + 2*nyquist, 2 + 2*nyquist + nfreq)))]
                    # if less than nfreqs, repeat last frequency value
                    elif nyquist < nfreq:
                        data = data[:, np.concatenate((range(2 + nyquist),
                                                       np.repeat(1 + nyquist, nfreq - nyquist),
                                                       range(2 + nyquist, 2 + 2*nyquist),
                                                       np.repeat(1 + 2*nyquist, nfreq - nyquist),
                                                       range(2 + 2*nyquist, 2 + 3*nyquist),
                                                       np.repeat(1 + 3*nyquist, nfreq - nyquist))
                                                      ).astype(int)]

                    features[-1].append(data)
                features[-1] = np.concatenate(features[-1], axis=0)
                feature_labels.append(['ID_set', 'ID_ic'] + ['psd_median']*nfreq
                                      + ['psd_var']*nfreq + ['psd_kurt']*nfreq)
            # autocorrelation
            with h5py.File(join(self.datapath, 'features', 'features_AutoCorr.mat'), 'r') as f:
                print('Loading AutoCorr features...')
                features.append(list())
                for element in f['features_out'][0]:
                    data = np.array(f[element]).T
                    if data.size > 2 and data.shape[1] == 102 and not len(data.dtype):
                        features[-1].append(data)
                features[-1] = np.concatenate(features[-1], axis=0)
                feature_labels.append(self.__load_matlab_cellstr(f, 'feature_labels')[:2] + ['Autocorr'] * 100)

            # find topomap duplicates
            print('Finding topo duplicates...')
            _, duplicate_order = np.unique(features[0][:, 2:742].astype(np.float32), return_inverse=True, axis=0)
            do_sortind = np.argsort(duplicate_order)
            do_sorted = duplicate_order[do_sortind]
            do_indices = np.where(np.diff(np.concatenate(([-1], do_sorted))))[0]
            group2indices = [do_sortind[do_indices[x]:do_indices[x + 1]] for x in range(0, duplicate_order.max())]
            del _

            # load labels
            if self.label_type == 'database':
                # load data from database
                conn = sqlite3.connect(join(self.datapath, 'labels', 'database.sqlite'))
                c = conn.cursor()
                dblabels = c.execute('SELECT * FROM labels '
                                     'INNER JOIN images ON labels.image_id = images.id '
                                     'WHERE user_id IN '
                                     '(SELECT user_id FROM labels '
                                     'GROUP BY user_id '
                                     'HAVING COUNT(*) >= 30)'
                                     ).fetchall()
                conn.close()
                # reformat as list of ndarrays
                dblabels = [(x[1], np.array(x[15:17]), np.array(x[3:11])) for x in dblabels]
                dblabels = [np.stack(x) for x in zip(*dblabels)]
                # organize labels by image
                udb = np.unique(dblabels[1], return_inverse=True, axis=0)
                dblabels = [(dblabels[0][y], dblabels[1][y][0], dblabels[2][y])
                            for y in (udb[1] == x for x in range(len(udb[0])))]
                label_index = np.stack((x[1] for x in dblabels))

            elif self.label_type == 'luca':
                # load data from database
                conn = sqlite3.connect(join(self.datapath, 'labels', 'database.sqlite'))
                c = conn.cursor()
                dblabelsluca = c.execute('SELECT * FROM labels '
                                         'INNER JOIN images ON labels.image_id = images.id '
                                         'WHERE user_id = 1').fetchall()
                conn.close()
                # remove low-confidence labels
                dblabelsluca = [x for x in dblabelsluca if x[10] == 0]
                # reformat as ndarray
                labels = np.array([x[3:10] for x in dblabelsluca]).astype(np.float32)
                labels /= labels.sum(1, keepdims=True)
                labels = [labels]
                label_index = np.array([x[15:17] for x in dblabelsluca])
                transforms = ['none']

            else:
                # load labels from files
                with open(join(self.datapath, 'labels', file_name), 'rb') as f:
                    print('Loading labels...')
                    data = pkl.load(f)
                    if 'transform' in data.keys():
                        transforms = data['transform']
                    else:
                        transforms = ['none']
                    labels = data['labels']
                    if isinstance(labels, np.ndarray):
                        labels = [labels]
                    if 'labels_cov' in data.keys():
                        label_cov = data['labels_cov']
                    label_index = np.stack((data['instance_set_numbers'], data['instance_ic_numbers'])).T
                del data

            # match components and labels
            print('Matching components and labels...')
            temp = self.__match_indices(label_index.astype(np.int), features[0][:, :2].astype(np.int))
            label2component = dict(zip(*temp))
            del temp
            # match feature-sets
            print('Matching features...')
            feature_inds = self.__match_indices(*[x[:, :2].astype(np.int) for x in features])

            # check which labels are not kept
            print('Rearanging components and labels...')
            kept_labels = [x for x, y in label2component.iteritems() if y in feature_inds[0]]
            dropped_labels = [x for x, y in label2component.iteritems() if y not in feature_inds[0]]

            # for each label, pick a new component that is kept (if any)
            ind_n_data_points = [x for x, y in enumerate(feature_labels[0]) if y == 'number of data points'][0]
            for ind in dropped_labels:
                group = duplicate_order[label2component[ind]]
                candidate_components = np.intersect1d(group2indices[group], feature_inds[0])
                # if more than one choice, pick the one from the dataset with the most samples unless one from this
                #   group has already been found
                if len(candidate_components) >= 1:
                    if len(candidate_components) == 1:
                        new_index = features[0][candidate_components, :2]
                    else:
                        new_index = features[0][candidate_components[features[0][candidate_components,
                                                                                 ind_n_data_points].argmax()], :2]
                    if not (new_index == label_index[dropped_labels]).all(1).any() \
                            and not any([(x == label_index[kept_labels]).all(1).any()
                                         for x in features[0][candidate_components, :2]]):
                        label_index[ind] = new_index
            del label2component, kept_labels, dropped_labels, duplicate_order

            # feature labels (change with features)
            psd_lims = np.where(np.char.startswith(feature_labels[0], 'psd'))[0][[0, -1]]
            feature_labels = np.concatenate((feature_labels[0][:psd_lims[0]],
                                             feature_labels[0][psd_lims[1] + 1:],
                                             feature_labels[1][2:],
                                             feature_labels[2][2:]))

            # combine features, keeping only components with all features
            print('Combining feature-sets...')

            def index_features(data, new_index):
                return np.concatenate((data[0][feature_inds[0][new_index], :psd_lims[0]].astype(np.float32),
                                       data[0][feature_inds[0][new_index], psd_lims[1] + 1:].astype(np.float32),
                                       data[1][feature_inds[1][new_index], 2:].astype(np.float32),
                                       data[2][feature_inds[2][new_index], 2:].astype(np.float32)),
                                      axis=1)

            # rematch with labels
            print('Rematching components and labels...')
            ind_labeled_labels, ind_labeled_features = self.__match_indices(
                label_index.astype(np.int),features[0][feature_inds[0], :2].astype(np.int))
            del label_index

            # find topomap duplicates
            _, duplicate_order = np.unique(features[0][feature_inds[0], 2:742].astype(np.float32), return_inverse=True,
                                           axis=0)
            do_sortind = np.argsort(duplicate_order)
            do_sorted = duplicate_order[do_sortind]
            do_indices = np.where(np.diff(np.concatenate(([-1], do_sorted))))[0]
            group2indices = [do_sortind[do_indices[x]:do_indices[x + 1]] for x in range(0, duplicate_order.max())]

            # aggregate data
            dataset = dict()
            try:
                dataset['transform'] = transforms
            except UnboundLocalError:
                pass
            if self.label_type == 'database':
                dataset['labeled_labels'] = [dblabels[x] for x in np.where(ind_labeled_labels)[0]]
            else:
                dataset['labeled_labels'] = [x[ind_labeled_labels, :] for x in labels]
                if 'label_cov' in locals():
                    dataset['labeled_label_covariances'] = [x[ind_labeled_labels, :].astype(np.float32)
                                                            for x in label_cov]
            dataset['labeled_features'] = index_features(features, ind_labeled_features)

            # find equivalent datasets with most samples
            unlabeled_groups = [x for it, x in enumerate(group2indices)
                                if not np.intersect1d(x, ind_labeled_features).size]
            ndata = features[0][feature_inds[0]][:, ind_n_data_points]
            ind_unique_unlabled = [x[ndata[x].argmax()] for x in unlabeled_groups]
            dataset['unlabeled_features'] = index_features(features, ind_unique_unlabled)

            # close h5py pscorr file and clean workspace
            del features, group2indices
            try:
                del labels
            except NameError:
                del dblabels
            if 'label_cov' in locals():
                del label_cov

            # remove inf columns
            print('Cleaning data of infs...')
            inf_col = [ind for ind, x in enumerate(feature_labels) if x == 'SASICA snr'][0]
            feature_labels = np.delete(feature_labels, inf_col)
            dataset['unlabeled_features'] = np.delete(dataset['unlabeled_features'], inf_col, axis=1)
            dataset['labeled_features'] = np.delete(dataset['labeled_features'], inf_col, axis=1)

            # remove nan total_rows
            print('Cleaning data of nans...')
            # unlabeled
            unlabeled_not_nan_inf_index = np.logical_not(
                np.logical_or(np.isnan(dataset['unlabeled_features']).any(axis=1),
                              np.isinf(dataset['unlabeled_features']).any(axis=1)))
            dataset['unlabeled_features'] = \
                dataset['unlabeled_features'][unlabeled_not_nan_inf_index, :]
            # labeled
            labeled_not_nan_inf_index = np.logical_not(np.logical_or(np.isnan(dataset['labeled_features']).any(axis=1),
                                                                     np.isinf(dataset['labeled_features']).any(axis=1)))
            dataset['labeled_features'] = dataset['labeled_features'][labeled_not_nan_inf_index, :]
            if self.label_type == 'database':
                dataset['labeled_labels'] = [dataset['labeled_labels'][x]
                                             for x in np.where(labeled_not_nan_inf_index)[0]]
            else:
                dataset['labeled_labels'] = [x[labeled_not_nan_inf_index, :] for x in dataset['labeled_labels']]
                if 'labeled_label_covariances' in dataset.keys():
                    dataset['labeled_label_covariances'] = [x[labeled_not_nan_inf_index, :, :]
                                                            for x in dataset['labeled_label_covariances']]
            if not self.unique:
                dataset['unlabeled_duplicates'] = dataset['unlabeled_duplicates'][unlabeled_not_nan_inf_index]
                dataset['labeled_duplicates'] = dataset['labeled_duplicates'][labeled_not_nan_inf_index]

            # save feature labels (names, e.g. psd)
            dataset['feature_labels'] = feature_labels

            # save the results
            print('Saving aggregated dataset...')
            joblib.dump(dataset, join(self.datapath, 'cache', processed_file_name), 0)

        # print time
        total = time() - start
        print('Time to load: ' + strftime("%H:%M:%S", gmtime(total)) +
              ':' + np.mod(total, 1).astype(str)[2:5] + '\t(HH:MM:SS:sss)')

        return dataset

    def load_semi_supervised(self):
        """
        Load the ICL dataset where only a fraction of data points are labeled.
        Follows the settings provided during initializations
        :return: (train set unlabeled, train set labeled, sample test set (unlabeled), validation set (labeled),
            output labels)
        """

        rng = np.random.RandomState(seed=self.seed)
        start = time()

        # get data
        icl = self.load_data()

        # copy full dataset
        icl['unlabeled_features'] = \
            OrderedDict([(key, icl['unlabeled_features'][:, ind]) for key, ind
                         in self.train_feature_indices.iteritems() if key in self.features])
        icl['labeled_features'] = \
            OrderedDict([(key, icl['labeled_features'][:, ind]) for key, ind
                         in self.train_feature_indices.iteritems() if key in self.features])

        # set ids to int
        icl['unlabeled_features']['ids'] = icl['unlabeled_features']['ids'].astype(int)
        icl['labeled_features']['ids'] = icl['labeled_features']['ids'].astype(int)

        # decide how to split into train / validation / test
        # validation set of random labeled components for overfitting / convergence estimation
        try:
            valid_ind = rng.choice(icl['labeled_features']['ids'].shape[0], size=100, replace=False)
        except:
            valid_ind = rng.choice(icl['labeled_features']['ids'].shape[0], size=100, replace=True)
        # random unlabeled datasets for manual analysis
        test_datasets = rng.choice(np.unique(icl['unlabeled_features']['ids'][:, 0]),
                                   size=self.n_test_datasets, replace=False)
        test_ind = np.where(np.array([x == icl['unlabeled_features']['ids'][:, 0] for x in test_datasets]).any(0))[0]

        # normalize other features
        if 'topo' in self.features:
            print('Normalizing topo features...')
            icl['unlabeled_features']['topo'], pca = self.normalize_topo_features(icl['unlabeled_features']['topo'])
            icl['labeled_features']['topo'] = self.normalize_topo_features(icl['labeled_features']['topo'], pca)[0]

        # normalize psd features
        if 'psd' in self.features:
            print('Normalizing psd features...')
            icl['unlabeled_features']['psd'] = self.normalize_psd_features(icl['unlabeled_features']['psd'])
            icl['labeled_features']['psd'] = self.normalize_psd_features(icl['labeled_features']['psd'])

        # normalize psd_var features
        if 'psd_var' in self.features:
            print('Normalizing psd_var features...')
            icl['unlabeled_features']['psd_var'] = self.normalize_psd_features(icl['unlabeled_features']['psd_var'])
            icl['labeled_features']['psd_var'] = self.normalize_psd_features(icl['labeled_features']['psd_var'])

        # normalize psd_kurt features
        if 'psd_kurt' in self.features:
            print('Normalizing psd_kurt features...')
            icl['unlabeled_features']['psd_kurt'] = self.normalize_psd_features(icl['unlabeled_features']['psd_kurt'])
            icl['labeled_features']['psd_kurt'] = self.normalize_psd_features(icl['labeled_features']['psd_kurt'])

        # normalize psd_kurt features
        if 'autocorr' in self.features:
            print('Normalizing autocorr features...')
            icl['unlabeled_features']['autocorr'] = self.normalize_autocorr_features(
                icl['unlabeled_features']['autocorr'])
            icl['labeled_features']['autocorr'] = self.normalize_autocorr_features(icl['labeled_features']['autocorr'])

        # normalize dipole features
        if 'dipole' in self.features:
            print('Normalizing dipole features...')
            icl['unlabeled_features']['dipole'] = self.normalize_dipole_features(icl['unlabeled_features']['dipole'])
            icl['labeled_features']['dipole'] = self.normalize_dipole_features(icl['labeled_features']['dipole'])

        # normalize handcrafted features
        if 'handcrafted' in self.features:
            print('Normalizing hand-crafted features...')
            icl['unlabeled_features']['handcrafted'] = \
                self.normalize_handcrafted_features(icl['unlabeled_features']['handcrafted'],
                                               icl['unlabeled_features']['ids'][:, 1])
            icl['labeled_features']['handcrafted'] = self.normalize_handcrafted_features(
                icl['labeled_features']['handcrafted'], icl['labeled_features']['ids'][:, 1])

        # normalize mi features
        if 'mi' in self.features:
            print('Normalizing mi features...')
            icl['unlabeled_features']['mi'] = self.normalize_mi_features(icl['unlabeled_features']['mi'])
            icl['labeled_features']['mi'] = self.normalize_mi_features(icl['labeled_features']['mi'])

        # recast labels
        if self.label_type == 'database':
            pass
        else:
            icl['labeled_labels'] = [x.astype(np.float32) for x in icl['labeled_labels']]
            if 'labeled_label_covariances' in icl.keys():
                icl['labeled_label_covariances'] = [x.astype(np.float32) for x in icl['labeled_label_covariances']]

        # separate data into train, validation, and test sets
        print('Splitting and shuffling data...')
        #  unlabeled training set
        ind = rng.permutation(np.setdiff1d(range(icl['unlabeled_features']['ids'].shape[0]), test_ind))
        x_u = OrderedDict([(key, val[ind]) for key, val in icl['unlabeled_features'].iteritems()])
        y_u = None
        # labeled training set
        ind = rng.permutation(np.setdiff1d(range(icl['labeled_features']['ids'].shape[0]), valid_ind))
        x_l = OrderedDict([(key, val[ind]) for key, val in icl['labeled_features'].iteritems()])
        if self.label_type == 'database':
            print(icl['labeled_labels'][0])
            y_l = [icl['labeled_labels'][x] for x in ind]
        else:
            y_l = [x[ind] for x in icl['labeled_labels']]
            if 'labeled_label_covariances' in icl.keys():
                c_l = [x[ind] for x in icl['labeled_label_covariances']]
        # validation set.
        rng.shuffle(valid_ind)
        x_v = OrderedDict([(key, val[valid_ind]) for key, val in icl['labeled_features'].iteritems()])
        if self.label_type == 'database':
            y_v = [icl['labeled_labels'][x] for x in valid_ind]
        else:
            y_v = [x[valid_ind] for x in icl['labeled_labels']]
            if 'labeled_label_covariances' in icl.keys():
                c_v = [x[valid_ind] for x in icl['labeled_label_covariances']]
        # unlabeled test set.
        rng.shuffle(test_ind)
        x_t = OrderedDict([(key, val[test_ind]) for key, val in icl['unlabeled_features'].iteritems()])
        y_t = None

        train_u = (x_u, y_u)
        if 'labeled_label_covariances' in icl.keys():
            train_l = (x_l, y_l, c_l)
        else:
            train_l = (x_l, y_l)
        test = (x_t, y_t)
        if 'labeled_label_covariances' in icl.keys():
            val = (x_v, y_v, c_v)
        else:
            val = (x_v, y_v)

        # print time
        total = time() - start
        print('Time to load: ' + strftime("%H:%M:%S", gmtime(total)) +
              ':' + np.mod(total, 1).astype(str)[2:5] + '\t(HH:MM:SS:sss)')

        return train_u, train_l, test, val, \
            ('train_unlabeled', 'train_labeled', 'test', 'validation', 'labels')

    def load_test_data(self, process_features=True):
        """
        Load the ICL test dataset used in the publication.
        Follows the settings provided during initializations.
        :param process_features: Whether to preprocess/normalize features.
        :return: (features, labels, channel_features)
        """

        # check for files and download if missing
        self.check_for_download(('test_labels', 'test_features'))

        # load features
        with h5py.File(join(self.datapath, 'features', 'features_testset_full.mat'), 'r') as f:
            features = np.asarray(f['features']).T
            feature_labels = self.__load_matlab_cellstr(f, 'feature_label')
            channel_features = []
            for dataset in f['channel_features'].value.flatten():
                # expand
                dataset = f[dataset].value.flatten()
                # expand and format
                id = f[dataset[0]].value.flatten()
                chans = [''.join(map(unichr, f[x].value.flatten())) for x in f[dataset[1]].value.flatten()]
                icamat = f[dataset[2]].value.T
                # append
                channel_features.append([id, chans, icamat[:, :3], icamat[:, 3:]])

        # load labels
        with open(join(self.datapath, 'labels', 'ICLabels_test.pkl'), 'rb') as f:
            labels = pkl.load(f)

        # match features and labels
        _, _, ind = np.intersect1d(labels['instance_id'], labels['instance_number'], return_indices=True)
        label_id = np.stack((labels['instance_study_numbers'][ind],
                             labels['instance_set_numbers'][ind],
                             labels['instance_ic_numbers'][ind]), axis=1)
        feature_id = features[:, :3].astype(int)
        match = self.__match_indices(label_id, feature_id)
        features = features[match[1, :][match[0, :]], :]

        # remove inf columns
        print('Cleaning data of infs...')
        inf_col = [ind for ind, x in enumerate(feature_labels) if x == 'SASICA snr'][0]
        feature_labels = np.delete(feature_labels, inf_col)
        features = np.delete(features, inf_col, axis=1)

        # convert to ordered dict
        features = \
            OrderedDict([(key, features[:, ind]) for key, ind
                         in self.test_feature_indices.iteritems() if key in self.features])

        # process features
        if process_features:

            # normalize other features
            if 'topo' in self.features:
                print('Normalizing topo features...')
                features['topo'] = self.normalize_topo_features(features['topo'])

            # normalize psd features
            if 'psd' in self.features:
                print('Normalizing psd features...')
                features['psd'] = self.normalize_psd_features(features['psd'])

            # normalize psd_var features
            if 'psd_var' in self.features:
                print('Normalizing psd_var features...')
                features['psd_var'] = self.normalize_psd_features(features['psd_var'])

            # normalize psd_kurt features
            if 'psd_kurt' in self.features:
                print('Normalizing psd_kurt features...')
                features['psd_kurt'] = self.normalize_psd_features(features['psd_kurt'])

            # normalize psd_kurt features
            if 'autocorr' in self.features:
                print('Normalizing autocorr features...')
                features['autocorr'] = self.normalize_autocorr_features(features['autocorr'])

            # normalize dipole features
            if 'dipole' in self.features:
                print('Normalizing dipole features...')
                features['dipole'] = self.normalize_dipole_features(features['dipole'])

            # normalize handcrafted features
            if 'handcrafted' in self.features:
                print('Normalizing hand-crafted features...')
                features['handcrafted'] = self.normalize_handcrafted_features(features['handcrafted'],
                                                                              features['ids'][:, 1])

        return features, labels, channel_features

    def load_channel_features(self):
        # load features
        with h5py.File(join(self.datapath, 'features', 'features_ICAChanlocs.mat'), 'r') as f:
            ids, chans, xyz, icamats = [], [], [], []
            for dataset in f['features_out'].value.flatten():
                # expand
                dataset = f[dataset].value.flatten()
                if np.array_equal(dataset, np.zeros(2)):
                    continue
                # expand and format
                ids.append(f[dataset[0]].value.flatten())
                chans.append([''.join(map(unichr, f[x].value.flatten())) for x in f[dataset[1]].value.flatten()])
                icamat = f[dataset[2]].value.T
                xyz.append(icamat[:, :3])
                icamats.append(icamat[:, 3:])

        return ids, chans, xyz, icamats

    def load_classifications(self, n_cls, ids=None):
        """
        Load classification of the ICLabel training set by several published and publicly available IC classifiers.
        Classifiers included are MARA, ADJUST, FASTER, IC_MARC, and EyeCatch. MARA, and FASTER are only included in
        the 2 class case. ADJUST is also included in the 3-class case. IC_MARC and EyeCatch are included in all
        cases. Note that EyeCatch only has two classes (Eye and Not-Eye) but does not follow the patter of label
        conflation used for the other classifiers as it has not Brain IC class.
        :param n_cls: How many IC classes to consider. Must be 2, 3, or 5.
        :param ids: If only a subset of ICs are desired, the relevant IC IDs may be passed here as an (n by 2) ndarray.
        :return: Dictionary of classifications separated by classifier.
        """
        # check inputs
        assert n_cls in (2, 3, 5), 'n_cls must be 2, 3, or 5'

        # load raw classifications
        raw = self._load_classifications(ids)

        # format and limit to number of desired classes
        #   2: brain, other
        #   3: brain, eye, other
        #   5: brain, muscle, eye, heart, other
        # exception for eye_catch which is always [eye] where eye >= 0.93 is the threshold for detection
        classifications = {}
        for cls, lab in raw.iteritems():
            if cls == 'adjust':
                if n_cls == 2:
                    non_brain = raw[cls].max(1, keepdims=True)
                    classifications[cls] = np.concatenate((1 - non_brain, non_brain), 1)
                elif n_cls == 3:
                    brain = 1 - raw[cls].max(1, keepdims=True)
                    eye = raw[cls][:, :-1].max(1, keepdims=True)
                    other = raw[cls][:, -1:]
                    classifications[cls] = np.concatenate((brain, eye, other), 1)
            elif cls == 'mara':
                if n_cls == 2:
                    classifications[cls] = np.concatenate((1 - raw[cls], raw[cls]), 1)
            elif cls == 'faster':
                if n_cls == 2:
                    classifications[cls] = np.concatenate((1 - raw[cls], raw[cls]), 1)
            elif cls == 'ic_marc': # ['blink', 'neural', 'heart', 'lat. eye', 'muscle', 'mixed']
                brain = raw[cls][:, 1:2]
                if n_cls == 2:
                    classifications[cls] = np.concatenate((brain, 1 - brain), 1)
                elif n_cls == 3:
                    eye = raw[cls][:, [0, 3]].sum(1, keepdims=True)
                    other = raw[cls][:,  [2, 4, 5]].sum(1, keepdims=True)
                    classifications[cls] = np.concatenate((brain, eye, other), 1)
                elif n_cls == 5:
                    muscle = raw[cls][:, 4:5]
                    eye = raw[cls][:, [0, 3]].sum(1, keepdims=True)
                    heart = raw[cls][:, 2:3]
                    other = raw[cls][:,  5:]
                    classifications[cls] = np.concatenate((brain, muscle, eye, heart, other), 1)
            elif cls == 'eye_catch':
                classifications[cls] = raw[cls]
            else:
                raise UserWarning('Unknown classifier: {}'.format(cls))

        # return
        return classifications

    def _load_classifications(self, ids=None):

        # check for files and download if missing
        self.check_for_download('classifications')

        # load classifications
        classifications = {}
        with h5py.File(join(self.datapath, 'other', 'other_classifiers.mat'), 'r') as f:
            print('Loading classifications...')
            for cls, lab in f.iteritems():
                classifications[cls] = lab[:].T

        # match to given ids
        if ids is not None:
            for cls, lab in classifications.iteritems():
                _, ind_id, ind_lab = np.intersect1d((ids * [100, 1]).sum(1), (lab[:, :2].astype(int) * [100, 1]).sum(1),
                                                    return_indices=True)
                classifications[cls] = np.empty((ids.shape[0], lab.shape[1] - 2))
                classifications[cls][:] = np.nan
                classifications[cls][ind_id] = lab[ind_lab, 2:]

        return classifications

    def generate_cache(self, refresh=False):
        """
        Generate all possible training set cache files to speed up later requests.
        :param refresh: If true, deletes previous cache files. Otherwise only missing cache files will be generated.
        """

        if refresh:
            rmtree(join(self.datapath, 'cache'))
            os.mkdir(join(self.datapath, 'cache'))

        urexpert = copy(self.label_type)
        for label_type in ('luca', 'all', 'database'):
            self.label_type = label_type
            self.load_data()
        self.label_type = urexpert

    @staticmethod
    def _download(url, filename, attempts=3):
        chunk_size = 256 * 1024

        for _ in range(attempts):
            try:
            # open connection to server
                with requests.get(url, stream=True) as r:
                    r.raise_for_status()
                    total_size_in_bytes = int(r.headers.get('content-length', 0))
                    # check if file is already downloaded
                    if os.path.exists(filename) and os.stat(filename).st_size == total_size_in_bytes:
                        print("File already downloaded. Skipping.")
                        return
                    # set up progress bar
                    with tqdm.tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True) as progress_bar:
                        # open file for writing
                        with open(filename, 'wb') as f:
                            for chunk in r.iter_content(chunk_size=chunk_size):
                                progress_bar.update(len(chunk))
                                f.write(chunk)

                        # check that file downloaded completely
                        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                            raise requests.exceptions.RequestException("Incomplete download.")
                        else:
                            break
            except requests.exceptions.RequestException:
                pass
        else:
            # all attempts failed
            raise requests.exceptions.RequestException("Download failed.")

    def download_trainset_cllabels(self):
        """
        Download labels for the ICLabel training set.
        """
        print('Downloading individual ICLabel training set CL label files...')
        folder = 'labels'
        if not isdir(join(self.datapath, folder)):
            os.mkdir(join(self.datapath, folder))
        for it, url in enumerate(self.label_train_urls):
            print('Downloading label file {} of {}...'.format(it, len(self.label_train_urls)))
            self._download(url, join(self.datapath, folder, basename(url)))

    def download_trainset_features(self):
        """
        Download features for the ICLabel training set.
        """
        folder = 'features'
        base_filename = join(self.datapath, folder, 'features')
        n_files = 25

        # check if files have already been downloaded
        for it, url in enumerate(self.feature_train_urls):
            if not isfile(join(self.datapath, folder, basename(url))):
                break
        else:
            print('Feature files already downloaded.')
            return
        print('Caution: this download is approximately 25GB and requires twice that space on your drive for unzipping!')

        print('Downloading zipped ICLabel training set features...')
        if not isdir(join(self.datapath, folder)):
            os.mkdir(join(self.datapath, folder))
        for it in range(n_files):
            print('Downloading file part {} of {}...'.format(it + 1, n_files))
            zip_name = base_filename + '{:02d}.zip'.format(it)
            self._download(self.feature_train_zip_parts_url.format(it), zip_name)

        print('Combining file parts...')
        with open(base_filename + '.zip', 'wb') as f:
            for it in range(n_files):
                with open(base_filename + '{:02d}.zip'.format(it), 'rb') as f_part:
                    f.write(f_part.read())
        for it in range(n_files):
            os.remove(base_filename + '{:02d}.zip'.format(it))

        print('Extracting zipped ICLabel training set features...')
        from zipfile import ZipFile
        with ZipFile(base_filename + '.zip') as myzip:
            myzip.extractall(path=join(self.datapath, folder))
        print('Deleting zip archive...')
        os.remove(base_filename + '.zip')

    def download_testset_cllabels(self):
        """
        Download labels for the ICLabel test set.
        """
        print('Downloading ICLabel test set CL label files...')
        folder = 'labels'
        if not isdir(join(self.datapath, folder)):
            os.mkdir(join(self.datapath, folder))
        self._download(self.label_test_url, join(self.datapath, folder, 'ICLabels_test.pkl'))

    def download_testset_features(self):
        """
        Download features for the ICLabel test set.
        """
        print('Downloading ICLabel test set features...')
        folder = 'features'
        if not isdir(join(self.datapath, folder)):
            os.mkdir(join(self.datapath, folder))
        self._download(self.feature_test_url, join(self.datapath, folder, 'features_testset_full.mat'))

    def download_database(self):
        """
        Download anonymized ICLabel website database.
        """
        print('Downloading anonymized ICLabel website database...')
        folder = 'labels'
        if not isdir(join(self.datapath, folder)):
            os.mkdir(join(self.datapath, folder))
        self._download(self.db_url, join(self.datapath, folder, 'database.sqlite'))

    def download_icclassifications(self):
        """
        Download precalculated classification for several publicly available IC classifiers.
        """
        print('Downloading classifications for some publicly available classifiers...')
        folder = 'other'
        if not isdir(join(self.datapath, folder)):
            os.mkdir(join(self.datapath, folder))
        self._download(self.cls_url, join(self.datapath, folder, 'other_classifiers.mat'))

    def check_for_download(self, data_type):
        """
        Check if something has been downloaded and, if not, get it.
        :param data_type: What data to check for. Can be: train_labels, train_features, test_labels, test_features,
            database, and/or 'classifications'.
        """

        if not isinstance(data_type, (tuple, list)):
            data_type = [data_type]

        for val in data_type:
            if val == 'train_labels':
                for it, url in enumerate(self.label_train_urls):
                    if not isfile(join(self.datapath, 'labels', basename(url))):
                        self.download_trainset_cllabels()
            elif val == 'train_features':
                for it, url in enumerate(self.feature_train_urls):
                    assert isfile(join(self.datapath, 'features', basename(url))), \
                        'Missing training feature file "' + basename(url) + '" and possibly others. ' \
                        'It is a large download which you may accomplish through calling the method ' \
                        '"download_trainset_features()".'
            elif val == 'test_labels':
                if not isfile(join(self.datapath, 'labels', 'ICLabels_test.pkl')):
                    self.download_testset_cllabels()
            elif val == 'test_features':
                if not isfile(join(self.datapath, 'features', 'features_testset_full.mat')):
                    self.download_testset_features()
            elif val == 'database':
                if not isfile(join(self.datapath, 'labels', 'database.sqlite')):
                    self.download_database()
            elif val == 'classifications':
                if not isfile(join(self.datapath, 'other', 'other_classifiers.mat')):
                    self.download_icclassifications()


    # data normalization

    @staticmethod
    def _clip_and_rescale(vec, min, max):
        return (np.clip(vec, min, max) - min) * 2. / (max - min) - 1

    @staticmethod
    def _unscale(vec, min, max):
        return (vec + 1) * (max-min) / 2 + min

    @staticmethod
    def normalize_dipole_features(data):
        """
        Normalize dipole features.
        :param data: dipole features
        :return: normalized dipole features
        """

        # indices
        ind_dipole_pos = np.array([1, 2, 3, 8, 9, 10, 14, 15, 16])
        ind_dipole1_mom = np.array([4, 5, 6])
        ind_dipole2_mom = np.array([11, 12, 13, 17, 18, 19])
        ind_rv = np.array([0, 7])

        # normalize dipole positions
        data[:, ind_dipole_pos] /= 100
        # clip dipole position
        max_dist = 1.5
        data[:, ind_dipole_pos] = np.clip(data[:, ind_dipole_pos], -max_dist, max_dist) / max_dist
        # normalize single dipole moments
        data[:, ind_dipole1_mom] /= np.abs(data[:, ind_dipole1_mom]).max(1, keepdims=True)
        # normalize double dipole moments
        data[:, ind_dipole2_mom] /= np.abs(data[:, ind_dipole2_mom]).max(1, keepdims=True)
        # center residual variance
        data[:, ind_rv] = data[:, ind_rv] * 2 - 1
        return data.astype(np.float32)

    def normalize_topo_features(self, data, pca=None):
        """
        Normalize scalp topography features.
        :param data: scalp topography features
        :param pca: A PCA matrix to use if for the test set if do_pca was set to true in __init__.
        :return: (normalized dipole features, pca matrix or None)
        """
        # apply pca
        if self.do_pca:
            if pca is None:
                pca = PCA(whiten=True)
                pca.fit_transform(data)
            else:
                data = pca.transform(data)

            # clip extreme values
            data = np.clip(data, -2, 2)

        else:
            # normalize to norm 1
            data /= np.linalg.norm(data, axis=1, keepdims=True)

        return data.astype(np.float32), pca

    def normalize_psd_features(self, data):
        """
        Normalize power spectral density features.
        :param data: power spectral density features
        :return: normalized power spectral density features
        """

        # undo notch filter
        for linenoise_ind in (49, 59):
            notch_ind = (
            data[:, [linenoise_ind - 1, linenoise_ind + 1]] - data[:, linenoise_ind, np.newaxis] > 5).all(1)
            data[notch_ind, linenoise_ind] = data[notch_ind][:, [linenoise_ind - 1, linenoise_ind + 1]].mean(1)

        # divide by max abs
        data /= np.amax(np.abs(data), axis=1, keepdims=True)

        return data.astype(np.float32)

    @staticmethod
    def normalize_autocorr_features(data):
        """
        Normalize autocorrelation function features.
        :param data: autocorrelation function features
        :return: normalized autocorrelation function features
        """
        # normalize to max of 1
        data[data > 1] = 1
        return data.astype(np.float32)

    def normalize_handcrafted_features(self, data, ic_nums):
        """
        Normalize hand crafted features.
        :param data: hand crafted features
        :param data: ic indices when sorted by power within their respective datasets. The 2nd ID number can be used for
            this in the training dataset
        :return: normalized handcrafted features
        """
        # autocorreclation
        data[:, 0] = self._clip_and_rescale(data[:, 0], -0.5, 1.)
        # SASICA focal topo
        data[:, 1] = self._clip_and_rescale(data[:, 1], 1.5, 12.)
        # SASICA snr REMOVED
        # SASICA ic variance
        data[:, 2] = self._clip_and_rescale(np.log(data[:, 2]), -6., 7.)
        # ADJUST diff_var
        data[:, 3] = self._clip_and_rescale(data[:, 3], -0.05, 0.06)
        # ADJUST Temporal Kurtosis
        data[:, 4] = self._clip_and_rescale(np.tanh(data[:, 4]), -0.5, 1.)
        # ADJUST Spatial Eye Difference
        data[:, 5] = self._clip_and_rescale(data[:, 5], 0., 0.4)
        # ADJUST spatial average difference
        data[:, 6] = self._clip_and_rescale(data[:, 6], -0.2, 0.25)
        # ADJUST General Discontinuity Spatial Feature
        # ADJUST maxvar/meanvar
        data[:, 8] = self._clip_and_rescale(data[:, 8], 1., 20.)
        # FASTER Median gradient value
        data[:, 9] = self._clip_and_rescale(data[:, 9], -0.2, 0.2)
        # FASTER Kurtosis of spatial map
        data[:, 10] = self._clip_and_rescale(data[:, 10], -50., 100.)
        # FASTER Hurst exponent
        data[:, 11] = self._clip_and_rescale(data[:, 11], -0.2, 0.2)
        # number of channels
        # number of ICs
        # ic number relative to number of channels
        ic_rel = self._clip_and_rescale(ic_nums * 1. / data[:, 13], 0., 1.)
        # topoplot plot radius
        data[:, 12] = self._clip_and_rescale(data[:, 14], 0.5, 1)
        # epoched?
        # sampling rate
        # number of data points

        return np.hstack((data[:, :13], ic_rel.reshape(-1, 1))).astype(np.float32)

    # plotting functions

    @staticmethod
    def _plot_grid(data, function):
        nax = data.shape[0]
        a = np.ceil(np.sqrt(nax)).astype(np.int)
        b = np.ceil(1. * nax / a).astype(np.int)
        f, axarr = plt.subplots(a, b, sharex='col', sharey='row')
        axarr = axarr.flatten()
        for x in range(nax):
            function(data[x], axis=axarr[x])
            axarr[x].set_title(str(x))

    def pad_topo(self, data):
        """
        Reshape scalp topography images features and pad with zeros to make 32x32 pixel images.
        :param data: Scalp topography features as provided by load_data() and load_semisupervised_data().
        :return: Padded scalp topography images.
        """
        if data.ndim == 1:
            ntopo = 1
        else:
            ntopo = data.shape[0]
        topos = np.zeros((ntopo, 32 * 32))
        topos[:, self.topo_ind] = data
        topos = topos.reshape(-1, 32, 32).transpose(0, 2, 1)
        return np.squeeze(topos)

    def plot_topo(self, data, axis=plt):
        """
        Plot an IC scalp topography.
        :param data: Scalp topography vector (unpadded).
        :param axis: Optional matplotlib axis in which to plot.
        """
        topo = self.pad_topo(data)
        topo = np.flipud(topo)
        maxabs = np.abs(data).max()
        axis.matshow(topo, cmap='jet', aspect='equal', vmin=-maxabs, vmax=maxabs)

    def plot_topo_grid(self, data):
        """
        Plot a grid of IC scalp topographies.
        :param data: Matrix of scalp topography vectors (unpadded).
        """
        if data.ndim == 1:
            self.plot_topo(data)
        else:
            nax = data.shape[0]
            if nax == 740:
                data = data.T
                nax = data.shape[0]
            if nax > self.max_grid_plot:
                print 'Too many plots requested.'
                return

            self._plot_grid(data, self.plot_topo)

    def plot_psd(self, data, axis=plt):
        """
        Plot an IC power spectral density.
        :param data: Power spectral density vector.
        :param axis: Optional matplotlib axis in which to plot.
        """
        if self.psd_limits is not None:
            data = self._unscale(data, *self.psd_limits)
        if self.psd_mean is not None:
            data = data + self.psd_mean
        axis.plot(self.psd_ind[:data.flatten().shape[0]], data.flatten())

    def plot_psd_grid(self, data):
        """
        Plot a grid of IC power spectral densities.
        :param data: Matrix of power spectral density vectors.
        """
        if data.ndim == 1:
            self.plot_psd(data)
        else:
            nax = data.shape[0]
            if nax > self.max_grid_plot:
                print 'Too many plots requested.'
                return

            self._plot_grid(data, self.plot_psd)

    @staticmethod
    def plot_autocorr(data, axis=plt):
        """
        Plot an IC autocorrelation function.
        :param data: autocorrelation function vector.
        :param axis: Optional matplotlib axis in which to plot.
        """
        axis.plot(np.linspace(0, 1, 101)[1:], data.flatten())

    def plot_autocorr_grid(self, data):
        """
        Plot a grid of IC autocorrelation functions.
        :param data: Matrix of autocorrelation function vectors.
        """
        if data.ndim == 1:
            self.plot_autocorr(data)
        else:
            nax = data.shape[0]
            if nax > self.max_grid_plot:
                print 'Too many plots requested.'
                return

            self._plot_grid(data, self.plot_autocorr)

    def web_image(self, component_id):
        """
        Open the component properties image from the ICLabel website (iclabel.ucsd.edu) for an IC. Not all ICs have
        images available.
        :param component_id: ID for the component which can be either 2 or 3 numbers if from the training set or test
            set, respectively.
        """
        if len(component_id) == 2:
            wb.open_new_tab(self.base_url_image + '{0:0>6}_{1:0>3}.png'.format(*component_id))
        elif len(component_id) == 3:
            wb.open_new_tab(self.base_url_image + '{0:0>2}_{1:0>2}_{2:0>3}.png'.format(*component_id))
        else:
            raise ValueError('component_id must have 2 or 3 elements.')
