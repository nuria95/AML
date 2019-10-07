#! /usr/bin/env python3

'''
Hey Nuria!

Just so you know my folder structure is as follows:

project0  # doesnt matter what you call this folder
    Data
        train.csv
        test.csv
    Code
        soln.py

You'll need the same structure for the code here to work

xxx

'''

import math
import os

import numpy as np
import pandas as pd


class Tools():
    import pandas as pd

    @staticmethod
    def read_csv(file_path):
        if not os.path.exists(file_path):
            raise ImportError(file_path + ' doesnt exist')
        else:
            return pd.read_csv(file_path, float_precision='high')

    @staticmethod
    def analyze(pd_DataFrame):
        print('Shape:')
        print(pd_DataFrame.shape, '\n')
        print('Col names:')
        print(pd_DataFrame.columns, '\n')
        print('Dtypes:')
        print(pd_DataFrame.dtypes, '\n')


class TrainData():
    ''' class unique to this exercise, do not copy'''
    def __init__(self, pd_data, with_id=True):
        nparray = pd_data.to_numpy(dtype=np.float64)
        self.y = nparray[:, 0 + with_id]
        self.x = nparray[:, 1 + with_id:]

    def n_features(self):
        return self.x.shape[1]

    def n_data(self):
        return self.x.shape[0]


class TestData():
    ''' class unique to this exercise, do not copy'''
    def __init__(self, pd_data, with_id=True):
        nparray = pd_data.to_numpy(dtype=np.float64)
        self.index = nparray[:, 0].astype(int) if with_id else None
        self.x = nparray[:, 0 + with_id:]
        self.y_pred = np.zeros((self.n_data(),))

    def n_features(self):
        return self.x.shape[1]

    def n_data(self):
        return self.x.shape[0]


def main(write_file=True):

    # import data
    data_path = os.path.dirname(os.getcwd())
    files = [f + '.csv' for f in ['train', 'test']]
    train_file = os.path.join(data_path, 'Data', files[0])
    test_file = os.path.join(data_path, 'Data', files[1])

    try:
        train_pd = Tools.read_csv(train_file)
        test_pd = Tools.read_csv(test_file)
    except ImportError as e:
        print(e)
        return

    print('Training data:')
    Tools.analyze(train_pd)

    train_data = TrainData(train_pd)
    y_train = train_data.y
    x_train = train_data.x

    # verify that yi is actually the mean of xi
    sqr_error = np.zeros((y_train.shape[0],), dtype=np.float64)
    for i in range(train_data.n_data()):
        sqr_error[i] = (y_train[i] - np.mean(x_train[i, :], dtype=np.float64))**2
    rms_error = math.sqrt(1/train_data.n_data() * np.sum(sqr_error))
    print('RMS_ERROR on training set: ', rms_error, '\n')

    print('Test data:')
    Tools.analyze(test_pd)

    # compute predictions for test set
    test_data = TestData(test_pd)
    for i in range(test_data.n_data()):
        test_data.y_pred[i] = np.mean(test_data.x[i, :])

    # write to file
    if write_file:
        df = pd.DataFrame(test_data.y_pred, index=test_data.index, columns=['y'])
        df.index.name = 'Id'
        df.to_csv(os.path.join(data_path, 'Data', 'pred_Michael.csv'))
        print(df)


if __name__ == '__main__':
    main(write_file=True)
