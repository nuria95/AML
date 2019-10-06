import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


class utilityTools():
    @staticmethod
    def read_csv(file_path):
        if not os.path.exists(file_path):
            raise ImportError(file_path + ' doesnt exist man')
        else:
            return pd.read_csv(file_path, float_precision='round_trip')

    @staticmethod
    def analyze(pd_DataFrame):
        print('Shape:')
        print(pd_DataFrame.shape, '\n')
        print('Col names:')
        print(pd_DataFrame.columns, '\n')
        print('Dtypes:')
        print(pd_DataFrame.dtypes, '\n')
        print('Example data:')
        print(pd_DataFrame.iloc[0])


class Predict():

    def __init__(self, pd_DataFrame):
        self.cols_with_feats = [
            col for col in pd_DataFrame.columns if 'x' in col]
        self.id = pd_DataFrame['Id']
        self.x = pd_DataFrame[self.cols_with_feats]
        self.y_true = pd_DataFrame['y'] if 'y' in pd_DataFrame.columns else None

    def make_predictions(self):
        # x_np_array = self.x.to_numpy(dtype=np.float64) #convert to np to allow dtype in np.mean --> doesnt change precision!
        # y_pred = np.mean(self.x.to_numpy(dtype=np.float64),axis=1,dtype=np.float64)
        y_pred = np.mean(self.x, axis=1)
        return y_pred


def main(write_file=False):

    # import data
    task_path = os.path.dirname(os.getcwd())
    files = [f + '.csv' for f in ['train', 'test']]
    train_file = os.path.join(task_path, 'Data', files[0])
    test_file = os.path.join(task_path, 'Data', files[1])

    # Read test and train data:
    try:
        train_pd = utilityTools.read_csv(train_file)
        test_pd = utilityTools.read_csv(test_file)
    except ImportError as e:
        print(e)
        return
    print('Training data:')
    utilityTools.analyze(train_pd)

    # Make predictions:
    train_data = Predict(train_pd)
    y_train_true = train_data.y_true

    y_train_pred = train_data.make_predictions()

    test_data = Predict(test_pd)
    y_test_pred = test_data.make_predictions()
    assert len(y_test_pred) == len(test_pd), 'dimension mismatch in TEST!'

    # Compute RMSE for training predicitions
    RMSE_train = mean_squared_error(y_train_true, y_train_pred)**0.5
    print('RMSE in training:', '\n', RMSE_train)

    # write to file
    if write_file:
        df = pd.DataFrame({'y': y_test_pred, 'Id': test_data.id})
        df = df.set_index('Id')
        df.to_csv(os.path.join(task_path, 'Data', 'pred_Nuria_round_trip.csv'))


if __name__ == '__main__':
    main(True)
