#! /usr/bin/env python3
import os

import numpy as np
import pandas as pd
import feature_selection
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.mixture import GaussianMixture
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
        



class Data():

    def __init__(self, pd_DataFrame_x, pd_DataFrame_y=None,fill_nans=True):
        self.cols_with_feats = [
            col for col in pd_DataFrame_x.columns if 'x' in col]
        self.x_pd = pd_DataFrame_x[self.cols_with_feats]
        if fill_nans:
            self.fill_na()
        
        self.x = self.x_pd.to_numpy(dtype=np.float64)
        self.id = pd_DataFrame_x['id']
        self.x_nfeatures = self.x.shape[1]
        self.x_ndata = self.x.shape[0]

        if pd_DataFrame_y is not None:
            self.y = pd_DataFrame_y['y'].to_numpy(dtype=np.float64)
    
    def percentage_empty_data(self):
        return (np.sum(np.isnan(self.x_pd),axis=0)/self.x_ndata).sort_values()

    def StandardizeData(self):
       self.x_standard = StandardScaler().fit_transform(self.x)
       #print(np.nanmean(self.x_standard,axis=0).shape)
       #print(np.nanstd(self.x_standard))
       
    def fill_na(self):
        for i in self.x_pd.columns:
            self.x_pd[i].fillna(self.x_pd[i].mean(),inplace=True)
        

        


class Predict():
    def __init__(self,x,y, test_size=0.1 ):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=test_size, random_state=42)

    def fit_ElasticNetCV(self, l1_ratio,fit_intercept=True, normalize = True, cv=5, random_state=0):
        self.regr = ElasticNetCV(l1_ratio=l1_ratio,fit_intercept=fit_intercept, normalize = normalize, cv=cv, max_iter=2000)
        self.regr.fit(self.x_train,self.y_train)
        return self.regr

    def fit_KernelRidge(self,alpha=1.0,kernel='rbf',degree=3):
        self.regr=KernelRidge(alpha=alpha,kernel=kernel, degree=degree)
        self.regr.fit(self.x_train,self.y_train)

    def print_attributes(self):
        #print('Grid of Alphas:', self.regr.alphas_)
        print('Best Alpha:', self.regr.alpha_)
        print('Best l1_ratio:',self.regr.l1_ratio_)
        print('w:',np.sort(self.regr.coef_))
        #self.regr.coef_=np.array([i if abs(i)>1e-8 else 0 for i in self.regr.coef_])
        #print('mse path:',self.regr.mse_path_)

    def predict(self, show_result=True):
        self.y_pred=self.regr.predict(self.x_test)

        #Create Dataframe for comparison of predictions:
        if show_result:
            comparison_pred_df = pd.concat([pd.DataFrame(self.y_pred),pd.DataFrame(self.y_test)],axis=1)
            comparison_pred_df.columns = ['Predicted','Groundtruth']  #naming the dataframe columns
            print(comparison_pred_df)
        
    def r2_score(self):
        return r2_score(self.y_test, self.y_pred)
    
class OutlierDetection():
    def __init__(self,x,y):
        self.x_outliers=x
        self.y_outliers=y
        gaussian_mixt = GaussianMixture(n_components=2)
        self.outputs=gaussian_mixt.fit_predict(x)
        if len(self.outputs[self.outputs==1]) > len(x)/2:
            self.outlier_class = 0
        else:
            self.outlier_class = 1
        print('Num outliers:',len(self.outputs[self.outputs==self.outlier_class]))
    
    def remove_outliers(self):
        x_outliers=self.x_outliers[self.outputs!=self.outlier_class]
        y_outliers=self.y_outliers[self.outputs!=self.outlier_class]

        return x_outliers, y_outliers




def main(write_file=True):

    # import data
    data_path = os.path.dirname(os.getcwd())
    files = [f + '.csv' for f in ['X_train', 'X_test','y_train']]
    train_file = os.path.join(data_path, 'Data', files[0])
    test_file = os.path.join(data_path, 'Data', files[1])
    train_y_file = os.path.join(data_path, 'Data', files[2])

    try:
        train_x_pd = Tools.read_csv(train_file)
        test_pd = Tools.read_csv(test_file)
        train_y_pd = Tools.read_csv(train_y_file)
    except ImportError as e:
        print(e)
        return

    print('Training data:')
    train_data = Data(train_x_pd, train_y_pd,fill_nans=True)
    print('num_features', train_data.x_nfeatures)
    print('lenght training data', train_data.x_ndata)
    print('percentage empty data per feature:\n',train_data.percentage_empty_data() )
   
    ##Feature selection
    #feature_selection.featureImportance(train_data.x_pd,train_data.y)
    
    ##Standardize Data
    train_data.StandardizeData()
    
    ##Kernel Ridge Regression:
    # pred=Predict(np.nan_to_num(train_data.x_standard),train_data.y, test_size=0.1)
    # pred.fit_KernelRidge()
    # pred.predict()
    # print('R2_SCORE in training:',pred.r2_score())
    

    #Outlier detection
    outliers=OutlierDetection(train_data.x,train_data.y) 
    [x_outliers, y_outliers]=outliers.remove_outliers()
    
    print('Len Filtered data:',len(x_outliers))

    pred=Predict(x_outliers,y_outliers, test_size=0.1)
    regr=pred.fit_ElasticNetCV(l1_ratio=[1],fit_intercept=True,normalize = True, cv=5) #find best is 1 (using grid suggested by sklearn)
    pred.print_attributes()
    pred.predict()
    print('R2_SCORE in training:',pred.r2_score())


    
    # # print('Test data:')
    # # #Tools.analyze(test_pd)
    test_data = Data(test_pd,fill_nans=True)
    y_test_pred=regr.predict(test_data.x)
    

     # write to file
    if write_file:
        df = pd.DataFrame({'y': y_test_pred, 'id': test_data.id})
        df = df.set_index('id')
        df.to_csv(os.path.join(data_path, 'Data', 'pred_Nuria.csv'))
    
if __name__ == '__main__':
    main(write_file=True)
