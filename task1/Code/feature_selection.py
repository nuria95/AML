import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt

plt.ioff()

def Univariate_Selection(X,y):
    bestfeatures = SelectKBest(score_func=chi2, k=10)
    X_not_nan=np.nan_to_num(X)
    print(X.columns)
    fit = bestfeatures.fit(X_not_nan,y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    #concat two dataframes for better visualization 
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Specs','Score']  #naming the dataframe columns
    print(featureScores.nlargest(10,'Score'))  #print 10 best features


def featureImportance(X,y):
    model = ExtraTreesClassifier()
    X_not_nan=np.nan_to_num(X)
    model.fit(X_not_nan,y)
    #print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
    #plot graph of feature importances for better visualization
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    print('Feature importance:\n',feat_importances.sort_values())
    # feat_importances.nlargest(10).plot(kind='barh')
    # plt.show()
