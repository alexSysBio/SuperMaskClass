# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 15:50:38 2021

@author: Alexandros Papagiannakis, Christine Jacobs-Wagner group, Stanford University 2021
"""

import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import os
#from sklearn.preprocessing import StandardScaler
#from sklearn.decomposition import PCA
#
#import pandas as pd
#cell_dataframe = pd.read_pickle(r"M:\Alex\cell_mesh_classification_ML\training_data", compression='zip')

def supervised_prediction(cell_dataframe, 
                  trained_model,
                  features_=['cell_area', 'cell_length', 'constriction', 'max_distance', 'cell_angle', 'central_intercept']):
    """Function used to make predictions based on cell features and a trained supervised model.

    Args:
        cell_dataframe (_Pandas DataFrame_): the dataframe which includes the SVM features for the prediction
        trained_mocel (_model_): Trained model object
        features_ (list, optional): List of features to be used. Must be compatible with the features used for training. 
                                    Defaults to ['cell_area', 'cell_length', 'constriction', 'max_distance', 'cell_angle', 'central_intercept'].
        
        
    Returns:
        [0] _list_: list of well-segmented cells
        [1] _list_: list of badly segmented cells
        [2] _Pandas DataFrame_: Pandas DataFrame with the prediction column ('prediction') applied in the input cell_dataframe
    """

    feat= features_
    svm_prediction_dataframe = cell_dataframe.dropna()
    # the cells that did not have a medial axis
    bad_cells = cell_dataframe[~cell_dataframe.cell_id.isin(svm_prediction_dataframe.cell_id.to_list())].cell_id.to_list()
    
    clf = trained_model
    features = svm_prediction_dataframe[feat]
    X = np.asarray(features)
    yhat = clf.predict(X)
    svm_prediction_dataframe['class'] = yhat
    
    # the good predicted cells
    good_cells = svm_prediction_dataframe[svm_prediction_dataframe['class']==1].cell_id.to_list()
    # the badly predicted cells
    bad_cells += svm_prediction_dataframe[svm_prediction_dataframe['class']==0].cell_id.to_list()
    
    cell_dataframe['prediction'] = cell_dataframe.cell_id
    cell_dataframe['prediction'] = cell_dataframe['prediction'].replace(good_cells, 1)
    cell_dataframe['prediction'] = cell_dataframe['prediction'].replace(bad_cells, 0)
    
    sns.scatterplot(data=cell_dataframe, x='cell_area', y='max_distance', hue='prediction')
    plt.xlim(0,2000)
    plt.show()
    
    sns.scatterplot(data=cell_dataframe, x='cell_angle', y='max_distance', hue='prediction')
    plt.show()
    
    sns.scatterplot(data=cell_dataframe, x='cell_area', y='cell_length', hue='prediction')
    plt.xlim(0,2000)
    plt.ylim(0,160)
    plt.show()
    
    sns.scatterplot(data=cell_dataframe, x='sinuosity', y='cell_angle', hue='prediction')
    plt.xlim(1,1.1)
    plt.ylim(130,185)
    plt.show()
    
    print(round(len(bad_cells)/len(good_cells)*100,1), '% of cells will be excluded from the analysis.')
    print(len(good_cells), 'good cells')
    print(len(bad_cells), 'bad cells')
    
    return good_cells, bad_cells, cell_dataframe
    


