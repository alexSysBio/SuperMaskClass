# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 15:50:38 2021

@author: Alexandros Papagiannakis, HHMI at Stanford 2021
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import classification_report, confusion_matrix, f1_score, jaccard_score
import itertools
from sklearn import svm
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler
#from sklearn.decomposition import PCA


def plot_confusion_matrix(cm, classes,
                            normalize=False,
                            title='Confusion matrix',
                            cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
        
        
        
def train_svm_model(train_df, features_=['cell_area', 'cell_length', 'constriction', 'max_distance', 'cell_angle', 'central_intercept'], 
                    kernel_='linear', weight=True, test_size_=0.3):
    """The function used to train the SVM model.

    Args:
        train_df (_Pandas DataFrame_): The training dataframe that includes a column called class 
        features_ (list, optional): list of features. Defaults to ['cell_area', 'cell_length', 'constriction', 'max_distance', 'cell_angle', 'central_intercept'].
        kernel_ (str, optional): The type of SVM model used. Defaults to 'linear'.
        weight (bool, optional): Choose True for a weighted SVM application. Defaults to True.
        test_size_ (float, optional): Fraction of test data. Defaults to 0.3.

    Returns:
        _SVM model_: The SVM model object
    """
    feat= features_
    svm_training_dataframe = train_df.dropna()
    
    if weight == False:
        clf = svm.SVC(kernel=kernel_)
    elif weight == True:
        print('Please wait. Weighted optimization takes time...')
        clf = svm.SVC(kernel=kernel_, class_weight='balanced')
    
    features = svm_training_dataframe[feat]
    X = np.asarray(features)
    y = np.asarray(svm_training_dataframe['class'])
    
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=test_size_, random_state=4)
    print('Train set:', X_train.shape, y_train.shape)
    print('Test set:', X_test.shape, y_test.shape)
    
    clf.fit(X_train, y_train)
    yhat = clf.predict(X_test)
    yhat[0:5]
    
    print(f1_score(y_test, yhat, average='weighted'), 'f1')
    print(jaccard_score(y_test, yhat), 'jaccard')
    
    
    cnf_matrix = confusion_matrix(y_test, yhat)
    np.set_printoptions(precision=2)
    
    print (classification_report(y_test, yhat))
    
    # Plot non-normalized confusion matrix
  
    plot_confusion_matrix(cnf_matrix, classes=['BAD', 'GOOD'],normalize= False,  title='Confusion matrix')
    plot_confusion_matrix(cnf_matrix, classes=['BAD', 'GOOD'],normalize= True,  title='Confusion matrix')
    
    return clf
