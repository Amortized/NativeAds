"""

__author__ = 'amortized'
"""

import numpy  as np;
from sklearn.preprocessing import Imputer;
from sklearn.grid_search import ParameterGrid;
from multiprocessing import Pool;
import copy;
import random;
import sys;
import warnings;
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix;
import matplotlib.pyplot as plt;
import numpy as np
import scipy.sparse
import pickle
import xgboost as xgb
import copy
from sklearn.preprocessing import OneHotEncoder
from random import randint
from random import shuffle
import math
import copy
from sklearn.preprocessing import OneHotEncoder;
from numpy import array;

def generateParams():
    # Parameters

    paramaters_grid    = {'eta': [0.1,0.08], 'min_child_weight' : [1],  'colsample_bytree' : [1.0], 'subsample' : [1.0],'max_depth' : [7]};

    paramaters_search  = list(ParameterGrid(paramaters_grid));

    parameters_to_try  = [];
    for ps in paramaters_search:
        params           = {'eval_metric' : 'auc', 'objective' : 'binary:logistic', 'nthread' : 8};
        for param in ps.keys():
            params[str(param)] = ps[param];
        parameters_to_try.append(copy.copy(params));

    return parameters_to_try;     



def train(train_X, train_Y, validation_X, validation_Y, feature_names):

    train_X      = array(train_X);
    train_Y      = array(train_Y);
    validation_X = array(validation_X);
    validation_Y = array(validation_Y);

    print(type(train_X));

    dtrain = xgb.DMatrix( train_X, label=train_Y, missing=float('NaN'));
    dvalidation = xgb.DMatrix( validation_X, label=validation_Y,missing=float('NaN'))

    #Clean up data
    del train_X, validation_X, train_Y, validation_Y;

    #Track metrics on the watchlist
    watchlist = [ (dtrain,'train'), (dvalidation, 'validation') ]

    parameters_to_try = generateParams();

    best_params          = None;
    overall_best_metric  = 0;
    overall_best_nrounds = 0;

    for i in range(0, len(parameters_to_try)):
        param      = parameters_to_try[i];
        num_round  = 1000;

        classifier = xgb.train(param,dtrain,num_round,evals=watchlist,early_stopping_rounds=100);
        
        metric     = classifier.best_score;
        itr        = classifier.best_iteration;

        print("\n Metric : " + str(metric) + " for Params " + str(param) + " occurs at " + str(itr));

        if metric > overall_best_metric:
            overall_best_metric  = metric;
            best_params          = copy.copy(param);
            overall_best_nrounds = itr;

    print("\n Training the model on the entire training set with the best params")

    bst = xgb.train(best_params, dtrain, 1+overall_best_nrounds);
    print("\n\n Overall Best AUC : " + str(overall_best_metric) + " for Params " + str(best_params) + " occurs at " + str(overall_best_nrounds));
    

    feature_imp = bst.get_fscore();

    print("Feature Importance ... ");
    for w in sorted(feature_imp, key=feature_imp.get, reverse=True):
        print( str(feature_names[int(w.replace("f",""))]) + " : "  + str(feature_imp[w]) );

    
    return bst;

