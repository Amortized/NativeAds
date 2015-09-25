"""
Truly Native
__author__ : Rahul

"""

from __future__ import print_function

from collections import Counter
import glob
import multiprocessing
import os
import re
import sys
import time
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import copy
from pandas import Series
from sklearn.cross_validation import train_test_split
import model;
import xgboost as xgb;
import numpy  as np;


def create_features(filepath):
    """
      Creates features
    """
    values = {}
    filename = os.path.basename(filepath)
    with open(filepath, 'rb') as infile:
        text = infile.read()
    values['file'] = filename
    if filename in train_keys:
        values['sponsored'] = train_keys[filename]
    values['lines'] = text.count('\n')
    values['spaces'] = text.count(' ')
    values['tabs'] = text.count('\t')
    values['braces'] = text.count('{')
    values['brackets'] = text.count('[')
    values['words'] = len(re.split('\s+', text))
    values['length'] = len(text)
    return values

print('--- Identify training and test files ---');
train        = pd.read_csv("./data/train_v2.csv");
sample_sub   = pd.read_csv("./data/sampleSubmission_v2.csv");

train_keys   = Series(train.sponsored.values,index=train.file).to_dict()
test_files   = set(sample_sub.file.values);	

#Get all the files to read 	
filepaths    = glob.glob('data/*/*.txt')	
num_tasks    = len(filepaths)

#Create features in parallel
p = multiprocessing.Pool(16);

results = p.map(create_features, filepaths);
p.close();
p.join();

#Split the data
df_full = pd.DataFrame(list(results));
train   = df_full[df_full.sponsored.notnull()];
test    = df_full[df_full.sponsored.isnull() & df_full.file.isin(test_files)];
train = train.drop(['file'], 1);
train_Y = train.sponsored.values;
train_X = train.drop(['sponsored'], 1).as_matrix();
feature_names = train.drop(['sponsored'], 1).columns;

#Create validation set
train_X, validation_X, train_Y, validation_Y = train_test_split(train_X, train_Y, test_size=0.20, random_state=0);

#Train
best_model = model.train(train_X, train_Y, validation_X, validation_Y, feature_names);

#predict
test_ids = test.file.values; 
test_X   = xgb.DMatrix(np.asarray(test.drop(['file','sponsored'], 1).as_matrix()));

submission = test[['file']].reset_index(drop=True)
submission['sponsored'] = best_model.predict(test_X);
submission.to_csv('submission.csv', index=False)



