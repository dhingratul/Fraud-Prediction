#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 09:26:37 2018

@author: dhingratul
"""
import pandas as pd
from sklearn import preprocessing
import numpy as np


def oneHot(df, y, clip=True, thresh=False):
    n = y.max() + 1
    m = df.shape[0]
    oh = np.zeros((m, n))
    for i in range(m):
        oh[i][y[i]] = 1 
    if clip is True:
        sum_ = oh.sum(axis=0)
        frac = sum_ / float(sum(sum_))
        if thresh is False:
            x = oh[:,frac > frac.mean()]
        else:
            x = oh[:,frac > thresh]
    else:
        x = oh
    return x


def getEncoded(df, feat, column, out_name, oh=False, clip=False, write_out=False):
    le = preprocessing.LabelEncoder()
    le.fit(df[column])
    y = le.transform(df[column]) 
    if write_out is True:
        df[out_name] = pd.Series(y)
    # One Hot encode features
    if oh is True:
        x = oneHot(df, y, clip, thresh=False)
        feat.append(x)
    return df, feat, le


def randomSample(train, num_samples):
    fraud = train[train['y'] == 1]
    n_fraud = train[train['y'] == 0].sample(n = num_samples, random_state=0)
    df = pd.concat([fraud, n_fraud], ignore_index=True)
    df = df.sample(frac=1, random_state=0).reset_index(drop=True)
    return df


def split_training_testing(X, Y, gnd, negative=10000, per=0.05):
    df_x = pd.DataFrame(X)
    df_x['y'] = Y
    df_x['gnd'] = gnd
    df_x.sort_values(by=['y'], inplace=True, ascending=False)
    frac_positive = (df_x[df_x['y'] == 1].shape[0])/float(df_x.shape[0])
    split = int(frac_positive * per * df_x.shape[0])
    df_x.reset_index(drop=True, inplace=True)
    test = df_x.iloc[:split]
    train = df_x.iloc[split:]
    train = randomSample(train, negative)
    y_train = train['y'].as_matrix()
    y_train_gnd = train['gnd'].as_matrix()
    train = train.drop(['y'], axis=1)
    train = train.drop(['gnd'], axis=1)
    
    y_test = test['y'].as_matrix()
    y_test_gnd = test['gnd'].as_matrix()
    test.drop(['y'], axis=1, inplace=True)
    test.drop(['gnd'], axis=1, inplace=True)
    return train.as_matrix(), y_train, y_train_gnd, test.as_matrix(), y_test, y_test_gnd


def voting(y_pred_test, gnd_te):
    df = pd.DataFrame({'y':y_pred_test, 'gnd':gnd_te})
    df.sort_values(by=['y'], inplace=True, ascending=False)
    out = df.groupby(['gnd']).mean()
    return len(out[out['y'] > 0])/float(len(out))


def evaluate(y_pred_X, gnd, thresh, le_y):
    df2 = pd.DataFrame({'y':y_pred_X, 'gnd':gnd})
    #df.sort_values(by=['y'], inplace=True, ascending=False) 
    out = df2.groupby(['gnd']).mean()
    out.reset_index(inplace=True)
    labels = out['gnd'].as_matrix()
    mask2 = labels[out['y'] > thresh]
    return list(le_y.inverse_transform(mask2))


def encodeDates(df, feat, column):
    y = pd.date_range(df[column].min(), df[column].max(), freq="5min")
    out = np.zeros((df.shape[0], len(y) - 1))
    for j in range(1, len(y) - 1):
        x=df[(df[column] < y[j]) & (df[column] >= y[j - 1])]
        out[x.index[0]:x.index[-1], j-1] = 1
    feat.append(out)
    return feat
