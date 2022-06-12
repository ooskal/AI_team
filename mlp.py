#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 29 17:09:06 2022

@author: siyeon
"""

from sklearn.neural_network import MLPClassifier
import data_reader
from sklearn.metrics import precision_score

dr = data_reader.DataReader()

h = 100

for i in range(8):  
    mlp = MLPClassifier(hidden_layer_sizes=(h),
                        learning_rate_init=0.001,
                        batch_size=256,
                        max_iter=300,
                        solver='adam',
                        verbose=True)
    mlp.fit(dr.x_train, dr.y_train)
    res = mlp.predict(dr.x_test)
    
    print("hidden_layer_sizes = " + str(h))
    print("훈련 세트 정확도: {:.3f}".format(mlp.score(dr.x_train, dr.y_train)))
    print("테스트 세트 정확도: {:.3f}".format(mlp.score(dr.x_test, dr.y_test)))
    print("정밀도: " + str(precision_score(dr.y_test, res)))
    
    h = h + 100



