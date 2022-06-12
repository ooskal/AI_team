#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 13:27:34 2022

@author: siyeon
"""

from sklearn import svm
import data_reader
from sklearn.metrics import precision_score


dr = data_reader.DataReader()

c = 0.01

for i in range(5):
    m = 1000
    
    for i in range(8):
        
        model = svm.SVC(kernel='linear', C=c, max_iter=m).fit(dr.x_train, dr.y_train)
        res = model.predict(dr.x_test)
        
        print("\nC = " + str(c) + " max_iter = " +str(m))
        print("훈련 세트 정확도: {:.3f}".format(model.score(dr.x_train, dr.y_train)))
        print("테스트 세트 정확도: {:.3f}".format(model.score(dr.x_test, dr.y_test)))
        print("정밀도: " + str(precision_score(dr.y_test, res)))
        
        m = m + 1000

    c = c * 10
    