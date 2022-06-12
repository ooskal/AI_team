#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 29 16:25:16 2022

@author: siyeon
"""

from sklearn.linear_model import Perceptron
import data_reader
from sklearn.metrics import precision_score

dr = data_reader.DataReader()



#max_iter:epoch수, eta0 : 학습률
m = 100

for i in range(3):
    e = 0.0001
    
    for i in range(7):
        
        p = Perceptron(max_iter=m, eta0=e, random_state=1)
        p.fit(dr.x_train, dr.y_train)
        res = p.predict(dr.x_test)
        
        print("\nmax_iter = " + str(m) + " eta0 = " + str(e))
        print("훈련 세트 정확도: {:.3f}".format(p.score(dr.x_train, dr.y_train)))
        print("테스트 세트 정확도: {:.3f}".format(p.score(dr.x_test, dr.y_test)))
        print("정밀도: " + str(precision_score(dr.y_test, res)))
        e = e * 10
    
    m = m + 200
    


