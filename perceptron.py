# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 09:50:47 2022

@author: user
"""

from sklearn.linear_model import Perceptron
import numpy as np
import data_reader

dr = data_reader.DataReader()
X=dr.x_train
Y=dr.y_train

p = Perceptron(max_iter=100, eta0=0.001,verbose=0,random_state=1)
p.fit(X, Y)

res = p.predict(dr.x_test)

print("정확률: {:.2f}".format(np.mean(res==dr.y_test)))