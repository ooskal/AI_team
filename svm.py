# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 08:59:37 2022

@author: user
"""

import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
import data_reader

dr = data_reader.DataReader()
X = dr.x_train
Y = dr.y_train


s = svm.SVC(gamma=0.001)
s.fit(X,Y) #분류 모델 훈련

res = s.predict(dr.x_test) #테스트
print("정확률: {:.2f}".format(np.mean(res==dr.y_test)))


#참고 사이트 : https://bskyvision.com/851