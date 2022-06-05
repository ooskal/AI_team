# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 09:57:59 2022

@author: user
"""

from sklearn.neural_network import MLPClassifier
import numpy as np
import data_reader

dr = data_reader.DataReader()
X=dr.x_train
Y=dr.y_train
mlp = MLPClassifier(hidden_layer_sizes=(100),
                    learning_rate_init=0.001,
                    batch_size=256,
                    max_iter=300,
                    solver='adam',
                    verbose=True)
mlp.fit(X, Y)

res = mlp.predict(dr.x_test)

print("정확률: {:.2f}".format(np.mean(res==dr.y_test)))