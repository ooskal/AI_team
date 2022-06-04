# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 08:58:14 2022

@author: user
"""

import numpy as np
import random
from matplotlib import pyplot as plt
import re
import pandas as pd

# 데이터를 떠먹여 줄 클래스를 제작합니다.
class DataReader():
    def __init__(self):
        self.label = ["email", "spam-email"]

        self.train_X, self.train_Y, self.test_X, self.test_Y = self.read_data()

        # 데이터 읽기가 완료되었습니다.
        # 읽어온 데이터의 정보를 출력합니다.
        print("\n\nData Read Done!")
        print("Training X Size : " + str(self.train_X.shape))
        print("Training Y Size : " + str(self.train_Y.shape))
        print("Test X Size : " + str(self.test_X.shape))
        print("Test Y Size : " + str(self.test_Y.shape) + '\n\n')

    def read_data(self):
        print("Reading Data...")
        file = open("emails.csv")
        data = []
        for line in file:
            splt = line.split(",")
            if len(splt) != 3002:
              break
            df = pd.read_csv('emails.csv')
            feature= float(df.iloc[:,:3001])
            label = self.label.index(splt[3001])
            data.append(((feature), label))

        random.shuffle(data)

        X = []
        Y = []

        for el in data:
            X.append(el[0])
            Y.append(el[1])

        X = np.asarray(X)
        Y = np.asarray(Y)

        X = X / np.max(X, axis=0)

        train_X = X[:int(len(X)*0.8)]
        train_Y = Y[:int(len(Y)*0.8)]
        test_X = X[int(len(X)*0.8):]
        test_Y = Y[int(len(Y)*0.8):]

        return train_X, train_Y, test_X, test_Y