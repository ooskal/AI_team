#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 09:19:12 2022

@author: siyeon
"""

import numpy as np
import random


class DataReader():
    
    def __init__(self):
        #훈련, 테스트 데이터
        self.x_train, self.y_train, self.x_test, self.y_test = self.read_data()
        #데이터 정보 출력 
        print("******** Data Read Done ********")
        print("Training Data Size: " + str(len(self.x_train)))
        print("Test Data Size: " + str(len(self.x_test)) + "\n\n")
        
        
    # 데이터 읽어오기 위한 메서드
    def read_data(self):
        # 파일 실행, 헤더 제거
        file = open("emails.csv")
        file.readline()
        
        # 데이터, 레이블 저장할 변수
        data = []
        
        # 파일 한 줄씩 읽어오기
        for line in file:
            splt = line.split(",") # 컴마 기준으로 split()실행, 문자열 > 리스트
            
            x, y = self.process_data(splt) # split 결과물 정리 후 x, y 값으로 나누기
            
            data.append((x, y)) # 나누어진 데이터 저장
            
        # 데이터 섞기
        random.shuffle(data)
        
        X = []
        Y = []
        
        for i in data:
            X.append(i[0]) # 피처 값 저장
            Y.append(i[1]) # 레이블 값 저장 
            
        X = np.asarray(X)
        Y = np.asarray(Y)
        
        # 훈련, 테스트 데이터 8:2로 나누기
        x_train = X[ :int(len(X) * 0.8)]
        y_train = Y[ :int(len(Y) * 0.8)]
        x_test =  X[int(len(X) * 0.8): ]
        y_test =  Y[int(len(Y) * 0.8): ]
        
        return x_train, y_train, x_test, y_test
    
    
    # split() 값 정리하는 메서드
    def process_data(self, splt):
        
        # 특징들 저장할 변수
        features = []
        
        for i in range(1, 3001):
            features.append(float(splt[i]))
        
        # 레이블 저장할 변수
        label = float(splt[3001])
        
        
        return features, label
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
            