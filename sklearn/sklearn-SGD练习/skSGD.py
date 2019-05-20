# -*- coding: utf-8 -*-
"""
Created on Wed May 15 16:24:42 2019
利用sklearn的SDG随机梯度下降
@author: Kylin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.linear_model import SGDClassifier

#生成大样本，高纬度特征数据
X, y = make_classification(200000, n_features=200, n_informative=25, 
                           n_redundant=0, n_classes=10, class_sep=2, 
                           random_state=0)

#用SGDClassifier做训练，使用L2级范数，学习率为0.001, 并画出batch在训练前后的得分差
est = SGDClassifier(penalty="l2", alpha=0.001)
progressive_validation_score = []
train_score = []
#进行1000次迭代
for datapoint in range(0, 199000, 1000):
    #每次挑选1000个样本和对应标签
    X_batch = X[datapoint:datapoint+1000]
    y_batch = y[datapoint:datapoint+1000]
    
    if datapoint > 0:
        progressive_validation_score.append(est.score(X_batch, y_batch))
    #在线批量学习
    est.partial_fit(X_batch, y_batch, classes=range(10))
    if datapoint > 0:
        train_score.append(est.score(X_batch, y_batch))
    
plt.plot(train_score, label="train score")
plt.plot(progressive_validation_score, label="progressive validation score")
plt.xlabel("Mini-batch")
plt.ylabel("Score")
plt.legend(loc='best')  
plt.show()             



