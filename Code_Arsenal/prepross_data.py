# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 10:38:17 2016
创建一个类，里面包含各种预处理的效果,创建实例需要传入
的参数是一个一维数组，该一维数组可以是音频文件
@author: 390672
"""

from sklearn import preprocessing
import matplotlib.pyplot as plt
import warnings


    
class Prepross_data(object):
    def __init__(self, inputdata = []):
        self.origin_data = inputdata
        self.scaled_data = []
        self.MinMaxScalered_data = []
        self.normalized_data = []
    # 规范化出高斯零均值和方差的数据，使数据符合标准正态分布
    def scale_data(self):
        after_data = preprocessing.scale(self.origin_data)
        self.scaled_data = after_data        
        return after_data
        
    # 规范化出一定范围内的数据,相当与minmapmax
    def MinMaxScaler_data(self,featurerange=(-1, 1)):
        min_max_scaler = preprocessing.MinMaxScaler(feature_range = featurerange)
        after_data = min_max_scaler.fit_transform(self.origin_data)
        self.MinMaxScalered_data = after_data        
        return after_data    

    # 正则化数据
    def normalize_data(self,norm = 'l1'):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        
        origin_data = self.origin_data
        after_data =preprocessing.normalize(origin_data,norm = norm)
        after_data = after_data.T        
        after_data = after_data[:,0]        
        self.normalized_data = after_data
 
        return after_data
        
    # 画图相关的方法
    def plot_scale_data(self):
        if len(self.scaled_data):
            plt.plot(self.scaled_data)
            plt.show()
        else:
            print('请先对数据做规范化')
    
    def plot_MinMaxScaler_data(self):
        if len(self.MinMaxScalered_data):
            plt.plot(self.MinMaxScalered_data)
            plt.show()
        else:
            print('请先对数据做规范化')
    
    def plot_normalize_data(self):
        if len(self.normalized_data):
            plt.plot(self.normalized_data)
            plt.show()
        else:
            print('请先对数据做规范化')
    
