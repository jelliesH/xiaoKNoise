# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 16:40:56 2016

@author: 390672
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn import cluster
from sklearn import preprocessing

def fullfill_nan(data_df,names):
    # input:
    # data_df - 数据表DataFrame结构，一张完整的数据表
    # names - 特征集的名称，list
    # output：
    # 将特征集中所有有效的特征（不含nan值）进行聚类，在根据聚类情况将值赋给nan的元素
    
    # 输入的names对应的特征集的所有数据
    cluster_data = data_df[names]
    # cluster_data2 - 只包含全部有效值数据表（完全没有nan值）
    cluster_data2 = cluster_data
    for each in names:
        if not(any(cluster_data2[each].isnull())):
            # 没有非空的值，全部是有效值
            pass
        else:
            cluster_data2 = cluster_data2.drop(each,axis=1)

    # col_valid 所有有效的特征的列名
    # col_rest 所有含无效值的特征的列名
    col_full = cluster_data.columns
    col_valid = cluster_data2.columns.values.tolist()
    col_rest = col_full.drop(col_valid).values.tolist()

    for each in col_rest:
        nan_ind = cluster_data2.index[data_df[each].isnull()]
        notnan_ind = cluster_data2.index[data_df[each].notnull()]

        train_data = cluster_data2.ix[notnan_ind,:] # 将nan的样本点剔除掉的样本库
        nan_data = cluster_data2.ix[nan_ind,:] # 只有nan的样本点的样本库
        try:
            ###################################################################################
            ################  对读入的数据进行聚类，再根据聚类结果填充空值 ####################
            ###################################################################################
            # 数据预处理（无量纲minmax）
            min_max_scaler = preprocessing.MinMaxScaler()
            X_train_minmax = min_max_scaler.fit_transform(train_data)
            X_nan_minmax = min_max_scaler.fit_transform(nan_data)
            # 进行聚类 K-Means
            km = cluster.KMeans(n_clusters=2).fit(X_train_minmax)
            labels = km.labels_
    
            notnan_ind_0 = train_data[labels == 0].index
            notnan_ind_1 = train_data[labels == 1].index
            mean_0 = data_df.ix[notnan_ind_0,each].mean()
            mean_1 = data_df.ix[notnan_ind_1,each].mean()
    
            nan_labels = km.predict(X_nan_minmax)
    
            nan_ind_0 = nan_data[nan_labels == 0].index
            nan_ind_1 = nan_data[nan_labels == 1].index
    
            data_df.loc[nan_ind_0,each] = mean_0
            data_df.loc[nan_ind_1,each] = mean_1
        except BaseException as e:
            print(each + '缺失值填充失败，出现如下错误：\n',e)
            # 因为每个属性都有缺失值，没办法进行聚类计算补充
            nan_ind = cluster_data2.index[data_df[each].isnull()]
            notnan_ind = cluster_data2.index[data_df[each].notnull()]
            mean = data_df.ix[notnan_ind,each].mean()
            data_df.loc[nan_ind,each] = mean
                    
    return data_df

if __name__ == '__main__':
    data_path = '/home/390672/WORKSHOP/noise_analysis/清洗后的特征数据/混合特征提取/6特征集全齐/normal_data_source.xlsx'
    
    data_df = pd.read_excel(data_path)
    data_df.index = range(data_df.shape[0])
    
    # 如果某列属性整列都是nan值，直接将该列删除
    data_df = data_df.dropna(how="all",axis=1)
    
    columns = data_df.columns
    # 对inf的数据进行填充
    for each in columns:
    
        temp = data_df.loc[:,each]
        temp.loc[temp == np.inf] = np.nan
        data_df.ix[:,each] = temp
    
    import time 
    time_a = time.time()
    list1 = dict() #缺失数量大于20 %
    list2 = dict() #缺失数量少于等于20%
    
    # 对于某列含有部分缺失值的数据进行处理
    for each in columns:
        temp = data_df.loc[:,each].notnull() # temp为True代表是非空的元素
        
        if all(temp):
            # 没有非空的值，全部是有效值
            pass
        elif len(temp[temp==False])/len(temp) > 0.2:
            # 1. 缺失数量大于20 %
            #  如果含有inf值的元素，使用nan值来替换
            #    用KNN找出最近邻的值作填充
            # 获取该列所处的特征集名称
            m = re.findall('.*_p',each) # each:'cl1_p2'; m :'cl1_p'
            group_name = m[0][:-2]  # group_name: 'cl1'
            list1[group_name] = [each_b for each_b in pd.Series(columns.values) if group_name in each_b]
            
        else:
            # 2. 缺失数量少于等于20%
            #    用平均值
            # 找出 缺失数量大于20 % 的列
            # 获取该列所处的特征集名称
            m = re.findall('.*_p',each) # each:'cl1_p2'; m :'cl1_p'
            group_name = m[0][:-2]  # group_name: 'cl1'
            list2[group_name] = [each_b for each_b in pd.Series(columns.values) if group_name in each_b]
            
    for each in list1.keys():
        data_df = fullfill_nan(data_df=data_df, names=list1[each])
        
    print(time.time() - time_a)