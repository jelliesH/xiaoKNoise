# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 09:37:11 2016

@author: 390672
"""

"""
将result的结果转换成DataFrame结构
result 的数据结构如下:
result - list
       result = [(A),(A)......]
A - tuple 每个噪音的6个特征集
        A = (B,B,B2,B3,B4,B5)
B - dict 每个特征集
B2,B3 - 多个滤波器滤出的信号对应的多个特征集
B4,B5 - 多个IMF滤出的信号对应的多个特征集
"""

keys = list()
    each = result[0]
    each = each[1:]
    for ii in range(len(each)):
        if ii <= 1:
            keys = keys + list(each[ii].keys())
        elif ii<=3:
            cl3_4 = each[ii]
            # 针对滤波器的特征集
            for each_n in cl3_4:
                keys = keys + list(cl3_4[each_n].keys())
        else:
            cl5_6 = each[ii]
            for each_n in cl5_6:
                keys = keys + list(cl5_6[each_n].keys())
            
    