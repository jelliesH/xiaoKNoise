# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 10:31:18 2016
参照matlab代码写成的EMD
@author: Steven Kwong
"""
import numpy as np
import scipy.interpolate as interpolate
from scipy import signal
from time import time
import pandas as pd

def get_IMF(data,max_modes = 5):
    imf_group = []
    
    # 进行模态分解，获取所有模态
    residue = data
    t_a1 = time()
    while ismonotonic(residue) == False:
    #   只要剩余项不是单调的就继续分解
        if len(imf_group) > max_modes:
            break
        else:
            pass
        
        x = residue
        m = 1
        max_n = 0
        min_n = max_n + 2
        sd = 10
        while isimf(x,m,max_n,min_n) == False or sd < 0.3 :
            t_b1 = time()
            e_max,e_min,max_n,min_n = getspline(x)
            m = (e_max + e_min)*.5
            h = x - m  # h 待判定的本征模态
            
            sd = np.sum((x-h)**2)/np.sum(x**2)        
            x = h
            t_b2 = time()
            
        print('总共 耗时：%f'%(t_b2 - t_b1))    
        print('第 %d 个IMF   获取成功'% (len(imf_group) + 1))
        imf_group.append(h)
        residue = residue - h
    t_a2 = time()
    print('总共耗时：%f'%(t_a2 - t_a1))    
    return imf_group

def ismonotonic(in_data):
    """
    -------------------------------------------
    ismonotonic
    -------------------------------------------
    用来判断输入的信号是否单调的
    input：
        in_data - 输入需要判定的信号
        
    output：
        result - 单调判定的结果
                    若单调递增或递减：True，非单调的返回False
    """ 
    dx = np.diff(in_data)
    return np.all(dx <= 0) or np.all(dx >= 0)
    
def isimf(in_data,in_mean,max_num,min_num):
    """
    -------------------------------------------
    isimf
    -------------------------------------------
    用来判断输入的信号是IMF分量
    有两个标准：
    1 局部均值为0，由局部极大值点构成的包络线和局部极小
        值点构成的包络线平均值为零
    2 极值点和过零点数目相等或者相差不超过1点
    
    input：
        in_data - 输入需要判定的信号
        in_mean - 极值包络线的均值线
        max_num - 极大值点的个数
        min_num - 极小值点的个数
        
    output：
        result - 单调判定的结果
                    若单调递增或递减：True，非单调的返回False
    """ 
    #m = np.sum(in_mean)

    # 求过零点个数zc_num
    N = len(in_data)
    product = in_data[0:(N-2)]*in_data[1:(N-1)]
    product = pd.Series(product)
    zc_num = len(product[product < 0])
    
    if abs(zc_num - (max_num + min_num)) <= 1:
        return True
    else:
        return False

def getspline(in_data):
    """
    -------------------------------------------
    getspline
    -------------------------------------------
    获取输入信号的条样插值连接得到的包络线
    
    input：
        in_data - 输入信号
        
    output：
        out_data - 输出包络
    """ 
    N = len(in_data)
    in_data = np.r_[in_data]
    
    #######################################
    ### max_ind是in_data的极大值所在的位置 ###
    #######################################
    max_ind = signal.argrelextrema(in_data,\
                            np.greater)[0].tolist()
    #   max_n是极大值点的个数
    max_n = len(max_ind)
    #   做条样曲线插值
    maxspline_x = np.array([0] + max_ind + [N-1])
    maxspline_y = in_data[maxspline_x.tolist()]
#-    spl = interpolate.InterpolatedUnivariateSpline(maxspline_x,maxspline_y)     
#-    maxspline_xx = np.arange(N)
#-    #   maxspline极大值包络
#-    maxspline = spl(maxspline_xx)    
    
    tck = interpolate.splrep(maxspline_x,maxspline_y)
    maxspline_xx = np.arange(N)
    maxspline = interpolate.splev(maxspline_xx,tck)
    #######################################          
    ### min_ind是in_data的极小值所在的位置 ###
    #######################################
    min_ind = signal.argrelextrema(in_data,\
                            np.less)[0].tolist()
    #   min_n是极大值点的个数
    min_n = len(min_ind)
    #   做条样曲线插值
    minspline_x = np.array([0] + min_ind + [N-1])
    minspline_y = in_data[minspline_x.tolist()]
#-    spl = interpolate.InterpolatedUnivariateSpline(minspline_x,minspline_y)     
#-    minspline_xx = np.arange(N)    
#-    #   minspline极大值包络
#-    minspline = spl(minspline_xx)
    tck = interpolate.splrep(minspline_x,minspline_y)
    minspline_xx = np.arange(N)
    minspline = interpolate.splev(minspline_xx,tck)
    return maxspline, minspline, max_n, min_n