# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 16:31:03 2016

@author: 390672
"""

import numpy as np
import matlab.engine
# 打开matlab引擎
eng = matlab.engine.start_matlab()

x = np.linspace(0,2*np.pi,10000)
y = np.sin(x) + np.sin(30*x)
ya = y.tolist()
# 得到的ya是matlab能读得进去的类型 list - cell
# 将list转变成 numeric array
yb = matlab.double(ya)

imfs = eng.emd(yb)
imfs_p = []
# 将结果转换成python的数据类型
for each in imfs:
    imfs_p.append(eng.num2cell(each))
# 关闭matlab引擎
eng.quit()


ta = time()
wave2 = matlab.double(wave.tolist())
imfs = eng.emd(wave2)
tb = time()
print('读取完成,共耗时：%f'%(tb - ta) + '秒')