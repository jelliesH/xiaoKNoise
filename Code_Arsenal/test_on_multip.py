# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 14:03:08 2016
尝试一种程序结构，看这样设计并行是否可行
@author: 390672
"""
from multiprocessing import Pool,Queue,Manager
import time,os,random
import matlab.engine



def long_time_task(ii,q):
    eng = matlab.engine.start_matlab()
    
    print('running task%s(%s)……'%(ii,os.getpid()))
    start = time.time()
    num = eng.rand(1)
    print('$%f'%num)
    time.sleep(num*2)
    end = time.time()
    eng.quit()
    q.put(num+ii)
    print('Task %s runs %0.2f seconds'%(ii,(end - start)))

if __name__ == '__main__':
    p = Pool(8)
    manager = Manager()
    q = manager.Queue()
#    # 计算matlab engine启动耗时
#    start = time.time()
#    eng = matlab.engine.start_matlab()
#    eng.quit()
#    end = time.time()
#    print("这是matlab engine启动耗时：%0.2f"%(end - start))
    start = time.time()
    #eng = matlab.engine.start_matlab()
    for ii in range(10):
        p.apply_async(long_time_task,args=(ii,q,))
    outcome = []
 
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    
    # 获取队列中的元素
    while not(q.empty()):
        item = q.get(True)
        outcome.append(item)
        print('>>>>>%f'%item)
        #q.task_done()
    end = time.time()   
    print('All subprocesses done，总耗时：%0.2f秒'%(end - start))
    #eng.quit()