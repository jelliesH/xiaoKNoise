# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 15:06:49 2016

@author: 390672
"""

from multiprocessing import Pool

def f(x):
    return x*x

if __name__ == '__main__':
    with Pool(processes=4) as pool:         # start 4 worker processes
        result = pool.apply_async(f, (10,)) # evaluate "f(10)" asynchronously
        print(result.get(timeout=1))        # prints "100" unless your computer is *very* slow

        print(pool.map(f, range(10)))       # prints "[0, 1, 4,..., 81]"

        it = pool.imap(f, range(10))
#        print(next(it))                     # prints "0"
#        print(next(it))                     # prints "1"
#        print(it.next(timeout=1))           # prints "4" unless your computer is *very* slow

#        import time
#        result = pool.apply_async(time.sleep, (10,))
#        print(result.get(timeout=1))        # raises TimeoutError