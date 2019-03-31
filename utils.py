import numpy as np
import math

def log_w_zero_mask_1D(arr):

    tmp = []
    for i in arr:
        if i==0:
            tmp.append(0)
        else:
            tmp.append(math.log(i))

    return np.asarray(tmp)

def log_w_zero_mask_2D(arr):

    tmp = []
    for i in arr:
        tm = []
        for j in i:
            if j==0:
                tm.append(0)
            else:
                tm.append(math.log(j))
        tmp.append(tm)

    return np.asarray(tmp)