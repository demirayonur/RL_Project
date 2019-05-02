import numpy as np
import math
import os


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


def ls_dir_idxs(base_dir):
    idx_dirs = filter(str.isdigit, os.listdir(base_dir))
    idx_dirs = sorted(idx_dirs, key=lambda x: int(x))
    return map(lambda x: os.path.join(base_dir, x), idx_dirs)


def aic(hmm, per_data, per_lens, mean=True):
    lower = 0
    n_states = hmm.n_components
    n_dim = per_data.shape[-1]
    n_params = 0.0

    # Means
    n_params += n_states * n_dim
    # Covars Diagonal
    n_params += n_states * n_dim
    # Trans mat
    n_params += n_states ** 2
    # Priors
    n_params += n_states

    total_aic = 0.0

    for i in range(len(per_lens)):
        data = per_data[lower:lower + per_lens[i]]
        lower += per_lens[i]
        ll = hmm.score(data)
        aic = np.log(len(data)) * n_params - 2*ll
        total_aic += aic

    return total_aic / len(per_lens) if mean else total_aic
