import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
from improve_klfd.plot_utils import plot_ellipse


base_dir = '/Users/cem/Desktop/experiments'
n_greedy = 10
exp_idxs = list(range(11, 21))

all_success = []
all_rewards = []

for exp in exp_idxs:
    success = []
    rewards = []

    for r in range(1,n_greedy+1):
        log = np.loadtxt(os.path.join(base_dir, str(exp), 'greedy_{}.csv'.format(r)), delimiter=',')
        suc, reward = log
        success.append(suc)
        rewards.append(reward)

    log = np.loadtxt(os.path.join(base_dir, str(exp), 'greedy_fin.csv'.format(r)), delimiter=',')
    suc, reward = log
    success.append(suc)
    rewards.append(reward)

    all_rewards.append(rewards)
    all_success.append(success)

all_success = np.array(all_success)
all_rewards = np.array(all_rewards)

X = np.arange(1, n_greedy+2)
mu_s = np.mean(all_success, axis=0)
std_s = np.var(all_success, axis=0)

plt.title("5 States 10 Episodes")
plt.ylabel("Success")
plt.xlabel("Episode")
plt.plot(X, mu_s)
plt.fill_between(X, np.clip(mu_s+std_s,0,1), mu_s-std_s, alpha=0.2)
plt.savefig('5s_10e_succ.png', bbox_inches='tight', dpi=300)