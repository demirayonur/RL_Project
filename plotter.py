import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
from improve_klfd.plot_utils import plot_ellipse


log_dir = '/home/user/Desktop/5'
n_greedy = 9

success = []
rewards = []

for r in range(1,n_greedy+1):
    log = np.loadtxt(os.path.join(log_dir, 'greedy_{}.csv'.format(r)), delimiter=',')
    suc, reward = log
    success.append(suc)
    rewards.append(reward)

action_model = pickle.load(open('/home/user/Desktop/5/action_model.pk', 'rb'))
means_initial = action_model.hmm.means
covars_initial = action_model.hmm.covars

means_final = np.loadtxt('/home/user/Desktop/5/means_8.csv', delimiter=',')
covars_final = np.load('/home/user/Desktop/5/covars_8.csv.npy')

plt.plot(success)
plt.show()

f, axs = plt.subplots(1,2)

axs[1].scatter(means_final[:, 0], means_final[:, 1])

for i in range(3):
    axs[0].scatter(means_initial[i, 0], means_initial[i, 1], color='C{}'.format(i+1))
    plot_ellipse(axs[0], means_initial[i], covars_initial[i, :2, :2], color='C{}'.format(i+1))

    axs[1].scatter(means_final[i, 0], means_final[i, 1], color='C{}'.format(i + 1))
    plot_ellipse(axs[1], means_final[i], covars_final[i, :2, :2], color='C{}'.format(i + 1))

xval = map(lambda x: x.get_xlim(), axs)
yval = map(lambda x: x.get_ylim(), axs)

for subplot in axs:
    subplot.set_xlim(min((xval[0][0], xval[1][0])), max((xval[0][1], xval[1][1])))
    subplot.set_ylim(min(yval[0][0], yval[1][0]), max(yval[0][1], yval[1][1]))

plt.show()


