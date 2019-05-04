from improve_klfd.learning import KFLFD
import numpy as np
import pickle
import matplotlib.pyplot as plt
from improve_klfd.plot_utils import plot_hmm
from mpl_toolkits.mplot3d import Axes3D


action_kfs = np.load('/home/user/Desktop/action_kf.npy')
goal_kfs = np.load('/home/user/Desktop/goal_kf.npy')
pca = pickle.load(open('/home/user/Desktop/pca.pk', 'rb'))

kf_lfd = KFLFD(0.05, 6)
kf_lfd.from_loaded_data(action_kfs, goal_kfs, pca)

test_per_d = pickle.load(open('/home/user/Desktop/sim_demo/1/pcae.pk', 'rb'))[1:]
test_per = np.array([p[1] for p in test_per_d])
test_ts = np.array([p[0] for p in test_per_d])
test_per = pca.transform(test_per)
test_ts -= test_ts[0]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for _ in range(6):
    t, x = kf_lfd.generate_rollout()
    ax.plot(x[:,0], x[:,1], x[:,2])


# plot_hmm(kf_lfd.action_model.hmm, plt.gca())

plt.show()