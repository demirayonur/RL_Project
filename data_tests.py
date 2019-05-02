from improve_klfd.learning import KFLFD
import numpy as np
import pickle
import matplotlib.pyplot as plt


action_kfs = np.load('/home/user/Desktop/action_kf.npy')
goal_kfs = np.load('/home/user/Desktop/goal_kf.npy')
pca = pickle.load(open('/home/user/Desktop/pca.pk', 'rb'))

kf_lfd = KFLFD(6)
kf_lfd.from_loaded_data(action_kfs, goal_kfs, pca)

test_per_d = pickle.load(open('/home/user/Desktop/sim_demo/1/pcae.pk', 'rb'))[1:]
test_per = np.array([p[1] for p in test_per_d])
test_ts = np.array([p[0] for p in test_per_d])
test_per = pca.transform(test_per)
test_ts -= test_ts[0]

exp_rews = []

for i in range(1, len(test_per)+1):
    exp_rews.append(kf_lfd.s2d.get_expected_return(test_per[:i]))

plt.plot(test_ts, exp_rews)
plt.show()

f, axs = plt.subplots(3)

for _ in range(6):
    t, x = kf_lfd.generate_rollout()

    for i in range(3):
        axs[i].plot(t, x[:, i])

plt.show()
