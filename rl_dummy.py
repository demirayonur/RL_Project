from sampling import sample_driver
from rl import HMMES
import matplotlib.pyplot as plt
from plot_utils import plot_hmm
from motion import generate_motion
import numpy as np

def reward(t, x):
    via_point = np.array([0.5, 0.2])
    idx = np.argmin(np.abs(t - 0.5))
    y = x[idx]
    return np.linalg.norm(via_point - y)

n_sample = 10
n_state_a = 3
n_dim_a = 2
n_offspring = 15
n_episode = 100
duration = 1
std_init = 1.
std_decay = 0.9

demos_action = sample_driver(n_sample, n_state_a, n_dim_a)

model = HMMES(demos_action, n_state_a, 15)

for d in demos_action:
    plt.scatter(d[:,0], d[:,1], color='black')

axs = plt.gca()
plot_hmm(model.hmm, axs)

t, x = model.generate_motion(duration)
plt.plot(x[:,0], x[:,1])

for _ in range(5):
    t_r, x_r = model.generate_rollout(1., duration)
    plt.plot(x_r[:,0], x_r[:,1], linestyle=':')


plt.legend()
plt.show()


model.reset_rollout()
rewards = []
std = std_init

for e in range(n_episode):
    print e
    for r in range(n_offspring):
        t_r, x_r = model.generate_rollout(std, duration)
        rollout_reward = reward(t_r, x_r)
        rewards.append(rollout_reward)
        model.update(rollout_reward)
    std = std*std_decay
    print np.mean(rewards[-n_offspring])

plt.plot(rewards)
plt.show()

f, axs = plt.subplots(2)
t, x = model.generate_motion(duration)
# TODO: Generete motion and generate rollout can pick different keyframe squences (is this because this is a dummy data??)
axs[0].scatter(0.5, 0.5)
axs[1].scatter(0.5, 0.2)
for i in range(2):
    axs[i].scatter(0.3, 0.5)
    axs[i].plot(t, x[:,i])
plt.show()