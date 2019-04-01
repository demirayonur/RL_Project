from sampling import sample_driver
from rl import HMMES, HMMPower
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

model = HMMES(demos_action, n_state_a, n_offspring, adapt_cov=False)

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
plt.savefig('figures/initial.png', dpi=200, bbox_inches='tight')


model.reset_rollout()
es_rewards = []
std = std_init

print "Training ES"
for e in range(n_episode):
    for r in range(n_offspring):
        t_r, x_r = model.generate_rollout(std, duration)
        rollout_reward = reward(t_r, x_r)
        es_rewards.append(rollout_reward)
        model.update(rollout_reward)
    std = std*std_decay


es_cov_rewards = []
std = std_init

model = HMMES(demos_action, n_state_a, n_offspring, adapt_cov=True)

print "Training ES CMA"
for e in range(n_episode):
    for r in range(n_offspring):
        t_r, x_r = model.generate_rollout(std, duration)
        rollout_reward = reward(t_r, x_r)
        es_cov_rewards.append(rollout_reward)
        model.update(rollout_reward)
    std = std*std_decay
model.hmm.save('.')
# f, axs = plt.subplots(2)
# t, x = model.generate_motion(duration)
# axs[0].scatter(0.5, 0.5)
# axs[1].scatter(0.5, 0.2)
# for i in range(2):
#     axs[i].scatter(0.3, 0.5)
#     axs[i].plot(t, x[:,i])
# plt.show()

model = HMMPower(demos_action, n_state_a, n_episode*n_offspring, n_offspring)

power_rewards = []
std = std_init

print "Training PoWER"
for e in range(n_episode*n_offspring):
    t_r, x_r = model.generate_rollout(std, duration)
    rollout_reward = reward(t_r, x_r)
    power_rewards.append(rollout_reward)
    model.update(-rollout_reward)
    std = std * std_decay

plt.plot(power_rewards, label='PoWER')
plt.plot(es_rewards, label='ES')
plt.plot(es_cov_rewards, label='ES-Cov')
plt.legend()
plt.savefig('figures/rewards.png', dpi=200, bbox_inches='tight')
