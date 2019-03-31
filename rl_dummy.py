from sampling import sample_driver
from rl import HMMES
import matplotlib.pyplot as plt
from plot_utils import plot_hmm
from motion import generate_motion


n_sample = 10
n_state_a = 3
n_dim_a = 2
n_offspring = 15
n_episode = 100

demos_action = sample_driver(n_sample, n_state_a, n_dim_a)

model = HMMES(demos_action, n_state_a, 15)

for d in demos_action:
    plt.scatter(d[:,0], d[:,1], color='black')

axs = plt.gca()
plot_hmm(model.hmm, axs)

t, x = generate_motion(model.hmm.means, 10)

plt.plot(x[:,0], x[:,1])

for _ in range(n_offspring):
    t_r, x_r = model.generate_rollout(1.)
    plt.plot(x_r[:,0], x_r[:,1], linestyle=':')

plt.show()