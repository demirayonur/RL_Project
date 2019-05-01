import matplotlib.pyplot as plt
from improve_klfd.plot_utils import plot_hmm
import numpy as np


class Trainer(object):
    def __init__(self, model, demos_action, name, dpi=250, duration=1.0):
        self.model = model
        self.dpi = dpi
        self.duration = duration
        self.via_point = np.array([0.5, 0.2])

        for d in demos_action:
            plt.scatter(d[:, 0], d[:, 1], color='black')

        axs = plt.gca()
        plot_hmm(self.model.hmm, axs)

        self.t_orig, self.x_orig = self.model.generate_motion(duration)
        plt.plot(self.x_orig[:, 0], self.x_orig[:, 1])

        for _ in range(5):
            t_r, x_r = self.model.generate_rollout(1., duration)
            plt.plot(x_r[:, 0], x_r[:, 1], linestyle=':')

        plt.scatter(self.via_point[0], self.via_point[1], label='w', marker='X')

        self.model.reset_rollout()

        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        self.name = name
        plt.savefig('figures/initial_{}.png'.format(self.name), dpi=self.dpi, bbox_inches='tight')
        plt.clf()

        self.rewards = []

        self.via_t = 0.5

    def run(self, n_episode, n_rollout, std_init, decay_std, negative=False):
        print "Running {}".format(self.name)
        std = std_init
        for e in range(n_episode):
            for r in range(n_rollout):
                t_r, x_r = self.model.generate_rollout(std, self.duration)
                rollout_reward = self.reward(t_r, x_r)
                self.rewards.append(rollout_reward)
                self.model.update(-rollout_reward if negative else rollout_reward)
            std = decay_std(std)
            print e, np.mean(self.rewards[:-n_rollout+1])
        self.save_result()

    def reward(self, t, x):
        idx = np.argmin(np.abs(t - self.via_t))
        y = x[idx]
        return np.linalg.norm(self.via_point - y)

    def save_result(self):
        t, x = self.model.generate_motion(self.duration)

        axs = plt.gca()
        plot_hmm(self.model.hmm, axs)
        plt.plot(x[:, 0], x[:, 1])
        plt.scatter(self.via_point[0], self.via_point[1], label='w', marker='X')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.savefig('figures/final_{}.png'.format(self.name), dpi=self.dpi, bbox_inches='tight')

        plt.clf()

        f, axs = plt.subplots(2)

        axs[0].scatter(self.via_t, self.via_point[0])
        axs[1].scatter(self.via_t, self.via_point[1])

        for i in range(2):
            axs[i].plot(t, x[:,i])
            axs[i].plot(self.t_orig, self.x_orig[:,i], linestyle=':')
            axs[i].set_xlabel('t')

        axs[0].set_ylabel('x')
        axs[1].set_ylabel('y')

        plt.legend()
        plt.savefig('figures/temporal_{}.png'.format(self.name), dpi=self.dpi, bbox_inches='tight')
        plt.clf()