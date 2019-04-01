from hmm import HMM
import numpy as np
from motion import generate_motion


class HMMES(object):
    def __init__(self, demos, n_state, n_offspring):
        '''
        HMM Evolution Strategy. Creates an hmm inside. Explores and updates hmm parameters. Generates motion for an
        episode. Uses an Evolution Strategy, PI^2-ES for each state separately.

         :param demos: List of 2D numpy arrays.
                      Each 2D numpy array is (n_keyframe, n_dim)
                      for action model, n_dim = 7
                      for goal model, n_dim = 8

        :param n_state: Integer, number of possible hidden states.
        :param n_offspring: Number of offsprings (rollouts) in an episode
        '''
        self.hmm = HMM(demos, n_state)

        self.n_offspring = n_offspring
        self.reset_rollout()

    def reset_rollout(self):
        self.exp_means = np.zeros((self.hmm.n_state, self.n_offspring, self.hmm.n_dims))
        self.rewards = np.zeros((self.n_offspring))
        self.rollout_count = 0

    def remove_rollout(self):
        self.rollout_count -= 1

    def generate_rollout(self, std, duration=10.):
        state_sequence = self.hmm.keyframe_generation(self.hmm.n_state)

        for state in state_sequence:
            mu_exp = np.random.multivariate_normal(self.hmm.means[state], self.hmm.covars[state]*std)
            self.exp_means[state, self.rollout_count] = mu_exp

        times, positions = generate_motion(self.exp_means[state_sequence,self.rollout_count,:], duration)
        self.std = std
        self.rollout_count += 1

        return times, positions

    def generate_motion(self, duration=10.):
        state_sequence = self.hmm.keyframe_generation(self.hmm.n_state)
        return generate_motion(self.hmm.means[state_sequence,:], duration)

    def update(self, reward):
        if type(reward) == list or type(reward) == np.ndarray:
            reward = np.sum(reward)

        self.rewards[self.rollout_count-1] = reward

        if self.rollout_count < self.n_offspring:
            return

        costs_range = max(self.rewards) - min(self.rewards)
        if costs_range == 0:
            weights = np.full(self.n_offspring, 1.0)
        else:
            costs_norm = np.asarray([-self.n_offspring * (x - min(self.rewards)) / costs_range for x in self.rewards])
            weights = np.exp(costs_norm)

        pr = weights/np.sum(weights)

        new_means = np.sum(self.exp_means*pr.reshape(1,-1,1), axis=1)
        self.hmm.update_means(new_means)

        self.reset_rollout()