from hmm import HMM
import numpy as np
from motion import generate_motion
import copy


class BaseRL(object):
    def __init__(self, demos, n_state, n_offspring):
        '''
        Base RL class. Creates an hmm inside. Explores and updates hmm parameters. Generates motion for an
        episode. Stores rollout (or episode) information

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
        '''
        Resets all memory
        :return: None
        '''
        self.exp_means = np.zeros((self.hmm.n_state, self.n_offspring, self.hmm.n_dims))
        self.rewards = np.zeros((self.n_offspring))
        self.rollout_count = 0

    def remove_rollout(self):
        self.rollout_count -= 1

    def generate_rollout(self, std, duration=10.):
        '''
        Generates a rollout. Gets the most likely state sequence and fits a fifth order spline between randomly sampled
        points from each state. Note that keeps the generated rollout in memory. To remove last rollout call remove_rollout
        to clear all memory, call reset_rollout

        :param std: Sampling step size
        :param duration: Total motion duration
        :return:
        '''
        state_sequence = self.hmm.keyframe_generation(self.hmm.n_state)

        for state in state_sequence:
            mu_exp = np.random.multivariate_normal(self.hmm.means[state], self.hmm.covars[state]*std)
            self.exp_means[state, self.rollout_count] = mu_exp

        times, positions = generate_motion(self.exp_means[state_sequence,self.rollout_count,:], duration)
        self.std = std
        self.rollout_count += 1

        return times, positions

    def generate_motion(self, duration=10.):
        '''
        Gets the most likely state sequence and fits a fifth order spline between state centers

        :param duration: Total motion duration in seconds.
        :return: duration, position tuple
        '''
        state_sequence = self.hmm.keyframe_generation(self.hmm.n_state)
        return generate_motion(self.hmm.means[state_sequence,:], duration)

    def update(self, reward):
        raise NotImplementedError()


class HMMES(BaseRL):
    def __init__(self, demos, n_state, n_offspring, adapt_cov=True):
        BaseRL.__init__(self, demos, n_state, n_offspring)
        self.adapt_cov = adapt_cov

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
            costs_norm = np.asarray([-10* (x - min(self.rewards)) / costs_range for x in self.rewards])
            weights = np.exp(costs_norm)

        pr = weights/np.sum(weights)

        new_means = np.sum(self.exp_means*pr.reshape(1,-1,1), axis=1)

        if self.adapt_cov:
            new_cov = np.zeros_like(self.hmm.covars)

            for b in range(self.hmm.n_state):
                cov_b = np.zeros_like(self.hmm.covars[b])
                for k in range(self.n_offspring):
                    diff = (self.exp_means[b, k, :] - self.hmm.means[b, :]).reshape(-1, 1)
                    cov_b += pr[k] * np.matmul(diff, diff.T)

                new_cov[b] = copy.deepcopy(cov_b)

            self.hmm.update_covars(new_cov)

        self.hmm.update_means(new_means)
        self.reset_rollout()


class HMMPower(BaseRL):
    def __init__(self, demos, n_state, n_offspring, n_episode=None):
        BaseRL.__init__(self, demos, n_state, n_offspring)
        self.n_offspring = n_episode*n_offspring
        self.reset_rollout()

    def update(self, reward):
        if type(reward) == list or type(reward) == np.ndarray:
            reward = np.sum(reward)

        self.rewards[self.rollout_count-1] = reward

        rewards = self.rewards[:self.rollout_count]
        exp_means = self.exp_means[:,:self.rollout_count,:]

        dW = exp_means - self.hmm.means.reshape(-1,1,self.hmm.n_dims)

        if len(rewards) <= self.n_offspring:
            idx = list(range(len(rewards)))
        else:
            idx = np.argsort(rewards, axis=0)[0:self.n_offspring]

        # Power Update
        pW = np.sum(dW[:,idx,:] * rewards[idx].reshape(1, -1, 1), axis=1) / np.sum(rewards[idx])
        new_means = self.hmm.means + pW

        self.hmm.update_means(new_means)