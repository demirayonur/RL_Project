import numpy as np
import copy


class Sparse2Dense(object):
    def __init__(self, goal_model, gamma=0.9, n_iter=50):
        self.goal_model = goal_model
        self.gamma = gamma
        self.v_table = np.zeros(self.goal_model.n_components)
        self.actions = np.arange(self.goal_model.n_components)
        self.states = np.arange(self.goal_model.n_components)

        values_iter = np.zeros((n_iter, self.goal_model.n_components))

        for n in range(n_iter):
            print "{}/{}".format(n+1, n_iter)
            values_iter[n] = self.learn(values_iter[n])

            self.v_table = np.mean(values_iter, axis=0)
            self.v_table = (self.v_table - self.v_table.min()) / (
                        self.v_table.max() - self.v_table.min())  # Normalize values

            succ_idx = np.argmax(self.v_table)
            self.v_table[succ_idx] *= 10.

    def learn(self, v, max_iter=1000):
        for _ in range(max_iter):
            features, states = self.goal_model.sample()
            v, diff = self.update(v, features, states)
            if diff < 1e-10:
                break
        return v

    def update(self, v_table, features, states):
        v_table_new = copy.deepcopy(v_table)

        is_success = self.goal_model.is_success(features)
        rewards = [0.] * len(states)
        rewards[-1] = 1.0 if is_success else 0.0

        for t in range(len(states)):
            s = states[t]
            r = rewards[t]

            qs = []
            for a in self.actions:
                v = 0.0
                for s_prime in self.states:
                    v += self.goal_model.hmm.transmat_[s][s_prime] * v_table[s_prime]
                qs.append(r + self.gamma * v)

            v_table_new[s] = np.max(qs)

        diff = np.sum(np.square(v_table_new-v_table))
        return v_table_new, diff

    def get_reward(self, per_seq):
        states = self.goal_model.hmm.predict(per_seq)
        rewards = []

        for t in range(len(states)):
            s = states[t]
            rewards.append(self.v_table[s])

        return np.array(rewards)

    def get_expected_reward_for_state(self, posterior):
        R = 0.
        for i, p_s in enumerate(posterior):
            R += self.v_table[i] * p_s
        return R

    def get_expected_reward(self, per_seq):
        posterior = self.goal_model.hmm.predict_proba(per_seq)[-1]
        return self.get_expected_reward_for_state(posterior)

    def get_expected_return(self, per_seq):
        posterior = self.goal_model.hmm.predict_proba(per_seq)

        expected_return = 0.0
        for i, post in enumerate(posterior):
            r = self.get_expected_reward_for_state(post)
            expected_return += r

        return expected_return