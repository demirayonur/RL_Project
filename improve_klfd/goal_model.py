import warnings
import numpy as np
from hmmlearn.hmm import GaussianHMM
from utils import aic


class HMMGoalModel(object):
    def __init__(self, per_data, per_lens=None, n_states=None):
        if per_lens is None:
            per_lens = list(map(len, per_data))

        if len(per_data.shape) > 2:
            per_data = per_data.reshape(-1, per_data.shape[-1])

        if n_states is None:
            components = [2, 4, 6, 8, 10]

            hmms = [GaussianHMM(n_components=c) for c in components]

            map(lambda g: g.fit(per_data, per_lens), hmms)
            scores = map(lambda g: aic(g, per_data, per_lens), hmms)

            max_score, self.hmm = sorted(zip(scores, hmms))[0]
        else:
            self.hmm = GaussianHMM(n_components=n_states)
            self.hmm.fit(per_data, per_lens)

        ll = self.hmm.score(per_data, per_lens)
        print "Goal HMM n_components", self.hmm.n_components, "Log likelihood", ll

        upper_idxs = [per_lens[0]-1]
        start_idxs = [0]
        for i in range(1, len(per_lens)):
            upper_idxs.append(upper_idxs[i-1]+per_lens[i])
            start_idxs.append(start_idxs[i-1]+per_lens[i-1])

        self.final_states = np.array(self.hmm.predict(per_data, per_lens))[upper_idxs]
        print self.final_states
        self.T = int(np.mean(per_lens))
        self.n_components = self.hmm.n_components

    def is_success(self, per_trj):
        per_trj = np.array(per_trj)
        states = self.hmm.predict(per_trj)
        final_state = states[-1]
        return final_state in self.final_states

    def sample(self, t=None):
        t = self.T if t is None else t
        return self.hmm.sample(t)
