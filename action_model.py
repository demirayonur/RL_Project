import numpy as np
from hmmlearn.hmm import GaussianHMM


action_kfs = np.load('/home/user/Desktop/action_kf.npy')

for s in [2,3,4,5]:
    hmm = GaussianHMM(s)
    hmm.fit(action_kfs[:, :, 1:].reshape(-1, 7), [6]*3)
    print hmm.score(action_kfs[:, :, 1:].reshape(-1, 7), [6]*3)
    print [np.linalg.norm(c) for c in hmm.covars_]