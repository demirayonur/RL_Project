from sampling import sample_driver
from hmm import HMM
from stats import log_multivariate_normal_density as log_pdf

n_sample = 100
n_state_a = 5
n_dim_a = 7

demos_action = sample_driver(n_sample, n_state_a, n_dim_a)

hmm_model = HMM(demos_action, n_state_a)