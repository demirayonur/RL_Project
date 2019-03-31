from sampling import sample_driver
from hmm import HMM
from stats import log_multivariate_normal_density as log_pdf

n_sample = 100

n_state_a = 5
n_state_g = 5

n_dim_a = 7
n_dim_g = 8

demos_action, demos_goal = sample_driver(n_sample, n_state_a, n_dim_a, n_state_g, n_dim_g)

hmm_model = HMM(demos_action, n_state_a)

x  = hmm_model.keyframe_generation(5)

