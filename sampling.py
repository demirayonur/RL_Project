import sklearn.datasets
import numpy as np
import random


def sum_normalize(arr):

    """
    This function normalizes the given 1D array so that
    the sum of the elements in it is equal to 1. Instead
    of inplacing, it returns a new 1D array.
    :param arr: 1D Numpy Array
    :return: 1D Numpy Normalized Array
    """

    return np.array([i/np.sum(arr) for i in arr])


def random_prior_prob_generation(n_state):
    """
    This function returns a prior distribution based on the
    number of states such that sum of the all probabilities
    is equal to 1.

    :param n_state: Integer
    :return: 1D Numpy Normalized Array
    """

    return sum_normalize(np.random.uniform(0, 1, n_state))


def random_transition_prob_generation(n_state):
    """
    This function returns a markov chain.
    Properties of markov chains:

    i) Each element of MC is greater than or equal to zero
       and less than or equal to one.

    ii) Sum of elements in each row has to be equal to 1.

    :param nstate: Integer
    :return: 2D Numpy Array
    """

    trans_prob = []
    for i in range(n_state):
        trans_prob.append(sum_normalize(np.random.uniform(0, 1, n_state)))

    return np.array(trans_prob)


def mean_vector_generation(n_state, n_dim):
    """
    This function return random mean vectors for each state.
    There is not any constraint on it.

    :param n_state: Integer
    :param n_dim: Integer
    :return: 2D Numpy Array
    """

    means = []
    for i in range(n_state):
        means.append(np.random.uniform(-1, +1, n_dim))

    return np.array(means)


def cov_matrix_generation(n_state, n_dim):
    """
    This function returns positive semidefinite covariance matrix
    for each state by using sklearn.datasets

    :param n_state: Integer
    :param n_dim: Integer
    :return: 3D Numpy Array
    """

    covs = np.empty(shape=(n_state, n_dim, n_dim))
    for i in range(n_state):
        covs[i, :] = sklearn.datasets.make_spd_matrix(n_dim)*0.01

    return covs

class RandHMM(object):

    def __init__(self, n_state, ndim):
        """
        :param n_state: Integer, number of hidden states
        :param ndim: Integer, dimension of the states and observations
        """

        self.n_state = n_state
        self.ndim = ndim
        self.prior_probs = random_prior_prob_generation(self.n_state)
        self.transition_probs = random_transition_prob_generation(self.n_state)
        self.means = mean_vector_generation(self.n_state, self.ndim)
        self.covars = cov_matrix_generation(self.n_state, self.ndim)

    def sample(self, size_sample, kf_in_each_demo):
        """

        :param size_sample: integer
        :return: list of 2D Numpy Arrays
        """
        demos = []

        sample_size = 0
        while sample_size < size_sample:

            state_seq = []
            current_state = which_index(self.prior_probs)
            state_seq.append(np.random.multivariate_normal(self.means[current_state], self.covars[current_state]))
            kf_count = 0
            while kf_count < kf_in_each_demo[sample_size]-1:

                current_state = which_index(self.transition_probs[current_state, ])
                state_seq.append(np.random.multivariate_normal(self.means[current_state], self.covars[current_state]))
                kf_count += 1

            demos.append(np.asarray(state_seq))
            sample_size += 1

        return demos


def which_index(arr):

    sorted_ind = sorted(range(len(list(arr))), key=lambda k: arr[k])
    cdf = []
    for i in sorted_ind:
        if len(cdf)==0:
            cdf.append(arr[i])
        else:
            cdf.append(cdf[-1]+arr[i])
    rand = random.uniform(0,1)

    ind = 0
    for i, p  in enumerate(cdf):
        if p >= rand:
            ind = i
            break
    return ind


def sample_driver(n_sample, n_state_a, n_dim_a):

    action_model = RandHMM(n_state_a, n_dim_a)

    lens = []

    count = 0
    while count < n_sample:
        lens.append(random.randint(4, 7))
        count += 1

    return action_model.sample(n_sample, lens)