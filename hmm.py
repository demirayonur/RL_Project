from hmmlearn import hmm
from utils import *
from pulp import *
import numpy as np
import scipy


class HMM(object):

    def __init__(self, demos, n_state):

        """
        Construction method

        :param demos: List of 2D numpy arrays.
                      Each 2D numpy array is (n_keyframe_in_the_corresponding_demo, n_dim)
                      for action model, n_dim = 7
                      for goal model, n_dim = 8

        :param n_state: Integer, number of possible hidden states.
        """

        self.demos = demos
        self.n_state = n_state

        """
        In this project, to learn and evaluate the Hidden Markov Model
        hmmlearn is used. hmmlearn deals with the multiple demos by
        taking all keyframes in one 2D Numpy array and length of all 
        sequences. In our case lengths is a list keeping the information
        of keyframe numbers in each demos. 
        """

        lengths = [demo.shape[0] for demo in self.demos]
        observations = np.concatenate([demo for demo in self.demos])

        self.model = hmm.GaussianHMM(n_components=self.n_state)
        self.model.fit(observations, lengths)

        """
        Statistical Parameters Learned by using Baum_Welch
        """

        self.prior_prob = self.model.startprob_
        self.transition_prob = self.model.transmat_
        self.means = self.model.means_
        self.covars = self.model.covars_

    def covariance_matrix_check(self, our_tol=1e-8):

        """
        This function checks covariance matrices of all states
        in terms of two things.

        * Whether it is symmetric or not
        * Whether it is positive semi definite or not

        :param our_tol: optional, default = 1e-8
        :return: If there is a problem raises error, otherwise nothing
        """

        def check_symmetric(a, tol=our_tol):
            return np.allclose(a, a.T, atol=tol)

        def is_psd(a, tol=our_tol):
            E, V = scipy.linalg.eigh(a)
            return np.all(E > -tol)

        for cov in self.covars:
            # Symmetry check
            if not check_symmetric(cov):
                raise ValueError('Covariance matrix must be symmetric')
            # PSD check
            if not is_psd(cov):
                raise ValueError('Covariance matrix must be positive semidefinite')

    def covariance_matrix_fix(self, our_tol=1e-8):

        """
        This function checks covariance matrices of all states
        in terms of symmetry and whether it is psd or not. If
        there is a problem it tries to fix it.

        This function applies the idea proposed in the paper
        "Computing a nearest symmetric positive semidefinite
        matrix" (1988) by N.J. Higham.
        """

        def is_psd(a, tol=our_tol):
            E, V = scipy.linalg.eigh(a)
            return np.all(E > -tol)

        for ind,cov in enumerate(self.covars):
            if not is_psd(cov):

                B = (cov + cov.T) / 2
                _, s, V = np.linalg.svd(B)
                H = np.dot(V.T, np.dot(np.diag(s), V))
                A2 = (B + H) / 2
                A3 = (A2 + A2.T) / 2

                if is_psd(A3):
                    self.covars[ind] = A3

                else:

                    spacing = np.spacing(np.linalg.norm(cov))
                    unit = np.eye(cov.shape[0])
                    k = 1
                    while not is_psd(A3):
                        mineig = np.min(np.real(np.linalg.eigvals(A3)))
                        A3 += unit * (-mineig * k ** 2 + spacing)
                        k += 1
                    self.covars[ind] = A3

    def keyframe_generation(self, num_keyframe):

        f1 = - log_w_zero_mask_1D(self.prior_prob)
        d1 = - log_w_zero_mask_2D(self.transition_prob)

        opt_model = LpProblem('Keyframe Sequence Generation', LpMinimize)

        # Parameters

        f, d, P, T = {}, {}, {}, {}

        for i in range(self.n_state):
            f[(i)] = f1[i]
            P[(i)] = self.prior_prob[i]
            for j in range(self.n_state):
                d[(i, j)] = d1[i,j]
                T[(i, j)] = self.transition_prob[i, j]

        # Decision Variables

        x = LpVariable.dicts('X', [(i, k) for i in range(self.n_state) for k in range(num_keyframe)], 0, 1, LpBinary)
        y = LpVariable.dicts('Y', [(i, j, k) for i in range(self.n_state) for j in range(self.n_state)
                                   for k in range(num_keyframe-1)], 0, 1, LpBinary)

        # Objective Functions

        opt_model += lpSum(x[(i, 0)]*f[(i)] for i in range(self.n_state)) + \
                     lpSum(y[(i,j,k)]*d[(i,j)] for i in range(self.n_state) for j in range(self.n_state)
                                               for k in range(num_keyframe-1))

        # Assignment Constraints

        for i in range(self.n_state):

            opt_model += lpSum(x[(i,k)] for k in range(num_keyframe)) <= 1

        for k in range(num_keyframe):

            opt_model += lpSum(x[(i, k)] for i in range(self.n_state)) == 1

        # Zero Probability Constraints

        for i in range(self.n_state):

            if P[(i)] == 0:
                opt_model += x[(i,0)] == 0

            for j in range(self.n_state):

                if T[(i,j)] == 0:
                    opt_model += lpSum(y[i,j,k] for k in range(num_keyframe-1)) == 0

        # Linearization Constraints

        for i in range(self.n_state):
            for j in range(self.n_state):
                for k in range(num_keyframe-1):
                    opt_model += y[(i,j,k)] <= x[(i,k)]
                    opt_model += y[(i,j,k)] <= x[(j,k+1)]
                    opt_model += y[(i,j,k)] >= x[(i,k)] + x[(j, k + 1)] -1

        # Model Solve

        opt_model.solve()

        # Extracting Optimal Solution
        seq = []
        for k in range(num_keyframe):
            for i in range(self.n_state):
                if x[(i,k)].varValue == 1:
                    seq.append(i)

        return seq

