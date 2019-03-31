from hmmlearn import hmm
import numpy as np
import scipy
import copy


class HMM(object):

    def __init__(self, demos, n_state):

        """
        Construction method

        :param demos: List of 2D numpy arrays.
                      Each 2D numpy array is (n_keyframe, n_dim)
                      for action model, n_dim = 7
                      for goal model, n_dim = 8

        :param n_state: Integer, number of possible hidden states.
        """

        self.demos = demos
        self.n_state = n_state
        self.n_dims = self.demos[0].shape[-1]

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

    @property
    def means(self):
        return self.model.means_

    @property
    def covars(self):
        return self.model.covars_

    def update_means(self, means):
        assert self.means.shape == means.shape
        self.model.means_ = copy.deepcopy(means)

    def update_covars(self, covars):
        assert self.covars.shape == covars.shape
        self.model.covars_ = copy.deepcopy(covars)

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