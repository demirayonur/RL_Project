from scipy import linalg
import numpy as np


def log_multivariate_normal_density(x, means, covariances, min_covariance=1e-7):

    """
    This function computes the log probability of all given
    observations for all components
    param x: array_like, shape (n_samples, n_features)
              Each row corresponds to a single data point
    param means: array_like, shape (n_components, n_features)
                  Each row corresponds to a single mean vector
    param covariances: array_like, shape (n_components, n_features, n_features)
                        List of n_components covariance parameters for each Gaussian
    param min_covariance: optional, default --> 1e-7
    return: array_like, shape (n_samples, n_components)
             Array containing the log probabilities of each data point in
             x under each of the n_components multivariate Gaussian distributions
    """

    n_samples, n_dim = x.shape
    nmix = len(means)
    log_prob = np.empty((n_samples, nmix))
    for c, (mu, cv) in enumerate(zip(means, covariances)):
        try:
            cv_chol = linalg.cholesky(cv, lower=True)
        except linalg.LinAlgError:
            # The model is most probably stuck in a component with too
            # few observations, we need to reinitialize this components
            try:
                cv_chol = linalg.cholesky(cv + min_covariance * np.eye(n_dim), lower=True)
            except linalg.LinAlgError:
                raise ValueError('Covariances must be symmetric, positive-definite')

        cv_log_det = 2 * np.sum(np.log(np.diagonal(cv_chol)))
        cv_sol = linalg.solve_triangular(cv_chol, (x - mu).T, lower=True).T
        log_prob[:, c] = - .5 * (np.sum(cv_sol ** 2, axis=1) + n_dim * np.log(2 * np.pi) + cv_log_det)

    return log_prob