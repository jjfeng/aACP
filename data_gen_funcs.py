import numpy as np

def linear_mu(xs):
    """
    Just sum all the covariates to get the mean
    """
    raw_mu = np.sum(xs, axis=1) * 0.5
    return raw_mu

def linear_sigma(xs):
    sigma_seq = np.ones((xs.shape[0], 1)) #0.3 * np.abs(xs[:,0] + 2*xs[:,1] + xs[:,2] + 2*xs[:,3]) + 0.5
    return sigma_seq
