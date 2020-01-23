import numpy as np

def linear_mu(xs):
    """
    Just sum all the covariates to get the mean
    """
    logit = np.maximum(np.minimum(np.sum(xs, axis=1) * 5, 5), -5)
    exp_logit = np.exp(logit)
    mu = exp_logit/(1 + exp_logit)
    return mu

def curvy_mu(xs):
    """
    Just sum all the covariates to get the mean
    """
    #logit = 0.5 * np.sum(np.sin(xs[:,:5]), axis=1) #+ 0.2 * np.power(np.sum(xs[:,5:10], axis=1), 2) + np.sum(xs[:,10:], axis=1)
    logit = 3 * np.sum((xs[:,:5]), axis=1) #+ 0.2 * np.power(np.sum(xs[:,5:10], axis=1), 2) + np.sum(xs[:,10:], axis=1)
    print("log", logit)
    logit = np.maximum(np.minimum(logit, 5), -5)
    exp_logit = np.exp(logit)
    mu = exp_logit/(1 + exp_logit)
    print("mu", mu)
    return mu
