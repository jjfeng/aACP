import numpy as np
import scipy.stats
import scipy.optimize

norm = scipy.stats.norm()

delta_b = 10
delta_scale = 1
def get_delta_t(t):
    return delta_scale * np.sqrt(np.log(t + delta_b))

t = 50
n = 200
eps_n = 0.05

b0 = 5.
beta = 1
def get_expected_loss(i):
    return np.power(i + b0, -beta)

print("initial performance", get_expected_loss(0))

def get_type_ii_error_simple():
    lag_time = np.floor(np.power(get_delta_t(t), 2))
    sigma_max = min(np.sqrt(get_expected_loss(0) + get_expected_loss(t)), 1)
    thres = eps_n * np.sqrt(n * lag_time)/sigma_max - get_delta_t(t)
    return 1 - norm.cdf(thres)

def get_sigma2_max(c0, t, i):
    # This is the sigma max used in the simple analysis
    sigma_max_init_v_t = get_expected_loss(0) + get_expected_loss(t)
    # This is the sigma max used in my fancy analysis
    sigma_max_c0 = np.power(c0, t + 1 - i) * eps_n + get_expected_loss(i - 1) + get_expected_loss(i)
    return min(sigma_max_init_v_t, sigma_max_c0)

def get_type_ii_error(c, debug=False):
    prob_tot = 0
    for i in range(t + 1):
        lag_time = np.floor(np.power(get_delta_t(i), 2))
        log_c_factor1 = np.sum(np.log(c[i + 1:])) if i < t else 0
        if log_c_factor1 > 4:
            prob_i = 0
        else:
            numerator = eps_n * np.sqrt(n * lag_time) * np.exp(log_c_factor1)
            c_factor0 = np.exp(np.sum(np.log(c[i:])))
            sigma2_max = get_sigma2_max(c_factor0, t, i)
            thres = numerator/min(1., np.sqrt(sigma2_max)) - get_delta_t(i)
            prob_i = 1 - norm.cdf(thres)
            if debug and prob_i > 1e-10:
                print("round i", i)
                print("  numer", numerator)
                print("  sigma2_max", sigma2_max)
                print("  prob i", prob_i)
        prob_tot += prob_i

    if debug:
        print("last lag_time", lag_time)
    return prob_tot

init_guess = np.ones(t + 1) * 1.1
bounds = [(1,1000)] * (t + 1)
res = scipy.optimize.minimize(get_type_ii_error, x0=init_guess, bounds=bounds)
print("final type ii error RECURSE", res.fun)
print(res.x)
get_type_ii_error(res.x, debug=True)

final_simple = get_type_ii_error_simple()
print("final type ii error SIMPLE", final_simple)
