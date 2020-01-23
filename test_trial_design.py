"""
Test trial designs. Evaluate what the max regret will be for these values
"""


import numpy as np
from scipy.stats import norm

b = 2
k = 0.5
w = 0.5
n = 30
tot_sum = 0
for t in range(1, 20):
    inner_sum = 0
    for j in range(1, t):
        denom1 = np.power(t + j + b, np.power(np.sqrt(k) + np.sqrt(w), 2))
        denom2 = np.sqrt(np.log(t + j + b) * (t - j + 1))
        inner_sum += 1/denom1/denom2
    tot_sum +=  inner_sum/np.sqrt(k * n)
print("tot", tot_sum)
