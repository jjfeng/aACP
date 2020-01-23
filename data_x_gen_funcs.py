import numpy as np


def min_x_func_constant(t):
    return -1

def max_x_func_constant(t):
    return 1

def std_func_changing(t):
    return 1

def mu_func_changing(t):
    return (t % 40 - 20) * 0.002
