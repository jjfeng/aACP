import os
import numpy as np
from numpy import ndarray
import scipy.stats
import pickle
import json
from itertools import chain, combinations
from operator import mul
from functools import reduce

def pickle_to_file(obj, file_name):
    with open(file_name, "wb") as f:
        pickle.dump(obj, f, protocol=-1)

def pickle_from_file(file_name):
    with open(file_name, "rb") as f:
        out = pickle.load(f)
    return out

def process_params(param_str, dtype, split_str=","):
    if param_str:
        return [dtype(r) for r in param_str.split(split_str)]
    else:
        return []
