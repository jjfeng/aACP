import numpy as np
from numpy import ndarray
from typing import List, Tuple

import data_x_gen_funcs

class SupportSimSettings:
    def generate_x(self, size: int):
        raise NotImplementedError("implement me")

class SupportSimSettingsUniform(SupportSimSettings):
    def __init__(self, num_p: int, min_func_name: str, max_func_name: str):
        self.num_p = num_p
        self.min_func_name = min_func_name
        self.max_func_name = max_func_name

    @property
    def max_func(self):
        return getattr(data_x_gen_funcs, self.max_func_name)

    @property
    def min_func(self):
        return getattr(data_x_gen_funcs, self.min_func_name)

    def generate_x(self, n: int, t_idx: int):
        """
        @return random vectors drawn from the support uniformly
        """
        min_x = self.min_func(t_idx)
        max_x = self.max_func(t_idx)
        return np.random.rand(n, self.num_p) * (max_x - min_x) + min_x

class SupportSimSettingsNormal(SupportSimSettings):
    def __init__(self, num_p: int, std_func_name, mu_func_name):
        self.num_p = num_p
        self.mu_func_name = mu_func_name
        self.std_func_name = std_func_name

    @property
    def std_func(self):
        return getattr(data_x_gen_funcs, self.std_func_name)

    @property
    def mu_func(self):
        return getattr(data_x_gen_funcs, self.mu_func_name)

    def generate_x(self, n: int, t_idx: int):
        """
        @return random vectors drawn from the support uniformly
        """
        mu_x = self.mu_func(t_idx)
        std_x = self.std_func(t_idx)
        return np.random.randn(n, self.num_p) * std_x + mu_x
