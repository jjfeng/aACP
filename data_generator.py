import numpy as np
from numpy import ndarray
import scipy.stats

import data_gen_funcs
import data_gen_funcs_bernoulli
from dataset import Dataset
from support_sim_settings import SupportSimSettings

class DataGenerator:
    """
    Simulation engine
    """
    def __init__(self,
            sim_func_form: str,
            sim_func_name: str,
            support_sim_settings: SupportSimSettings,
            num_classes: int =1,
            noise_sd: float = 1,
            std_dev_x: float =1,
            max_y: float =1,
            min_y: float =0):
        self.num_p = support_sim_settings.num_p
        self.std_dev_x = std_dev_x
        self.num_classes = num_classes
        self.min_y = min_y
        self.max_y = max_y
        self.noise_sd = noise_sd
        self.sim_func_form = sim_func_form
        self.support_sim_settings = support_sim_settings
        if sim_func_form in ["gaussian", "bounded_gaussian"]:
            self.mu_func = getattr(data_gen_funcs, sim_func_name + "_mu")
            self.raw_sigma_func = getattr(data_gen_funcs, sim_func_name + "_sigma")
        elif sim_func_form == "bernoulli":
            self.mu_func = getattr(data_gen_funcs_bernoulli, sim_func_name + "_mu")
        else:
            print(sim_func_form)
            raise ValueError("huh?")

    def sigma_func(self, xs: ndarray):
        """
        @return sigma when Y|X is gaussian
        """
        if self.sim_func_form == "gaussian":
            return self.noise_sd * self.raw_sigma_func(xs)
        elif self.sim_func_form == "bounded_gaussian":
            return self.noise_sd * self.raw_sigma_func(xs)
        elif self.sim_func_form == "bernoulli":
            raise ValueError("sure?")

    def create_data(self, num_obs: int, t_idx: int, seed:int = None):
        """
        @param num_obs: number of observations
        @param seed: if given, set the seed before generating data

        @param tuple with Dataset, SupportSimSettingsContinuous
        """
        if seed is not None:
            np.random.seed(seed)
        data_gen_xs = self.support_sim_settings.generate_x(num_obs, t_idx)
        dataset = self.create_data_given_x(data_gen_xs)
        return dataset

    def create_data_given_x(self, xs: ndarray):
        """
        For the given Xs, generate responses and dataset
        regression-type data only
        @return Dataset
        """
        size_n = xs.shape[0]
        mu_true = self.mu_func(xs)
        if len(mu_true.shape) == 1:
            mu_true = np.reshape(mu_true, (size_n, 1))
        if self.sim_func_form == "gaussian":
            sigma_true = np.reshape(self.sigma_func(xs), (size_n, 1))

            true_distribution = scipy.stats.norm(mu_true, sigma_true)
            y = true_distribution.rvs(size=mu_true.shape)
        elif self.sim_func_form == "bounded_gaussian":
            sigma_true = np.reshape(self.sigma_func(xs), (size_n, 1))

            true_distribution = scipy.stats.norm(mu_true, sigma_true)
            raw_y = true_distribution.rvs(size=mu_true.shape)
            y = np.maximum(np.minimum(raw_y, self.max_y), self.min_y)
        elif self.sim_func_form == "bernoulli":
            true_distribution = scipy.stats.bernoulli(p=mu_true)
            y = true_distribution.rvs(size=mu_true.shape)
        elif self.sim_func_form == "multinomial":
            # We have to do per row because multinomial is not nice and doesn't take in
            # 2D probability matrices
            all_ys = []
            for i in range(mu_true.shape[0]):
                mu_row = mu_true[i,:]
                true_distribution = scipy.stats.multinomial(n=1, p=mu_row)
                y = true_distribution.rvs(size=1)
                all_ys.append(y)
            y = np.vstack(all_ys)

        return Dataset(xs, y, num_classes=self.num_classes)
