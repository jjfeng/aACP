from typing import List
import numpy as np
from numpy import ndarray

from data_generator import DataGenerator
from dataset import Dataset
from support_sim_settings import SupportSimSettings

class TrialMetaData:
    def __init__(self,
            batch_sizes: ndarray,
            sim_func_form: str,
            support_sim_settings: SupportSimSettings,
            min_y: float,
            max_y: float,
            num_classes: int =1):
        self.batch_sizes = batch_sizes
        self.sim_func_form = sim_func_form
        self.num_p = support_sim_settings.num_p
        self.support_sim_settings = support_sim_settings
        self.max_y = max_y
        self.min_y = min_y
        self.num_classes = num_classes
        self.num_batches = len(self.batch_sizes)

    @property
    def num_scores(self):
        return 1 if self.sim_func_form == "bounded_gaussian" else 2

    @property
    def score_names(self):
        if self.sim_func_form == "bounded_gaussian":
            return ["neg_sq_err"]
        elif self.sim_func_form == "bernoulli":
            return ["Specificity", "Sensitivity"]

    def score_func(self, pred_y, true_y):
        """
        @return the score of each prediction
        """
        if self.sim_func_form == "bounded_gaussian":
            sq_err = np.power(pred_y.flatten() - true_y.flatten(), 2)
            return {"neg_sq_err": -sq_err}
        elif self.sim_func_form == "bernoulli":
            pred_y = pred_y.flatten().astype(int)
            true_y = true_y.flatten().astype(int)
            negatives = true_y == 0
            positives = true_y == 1
            true_negative = (pred_y[negatives] == 0).astype(int)
            true_positive = (pred_y[positives] == 1).astype(int)
            return {
                    "Specificity": true_negative,
                    "Sensitivity": true_positive}


class TrialData:
    def __init__(self,
            data_generator: DataGenerator,
            batch_sizes: ndarray,
            batch_data: List[Dataset] = []):
        self.data_generator = data_generator
        self.batch_sizes = batch_sizes

        self.batch_data = batch_data

    @property
    def num_batches(self):
        return len(self.batch_data)

    def get_start_to_end_data(self, start_index: int, end_index: int = None):
        cum_data = self.batch_data[start_index]
        if end_index is None:
            for data in self.batch_data[start_index + 1:]:
                cum_data = cum_data.merge(data)
        else:
            for data in self.batch_data[start_index + 1:end_index]:
                cum_data = cum_data.merge(data)
        return cum_data

    def make_new_batch(self):
        new_data = self.data_generator.create_data(
                self.batch_sizes[self.num_batches],
                self.num_batches)
        self.batch_data.append(new_data)

    def make_meta_data(self):
        return TrialMetaData(
            self.batch_sizes,
            self.data_generator.sim_func_form,
            self.data_generator.support_sim_settings,
            self.data_generator.min_y,
            self.data_generator.max_y,
            self.data_generator.num_classes,
        )

    def subset(self, end_index: int):
        return TrialData(
                self.data_generator,
                self.batch_sizes,
                self.batch_data[:end_index]
            )
