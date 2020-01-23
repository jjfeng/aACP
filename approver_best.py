import numpy as np
from numpy import ndarray
import scipy.optimize
import logging

from trial_data import TrialData
from dataset import Dataset
from data_generator import DataGenerator
from approver import Approver

class BestApprover(Approver):
    """
    Approves the one with the smallest empirical loss
    """

    def eval_models(self, trial_data: TrialData, little_neg: float=-1e-10):
        if self.num_submitted == self.curr_model_idx + 1:
            return None

        assert trial_data.num_batches == self.num_submitted + 1
        batch_index = self.num_submitted

        # Calculate model scores
        candidate_scores = []
        for candidate_idx in range(self.num_submitted):
            scores = self.get_model_scores(candidate_idx, candidate_idx + 1)
            candidate_scores.append(np.mean(scores))
        candidate_scores = np.array(candidate_scores)

        approved_idx = np.argmax(candidate_scores)
        best_score= np.max(candidate_scores)
        logging.info("   BEST score %f", best_score)
        self.approve_model(approved_idx)
        return approved_idx
