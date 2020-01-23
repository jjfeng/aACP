import numpy as np
from numpy import ndarray
import scipy.stats
import scipy.optimize
import logging

from dataset import Dataset
from trial_data import TrialData, TrialMetaData
from approver_ttest_reset import TTestResetApprover

class TTestBaselineResetApprover(TTestResetApprover):
    """
    Do Ttest with the first ever proposed algo
    Always same level alpha
    """
    def _update_approval_standards(self, approved_idx: int):
        self.approve_model(approved_idx)
        self.make_standard(None)

    def _eval_candidate(self, trial_data: TrialData, candidate_idx: int, eps = 0):
        """
        The reference model is always the initial one.
        """
        batch_index = self.num_submitted

        # Calculate model scores
        assert candidate_idx > 0

        eval_dataset = trial_data.get_start_to_end_data(candidate_idx + 1)
        ref_idx = 0

        inflation_factor = self.get_inflation_factor(
                candidate_idx,
                ref_idx,
                batch_index)
        if np.all(np.isfinite(inflation_factor)):
            improvement_score = self.get_improvement(candidate_idx, ref_idx)
            score = improvement_score - inflation_factor
            not_too_bad = np.all(score >= -eps)
            some_good = np.any(score > 0)
            return not_too_bad and some_good
        else:
            return False
