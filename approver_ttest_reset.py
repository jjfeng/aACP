import numpy as np
from numpy import ndarray
import scipy.stats
import scipy.optimize
import logging

from dataset import Dataset
from trial_data import TrialData, TrialMetaData
from approver import Approver
import group_sequential as gsm

class TTestResetApprover(Approver):
    """
    Always test with level alpha
    """
    def eval_models(self, trial_data: TrialData):
        batch_index = self.num_submitted
        if self.num_submitted == self.curr_approved_idx + 1:
            self.test_bounds.append(
                self._get_delta_seq(self.num_submitted, self.type_i_error))
            return None

        assert trial_data.num_batches == self.num_submitted + 1
        batch_index = self.num_submitted

        # Gatekeeping with group sequential mixed together?
        # Determine new approvals
        candidate_idxs = range(
                max(self.num_submitted - self.max_wait - 1, self.curr_approved_idx + 1),
                self.num_submitted)
        approved_status = [
                self._eval_candidate(trial_data, candidate_idx, self.ni_margin)
                for candidate_idx in candidate_idxs]

        approved_idx = None
        if np.any(approved_status):
            approved_idx = candidate_idxs[np.max(np.where(approved_status)[0])]
            print("APPROVE", approved_idx)

        self._update_approval_standards(approved_idx)

        self.test_bounds.append(
            self._get_delta_seq(self.num_submitted, self.type_i_error))

        return approved_idx

    def _update_approval_standards(self, approved_idx: int):
        self.approve_model(approved_idx)
        self.make_standard(approved_idx)


    def _eval_candidate(self, trial_data: TrialData, candidate_idx: int, eps = 0):
        batch_index = self.num_submitted

        # Calculate model scores
        assert candidate_idx > 0

        ref_idx  = self.curr_approved_idx
        assert ref_idx != candidate_idx

        start_eval_idx = candidate_idx + 1
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
