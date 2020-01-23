import numpy as np
from numpy import ndarray
import scipy.stats
import scipy.optimize
import logging

from dataset import Dataset
from trial_data import TrialData, TrialMetaData
from approver import Approver

class TTestBACApprover(Approver):
    """
    BAC control with window
    """
    def __init__(self,
            trial_meta: TrialMetaData,
            init_model,
            type_i_error: float = None,
            mptvr: float = None,
            decay: float = 1,
            window: int = 5,
            ni_margin: float = 0.005,
            max_wait: int = 1000,
            scratch_dir: str = "",
            alpha_use_factor: float = 0.1,
            denom_min: float = 0.5):
        super().__init__(
                trial_meta,
                init_model,
                type_i_error=type_i_error,
                mptvr=mptvr,
                decay=decay,
                ni_margin=ni_margin,
                max_wait=max_wait,
                window=window,
                denom_min=denom_min,
                scratch_dir=scratch_dir)
        self.alpha_spendings = [0]

    def _pick_alpha_spend(self, batch_idx):
        alpha_spend = self.mptvr/(self.window + self.max_wait)
        self.alpha_spendings.append(
                int(alpha_spend * 1000)/1000.)
        self.test_bounds.append(
                self._get_delta_seq(batch_idx, self.alpha_spendings[batch_idx]))
        logging.info("batch %d spend %f", batch_idx, self.alpha_spendings[batch_idx])

    def eval_models(self, trial_data: TrialData):
        batch_index = self.num_submitted
        if self.num_submitted == self.curr_approved_idx + 1:
            self._pick_alpha_spend(batch_index)
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
            print("BAC-APPROVE", approved_idx)

        self.approve_model(approved_idx)
        self.make_standard(None)

        self._pick_alpha_spend(batch_index)
        return approved_idx

    def _eval_candidate(self, trial_data: TrialData, candidate_idx: int, ni_margin = 0):
        """
        Performing gatekeeping for evaluating candidate
        """
        batch_index = self.num_submitted

        # Calculate model scores
        assert candidate_idx > 0

        prev_approved_models = list(set(self.cum_approved_models[:candidate_idx - 1]))
        upcoming_models = list(range(self.cum_approved_models[candidate_idx - 1], self.curr_approved_idx + 1))
        comparison_models = prev_approved_models + upcoming_models
        for ref_idx in comparison_models:
            improvement_score = self.get_improvement(candidate_idx, ref_idx)
            inflation_factor = self.get_inflation_factor(
                    candidate_idx,
                    ref_idx,
                    batch_index)
            score = improvement_score - inflation_factor

            if np.any(score <= -ni_margin):
                return False
            if score.size > 1 and np.all(score < 0):
                return False
        return True
