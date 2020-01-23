import numpy as np
from numpy import ndarray
import scipy.stats
import scipy.optimize
import logging

from dataset import Dataset
from trial_data import TrialData, TrialMetaData
from approver import Approver

class TTestBABRApprover(Approver):
    """
    BAR and BBR control with window
    Note that the code refers to benchmarks as standards
    """
    def __init__(self,
            trial_meta: TrialMetaData,
            init_model,
            type_i_error: float = None,
            mptvr: float = None,
            decay: float = 1,
            window: int = 5,
            ni_margin: float = 0.005,
            max_wait: int = 1,
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
        self.approval_alpha_spendings = [0]
        self.standard_alpha_spendings = [0]
        self.approval_max_wait = self.max_wait
        self.standard_max_wait = self.max_wait * 2
        self.alpha_use_factor = alpha_use_factor
        self.test_bounds_approval = [[]]
        self.test_bounds_standard = [[]]

    def _get_delta_t(self, t, j1, is_std):
        """
        @return delta_{t,j}
        """
        my_bounds = self.test_bounds_standard[j1] if is_std else self.test_bounds_approval[j1]
        if t - j1 - 1 >= len(my_bounds):
            return np.inf
        else:
            return my_bounds[t - j1 - 1]

    def get_inflation_factor(self,
            candidate_idx,
            ref_idx,
            batch_index: int,
            is_std: bool = False):
        delta_t = self._get_delta_t(batch_index, candidate_idx, is_std)
        if np.isfinite(delta_t):
            std_err = self.get_std_err_estimate(candidate_idx, ref_idx)
            inflation_factor = delta_t * std_err
            return inflation_factor
        else:
            return np.inf

    def _get_max_alpha_spend(self, window, is_std: bool=False, lag_raw=0):
        """
        @param window: this particular window length
        @return get max alpha to spend for this particular window length
        """
        lag = lag_raw if self.num_submitted < (self.trial_meta.num_batches - lag_raw) else 0
        standard_binary_mask = np.array([m is not None for m in self.standard_models], dtype=int)
        if window == 0:
            denom = self.denom_min
        else:
            window_start = max(0, standard_binary_mask.size - window + lag)
            denom = self.denom_min + np.sum(standard_binary_mask[window_start:])
        if is_std:
            numerator = np.sum(self.standard_alpha_spendings[-(self.standard_max_wait + window):])
            wiggle_room = self.mptvr * denom - numerator
        else:
            numerator = np.sum(self.approval_alpha_spendings[-(self.approval_max_wait + window):])
            wiggle_room = self.mptvr * denom - numerator
        assert wiggle_room >= 0
        return wiggle_room

    def _pick_alpha_spend(self, batch_idx):
        standard_binary_mask = np.array([m is not None for m in self.standard_models], dtype=int)
        window_range = range(min(self.window + 1, batch_idx + int(self.window/2)))

        max_alphas = [self._get_max_alpha_spend(0, is_std=False)/(self.approval_max_wait + self.window) * self.alpha_use_factor] + [
                self._get_max_alpha_spend(w, is_std=False) * 0.5 if w < self.window * 0.7
                else self._get_max_alpha_spend(w, is_std=False)/(self.window + 1 - w)
                for w in window_range[1:-1]]
        alpha_spend = min(max_alphas)
        self.approval_alpha_spendings.append(
                int(min(alpha_spend, self.type_i_error) * 1000)/1000.)
        self.test_bounds_approval.append(
                self._get_delta_seq(batch_idx, self.approval_alpha_spendings[batch_idx], self.approval_max_wait))


        max_alphas = [self._get_max_alpha_spend(0, is_std=True)/(self.standard_max_wait + self.window) * self.alpha_use_factor] + [
                self._get_max_alpha_spend(w, is_std=True) * 0.5 if w < self.window * 0.7
                else self._get_max_alpha_spend(w, is_std=True)/(self.window + 1 - w)
                for w in window_range[1:-1]]

        alpha_spend = min(max_alphas)
        self.standard_alpha_spendings.append(
                int(min(alpha_spend, self.type_i_error) * 1000)/1000.)
        self.test_bounds_standard.append(
                self._get_delta_seq(batch_idx, self.standard_alpha_spendings[batch_idx], self.standard_max_wait))

        logging.info("batch %d appro spend %f", batch_idx, self.approval_alpha_spendings[batch_idx])
        logging.info("batch %d std spend %f", batch_idx, self.standard_alpha_spendings[batch_idx])

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
                max(self.num_submitted - self.max_wait, self.curr_approved_idx + 1),
                self.num_submitted)
        approved_status = [
                self._eval_for_approval(trial_data, candidate_idx, self.ni_margin)
                for candidate_idx in candidate_idxs]

        approved_idx = None
        if np.any(approved_status):
            approved_idx = candidate_idxs[np.max(np.where(approved_status)[0])]
            print("BAR-APPROVE", approved_idx)

        self.approve_model(approved_idx)

        # Determine new standards
        candidate_bench_idxs = np.array([
            a for a in self.approved_models_uniq if a > self.curr_standard_idx])
        standard_status = [
                self._eval_for_standard(trial_data, idx)
                for idx in candidate_bench_idxs]

        standard_idx = None
        if np.any(standard_status):
            standard_idx = candidate_bench_idxs[np.min(np.where(standard_status)[0])]
            print("BSR-STANDARD", standard_idx)
        self.make_standard(standard_idx)

        self._pick_alpha_spend(batch_index)
        logging.info("TOT ALPHA APP SPEND %f", sum(self.approval_alpha_spendings))
        logging.info("TOT ALPHA STD SPEND %f", sum(self.standard_alpha_spendings))
        return approved_idx

    def _eval_for_approval(self, trial_data: TrialData, candidate_idx: int, ni_margin = 0):
        """
        Performing gatekeeping for evaluating candidate for approval
        """
        batch_index = self.num_submitted

        # Calculate model scores
        assert candidate_idx > 0

        prev_approved_models = [a for a in set(self.cum_approved_models[:candidate_idx - 1])]
        upcoming_models = list(range(self.cum_approved_models[candidate_idx - 1], self.curr_approved_idx + 1))
        comparison_models = prev_approved_models + upcoming_models
        for ref_idx in comparison_models:
            improvement_score = self.get_improvement(candidate_idx, ref_idx)
            inflation_factor = self.get_inflation_factor(
                    candidate_idx,
                    ref_idx,
                    batch_index,
                    is_std=False)
            score = improvement_score - inflation_factor
            if np.any(score <= -ni_margin):
                return False
            if score.size > 1 and np.all(score < 0):
                return False
        return True

    def _eval_for_standard(self, trial_data: TrialData, candidate_idx: int):
        """
        Performing gatekeeping for evaluating candidate for standard
        """
        batch_index = self.num_submitted

        # Calculate model scores
        assert candidate_idx > 0

        for ref_idx in range(self.cum_standard_models[candidate_idx - 1], self.curr_standard_idx + 1):
            improvement_score = self.get_improvement(candidate_idx, ref_idx)
            inflation_factor = self.get_inflation_factor(
                    candidate_idx,
                    ref_idx,
                    batch_index,
                    is_std=True)
            score = improvement_score - inflation_factor

            if np.any(score <= 0):
                return False
        return True
