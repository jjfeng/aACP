import numpy as np
from numpy import ndarray
import scipy.optimize
import logging

from approver import Approver
from trial_data import TrialData

class BlindApprover(Approver):
    """
    This blindly approves model
    """
    def eval_models(self, trial_data: TrialData):
        """
        Evaluate the current models with the latest trial data and approve the best one if one is good enough.
        """
        if self.num_submitted == self.curr_approved_idx + 1:
            return None

        assert trial_data.num_batches == self.num_submitted + 1
        approved_idx = self.curr_approved_idx + 1
        self.approve_model(approved_idx)
        self.make_standard(None)
        return approved_idx
