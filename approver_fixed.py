import numpy as np
from numpy import ndarray
import scipy.optimize
import logging

from approver import Approver
from trial_data import TrialData

class FixedApprover(Approver):
    """
    This only approves the first model
    """
    def eval_models(self, trial_data: TrialData):
        """
        Evaluate the current models with the latest trial data and approve the best one if one is good enough.
        """
        if self.num_submitted == self.curr_approved_idx + 1:
            return None

        self.approve_model(None)
        self.make_standard(None)
        return None
