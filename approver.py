import numpy as np
import logging

from trial_data import TrialData, TrialMetaData
from dataset import Dataset
from data_generator import DataGenerator
import group_sequential as gsm

class Approver:
    """
    Does two-way comparisons to pick out the best model.
    Does not look backwards.
    """
    def __init__(self,
            trial_meta: TrialMetaData,
            init_model,
            type_i_error: float = None,
            mptvr: float = None,
            decay: float = None,
            ni_margin: float = 0,
            max_wait: int = 5,
            window: int = 5,
            denom_min: float = 1,
            scratch_dir: str = ""):
        """
        approved_models: list of model indices, None if no model approved in that round
        """
        self.trial_meta = trial_meta
        self.submitted_models = [init_model]
        self.standard_models = [0]
        self.approved_models = [0]
        self.score_func = trial_meta.score_func
        self.support_sim_settings = trial_meta.support_sim_settings
        self.min_y = trial_meta.min_y
        self.max_y = trial_meta.max_y
        self.type_i_error = type_i_error
        self.mptvr = mptvr
        self.decay = decay
        self.ni_margin = ni_margin
        self.max_wait = max_wait
        self.window = window
        self.denom_min = denom_min
        self.scratch_dir = scratch_dir
        self.model_scores = [[None]]
        self.cached_deltas = {}
        self.test_bounds = [[]]

    def compress(self):
        self.keep_models = list(sorted(list(set(self.standard_models_uniq.tolist() + self.approved_models_uniq.tolist()))))
        self.compressed_idx_dict = {}
        counter = -1
        for i in self.keep_models:
            self.compressed_idx_dict[i] = counter + 1
            counter += 1

        self.submitted_models = [
                self.submitted_models[i] if i in self.keep_models else None
                for i in range(self.num_submitted)]
        self.cached_deltas = None
        self.model_scores = None

    def get_model(self, j):
        return self.submitted_models[j]

    @property
    def sigma_max(self):
        return self.max_y - self.min_y

    @property
    def cum_approved_models(self):
        cum_approved_models = []
        for m in self.approved_models:
            if m is not None:
                cum_approved_models.append(m)
            else:
                cum_approved_models.append(cum_approved_models[-1])
        return cum_approved_models
        return self.max_y - self.min_y

    @property
    def cum_standard_models(self):
        cum_standard_models = []
        for m in self.standard_models:
            if m is not None:
                cum_standard_models.append(m)
            else:
                cum_standard_models.append(cum_standard_models[-1])
        return cum_standard_models

    @property
    def curr_approved_idx(self):
        return max([m for m in self.approved_models if m is not None])

    @property
    def curr_approved(self):
        return self.submitted_models[self.curr_approved_idx]

    @property
    def num_submitted(self):
        return len(self.submitted_models)

    @property
    def num_approved(self):
        return np.unique([m for m in self.approved_models if m is not None]).size

    @property
    def curr_standard_idx(self):
        return max([m for m in self.standard_models if m is not None])

    @property
    def curr_standard(self):
        return self.submitted_models[self.curr_standard_idx]

    @property
    def num_standards(self):
        return np.unique([m for m in self.standard_models if m is not None]).size

    def submit_model(self, new_model):
        self.submitted_models.append(new_model)
        self.model_scores.append([None] * self.num_submitted)

    def approve_model(self, approved_idx: int):
        self.approved_models.append(approved_idx)

    def make_standard(self, approved_idx: int):
        self.standard_models.append(approved_idx)

    @property
    def approved_models_uniq(self):
        return np.unique([a for a in self.approved_models if a is not None])

    @property
    def index_approved(self):
        """
        @return the number of approvals up to that time point
        """
        return [self.compressed_idx_dict[a] for a in self.cum_approved_models]

    @property
    def index_standard(self):
        """
        @return the number of standards up to that time point
        """
        return [self.compressed_idx_dict[a] for a in self.cum_standard_models]

    @property
    def standard_models_uniq(self):
        return np.unique([a for a in self.standard_models if a is not None])

    def get_model_scores(self, model_idx, batch_start_idx):
        """
        Retrieves the model scores for that model
        from when that model predicted things
        (it does not get to try again)
        """
        score_dicts = self.model_scores[model_idx][batch_start_idx:]
        metric_keys = list()
        all_metrics = {}
        for k in score_dicts[0].keys():
            scores = [score_dict[k] for score_dict in score_dicts]
            all_metrics[k] = np.concatenate(scores)
        return all_metrics

    def _get_model_scores(self, model, dataset: Dataset, batch_index: int):
        """
        Query the model for a prediction and score it immediately.
        Store the score.
        """
        predictions = model.predict(dataset.x, batch_index)
        return self.score_func(predictions, dataset.y)

    def score_models_batch(self, batch_data: Dataset, batch_index: int):
        """
        Score all the submitted models on this new incoming batch of data
        """
        for idx, model in enumerate(self.submitted_models):
            self.model_scores[idx].append(
                self._get_model_scores(model, batch_data, batch_index))

    def eval_models(self, trial_data: TrialData):
        """
        """
        raise NotImplementedError("please inherit this func")

    def _get_delta_t(self, t, j1):
        """
        @return delta_{t,j}
        """
        my_bounds = self.test_bounds[j1]
        if t - j1 - 1 >= len(my_bounds):
            return np.inf
        else:
            return my_bounds[t - j1 - 1]

    def _get_delta_seq(self, new_model_idx, alpha_spend, max_wait=None):
        if max_wait is None:
            max_wait = self.max_wait
        num_checks = int(min(
            self.trial_meta.num_batches - 1 - new_model_idx,
            max_wait))

        if num_checks == 0:
            return []
        elif alpha_spend == 0:
            return [8] * num_checks
        else:
            # This is a repeated confidence interval approach
            if self.trial_meta.num_scores == 2:
                #check_factor = 1.0/num_checks + (num_checks - 1) * (num_checks - 2)/np.power(num_checks, 2)
                #alpha_spend_gsm = (1 - np.sqrt(1 - alpha_spend * check_factor))/check_factor
                alpha_spend_gsm = 1 - np.sqrt(1 - alpha_spend)
            elif self.trial_meta.num_scores == 1:
                alpha_spend_gsm = alpha_spend
            else:
                raise ValueError("huh?")

            #print("alpha spend gsm", alpha_spend_gsm)
            delta_key = (num_checks, alpha_spend)
            if delta_key in self.cached_deltas:
                gsm_bounds = self.cached_deltas[delta_key]
            else:
                gsm_bounds = gsm.get_gsm_bounds(
                    alpha_spend_gsm,
                    num_checks,
                    scratch_folder=self.scratch_dir)
                self.cached_deltas[delta_key] = gsm_bounds
            return gsm_bounds

    def get_std_err_estimate(self, idx1, idx2):
        """
        Estimate the sigma for the model score difference
        """
        start_eval_idx = max(idx1, idx2) + 1
        scores_dict1 = self.get_model_scores(idx1, start_eval_idx)
        scores_dict2 = self.get_model_scores(idx2, start_eval_idx)
        all_std_errs = []
        for k in scores_dict1.keys():
            scores1 = scores_dict1[k]
            scores2 = scores_dict2[k]
            if np.all(scores1 == scores2):
                min_loss = 0 #np.min([scores1, scores2])
                max_loss = 1 #np.max([scores1, scores2])
                scores1_me = np.concatenate([scores1, [min_loss]])
                scores2_me = np.concatenate([scores2, [max_loss]])
            else:
                scores1_me = scores1
                scores2_me = scores2

            sigma = np.sqrt(np.var(scores1_me - scores2_me))
            all_std_errs.append(
                    sigma/np.sqrt(scores1_me.size))
            #print("sikxe", scores1_me.size)
        return np.array(all_std_errs)

    def get_improvement(self, idx1, idx2):
        """
        get the difference in avg model scores
        """
        start_eval_idx = max(idx1, idx2) + 1
        scores_dict1 = self.get_model_scores(idx1, start_eval_idx)
        scores_dict2 = self.get_model_scores(idx2, start_eval_idx)
        all_mean_diffs = []
        for k in scores_dict1.keys():
            scores1 = scores_dict1[k]
            scores2 = scores_dict2[k]
            all_mean_diffs.append(np.mean(scores1 - scores2))
        return np.array(all_mean_diffs)

    def get_inflation_factor(self,
            candidate_idx,
            ref_idx,
            batch_index: int):
        delta_t = self._get_delta_t(batch_index, candidate_idx)
        if np.isfinite(delta_t):
            std_err = self.get_std_err_estimate(candidate_idx, ref_idx)
            inflation_factor = delta_t * std_err
            return inflation_factor
        else:
            return np.inf
