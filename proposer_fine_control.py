import scipy.stats
import numpy as np

from trial_data import TrialData
from proposer import Proposer

class FakeGaussianModel:
    def __init__(self, data_gen, eps, decay=0):
        self.data_gen = data_gen
        self.eps = eps
        self.decay = decay

    def predict(self, x, t):
        mus = self.data_gen.mu_func(x)
        eps = self.get_eps(t)
        return scipy.stats.norm.rvs(loc=mus, scale=eps, size=x.shape[0])

    def get_eps(self, t):
        return max(0, self.eps + self.decay * t)

class FakeBernoulliModel:
    def __init__(self, data_gen, eps, decay=0, offset=0):
        self.data_gen = data_gen
        self.eps = eps
        self.decay = decay
        self.offset = offset
        #assert eps <= 1 and eps >= 0

    def predict(self, x, t):
        true_mus = self.data_gen.mu_func(x)
        eps0, eps1 = self.get_eps(t)
        new_mus0 = ((1 - eps0) * true_mus + eps0 * 0.5) * (true_mus < 0.5)
        new_mus1 = ((1 - eps1) * true_mus + eps1 * 0.5) * (true_mus >= 0.5)
        new_mus = new_mus0 + new_mus1
        return scipy.stats.bernoulli.rvs(p=new_mus)

    def get_eps(self, t):
        eps0 = min(1, max(0, self.eps[0] - self.decay * np.sin(t + self.offset)))
        eps1 = min(1, max(0, self.eps[1] - self.decay * np.sin(t + self.offset)))
        return eps0, eps1

class FineControlProposer(Proposer):
    """
    This produces the same exact models over time with injected noise

    The proposed models are proposed by two noise params eps[0] and eps[1]
    eps[0] controls the mixture proportion between true model and bernoull 0.5 when mu(x) < 0.5
    eps[1] controls the mixture proportion between true model and bernoull 0.5 when mu(x) >= 0.5
    """
    def __init__(self,
            data_gen,
            noise=0.2,
            increment=0,
            decay=0,
            offset_scale=0):
        """
        @param increment: how much worse to decrease the true positive, true negative performance by
        """
        assert data_gen.sim_func_form == "bernoulli"

        self.proposal_history = []
        self.approval_history = []
        self.data_gen = data_gen
        self.noise = noise
        self.increment = increment
        self.decay = decay
        self.offset_scale = offset_scale
        self._get_true_params()

    def _get_true_params(self, num_obs=80000):
        mus = self.data_gen.mu_func(self.data_gen.support_sim_settings.generate_x(num_obs, 0))
        mu_sqs = np.power(mus, 2)
        self.expected_match = np.array([
                [np.mean(mus[mus < 0.5]), np.mean(mus[mus >= 0.5])],
                [np.mean(1 - mus[mus < 0.5]), np.mean(1 - mus[mus >= 0.5])]])
        self.expected_match_sqs = np.array([
                [np.mean(np.power(mus[mus < 0.5], 2)), np.mean(np.power(mus[mus >= 0.5], 2))],
                [np.mean(np.power(1 - mus[mus < 0.5], 2)), np.mean(np.power(1 - mus[mus >= 0.5], 2))]])
        self.prob_regions = [np.mean(mus < 0.5), np.mean(mus >= 0.5)]
        self.perf_tp = self.get_true_positive([0,0])
        self.perf_tn = self.get_true_negative([0,0])

    def get_true_negative(self, eps):
        """
        @return the estimated TN for this particular noise injection
        """
        prob_neg0 = self.expected_match[1,0] * self.prob_regions[0]
        prob_neg_neg0 = ((1 - eps[0]) * self.expected_match_sqs[1,0] + eps[0]/2 * self.expected_match[1,0]) * self.prob_regions[0]
        prob_neg1 = self.expected_match[1,1] * self.prob_regions[1]
        prob_neg_neg1 = ((1 - eps[1]) * self.expected_match_sqs[1,1] + eps[1]/2 * self.expected_match[1,1]) * self.prob_regions[1]
        true_neg = (prob_neg_neg0 + prob_neg_neg1)/(prob_neg0 + prob_neg1)
        return true_neg

    def get_true_positive(self, eps):
        """
        @return the estimated TP for this particular noise injection
        """
        prob_pos0 = self.expected_match[0,0] * self.prob_regions[0]
        prob_pos_pos0 = ((1 - eps[0]) * self.expected_match_sqs[0,0] + eps[0]/2 * self.expected_match[0,0]) * self.prob_regions[0]
        prob_pos1 = self.expected_match[0,1] * self.prob_regions[1]
        prob_pos_pos1 = ((1 - eps[1]) * self.expected_match_sqs[0,1] + eps[1]/2 * self.expected_match[0,1]) * self.prob_regions[1]
        true_pos = (prob_pos_pos0 + prob_pos_pos1)/(prob_pos0 + prob_pos1)
        return true_pos

    def propose_model(self, trial_data, curr_model_idx: int):
        #did_append = len(self.approval_history) <= 1 or curr_model_idx != self.approval_history[-1]
        if len(self.approval_history) == 0 or curr_model_idx != self.approval_history[-1]:
            self.approval_history.append(curr_model_idx)
        if self.num_models == 0:
            noise = np.array([self.noise, self.noise])
        else:
            if self.increment < 0:
                # We are going to be very adversarial
                # Figure out the correct noise injection to decrease the TN or TP at the desired increment
                # To do this, we need to use optimization
                # Also, we alternate between decreasing TN and TF by tracking the approval history
                prev_noise = self.proposal_history[self.approval_history[-2]].eps if len(self.approval_history) >= 2 else self.proposal_history[self.approval_history[-1]].eps
                curr_noise = self.proposal_history[curr_model_idx].eps
                curr_tp = self.get_true_positive(curr_noise)
                curr_tn = self.get_true_negative(curr_noise)
                if prev_noise[0] < curr_noise[0] or (prev_noise[0] == curr_noise[0] and np.random.choice(2) == 0):
                    goal_tp = max(0.5, curr_tp + self.increment)
                    goal_tn = curr_tn - self.increment/3 if curr_tn > 0.5 else 0.5
                else:
                    goal_tp = curr_tp - self.increment/3 if curr_tp > 0.5 else 0.5
                    goal_tn = max(0.5, curr_tn + self.increment)
            else:
                curr_noise = self.proposal_history[curr_model_idx].eps
                curr_tp = self.get_true_positive(curr_noise)
                curr_tn = self.get_true_negative(curr_noise)
                goal_tp = min(self.perf_tp, curr_tp + self.increment)
                goal_tn = min(self.perf_tn, curr_tn + self.increment)

            def get_acceptability_score(x):
                proposal_tp = self.get_true_positive(x)
                proposal_tn = self.get_true_negative(x)
                return np.power(proposal_tp - goal_tp, 2) + np.power(proposal_tn - goal_tn, 2)
            res = scipy.optimize.minimize(get_acceptability_score, x0=curr_noise)
            noise = res.x
            assert res.success
            #if did_append:
                #print("curr", curr_model_idx)
                #print(self.proposal_history[0].eps)
                #print("noise", curr_noise)
                #print("curr", curr_tp, curr_tn)
                #print("goal", goal_tp, goal_tn, res.x)

        model = FakeBernoulliModel(self.data_gen, noise, self.decay, offset=np.random.rand() * self.offset_scale)
        self.proposal_history.append(model)
        return model

class MoodyFineControlProposer(FineControlProposer):
    """
    This will propose some good models sometimes and bad models other times
    """
    def __init__(self, data_gen, noise=0.2, increment=0, init_period=20, period=1, decay=0):
        self.proposal_history = []
        self.proposal_noise = []
        self.data_gen = data_gen
        self.noise = noise
        self.increment = increment
        self.init_mood_num_batch = init_period
        self.mood_num_batch = period
        self.decay = 0

        self._get_true_params()

    def propose_model(self, trial_data, curr_model_idx: int):
        if self.num_models == 0:
            noise = np.array([self.noise, self.noise])
        else:
            increment_sign = 1 if (self.num_models < self.init_mood_num_batch or int(self.num_models/self.mood_num_batch) % 2 == 0) else -1
            increment = self.increment * increment_sign
            curr_noise = self.proposal_history[-1].eps
            curr_tp = self.get_true_positive(curr_noise)
            curr_tn = self.get_true_negative(curr_noise)
            goal_tp = min(self.perf_tp, curr_tp + increment)
            goal_tn = min(self.perf_tn, curr_tn + increment)

            def get_acceptability_score(x):
                proposal_tp = self.get_true_positive(x)
                proposal_tn = self.get_true_negative(x)
                return np.power(proposal_tp - goal_tp, 2) + np.power(proposal_tn - goal_tn, 2)
            res = scipy.optimize.minimize(get_acceptability_score, x0=curr_noise)
            noise = res.x
            assert res.success
            #print(self.num_models, "noise", curr_noise)
            #print("curr", curr_tp, curr_tn)
            #print("goal", goal_tp, goal_tn, res.x)

        model = FakeBernoulliModel(self.data_gen, noise, self.decay)

        self.proposal_history.append(model)
        return model
