from trial_data import TrialData


class Proposer:
    def __init__(self):
        self.proposal_history = []

    @property
    def num_models(self):
        return len(self.proposal_history)

    def propose_model(self, trial_data: TrialData):
        raise NotImplementedError()
