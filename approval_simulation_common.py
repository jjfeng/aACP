import sys
import time
import os
import shutil
import argparse
import logging
import numpy as np
from numpy import ndarray
from typing import List, Dict
import pandas as pd

from proposer import Proposer
from proposer_fine_control import MoodyFineControlProposer, FineControlProposer
from proposer_lasso import LassoProposer
from trial_data import TrialMetaData, TrialData
from approver import Approver
from approver_blind import BlindApprover
from approver_fixed import FixedApprover
from approver_ttest_reset import TTestResetApprover
from approver_best import BestApprover
from approver_ttest_bac import TTestBACApprover
from approver_ttest_babr import TTestBABRApprover
from approver_ttest_baseline import TTestBaselineResetApprover
from data_generator import DataGenerator
from dataset import Dataset
import evaluator
from common import pickle_to_file, pickle_from_file


def make_scratch_dir(args):
    args.scratch_dir = os.path.join(
            "scratch",
            "scratch%d_%d" % (args.seed, np.random.randint(10000)))
    for i in range(10):
        try:
            os.mkdir(args.scratch_dir)
            print("made scratch dir", args.scratch_dir)
            break
        except:
            time.sleep(np.random.randint(10))

def clean_scratch_dir(args):
    shutil.rmtree(args.scratch_dir)

def load_data(args):
    trial_data_dict = pickle_from_file(args.data_file)
    trial_data = trial_data_dict["data"]
    trial_meta = trial_data_dict["meta"]
    return trial_data, trial_meta

def get_update_engine(args, trial_meta: TrialMetaData, data_gen):
    UPDATE_DICT = {
            "lasso": LassoProposer,
            "fine_control": FineControlProposer,
            "moody": MoodyFineControlProposer,
    }
    if args.update_engine == "moody":
        return MoodyFineControlProposer(
                data_gen,
                noise=args.proposer_noise,
                init_period=args.proposer_init_period,
                period=args.proposer_period,
                increment=args.proposer_increment,
                decay=args.proposer_decay)
    elif args.update_engine == "fine_control":
        return FineControlProposer(
                data_gen,
                noise=args.proposer_noise,
                increment=args.proposer_increment,
                decay=args.proposer_decay,
                offset_scale=args.proposer_offset_scale)
    elif args.update_engine == "lasso":
        return LassoProposer(
            trial_meta.sim_func_form,
            min_val=trial_meta.min_y,
            max_val=trial_meta.max_y)
    else:
        raise ValueError("which proposer?")

def init_approval_policy(
        policy_name: str,
        trial_meta: TrialMetaData,
        init_model,
        args):
    MY_AACPS = ["BAC", "BABR"]
    POLICY_DICT = {
        "BAC": TTestBACApprover,
        "BABR": TTestBABRApprover,
        "Baseline": TTestBaselineResetApprover,
        "Best": BestApprover,
        "Blind": BlindApprover,
        "Fixed": FixedApprover,
        "Reset": TTestResetApprover}

    if policy_name not in POLICY_DICT:
        raise ValueError("Don't recognize policy specified")
    elif policy_name in MY_AACPS:
        policy_cls = POLICY_DICT[policy_name]
        return policy_cls(
                trial_meta,
                init_model,
                type_i_error=args.type_i_error,
                mptvr=args.mptvr,
                decay=args.decay,
                window=args.window,
                ni_margin=args.ni_margin,
                max_wait=args.max_wait,
                scratch_dir=args.scratch_dir,
                alpha_use_factor=args.alpha_use_factor,
                denom_min=args.denom_min)
    else:
        policy_cls = POLICY_DICT[policy_name]
        return policy_cls(
                trial_meta,
                init_model,
                type_i_error=args.type_i_error,
                ni_margin=args.ni_margin,
                max_wait=args.max_wait,
                scratch_dir=args.scratch_dir)

def run_simulation(
        trial_data: TrialData,
        trial_meta: TrialMetaData,
        proposer: Proposer,
        args):
    # Create the data generated each batch
    init_model = proposer.propose_model(trial_data.subset(1), curr_model_idx=0)
    approver = init_approval_policy(
            args.approval_policy,
            trial_meta,
            init_model,
            args)

    # Run the platform trial
    for batch_index in range(1, trial_meta.num_batches):
        sub_trial_data = trial_data.subset(batch_index + 1)

        approver.score_models_batch(trial_data.batch_data[batch_index], batch_index)
        approved_idx = approver.eval_models(sub_trial_data)
        logging.info("Approved in batch %d: model %s", batch_index, str(approved_idx))

        new_model = proposer.propose_model(sub_trial_data, curr_model_idx=approver.curr_approved_idx)
        approver.submit_model(new_model)
    logging.info("approval hist %s", approver.cum_approved_models)
    logging.info("standard hist %s", approver.cum_standard_models)
    return approver
