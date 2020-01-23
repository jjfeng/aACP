import sys
import os
import shutil
import argparse
import logging
import numpy as np
from numpy import ndarray
from typing import List, Dict
import pandas as pd

from approval_simulation_common import load_data
from common import *
from approver import Approver
import evaluator


def parse_args(args):
    ''' parse command line arguments '''

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--seed',
        type=int,
        help='Random number generator seed for replicability',
        default=12)
    parser.add_argument('--approver',
        type=str,
        default="_output/approver.pkl")
    parser.add_argument('--data-file',
        type=str,
        default="_output/obs_data.pkl")
    parser.add_argument('--window',
        type=int,
        help="window value",
        default=3)
    parser.add_argument('--ni-margin',
        type=float,
        help="non-inferior margin",
        default=0.01)
    parser.add_argument('--num-test-obs',
        type=int,
        help="This is number of test observations to evaluate true performance of proposed model updates.",
        default=80000)
    parser.add_argument('--log-file',
        type=str,
        default="_output/log.txt")
    parser.add_argument('--out-file',
        type=str,
        default="_output/eval_results.pkl")
    parser.set_defaults()
    args = parser.parse_args()

    return args

def print_approval_summaries(
        model_approver: Approver,
        true_model_score_dict: Dict,
        args):
    logging.info(str(model_approver.__class__))
    print(model_approver.__class__)
    logging.info("Model approval hist %s", model_approver.cum_approved_models)
    logging.info("Model standard hist %s", model_approver.cum_standard_models)
    num_approved = len(model_approver.approved_models_uniq)
    #num_std = len(model_approver.standard_models_uniq)
    final_score_dict = evaluator.get_final_score(
            model_approver,
            true_model_score_dict)
    efficiency_dict = evaluator.get_efficiency(
            model_approver,
            true_model_score_dict)
    num_approved_window = evaluator.get_num_window_approvals(
            args.window,
            model_approver)
    num_std_window = evaluator.get_num_window_standards(
            args.window,
            model_approver)
    num_bad_std = evaluator.count_bad_standard(
            args.window,
            args.ni_margin,
            model_approver,
            true_model_score_dict)
    num_bad_approval = evaluator.count_bad_approval(
            args.window,
            args.ni_margin,
            model_approver,
            true_model_score_dict)
    num_bad_approval_tot = evaluator.count_bad_approval(
            len(model_approver.cum_approved_models),
            args.ni_margin,
            model_approver,
            true_model_score_dict)
    summaries = {
                "approver": model_approver.__class__.__name__,
                "index": np.arange(num_bad_approval.size),
                "num_bad_approval": num_bad_approval,
                "num_bad_approv_tot": num_bad_approval_tot,
                "num_bad_std": num_bad_std,
                "tot_num_approved": num_approved,
                "num_approved_window": num_approved_window,
                "num_std_window": num_std_window,
            }
    all_dicts = {
            "final": final_score_dict,
            "efficiency": efficiency_dict}
    for measure, measure_dict in all_dicts.items():
        for endpoint, endpoint_val in measure_dict.items():
            summaries["%s_%s" % (measure, endpoint)] = endpoint_val

    all_approved_scores = evaluator.get_approver_scores(
            model_approver,
            true_model_score_dict)
    return pd.DataFrame(summaries), all_approved_scores

def main(args=sys.argv[1:]):
    args = parse_args(args)
    logging.basicConfig(format="%(message)s", filename=args.log_file, level=logging.DEBUG)
    print(args)
    logging.info(args)
    np.random.seed(args.seed)

    trial_data, trial_meta = load_data(args)
    approver = pickle_from_file(args.approver)
    true_model_scores_dict = evaluator.get_true_model_scores(
            args.seed,
            trial_data.data_generator,
            trial_meta,
            args.num_test_obs,
            [approver.submitted_models[i] for i in approver.keep_models])
    for k, v in true_model_scores_dict.items():
        logging.info("Endpoint %s", k)
        scores = evaluator._get_selected_scores(approver.index_approved, v)
        logging.info("selected vals %s", scores)
        print("Endpoint", k)
        print("selected vals", scores)

    summary_df, scores_df = print_approval_summaries(
        approver,
        true_model_scores_dict,
        args)
    pickle_to_file({
        "summary": summary_df,
        "scores": scores_df,
    }, args.out_file)

if __name__ == "__main__":
    main(sys.argv[1:])
