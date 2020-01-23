import sys
import os
import shutil
import argparse
import logging
import numpy as np
from numpy import ndarray
from typing import List
import pandas as pd

from approval_simulation_common import *
from common import process_params
import evaluator


def parse_args(args):
    ''' parse command line arguments '''

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--seed',
        type=int,
        help='Random number generator seed for replicability',
        default=12)
    parser.add_argument('--data-file',
        type=str,
        default="_output/obs_data.pkl")
    parser.add_argument('--proposer-noise',
        type=float,
        help='how much noise to add to truth',
        default=0.1)
    parser.add_argument('--proposer-increment',
        type=float,
        help='how much model does worse with time',
        default=0)
    parser.add_argument('--proposer-init-period',
        type=int,
        help='how often mood changes',
        default=20)
    parser.add_argument('--proposer-period',
        type=int,
        help='how often mood changes',
        default=1)
    parser.add_argument('--proposer-decay',
        type=float,
        help='how much worse a fixed model becomes',
        default=0)
    parser.add_argument('--proposer-offset-scale',
        type=float,
        help='how much to offset',
        default=0)
    parser.add_argument('--update-engine',
        type=str,
        help='which model updater to use',
        default="fine_control",
        choices=["fine_control", "lasso", "moody"])
    parser.add_argument('--approval-policy',
        type=str,
        help="name of approval policy",
        default="ttest_reset",
        choices=[
            "Fixed",
            "Blind",
            "Reset",
            "Baseline",
            ])
    parser.add_argument('--ni-margin',
        type=float,
        help="non-inferior margin")
    parser.add_argument('--max-wait',
        type=int,
        help="max wait for approval",
        default=5)
    parser.add_argument('--type-i-error',
        type=float,
        help="what is the biggest alpha we can use",
        default=0.05)
    parser.add_argument('--log-file',
        type=str,
        default="_output/log.txt")
    parser.add_argument('--out-file',
        type=str,
        default="_output/approver.pkl")
    parser.set_defaults()
    args = parser.parse_args()

    make_scratch_dir(args)
    return args

def main(args=sys.argv[1:]):
    args = parse_args(args)
    logging.basicConfig(format="%(message)s", filename=args.log_file, level=logging.DEBUG)
    print(args)
    logging.info(args)
    np.random.seed(args.seed)

    trial_data, trial_meta = load_data(args)

    update_engine = get_update_engine(args, trial_meta, trial_data.data_generator)
    approver = run_simulation(
                trial_data,
                trial_meta,
                update_engine,
                args)

    approver.compress()
    pickle_to_file(approver, args.out_file)
    clean_scratch_dir(args)

if __name__ == "__main__":
    main(sys.argv[1:])
