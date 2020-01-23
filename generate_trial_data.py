import sys
import argparse
import logging
import numpy as np
from numpy import ndarray
from typing import List

from trial_data import TrialData
from data_generator import DataGenerator
from support_sim_settings import *
from dataset import Dataset
from common import pickle_to_file

def parse_args(args):
    ''' parse command line arguments '''

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--seed',
        type=int,
        help='Random number generator seed for replicability',
        default=0)
    parser.add_argument('--density-parametric-form',
        type=str,
        default="bounded_gaussian",
        help="The parametric form we are going to use for Y|X",
        choices=["bounded_gaussian", "bernoulli"])
    parser.add_argument('--sim-func-name',
        type=str,
        default="linear",
        choices=["linear", "curvy"])
    parser.add_argument('--num-p',
        type=int,
        default=50)
    parser.add_argument('--support-setting',
        type=str,
        default="constant",
        choices=["constant"])
    parser.add_argument('--min-y',
        type=float,
        default=-1)
    parser.add_argument('--max-y',
        type=float,
        default=1)
    parser.add_argument('--num-batches',
        type=int,
        default=20)
    parser.add_argument('--first-batch-size',
        type=int,
        default=40)
    parser.add_argument('--batch-size',
        type=int,
        default=40)
    parser.add_argument('--batch-incr',
        type=int,
        default=0)
    parser.add_argument('--log-file',
        type=str,
        default="_output/data_log.txt")
    parser.add_argument('--out-file',
        type=str,
        default="_output/obs_data.pkl")
    parser.set_defaults()
    args = parser.parse_args()

    assert args.num_batches > 1
    assert args.min_y < args.max_y
    args.batch_sizes = [args.first_batch_size] + [
            args.batch_size + args.batch_incr * i
            for i in range(args.num_batches - 1)]
    return args

def const(t):
    return -1

def main(args=sys.argv[1:]):
    args = parse_args(args)
    logging.basicConfig(format="%(message)s", filename=args.log_file, level=logging.DEBUG)
    print(args)
    logging.info(args)

    np.random.seed(args.seed)
    if args.support_setting == "constant":
        support_sim_settings = SupportSimSettingsUniform(
            args.num_p,
            min_func_name="min_x_func_constant",
            max_func_name="max_x_func_constant")
    elif args.support_setting == "changing":
        raise ValueError("huh? i can get here?")
        support_sim_settings = SupportSimSettingsNormal(
            args.num_p,
            std_func_name="std_func_changing",
            mu_func_name="mu_func_changing")
    else:
        raise ValueError("Asdfasdf")

    data_gen = DataGenerator(
            args.density_parametric_form,
            args.sim_func_name,
            support_sim_settings,
            max_y=args.max_y,
            min_y=args.min_y)
    trial_data = TrialData(data_gen, args.batch_sizes)
    for batch_index in range(args.num_batches):
        trial_data.make_new_batch()

    out_dict = {
        "meta": trial_data.make_meta_data(),
        "data": trial_data
    }
    print(out_dict["meta"])
    pickle_to_file(out_dict, args.out_file)

if __name__ == "__main__":
    main(sys.argv[1:])
