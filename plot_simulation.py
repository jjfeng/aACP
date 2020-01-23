import sys
import glob
import argparse
import logging
import numpy as np
from numpy import ndarray
from typing import List
import pandas as pd

from common import pickle_to_file, pickle_from_file, process_params


def parse_args(args):
    ''' parse command line arguments '''

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--denom-min',
        type=float,
        default=1)
    parser.add_argument('--result-names',
        type=str)
    parser.add_argument('--result-files',
        type=str)
    parser.add_argument('--out-csv',
        type=str)

    parser.set_defaults()
    args = parser.parse_args()

    args.result_files = args.result_files.split(",")
    args.result_names = args.result_names.split(",")
    return args

def main(args=sys.argv[1:]):
    args = parse_args(args)

    results = []
    for res_file_template, res_name in zip(args.result_files, args.result_names):
        for res in glob.glob(res_file_template):
            res_df = pickle_from_file(res)["summary"]
            res_df["approver"] = res_name
            results.append(res_df)
    results = pd.concat(results)

    mean_res = results.groupby(["approver", "index"]).mean()
    mean_res["meBAR"] = mean_res["num_bad_approval"]/(args.denom_min + mean_res["num_std_window"])
    mean_res["meBSR"] = mean_res["num_bad_std"]/(args.denom_min + mean_res["num_std_window"])
    print(mean_res.groupby(["approver"]).max())
    mean_res.groupby(["approver"]).max().to_latex(args.out_csv, float_format="%.3f")

if __name__ == "__main__":
    main(sys.argv[1:])
