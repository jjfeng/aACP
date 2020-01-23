import sys
import glob
import argparse
import logging
import numpy as np
from numpy import ndarray
from typing import List
import pandas as pd
import matplotlib
matplotlib.use('agg')
import seaborn as sns

from approval_simulation_common import load_data
from common import *
import evaluator


def parse_args(args):
    ''' parse command line arguments '''

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--seed',
        type=int,
        help='Random number generator seed for replicability',
        default=12)
    parser.add_argument('--result-names',
        type=str)
    parser.add_argument('--result-files',
        type=str)
    parser.add_argument('--meta-name', type=str, default=None)
    parser.add_argument('--meta-type', type=str, default='int')
    parser.add_argument('--endpoint', type=str, default=None)
    parser.add_argument('--meta-vals',
        type=str, default="")
    parser.add_argument('--out-plot',
        type=str)
    parser.add_argument('--hide-legend', action="store_true", default=False)

    parser.set_defaults()
    args = parser.parse_args()

    args.result_files = args.result_files.split(",")
    args.result_names = args.result_names.split(",")
    args.meta_vals = args.meta_vals.split(",") if args.meta_vals else []
    return args

def plot_scores(approver_scores, out_plot, hue_order, meta_name=None, hide_legend=False):
    if meta_name is None:
        sns_plot = sns.relplot(
            x = "Time",
            y = "Value",
            col = "endpoint",
            hue = "aACP",
            hue_order = hue_order,
            kind = "line",
            data=approver_scores,
            legend=False if hide_legend else "brief")
    else:
        sns_plot = sns.relplot(
            x = "Time",
            y = "Value",
            col = meta_name,
            row = "endpoint",
            hue = "aACP",
            hue_order = hue_order,
            kind = "line",
            data=approver_scores,
            legend=False if hide_legend else "brief")
    sns_plot.savefig(out_plot)

def plot_endpoint_scores(approver_scores, endpoint_name, out_plot, hue_order, meta_name=None, hide_legend=False):
    if meta_name is None:
        sns_plot = sns.relplot(
            x = "Time",
            y = endpoint_name,
            hue = "aACP",
            hue_order = hue_order,
            kind = "line",
            data=approver_scores,
            legend=False if hide_legend else "brief")
    else:
        sns_plot = sns.relplot(
            x = "Time",
            y = "Value",
            col = meta_name,
            row = "endpoint",
            hue = "aACP",
            hue_order = hue_order,
            kind = "line",
            data=approver_scores,
            legend=False if hide_legend else "brief")
    sns_plot.savefig(out_plot)

def main(args=sys.argv[1:]):
    args = parse_args(args)
    #logging.basicConfig(format="%(message)s", filename=args.log_file, level=logging.DEBUG)
    print(args)
    #logging.info(args)
    np.random.seed(args.seed)

    results = []
    for idx in range(len(args.result_files)):
        res_file_template = args.result_files[idx]
        res_name = args.result_names[idx]
        for res in glob.glob(res_file_template):
            res_df = pickle_from_file(res)["scores"]
            res_df["aACP"] = res_name
            if args.meta_vals:
                meta_val = args.meta_vals[idx]
                res_df[args.meta_name] = meta_val
            results.append(res_df)
    results = pd.concat(results)
    results = results.rename(columns={"time": "Time", "val": "Value"})
    if args.endpoint is not None:
        results = results[results.endpoint == args.endpoint]
    if args.meta_name is not None:
        results = results.astype({args.meta_name: args.meta_type})

    sns.set_context("paper", font_scale=1.4)

    if args.endpoint is not None:
        results[args.endpoint] = results.Value
        plot_endpoint_scores(results, args.endpoint, args.out_plot, hue_order=args.result_names, meta_name=args.meta_name, hide_legend=args.hide_legend)
    else:
        plot_scores(results, args.out_plot, hue_order=args.result_names, meta_name=args.meta_name, hide_legend=args.hide_legend)

if __name__ == "__main__":
    main(sys.argv[1:])
