import subprocess
from os import path
import pandas as pd

def get_gsm_bounds(alpha, num_batches, asf_factor=0.05, scratch_folder="scratch"):
    out_file = path.join(scratch_folder, "out.txt")
    cmd = [
            "Rscript",
            "R/get_gsm_bounds.R",
            "-a",
            str(alpha),
            "-b",
            str(num_batches),
            "-f",
            str(asf_factor),
            "-o",
            out_file]
    res = subprocess.call(cmd)
    bounds = pd.read_csv(str(out_file), header=None)
    return bounds.values.flatten()

