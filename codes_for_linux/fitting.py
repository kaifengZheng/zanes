import numpy as np
import pandas as pd
from larch import Group
from zanes.data_analysis import data_processing as dp
from zanes.data_analysis import zanes_data_analysis as zda
import matplotlib.pyplot as plt
import yaml
import toml
import os
import argparse
from os.path import basename, join
from os import mkdir
from glob import glob

parser = argparse.ArgumentParser(description="multi-dataset EXAFS fitting")
parser.add_argument("--config", type=str, default="input.yaml", help="config file path")
parser.add_argument("--input", type=str, default="data", help="input file path")
parser.add_argument(
    "-o", "--outputfolder", type=str, default="results", help="output folder path"
)
parser.add_argument(
    "-os",
    "--outputsurfix",
    type=str,
    default="",
    help="output file surfix(describe the fitting condition)",
)
parser.add_argument(
    "-dt", "--datatype", type=str, default="ProQEXAFS", help="data type"
)
parser.add_argument(
    "-bs", "--batchsize", type=int, default=-1, help="batch size for fitting"
)


args = parser.parse_args()
config_file = args.config
inputfile = args.input
output_folder = args.outputfolder
output_surfix = args.outputsurfix
datatype = args.datatype
batch_size = args.batchsize


with open(config_file, "r", encoding="utf-8") as yaml_file:
    dict_get = yaml.load(yaml_file, Loader=yaml.FullLoader)
norm_param = dict_get["norm_param"]
fitting_param = dict_get["fitting_param"]
fitting_range = dict_get["fitting_range"]
file_name = basename(inputfile)
foldername = join(output_folder, file_name[:-4] + output_surfix)
if not os.path.exists(foldername):
    mkdir(foldername)
exp_data = zda()
exp_data.read_datacollection(inputfile, datatype=datatype)
if batch_size < 0:
    batch_size = len(exp_data.data)
exp_data.process_data(norm_param)
dset, out, report, path = exp_data.run_fit_batch(
    fitting_param,
    fitting_range,
    [[0], [0]],
    ["feff_PtO", "feff"],
    save_name=join(foldername, file_name[:-4] + output_surfix),
    write=True,
    mpi=False,
    core=10,
    batch_size=batch_size,
)
report_sort = exp_data.write_sorted_report(out, report, fitting_param, path[0])
exp_data.write_fitted_data(dset, inputfile, foldername, suffix=output_surfix)
output_fit = join(foldername, file_name[:-4] + output_surfix + ".toml")
with open(output_fit, "w") as f:
    toml.dump(report_sort, f)
