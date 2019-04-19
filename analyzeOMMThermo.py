import numpy as np
import matplotlib.pyplot as plt
import argparse as ap

parser = ap.ArgumentParser(description="Analyze OpenMM Thermo log")
parser.add_argument('file', type=str, help='thermo log filename')
args = parser.parse_args()

print("Parsing...\n")
print("Log file name \t {}".format(args.file))

data = np.loadtxt(args.file, skiprows=1, delimiter=",")


