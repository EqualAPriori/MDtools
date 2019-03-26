# Template for writing a python script
import numpy as np
import sys
import os
import shutil
import mdtraj
import timeit
import argparse as ap


parser = ap.ArgumentParser(description="Strip Water from a trajectory")
parser.add_argument('top', type=str, help = "topology file")
args = parser.parse_args()

print("Parsing...\n")
print("topology file \t= {}".format( args.top ) )


