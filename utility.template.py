# 2019.03 Template for writing a utility that python can also call
import numpy as np
import sys
import os
import shutil
import mdtraj
import timeit
import argparse as ap


def main(topname):
    """ Strips the water from a trajectory file. Uses MDtraj

    Parameters
    ----------
    topname : str
        name of topology file
    Returns
    -------
    """

if __name__ == "__main__":
    parser = ap.ArgumentParser(description="Strip Water from a trajectory")
    parser.add_argument('top', type=str, help = "topology file")
    args = parser.parse_args()

    print("Parsing...\n")
    print("topology file \t= {}".format( args.top ) )
    main( args.top )

