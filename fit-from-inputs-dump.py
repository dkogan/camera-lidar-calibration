#!/usr/bin/python3
r'''Calibrate from a binary input dump

For debugging

'''


import sys
import argparse
import argparse_helpers
import re
import os


def parse_args():

    parser = \
        argparse.ArgumentParser(description = __doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--inject-noise',
                        action='store_true',
                        help = '''If given, we add expected noise to the
                        observations. If --inject-noise and no --fit-seed then I
                        will fit() from the previous fit() result, NOT from the
                        previous fit_seed() result.''')
    parser.add_argument('--fit-seed',
                        action='store_true',
                        help = '''If given, we fit_seed() and then fit(). By
                        default (no --fit-seed), we only call fit(): from the
                        previous fit_seed() result if no --inject-noise or from
                        the previous fit() result if --inject-noise''')
    parser.add_argument('--exclude',
                        type=argparse_helpers.comma_separated_list_of_non_negative_integers,
                        help = '''Optional comma-separated list of integers >=
                        0. If given, exclude these snapshots''')
    parser.add_argument('--verbose',
                        action = 'store_true',
                        help='''Report details about the solve''')
    parser.add_argument('--dump',
                        type=str,
                        help = '''Write solver diagnostics into the given
                        .pickle file. Primarily to feed
                        show-transformation-uncertainty.py''')
    parser.add_argument('context',
                        help = '''.pickle file from fit.py --dump or
                        buf_inputs_dump from the clc_...() C functions''')
    args = parser.parse_args()

    return args


args = parse_args()

import numpy as np
import numpysane as nps
import pickle
import clc
import mrcal

import testutils

with open(args.context, "rb") as f:
    try:
        context = pickle.load(f)
        dump    = context['result']['inputs_dump']
    except pickle.UnpicklingError:
        # maybe it's a binary dump
        dump = f.read()

if args.exclude:
    args.exclude = \
        np.array(args.exclude,
                 dtype=np.int32)


result = clc.fit_from_inputs_dump(dump,
                                  isnapshot_exclude = args.exclude,
                                  do_inject_noise = args.inject_noise,
                                  do_fit_seed     = args.fit_seed,
                                  verbose         = args.verbose,
                                  do_skip_plots   = False)


Nlidars  = len(result["rt_lidar0_lidar" ])
Ncameras = len(result["rt_lidar0_camera"])


topics = \
    [f"lidar{i}"  for i in range(Nlidars )] + \
    [f"camera{i}" for i in range(Ncameras)]

context = \
    dict(result           = result,
         topics           = topics,
         kwargs_calibrate = dict())
if args.dump is not None:
    with open(args.dump, 'wb') as f:
        pickle.dump( context,
                     f )
    print(f"Wrote '{args.dump}'")
