#!/usr/bin/python3

r'''Find the board in a point cloud

SYNOPSIS

  $ ./point_segmentation_test.py     \
      /lidar/vl_points_1     \
      'camera-lidar*.bag'

This tool is primarily for developing and debugging C code that interacts with
the LIDAR data. This tool makes various assumptions. Read the code before
blindly using this

'''


import sys
import argparse
import re
import os


def parse_args():

    parser = \
        argparse.ArgumentParser(description = __doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--dump',
                        action='store_true',
                        help = '''dump the c-level diagnostics; do not try to make Python plots''')
    parser.add_argument('lidar-topic',
                        type=str,
                        help = '''The LIDAR topic we're looking at''')
    parser.add_argument('bag',
                        type=str,
                        help = '''The rosbag that contain the lidar data.''')


    args = parser.parse_args()
    return args


args = parse_args()



import bag_interface
import numpy as np
import numpysane as nps
import camera_lidar_calibration

points,r = camera_lidar_calibration.point_segmentation(args.bag,
                                                       getattr(args, 'lidar-topic'),
                                                       dump    = args.dump)
if args.dump:
    sys.exit()


import gnuplotlib as gp
i = 2

p = r['plane_pn'][i,:3]
n = r['plane_pn'][i,3:]

# plane is all x such that inner(x-p,n) = 0
# -> nt p = nt x
# -> z = nt p / n2 - n0/n2 x - n1/n2 y
gp.plot(points[r['ipoint'][i]],
        square=1,
        _3d=1,
        tuplesize=-3,
        _with='points',
        equation = f"{nps.inner(n,p) / n[2]} - {n[0]/n[2]}*x - {n[1]/n[2]}*y")


import IPython
IPython.embed()
sys.exit()

