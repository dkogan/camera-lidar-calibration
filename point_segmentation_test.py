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

lidar_topic = getattr(args, 'lidar-topic')

array = next(bag_interface.bag_messages_generator(args.bag, (lidar_topic,) ))['array']

points    = array['xyz']
intensity = array['intensity']
ring      = array['ring']

rings = np.unique(ring)

# I need to sort by ring and then by th
th = np.arctan2( points[:,1], points[:,0] )
def points_from_rings():
    for iring in rings:
        idx = ring==iring
        yield points[idx][ np.argsort(th[idx]) ]

points_sorted = nps.glue( *points_from_rings(),
                          axis = -2 )

Npoints = np.array([np.count_nonzero(ring==iring) for iring in rings],
                   dtype = np.int32)

r = camera_lidar_calibration.point_segmentation(points  = points_sorted,
                                                Npoints = Npoints,
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
gp.plot(points_sorted[r['ipoint'][i]],
        square=1,
        _3d=1,
        tuplesize=-3,
        _with='points',
        equation = f"{nps.inner(n,p) / n[2]} - {n[0]/n[2]}*x - {n[1]/n[2]}*y")


import IPython
IPython.embed()
sys.exit()

