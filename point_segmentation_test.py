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


if not (np.min(ring) == 0 and np.max(ring) == 31):
    raise Exception("I assume EXACTLY 32 rings for now")
Nrings = 32


# I need to sort by ring and then by th
th = np.arctan2( points[:,1], points[:,0] )
def points_from_rings():
    for iring in range(Nrings):
        idx = ring==iring
        yield points[idx][ np.argsort(th[idx]) ]

points_sorted = nps.glue( *points_from_rings(),
                          axis = -2 )

Npoints = np.array([np.count_nonzero(ring==iring) for iring in range(Nrings)],
                   dtype = np.int32)

camera_lidar_calibration.point_segmentation(points  = points_sorted,
                                            Npoints = Npoints )
