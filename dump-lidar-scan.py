#!/usr/bin/python3

r'''Dump a LIDAR scans to a dense binary file

SYNOPSIS

  $ ./dump-lidar-scan.py     \
      /lidar/vl_points_1     \
      'camera-lidar*.bag' > lidar.dump

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
import gnuplotlib as gp


lidar_topic = getattr(args, 'lidar-topic')

array = next(bag_interface.bag_messages_generator(args.bag, (lidar_topic,) ))['array']

points    = array['xyz']
intensity = array['intensity']
ring      = array['ring']

if not (np.min(ring) == 0 and np.max(ring) == 31):
    raise Exception("I assume EXACTLY 32 rings for now")
Nrings = 32

th = np.arctan2( points[:,1], points[:,0] )

# Obtained empirically ...
Nth_per_rotation = 1809
# ... like this ...:
# dth = 2.*np.pi / Nth_per_rotation
# iring = 10
# th0 = th[ring==iring][np.argmin(np.abs(th[ring==iring]))]
# ith = np.round((th[ring==iring]-th0)/dth).astype(int)
# gp.plot( ith*dth+th0 - th[ring==iring] )
# ... I'm looking for smallest offsets


points_dense = np.zeros((Nrings,Nth_per_rotation,3), dtype=np.float32)

dth = 2.*np.pi / Nth_per_rotation
ith0 = -(Nth_per_rotation//2)
for iring in range(Nrings):
    th_here     = th    [ring==iring]
    points_here = points[ring==iring]

    th0 = th_here[np.argmin(np.abs(th_here))]
    ith = np.round((th[ring==iring]-th0)/dth).astype(int) - ith0
    ith -= np.min(ith) # needed because sometimes I see min,max=1,N and I want 0,N-1

    if False:
        print(f"{np.min(ith)=} {np.max(ith)=}")
        continue
    if not (np.min(ith) >= 0 and np.max(ith) < Nth_per_rotation):
        raise Exception("Inconsistent azimuths")

    points_dense[iring,ith,:] = points_here

points_dense.tofile(sys.stdout)
