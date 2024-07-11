#!/usr/bin/python3
r'''Display a set of LIDAR point clouds in an aligned coordinate system

SYNOPSIS

  $ ./show-aligned-lidar-pointclouds.py \
      camera-lidar.bag          \
      /lidar/vl_points_0        \
      /lidar/vl_points_1        \
    [runs through this one scenario]

Displays aligned point clouds. Usefule for debugging

'''


import sys
import argparse
import re
import os

import numpy as np

def parse_args():

    parser = \
        argparse.ArgumentParser(description = __doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--threshold',
                        type=float,
                        default = 20.,
                        help = '''Max distance where we cut the plot''')

    parser.add_argument('--rt-lidar-ref',
                        type = str,
                        action = 'append',
                        help = '''Transforms for each LIDAR. Each transform is
                        given as r,r,r,t,t,t. Exactly as many --rt-lidar-ref
                        arguments must be given as LIDAR topics. This is
                        exclusive with --rt-ref-lidar''')
    parser.add_argument('--rt-ref-lidar',
                        type = str,
                        action = 'append',
                        help = '''Transforms for each LIDAR. Each transform is
                        given as r,r,r,t,t,t. Exactly as many --rt-lidar-ref
                        arguments must be given as LIDAR topics. This is
                        exclusive with --rt-lidar-ref''')

    parser.add_argument('bag',
                        type=str,
                        help = '''Glob for the rosbags that contains the lidar data''')

    parser.add_argument('lidar-topics',
                        type=str,
                        nargs = '+',
                        help = '''The LIDAR topics we're looking at. At least one must be given''')


    args = parser.parse_args()

    args.lidar_topics = getattr(args, 'lidar-topics')

    if args.rt_lidar_ref is None and \
       args.rt_ref_lidar is None:
        print("Exactly one of --rt-lidar-ref or --rt-ref-lidar must be given; got neither",
              file=sys.stderr)
        sys.exit(1)
    if args.rt_lidar_ref is not None and \
       args.rt_ref_lidar is not None:
        print("Exactly one of --rt-lidar-ref or --rt-ref-lidar must be given; got both",
              file=sys.stderr)
        sys.exit(1)

    def parse_rt(rt):
        s = rt.split(',')
        if len(s) != 6:
            print(f"Each --rt-... MUST be a comma-separated list of exactly 6 values: '{rt}' has the wrong number of values",
                  file=sys.stderr)
            sys.exit(1)
        try:
            s = [float(x) for x in s]
        except:
            print(f"Each --rt-... MUST be a comma-separated list of exactly 6 values: '{rt}' not parseable as a list of floats",
                  file=sys.stderr)
            sys.exit(1)
        return s

    Nlidar = len(args.lidar_topics)
    if args.rt_lidar_ref is not None:
        if len(args.rt_lidar_ref) != Nlidar:
            print(f"MUST have been given a matching number of --rt-lidar-ref and topics. Got {len(args.rt_lidar_ref)} and {Nlidar} respectively instead",
                  file=sys.stderr)
            sys.exit(1)

        args.rt_lidar_ref = np.array([parse_rt(rt) for rt in args.rt_lidar_ref])


    else:
        if len(args.rt_ref_lidar) != Nlidar:
            print(f"MUST have been given a matching number of --rt-ref-lidar and topics. Got {len(args.rt_ref_lidar)} and {Nlidar} respectively instead",
                  file=sys.stderr)
            sys.exit(1)

        args.rt_ref_lidar = np.array([parse_rt(rt) for rt in args.rt_ref_lidar])

    return args


args = parse_args()

import sys
import numpysane as nps
import mrcal
import gnuplotlib as gp

import calibration_data_import
import debag

try:
    pointcloud_msgs = \
        [ next(calibration_data_import.bag_messages_generator(args.bag, (topic,))) \
          for topic in args.lidar_topics ]
except:
    raise Exception(f"Bag '{args.bag}' doesn't have at least one message of {topic=}")

# Package into a numpy array
pointclouds = [ msg['array']['xyz'] \
                for msg in pointcloud_msgs ]

# Throw out everything that's too far, in the LIDAR's own frame
pointclouds = [ p[ nps.mag(p) < args.threshold ] for p in pointclouds ]

# Transform
if args.rt_lidar_ref is not None:
    args.rt_ref_lidar = mrcal.invert_rt(args.rt_lidar_ref)
pointclouds = [ mrcal.transform_point_rt(args.rt_ref_lidar[i],p) for i,p in enumerate(pointclouds) ]

data_tuples = [ ( p, dict( tuplesize = -3,
                           legend    = args.lidar_topics[i],
                           _with     = 'points pt 7 ps 1')) \
                for i,p in enumerate(pointclouds) ]

gp.plot(*data_tuples,
        _3d = True,
        square = True,
        wait = True)
