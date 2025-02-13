#!/usr/bin/python3
r'''Display a set of LIDAR point clouds in the aligned coordinate system

SYNOPSIS

  $ ./show-aligned-lidar-pointclouds.py                   \
      --bag camera-lidar.bag                              \
      --topic /lidar/vl_points_0,/lidar/vl_points_1 \
      /tmp/lidar[01]-mounted.cameramodel
    [plot pops up to show the aligned points]

Displays the point clouds in the lidar0 coord system

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

    parser.add_argument('--topic',
                        type=str,
                        required = True,
                        help = '''The LIDAR topics to visualize. This is a
                        comma-separated list of topics''')
    parser.add_argument('--bag',
                        required = True,
                        help = '''The one bag we're visualizing''')
    parser.add_argument('--after',
                        type=str,
                        help = '''If given, start reading the bags at this time.
                        Could be an integer (s since epoch or ns since epoch), a
                        float (s since the epoch) or a string, to be parsed with
                        dateutil.parser.parse()''')
    parser.add_argument('--threshold',
                        type=float,
                        default = 20.,
                        help = '''Max distance where we cut the plot''')
    parser.add_argument('lidar-models',
                        nargs   = '+',
                        help = '''The .cameramodel for the lidars in question.
                        Must correspond to the set in --topic.''')


    args = parser.parse_args()

    args.topic = args.topic.split(',')
    args.lidar_models = getattr(args, 'lidar-models')

    if len(args.lidar_models) != len(args.topic):
        print(f"MUST have been given a matching number of lidar models and topics. Got {len(args.lidar_models)=} and {len(args.topic)=} instead",
              file=sys.stderr)
        sys.exit(1)

    return args


args = parse_args()


import numpy as np
import mrcal
import clc


rt_lidar0_lidar = [mrcal.cameramodel(f).extrinsics_rt_toref() for f in args.lidar_models]

data_tuples = \
    clc.get_pointcloud_plot_tuples(args.bag, args.topic, args.threshold,
                                   rt_lidar0_lidar,
                                   start = args.after)
clc.plot(*data_tuples,
         _3d = True,
         square = True,
         xlabel = 'x (lidar0)',
         ylabel = 'y (lidar0)',
         zlabel = 'z (lidar0)',
         wait = True)
