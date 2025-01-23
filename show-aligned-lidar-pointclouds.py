#!/usr/bin/python3
r'''Display a set of LIDAR point clouds in the aligned coordinate system

SYNOPSIS

  $ ./show-aligned-lidar-pointclouds.py                   \
      --bag camera-lidar.bag                              \
      --topic /lidar/vl_points_0,/lidar/vl_points_1 \
      /tmp/lidar[01]-mounted.cameramodel
    [plot pops up to show the aligned points]

Displays the point clouds in a common vehicle coordinate system

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

    parser.add_argument('--rt-vehicle-lidar0',
                        type=argparse_helpers.comma_separated_list_of_floats,
                        help='''The vehicle-lidar0 transform. The solve is
                        always done in lidar0 coordinates, but we may want to
                        operate in a different "vehicle" frame. This argument
                        specifies the relationship between those frames. If
                        omitted, we assume an identity transform: the vehicle
                        frame is the lidar0 frame''')
    parser.add_argument('--topic',
                        type=str,
                        required = True,
                        help = '''The LIDAR topics to visualize. This is a
                        comma-separated list of topics''')
    parser.add_argument('--bag',
                        required = True,
                        help = '''The one bag we're visualizing''')
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



if args.rt_vehicle_lidar0 is not None:
    args.rt_vehicle_lidar0 = np.array(args.rt_vehicle_lidar0, dtype=float)
    args.Rt_vehicle_lidar0 = mrcal.Rt_from_rt(args.rt_vehicle_lidar0)
else:
    args.Rt_vehicle_lidar0 = None



rt_lidar0_lidar = [mrcal.cameramodel(f).extrinsics_rt_toref() for f in args.lidar_models]

data_tuples = \
    clc.get_pointcloud_plot_tuples(args.bag, args.topic, args.threshold,
                                   rt_lidar0_lidar,
                                   Rt_vehicle_lidar0 = args.Rt_vehicle_lidar0)
clc.plot(*data_tuples,
         _3d = True,
         square = True,
         xlabel = 'x (vehicle)',
         ylabel = 'y (vehicle)',
         zlabel = 'z (vehicle)',
         wait = True)
