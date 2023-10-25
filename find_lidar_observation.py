#!/usr/bin/python3

r'''Calibrate a set of cameras and LIDARs into a common coordinate system

SYNOPSIS

  $ ./find_lidar_observation.py \
      camera-lidar.bag          \
      /lidar/vl_points_0

This is a development tool to debug the board detection in LIDAR data. This tool
processes one topic from one bag, and produces the result visualizations.

'''


import sys
import argparse
import re
import os


def parse_args():

    parser = \
        argparse.ArgumentParser(description = __doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--viz',
                        action = argparse.BooleanOptionalAction,
                        help = '''By default, we produce visualizations ONLY if
                        we have ONE bag and ONE lidar-topic; if we have more of
                        either, we do NOT visualize. To force visualization,
                        pass --viz; to force NO visualization, pass --no-viz''')

    parser.add_argument('--viz-show-point-cloud-context',
                        action='store_true',
                        help = '''If given, display ALL the points in the scene
                        to make it easier to orient ourselves''')

    parser.add_argument('bag',
                        type=str,
                        help = '''Glob for the rosbags that contains the lidar data''')

    parser.add_argument('lidar-topic',
                        type=str,
                        nargs = '+',
                        help = '''The LIDAR topics we're looking at. At least one must be given''')


    args = parser.parse_args()

    return args


args = parse_args()

import calibration_data_import
import glob


lidar_topics = getattr(args, 'lidar-topic')
bags = glob.glob(args.bag)
if len(bags) == 0:
    print(f"No files matched the glob '{args.bag}'", file=sys.stderr)
    sys.exit(1)

if args.viz is not None:
    # user asked for specific visualization settings
    viz = args.viz
else:
    # default viz
    viz = (len(bags) == 1 and len(lidar_topics) == 1)


for lidar_topic in lidar_topics:
    for bag in bags:

        bagname = os.path.split(os.path.splitext(os.path.basename(bag))[0])[1]
        what = f"{bagname}-{os.path.split(lidar_topic)[1]}"

        calibration_data_import.get_lidar_observation( \
                                bag,
                                lidar_topic,
                                what                         = what,
                                viz                          = viz,
                                viz_show_only_accepted       = False,
                                viz_show_point_cloud_context = args.viz_show_point_cloud_context)
