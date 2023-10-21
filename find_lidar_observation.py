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

    parser.add_argument('--viz-show-point-cloud-context',
                        action='store_true',
                        help = '''If given, display ALL the points in the scene
                        to make it easier to orient ourselves''')

    parser.add_argument('bag',
                        type=str,
                        help = '''The rosbag that contains the lidar data''')

    parser.add_argument('lidar-topic',
                        type=str,
                        help = '''The one LIDAR topic we're looking at''')

    args = parser.parse_args()

    return args


args = parse_args()



import calibration_data_import

lidar_topic = getattr(args, 'lidar-topic')
bagname = os.path.split(os.path.splitext(os.path.basename(args.bag))[0])[1]
what = f"{bagname}-{os.path.split(lidar_topic)[1]}"

calibration_data_import.get_lidar_observation( \
                        args.bag,
                        lidar_topic,
                        what                         = what,
                        viz                          = True,
                        viz_show_only_accepted       = True,
                        viz_show_point_cloud_context = args.viz_show_point_cloud_context)
