#!/usr/bin/python3

r'''Report some basic info about the contents of a ros bag

SYNOPSIS

  $ ./bag_info.py lidar-and-camera.bag

  /lidar/lidar_points_0
  /lidar/lidar_points_1
  /front/multisense/left/image_mono
  /front/multisense/right/image_mono
  /front/multisense/aux/image_color

This is roughly similar to the "rosbag" and "ros2 bag" tools, but uses the
"rosbags" Python package, and does NOT require ROS to be installed

'''


import sys
import argparse
import re
import os


def parse_args():

    parser = \
        argparse.ArgumentParser(description = __doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('bag',
                        type=str,
                        help = '''The rosbag''')


    args = parser.parse_args()

    return args


args = parse_args()



import bag_interface

for topic in bag_interface.topics(args.bag):
    print(topic)
