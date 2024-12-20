#!/usr/bin/python3

r'''Display the LIDAR scans from a rosbag

SYNOPSIS

  $ ./show-bag.py             \
      --period 2              \
      --set "view 60,30,2"    \
      --set "xrange [-20:20]" \
      --set "yrange [-20:20]" \
      --set "zrange [-2:2]"   \
      /lidar/vl_points_1      \
      'camera-lidar*.bag'
    [A plot pops up showing the scans from the bags, updating every 2 sec]

This is a display tool to show the LIDAR scans one at a time. It's a very poor
man's rviz display, without requiring any ros tools to be installed

'''


import sys
import argparse
import re
import os


def parse_args():

    parser = \
        argparse.ArgumentParser(description = __doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--maxrange',
                        type=float,
                        help = '''If given, cut off the points at this range''')
    parser.add_argument('--ring',
                        type=int,
                        help = '''If given, show ONLY data from this ring.
                        Otherwise, display all of them''')
    parser.add_argument('--xy',
                        action='store_true',
                        help = '''If given, I make a 2D plot, ignoring the z axis''')
    parser.add_argument('--period',
                        type=float,
                        default = 0,
                        help = '''How much time to wait between moving to the
                        next bag; if <= 0 (the default), we wait until each
                        window is manually closed''')
    parser.add_argument('--set',
                        type=str,
                        action='append',
                        help='''Extra 'set' directives to gnuplotlib. Can be given multiple times''')
    parser.add_argument('--unset',
                        type=str,
                        action='append',
                        help='''Extra 'unset' directives to gnuplotlib. Can be given multiple times''')

    parser.add_argument('lidar-topic',
                        type=str,
                        help = '''The LIDAR topic we're looking at''')

    parser.add_argument('bags',
                        type=str,
                        nargs='+',
                        help = '''Glob(s) for the rosbags that contain the lidar
                        data. Each of these will be sorted alphanumerically''')


    args = parser.parse_args()
    return args


args = parse_args()

import bag_interface
import glob
import numpy as np
import numpysane as nps
import gnuplotlib as gp
import time


lidar_topic = getattr(args, 'lidar-topic')

for bag_glob in args.bags:
    bags = sorted(glob.glob(bag_glob))
    if len(bags) == 0:
        print(f"No files matched the glob '{bag_glob}'", file=sys.stderr)
        sys.exit(1)

    for bag in bags:
        try:
            p = next(bag_interface.messages(bag, (lidar_topic,) ))['array']
        except StopIteration:
            print(f"No messages with {lidar_topic=} in {bag=}. Continuing to next bag, if any",
                  file = sys.stderr)
            continue

        xyz       = p['xyz']
        intensity = p['intensity']

        if args.ring is not None:
            i = p['ring'] == args.ring
            xyz       = xyz[i]
            intensity = intensity[i]

        if args.maxrange is not None:
            i = nps.norm2(xyz) <= args.maxrange * args.maxrange
            xyz       = xyz[i]
            intensity = intensity[i]

        if not args.xy:
            data_tuple = (xyz[:,0], xyz[:,1], xyz[:,2])
        else:
            data_tuple = (xyz[:,0], xyz[:,1])
        gp.plot(*data_tuple,
                intensity,
                tuplesize = len(data_tuple)+1,
                _with  = 'dots palette',
                square = True,
                _3d    = not args.xy,
                _set   = args.set,
                _unset = args.unset,
                title  = f"{bag=} {lidar_topic=}",
                xlabel = 'x',
                ylabel = 'y',
                wait   = args.period <= 0)

        if args.period > 0:
            time.sleep(args.period)
