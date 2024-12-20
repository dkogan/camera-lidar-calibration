#!/usr/bin/python3

r'''Display the LIDAR scans from a rosbag

SYNOPSIS

  $ ./show-bag.py             \
      'camera-lidar*.bag'

  Bag 'camera-lidar-0.bag':
    /points0
    /points1
  Bag 'camera-lidar-1.bag':
    /points0
    /points1
    /image0
  ....


  $ ./show-bag.py             \
      --period 2              \
      --set "view 60,30,2"    \
      --set "xrange [-20:20]" \
      --set "yrange [-20:20]" \
      --set "zrange [-2:2]"   \
      --topic /points0 \
      'camera-lidar*.bag'

    [A plot pops up showing the LIDAR scans from the bags, updating every 2 sec]


  $ ./show-bag.py     \
      --topic /image0 \
      'camera-lidar*.bag'

    [A plot pops up showing the images in the bags. Updated with the next bag
    when the user closes the plot]

This is a display tool to show the contents of a ros bag. This is roughly
similar to the "rosbag" and "ros2 bag" and "rviz" tools, but uses the "rosbags"
Python package, and does NOT require ROS to be installed

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
                        help = '''Applies to LIDAR data. If given, cut off the
                        points at this range''')
    parser.add_argument('--ring',
                        type=int,
                        help = '''Applies to LIDAR data. If given, show ONLY
                        data from this ring. Otherwise, display all of them''')
    parser.add_argument('--xy',
                        action='store_true',
                        help = '''Applies to LIDAR data. If given, I make a 2D
                        plot, ignoring the z axis''')
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

    parser.add_argument('--topic',
                        type=str,
                        help = '''The topic we're visualizing. Can select LIDAR
                        or camera data. If omitted, we report the topics present
                        in the bag, and we exit''')

    parser.add_argument('bags',
                        type=str,
                        nargs='+',
                        help = '''Glob(s) for the rosbags that contain the lidar
                        data. Each of these will be sorted alphanumerically''')


    args = parser.parse_args()
    return args


args = parse_args()



def bags():
    import glob
    for bag_glob in args.bags:
        bags = sorted(glob.glob(bag_glob))
        if len(bags) == 0:
            print(f"No files matched the glob '{bag_glob}'", file=sys.stderr)
            sys.exit(1)

        for bag in bags:
            yield(bag)


import bag_interface

if args.topic is None:
    for bag in bags():
        print(f"Bag '{bag}':")
        for topic in bag_interface.topics(bag):
            print(f"  {topic}")
    sys.exit()



import numpy as np
import numpysane as nps
import gnuplotlib as gp
import time


def show_lidar(bag, p):
    kwargs = dict( _set   = args.set,
                   _unset = args.unset)

    gp.add_plot_option(kwargs,
                       title = bag,
                       overwrite = False)


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
            title  = f"{bag=} {args.topic=}",
            xlabel = 'x',
            ylabel = 'y',
            wait   = args.period <= 0,
            **kwargs)

    if args.period > 0:
        time.sleep(args.period)



def show_image(bag, p):
    kwargs = dict( _set   = args.set,
                   _unset = args.unset)

    gp.add_plot_option(kwargs,
                       title = bag,
                       overwrite = False)

    gp.add_plot_option(kwargs,
                       set = ('xrange [:] noextend',
                              'yrange [:] noextend reverse',),
                       unset = 'colorbox')

    if p.ndim == 2:
        # grayscale
        gp.add_plot_option(kwargs,
                           set = 'palette gray')

        gp.plot(p,
                _with     = 'image',
                tuplesize = 3,
                square    = True,
                wait      = args.period <= 0,
                **kwargs)
    else:
        # color
        gp.plot(p[...,2], p[...,1], p[...,0],
                _with     = 'rgbimage',
                tuplesize = 5,
                square    = True,
                wait      = args.period <= 0,
                **kwargs)

    if args.period > 0:
        time.sleep(args.period)



for bag in bags():
    try:
        p = next(bag_interface.messages(bag, (args.topic,) ))['array']
    except StopIteration:
        print(f"No messages with {args.topic=} in {bag=}. Continuing to next bag, if any",
              file = sys.stderr)
        continue


    def has_xyz(p):
        try:
            if 'xyz' in p.dtype.names:
                return True
        except:
            return False

    if has_xyz(p):
        show_lidar(bag, p)
    elif p.dtype == np.uint8 and \
         (p.ndim == 2 or (p.ndim==3 and p.shape[-1] == 3)):
        show_image(bag, p)
    else:
        print(f"Cannot interpret message from {args.topic}",
              file=sys.stderr)
        sys.exit(1)
