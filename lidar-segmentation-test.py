#!/usr/bin/python3

r'''Find the board in a point cloud

SYNOPSIS

  $ ./lidar-segmentation-test.py     \
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

import clc
lidar_segmentation_parameters = clc.lidar_segmentation_parameters()

def parse_args():

    parser = \
        argparse.ArgumentParser(description = __doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--debug',
                        type=float,
                        nargs=5,
                        help = '''If given, overrides --debug-iring,
                        --debug-xmin, --debug-ymin, --debug-xmax, --debug-ymax''')
    parser.add_argument('--after',
                        type=str,
                        help = '''If given, start reading the bag at this time.
                        Could be an integer (s since epoch or ns since epoch), a
                        float (s since the epoch) or a string, to be parsed with
                        dateutil.parser.parse()''')
    parser.add_argument('lidar-topic',
                        type=str,
                        help = '''The LIDAR topic we're looking at''')
    parser.add_argument('bag',
                        type=str,
                        help = '''The rosbag that contain the lidar data.''')

    for param,metadata in lidar_segmentation_parameters.items():
        if metadata['pyparse'] == 'p':
            # special case for boolean
            parser.add_argument(f'--{param.replace("_","-")}',
                                action  = 'store_true',
                                help = metadata['doc'])
        else:
            parser.add_argument(f'--{param.replace("_","-")}',
                                type    = type(metadata['default']),
                                default = metadata['default'],
                                help = metadata['doc'])

    args = parser.parse_args()
    return args


args = parse_args()



import numpy as np
import numpysane as nps

ctx = dict()
for k in lidar_segmentation_parameters.keys():
    ctx[k] = getattr(args, k)

if args.debug is not None:
    ctx['debug_iring'] = int(args.debug[0])
    ctx['debug_xmin']  = args.debug[1]
    ctx['debug_ymin']  = args.debug[2]
    ctx['debug_xmax']  = args.debug[3]
    ctx['debug_ymax']  = args.debug[4]

segmentation = \
    clc.lidar_segmentation(bag         = args.bag,
                           lidar_topic = getattr(args, 'lidar-topic'),
                           start       = args.after,
                           **ctx)
if args.dump or \
   args.debug_xmin <  1e10 or \
   args.debug_xmax > -1e10 or \
   args.debug_ymin <  1e10 or \
   args.debug_ymax > -1e10:
    # Write the planes out to stdout, in a way that can be cut/pasted into
    # test/test-lidar-segmentation.py
    plane_p = segmentation['plane_p']
    plane_n = segmentation['plane_n']
    for i in range(len(plane_p)):
        print(f"### plane {i}",
              file = sys.stderr)
        print(f"         plane_p = np.array(({plane_p[i,0]:.3f},{plane_p[i,1]:.3f},{plane_p[i,2]:.3f})),",
              file = sys.stderr)
        print(f"         plane_n = np.array(({plane_n[i,0]:.4f},{plane_n[i,1]:.4f},{plane_n[i,2]:.4f})),",
              file = sys.stderr)
    sys.exit()


import gnuplotlib as gp

Nplanes = len(segmentation['plane_p'])
if Nplanes == 0:
    print("No planes found")
    sys.exit()


i = 0

plane_p = segmentation['plane_p'][i]
plane_n = segmentation['plane_n'][i]

# plane is all x such that inner(x-p,n) = 0
# -> nt p = nt x
# -> z = nt p / n2 - n0/n2 x - n1/n2 y
gp.plot(segmentation['points'][i],
        square=1,
        _3d=1,
        tuplesize=-3,
        _with='points',
        title = "First reported plane",
        xlabel = 'x',
        ylabel = 'y',
        zlabel = 'z',
        equation = f"{nps.inner(plane_n,plane_p) / plane_n[2]} - {plane_n[0]/plane_n[2]}*x - {plane_n[1]/plane_n[2]}*y")


import IPython
IPython.embed()
sys.exit()

