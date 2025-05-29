#!/usr/bin/python3

r'''Report the az and el layout present in a LIDAR dataset

SYNOPSIS

  $ ./infer-lidar-spacing.py \
      /points0 \
      camera-lidar0.bag

  Nrings=128
  Npoints_per_rotation=512

  ....

These parameters are used by the lidar segmentation routine
'''


import sys
import argparse
import re
import os


def parse_args():

    parser = \
        argparse.ArgumentParser(description = __doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--show-histogram',
                        action = 'store_true',
                        help = '''If given, we display the histogram of
                        potential Npoints_per_rotation values. Hopefully we see
                        a sharp peak at the "right" value and that it is a power
                        of 2'''),
    parser.add_argument('lidar-topic',
                        type=str,
                        help = '''The LIDAR topic we're looking at''')
    parser.add_argument('bag',
                        type=str,
                        help = '''The rosbag that contain the lidar data.''')


    args = parser.parse_args()
    args.topic = getattr(args, 'lidar-topic')
    return args


args = parse_args()






import clc.bag_interface
import numpy as np
import numpysane as nps


msg_iter = \
    clc.bag_interface.messages(args.bag, (args.topic,),
                               ignore_unknown_message_types = True)

msg = next(msg_iter)

# for msg in msg_iter:
#     p    = msg['array']['xyz']
#     ring = msg['array']['ring']
#     ring_min = np.min(ring)
#     ring_max = np.max(ring)
#     print(f"{ring_min=} {ring_max=}")
# sys.exit()



p    = msg['array']['xyz']
ring = msg['array']['ring']

Nrings = np.max(ring)+1

i = nps.norm2(p) > 0

p    = p[i]
ring = ring[i]

az = np.arctan2(p[:,1], p[:,0])
el = np.arctan2(p[:,2], nps.mag(p[:,:2]))

Npoints_per_rotation_per_ring = \
    [ np.round( 2.*np.pi / np.abs(np.diff(az[ring == iring]))).astype(int) \
      for iring in range(Nrings) ]
Npoints_per_rotation = nps.glue(*Npoints_per_rotation_per_ring, axis=-1)


if args.show_histogram:
    import gnuplotlib as gp
    gp.plot(Npoints_per_rotation[Npoints_per_rotation < 1e6],
            histogram = True,
            binwidth  = True,
            wait      = True)

counts = np.bincount(Npoints_per_rotation[Npoints_per_rotation < 1e6])
Npoints_per_rotation,Npoints_per_rotation_second = np.argsort(-counts)[:2]

ratio = counts[Npoints_per_rotation] / counts[Npoints_per_rotation_second]
if ratio < 5.:
    print(f"WARNING: the delta-az histogram doesn't have a single 5x spike. count(first)/count(second) = {ratio:.1f}; Npoints_per_rotation might be wrong")

print(f"{Nrings=}")
print(f"{Npoints_per_rotation=}")
