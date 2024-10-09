#!/usr/bin/python3

r'''Dump a LIDAR scans to a dense binary file

SYNOPSIS

  $ ./dump-lidar-scan.py     \
      /lidar/vl_points_1     \
      'camera-lidar*.bag'
  Wrote '/tmp/tst.dat'

This tool is primarily for developing and debugging C code that interacts with
the LIDAR data. This tool makes various assumptions. Read the code before
blindly using this

'''


import sys
import argparse
import re
import os


def parse_args():

    parser = \
        argparse.ArgumentParser(description = __doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--dense',
                        action = 'store_true',
                        help = '''If given, generate dense data. This includes
                        (0,0,0) points in the no-data spots''')
    parser.add_argument('lidar-topic',
                        type=str,
                        help = '''The LIDAR topic we're looking at''')
    parser.add_argument('bag',
                        type=str,
                        help = '''The rosbag that contain the lidar data.''')


    args = parser.parse_args()
    return args


args = parse_args()



import bag_interface
import numpy as np
import numpysane as nps
import gnuplotlib as gp
import time
import shlex


lidar_topic = getattr(args, 'lidar-topic')

array = next(bag_interface.bag_messages_generator(args.bag, (lidar_topic,) ))['array']

points    = array['xyz']
intensity = array['intensity']
ring      = array['ring']


if not (np.min(ring) == 0 and np.max(ring) == 31):
    raise Exception("I assume EXACTLY 32 rings for now")
Nrings = 32


if not args.dense:

    filename = "/tmp/tst.dat"
    with open(filename, "wb") as f:


        f.write(f"# generated on {time.strftime('%Y-%m-%d %H:%M:%S')} with   {' '.join(shlex.quote(s) for s in sys.argv)}\n".encode())
        f.write("version = 1\n".encode())
        f.write(f"Nrings = {Nrings}".encode()) # no trailing \n is intentional. See following

        # I want to align the data to 16 bytes (probably overkill, but why not).
        # So I add (' '*N + '\n') to the text header to make it line up. The
        # spaces are optional, but the newline is not, so after the spaces I
        # want f.tell % 16 == 15.

        # Might be needed for some weird architectures?
        if len(' ') != 1 or len('\n') != 1:
            raise Exception("I'm assuming spaces and newlines each weight in at 1 byte")

        # I want
        #
        # f.tell%16 Nspaces_to_add
        # 15        0
        # 14        1
        # ...
        #  1        14
        #  0        15
        Nspaces = 15 - (f.tell() % 16)
        f.write( (Nspaces * b' ') + b'\n')
        if f.tell()%16:
            raise Exception("Data not aligned as expected")

        for iring in range(Nrings):
            idx = ring==iring
            points_thisring  = points[idx]
            th_thisring = np.arctan2( points_thisring[:,1], points_thisring[:,0] )
            points_thisring_sorted = points_thisring[np.argsort(th_thisring)]

            # I write a 16-byte block containing the number of points in this
            # ring, and then the point data itself
            f.write(f'{len(points_thisring_sorted):<16}'.encode())
            points_thisring_sorted.tofile(f)

    print(f"Wrote '{filename}'")

    sys.exit(0)




print("--dense is deprecated. I no longer use it, and it will probably be removed")


th = np.arctan2( points[:,1], points[:,0] )

# Obtained empirically ...
Nth_per_rotation = 1809
# ... like this ...:
# dth = 2.*np.pi / Nth_per_rotation
# iring = 10
# th0 = th[ring==iring][np.argmin(np.abs(th[ring==iring]))]
# ith = np.round((th[ring==iring]-th0)/dth).astype(int)
# gp.plot( ith*dth+th0 - th[ring==iring] )
# ... I'm looking for smallest offsets


points_dense = np.zeros((Nrings,Nth_per_rotation,3), dtype=np.float32)

dth = 2.*np.pi / Nth_per_rotation
ith0 = -(Nth_per_rotation//2)
for iring in range(Nrings):
    th_here     = th    [ring==iring]
    points_here = points[ring==iring]

    th0 = th_here[np.argmin(np.abs(th_here))]
    ith = np.round((th[ring==iring]-th0)/dth).astype(int) - ith0
    ith -= np.min(ith) # needed because sometimes I see min,max=1,N and I want 0,N-1

    if False:
        print(f"{np.min(ith)=} {np.max(ith)=}")
        continue
    if not (np.min(ith) >= 0 and np.max(ith) < Nth_per_rotation):
        raise Exception("Inconsistent azimuths")

    points_dense[iring,ith,:] = points_here

filename = "/tmp/tst.dat"
points_dense.tofile(filename)
print(f"Wrote '{filename}'")
