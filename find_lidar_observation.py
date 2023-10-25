#!/usr/bin/python3

r'''Calibrate a set of cameras and LIDARs into a common coordinate system

SYNOPSIS

  $ ./find_lidar_observation.py \
      camera-lidar.bag          \
      /lidar/vl_points_0
    [runs through this one scenario]

  $ ./find_lidar_observation.py     \
      --check                       \
      'camera-lidar*.bag'           \
      /lidar/vl_points_{0,1,2}
    [unit test to make sure this scenario works properly]

  $ ./find_lidar_observation.py     \
      --generate-ground-truth       \
      'camera-lidar*.bag'           \
      /lidar/vl_points_{0,1,2}

    [Produces ground-truth for --check. These results must be added to this]
    [script]

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

    parser.add_argument('--check',
                        action = 'store_true',
                        help = '''If given, we run the test suite''')

    parser.add_argument('--generate-ground-truth',
                        action = 'store_true',
                        help = '''If given, we generate the ground-truth data''')

    parser.add_argument('bag',
                        type=str,
                        help = '''Glob for the rosbags that contains the lidar data''')

    parser.add_argument('lidar-topic',
                        type=str,
                        nargs = '+',
                        help = '''The LIDAR topics we're looking at. At least one must be given''')


    args = parser.parse_args()

    if args.check and args.generate_ground_truth:
        print("--check and --generate-ground-truth are mutually exclusive",
              file = sys.stderr)
        sys.exit(1)
    if args.check and args.viz:
        print("--check and --viz are mutually exclusive",
              file = sys.stderr)
        sys.exit(1)

    return args


args = parse_args()

import calibration_data_import
import glob
import numpy as np
import numpysane as nps
import pprint


def print_red(x):
    """print the message in red"""
    sys.stdout.write("\x1b[31m" + x + "\x1b[0m\n")


def print_green(x):
    """Print the message in green"""
    sys.stdout.write("\x1b[32m" + x + "\x1b[0m\n")

def print_blue(x):
    """Print the message in blue"""
    sys.stdout.write("\x1b[34m" + x + "\x1b[0m\n")

def ok(msg):
    print_green(f"OK: bag={bagname} topic={lidar_topic}: {msg}")
def fail(msg):
    global Nfail
    print_red(f"FAILED: bag={bagname} topic={lidar_topic}: {msg}")
    Nfail += 1



if args.generate_ground_truth:
    expected = dict()
elif args.check:
    # unit test data
    array = np.array
    expected = \
{'/lidar/velodyne_front_horiz_points': {'one_cal_data_2023-10-19-20-36-36': None,
                                          'one_cal_data_2023-10-19-20-36-47': array([ 2.72895943,  0.1553812 , -0.10712708]),
                                          'one_cal_data_2023-10-19-20-36-57': None,
                                          'one_cal_data_2023-10-19-20-37-04': None,
                                          'one_cal_data_2023-10-19-20-37-12': None,
                                          'one_cal_data_2023-10-19-20-37-27': array([0.93806243, 3.55218633, 0.28569284]),
                                          'one_cal_data_2023-10-19-20-37-32': array([0.937604  , 3.58339717, 0.36106848]),
                                          'one_cal_data_2023-10-19-20-37-40': array([0.35613021, 3.17405553, 0.30846942]),
                                          'one_cal_data_2023-10-19-20-37-44': array([0.46325054, 3.15986437, 0.24403726]),
                                          'one_cal_data_2023-10-19-20-37-48': array([0.46972783, 2.88742341, 0.20659553]),
                                          'one_cal_data_2023-10-19-20-37-53': array([ 0.51652713,  2.67618973, -0.00732552]),
                                          'one_cal_data_2023-10-19-20-38-02': None,
                                          'one_cal_data_2023-10-19-20-38-06': None,
                                          'one_cal_data_2023-10-19-20-38-10': None,
                                          'one_cal_data_2023-10-19-20-38-14': None,
                                          'one_cal_data_2023-10-19-20-38-18': None,
                                          'one_cal_data_2023-10-19-20-38-23': None,
                                          'one_cal_data_2023-10-19-20-39-03': None,
                                          'one_cal_data_2023-10-19-20-39-15': None,
                                          'one_cal_data_2023-10-19-20-39-32': None,
                                          'one_cal_data_2023-10-19-20-39-39': None,
                                          'one_cal_data_2023-10-19-20-39-50': None,
                                          'one_cal_data_2023-10-19-20-39-57': None,
                                          'one_cal_data_2023-10-19-20-40-02': None,
                                          'one_cal_data_2023-10-19-20-40-09': None,
                                          'one_cal_data_2023-10-19-20-40-13': None,
                                          'one_cal_data_2023-10-19-20-40-17': None,
                                          'one_cal_data_2023-10-19-20-40-22': None,
                                          'one_cal_data_2023-10-19-20-40-28': None,
                                          'one_cal_data_2023-10-19-20-40-56': array([-2.29827357, -3.78845136,  0.43809813]),
                                          'one_cal_data_2023-10-19-20-41-00': array([-2.35151366, -3.73806784,  0.34331007]),
                                          'one_cal_data_2023-10-19-20-41-03': array([-2.21579325, -3.60456775,  0.32505068]),
                                          'one_cal_data_2023-10-19-20-41-07': None,
                                          'one_cal_data_2023-10-19-20-41-11': None,
                                          'one_cal_data_2023-10-19-20-41-18': None,
                                          'one_cal_data_2023-10-19-20-41-23': None,
                                          'one_cal_data_2023-10-19-20-41-27': None,
                                          'one_cal_data_2023-10-19-20-41-36': None,
                                          'one_cal_data_2023-10-19-20-41-40': None,
                                          'one_cal_data_2023-10-19-20-41-45': None,
                                          'one_cal_data_2023-10-19-20-41-50': None,
                                          'one_cal_data_2023-10-19-20-41-54': None,
                                          'one_cal_data_2023-10-19-20-41-59': None,
                                          'one_cal_data_2023-10-19-20-42-03': None,
                                          'one_cal_data_2023-10-19-20-42-07': None,
                                          'one_cal_data_2023-10-19-20-42-11': None},
 '/lidar/velodyne_front_tilted_points': {'one_cal_data_2023-10-19-20-36-36': array([ 3.38952164,  0.14083188, -0.47273965]),
                                           'one_cal_data_2023-10-19-20-36-47': array([ 2.77606531, -0.07918332,  0.11901872]),
                                           'one_cal_data_2023-10-19-20-36-57': array([ 3.00685403, -1.29609006, -0.62675718]),
                                           'one_cal_data_2023-10-19-20-37-04': array([ 3.17103141, -1.38000088, -0.66931502]),
                                           'one_cal_data_2023-10-19-20-37-12': array([ 3.09043594, -1.2802634 , -0.63795686]),
                                           'one_cal_data_2023-10-19-20-37-27': array([ 1.07967753, -3.51195878, -0.47078395]),
                                           'one_cal_data_2023-10-19-20-37-32': array([ 1.0687928 , -3.5820186 , -0.46736345]),
                                           'one_cal_data_2023-10-19-20-37-40': None,
                                           'one_cal_data_2023-10-19-20-37-44': None,
                                           'one_cal_data_2023-10-19-20-37-48': None,
                                           'one_cal_data_2023-10-19-20-37-53': None,
                                           'one_cal_data_2023-10-19-20-38-02': None,
                                           'one_cal_data_2023-10-19-20-38-06': None,
                                           'one_cal_data_2023-10-19-20-38-10': None,
                                           'one_cal_data_2023-10-19-20-38-14': None,
                                           'one_cal_data_2023-10-19-20-38-18': None,
                                           'one_cal_data_2023-10-19-20-38-23': None,
                                           'one_cal_data_2023-10-19-20-39-03': None,
                                           'one_cal_data_2023-10-19-20-39-15': None,
                                           'one_cal_data_2023-10-19-20-39-32': None,
                                           'one_cal_data_2023-10-19-20-39-39': None,
                                           'one_cal_data_2023-10-19-20-39-50': None,
                                           'one_cal_data_2023-10-19-20-39-57': None,
                                           'one_cal_data_2023-10-19-20-40-02': None,
                                           'one_cal_data_2023-10-19-20-40-09': None,
                                           'one_cal_data_2023-10-19-20-40-13': None,
                                           'one_cal_data_2023-10-19-20-40-17': None,
                                           'one_cal_data_2023-10-19-20-40-22': None,
                                           'one_cal_data_2023-10-19-20-40-28': None,
                                           'one_cal_data_2023-10-19-20-40-56': None,
                                           'one_cal_data_2023-10-19-20-41-00': None,
                                           'one_cal_data_2023-10-19-20-41-03': None,
                                           'one_cal_data_2023-10-19-20-41-07': None,
                                           'one_cal_data_2023-10-19-20-41-11': None,
                                           'one_cal_data_2023-10-19-20-41-18': None,
                                           'one_cal_data_2023-10-19-20-41-23': None,
                                           'one_cal_data_2023-10-19-20-41-27': None,
                                           'one_cal_data_2023-10-19-20-41-36': None,
                                           'one_cal_data_2023-10-19-20-41-40': None,
                                           'one_cal_data_2023-10-19-20-41-45': None,
                                           'one_cal_data_2023-10-19-20-41-50': None,
                                           'one_cal_data_2023-10-19-20-41-54': None,
                                           'one_cal_data_2023-10-19-20-41-59': None,
                                           'one_cal_data_2023-10-19-20-42-03': None,
                                           'one_cal_data_2023-10-19-20-42-07': None,
                                           'one_cal_data_2023-10-19-20-42-11': None}}


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

Nfail = 0
for lidar_topic in lidar_topics:
    for bag in bags:

        bagname = os.path.split(os.path.splitext(os.path.basename(bag))[0])[1]
        what = f"{bagname}-{os.path.split(lidar_topic)[1]}"

        if not args.check:
            plidar = \
                calibration_data_import.get_lidar_observation( \
                                        bag,
                                        lidar_topic,
                                        what                         = what,
                                        viz                          = viz,
                                        viz_show_only_accepted       = False,
                                        viz_show_point_cloud_context = args.viz_show_point_cloud_context)

            if args.generate_ground_truth:
                if lidar_topic not in expected: expected[lidar_topic] = dict()

                if plidar is None:
                    expected[lidar_topic][bagname] = None
                else:
                    expected[lidar_topic][bagname] = np.mean(plidar, axis=-2)

        else:
            # unit test

            try:
                pmean_expected = expected[lidar_topic][bagname]
            except KeyError:
                fail("no ground-truth available")
                continue

            try:
                plidar = \
                    calibration_data_import.get_lidar_observation( \
                                            bag,
                                            lidar_topic,
                                            what                         = what,
                                            viz                          = viz,
                                            viz_show_only_accepted       = False,
                                            viz_show_point_cloud_context = args.viz_show_point_cloud_context)
            except Exception as e:
                fail(f"Exception={e}")
                continue

            if plidar is None:
                if pmean_expected is None:
                    ok("true no-board result")
                else:
                    fail("did not detect existing board")

                continue

            if pmean_expected is None:
                fail("false detection of board")

            else:

                pmean = np.mean(plidar, axis=-2)
                d = nps.mag(pmean - pmean_expected)
                if d < 0.2:
                    ok("detected board in the correct location")
                else:
                    fail(f"detected board {d:.2f}m away from the expected location")


if args.check:
    if Nfail == 0:
        print_green("ALL TESTS PASSED!")
        sys.exit(0)
    else:
        print_green(f"ERROR! {Nfail} TESTS FAILED!")
        sys.exit(1)

if args.generate_ground_truth:
    pprint.pprint(expected)

sys.exit(0)
