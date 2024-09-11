#!/usr/bin/python3

r'''Find the calibration board in the LIDAR point cloud

SYNOPSIS

  $ ./find_lidar_observation.py     \
      --board-size 1                \
      camera-lidar.bag              \
      /lidar/vl_points_0
    [runs through this one scenario]

  $ ./find_lidar_observation.py     \
      --board-size 1                \
      --check                       \
      'camera-lidar*.bag'           \
      /lidar/vl_points_{0,1,2}
    [unit test to make sure this scenario works properly]

  $ ./find_lidar_observation.py     \
      --board-size 1                \
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

    #### identical logic to fit.py. please consolidate
    parser.add_argument('--board-size',
                        type = str,
                        required = True,
                        help = '''Must be given. This is the "width", but
                        assumes the board is square. Will mostly work for
                        non-square boards also, but the logic could be improved
                        in those cases. In rare cases we want separate board
                        sizes for min and max checks; if we need that, specify
                        --board-size MIN,MAX''')
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

    args.board_size_for_min,args.board_size_for_max = None,None
    if args.board_size is not None:
        try:
            args.board_size_for_min = args.board_size_for_max = float(args.board_size)
        except:
            pass
        if args.board_size_for_min is None:
            minmax = args.board_size.split(',')
            if len(minmax) != 2:
                print("--board-size must be either a number OR a string MIN,MAX: exactly TWO ,-separated numbers",
                      file=sys.stderr)
                sys.exit(1)
            try:
                args.board_size_for_min,args.board_size_for_max = [float(x) for x in minmax]
            except:
                print("--board-size must be either a number OR a string MIN,MAX: exactly two ,-separated NUMBERS",
                      file=sys.stderr)
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
                                           'one_cal_data_2023-10-19-20-42-11': None},
 '/lidar/velodyne_back_points': {'one_cal_data_2023-10-19-20-36-36': None,
                                   'one_cal_data_2023-10-19-20-36-47': None,
                                   'one_cal_data_2023-10-19-20-36-57': None,
                                   'one_cal_data_2023-10-19-20-37-04': None,
                                   'one_cal_data_2023-10-19-20-37-12': None,
                                   'one_cal_data_2023-10-19-20-37-27': None,
                                   'one_cal_data_2023-10-19-20-37-32': None,
                                   'one_cal_data_2023-10-19-20-37-40': None,
                                   'one_cal_data_2023-10-19-20-37-44': None,
                                   'one_cal_data_2023-10-19-20-37-48': None,
                                   'one_cal_data_2023-10-19-20-37-53': None,
                                   'one_cal_data_2023-10-19-20-38-02': array([ 0.48146572, -4.30331232,  0.25041326]),
                                   'one_cal_data_2023-10-19-20-38-06': array([ 0.56969363, -4.25879932,  0.32797712]),
                                   'one_cal_data_2023-10-19-20-38-10': array([ 0.4675303 , -4.35620861,  0.25074202]),
                                   'one_cal_data_2023-10-19-20-38-14': array([ 0.48355672, -4.26374425,  0.09983538]),
                                   'one_cal_data_2023-10-19-20-38-18': array([ 0.3797683 , -4.33577049, -0.12311108]),
                                   'one_cal_data_2023-10-19-20-38-23': None,
                                   'one_cal_data_2023-10-19-20-39-03': array([ 0.50288095, -4.12635376,  0.22477795]),
                                   'one_cal_data_2023-10-19-20-39-15': array([ 2.78141046, -0.64689985, -0.03070331]),
                                   'one_cal_data_2023-10-19-20-39-32': None,
                                   'one_cal_data_2023-10-19-20-39-39': None,
                                   'one_cal_data_2023-10-19-20-39-50': array([ 2.60073218, -0.33900763,  0.42959235]),
                                   'one_cal_data_2023-10-19-20-39-57': array([ 2.25811072, -0.27795991, -0.05176762]),
                                   'one_cal_data_2023-10-19-20-40-02': array([ 2.21846255, -0.17385532, -0.11853931]),
                                   'one_cal_data_2023-10-19-20-40-09': array([ 2.41068809,  1.1071917 , -0.13246111]),
                                   'one_cal_data_2023-10-19-20-40-13': array([ 2.32251938,  1.17080723, -0.1287162 ]),
                                   'one_cal_data_2023-10-19-20-40-17': array([ 2.23736822,  1.00981739, -0.1103475 ]),
                                   'one_cal_data_2023-10-19-20-40-22': array([ 2.31317922,  1.06193087, -0.24728878]),
                                   'one_cal_data_2023-10-19-20-40-28': array([ 2.41921947,  0.32112362, -0.07236682]),
                                   'one_cal_data_2023-10-19-20-40-56': array([0.13629973, 3.7390201 , 0.13720953]),
                                   'one_cal_data_2023-10-19-20-41-00': array([0.16007219, 3.6714972 , 0.09928238]),
                                   'one_cal_data_2023-10-19-20-41-03': array([0.03508815, 3.54555075, 0.12806585]),
                                   'one_cal_data_2023-10-19-20-41-07': array([0.14852151, 3.78132658, 0.42447379]),
                                   'one_cal_data_2023-10-19-20-41-11': array([0.18068037, 3.80079276, 0.28374882]),
                                   'one_cal_data_2023-10-19-20-41-18': None,
                                   'one_cal_data_2023-10-19-20-41-23': None,
                                   'one_cal_data_2023-10-19-20-41-27': None,
                                   'one_cal_data_2023-10-19-20-41-36': None,
                                   'one_cal_data_2023-10-19-20-41-40': None,
                                   'one_cal_data_2023-10-19-20-41-45': None,
                                   'one_cal_data_2023-10-19-20-41-50': None,
                                   'one_cal_data_2023-10-19-20-41-54': None,
                                   'one_cal_data_2023-10-19-20-41-59': array([-0.04031167,  4.73718271,  0.46243637]),
                                   'one_cal_data_2023-10-19-20-42-03': array([0.40927168, 4.39839728, 0.29325747]),
                                   'one_cal_data_2023-10-19-20-42-07': array([0.26970425, 4.36155094, 0.06847827]),
                                   'one_cal_data_2023-10-19-20-42-11': array([ 0.22857706,  4.50893072, -0.22291771])}}


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
    print(f"======== Looking for observations of {lidar_topic}'")

    for bag in bags:
        bagname = os.path.split(os.path.splitext(os.path.basename(bag))[0])[1]
        print(f"===== Looking for observations of {lidar_topic} in '{bagname}'")

        what = f"{bagname}-{calibration_data_import.canonical_lidar_topic_name(lidar_topic)}"

        if not args.check:
            plidar = \
                calibration_data_import.get_lidar_observation( \
                                        bag,
                                        lidar_topic,
                                        what                         = what,
                                        viz                          = viz,
                                        viz_show_only_accepted       = False,
                                        viz_show_point_cloud_context = args.viz_show_point_cloud_context,
                                        board_size_for_min           = args.board_size_for_min,
                                        board_size_for_max           = args.board_size_for_max)

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
                                            viz_show_point_cloud_context = args.viz_show_point_cloud_context,
                                            board_size_for_min           = args.board_size_for_min,
                                            board_size_for_max           = args.board_size_for_max)
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
