#!/usr/bin/python3

r'''LIDAR point-cloud-segmentation test suite

SYNOPSIS

  $ ./lidar-segmentation-auto-test.py testdata/     \
    ....
    All tests passed

'''


import sys
import os
import argparse

def parse_args():
    parser = \
        argparse.ArgumentParser(description = __doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('root',
                        help = '''The root path to the test data''')
    args = parser.parse_args()
    return args
args = parse_args()


import numpy as np
import numpysane as nps
import clc
import testutils

try:
    from tests_private import tests as tests_private
except:
    tests_private = ()

ctx = clc.lidar_segmentation_default_context()
max_range = ctx['threshold_max_range']

tests = (

    # This contains a board, but there's a ground scan beneath that lines up
    # well with it. This makes it look too bad, and I'm fine with nothing being
    # found here
    dict(bag     = '2023-10-19/one_cal_data_2023-10-19-20-36-36.bag',
         topic   = '/lidar/velodyne_front_tilted_points',
         plane_p = None,
         plane_n = None),
    dict(bag     = '2023-10-19/one_cal_data_2023-10-19-20-36-36.bag',
         topic   = '/lidar/velodyne_front_horiz_points',
         plane_p = np.array((3.368,-0.050,0.645)),
         plane_n = np.array((0.7032,-0.6637,0.2550)),
         ),
    dict(bag     = '2023-10-19/one_cal_data_2023-10-19-20-36-36.bag',
         topic   = '/lidar/velodyne_back_points',
         plane_p = None,
         plane_n = None),

    dict(bag     = '2023-10-19/one_cal_data_2023-10-19-20-36-47.bag',
         topic   = '/lidar/velodyne_front_tilted_points',
         plane_p = np.array((2.875,-0.078,0.033)),
         plane_n = np.array((0.6558,-0.0045,0.7550)),
         ),
    dict(bag     = '2023-10-19/one_cal_data_2023-10-19-20-36-47.bag',
         topic   = '/lidar/velodyne_front_horiz_points',
         plane_p = np.array((2.797,0.154,-0.028)),
         plane_n = np.array((-0.7487,-0.0133,0.6628)),
         ),
    dict(bag     = '2023-10-19/one_cal_data_2023-10-19-20-36-47.bag',
         topic   = '/lidar/velodyne_back_points',
         plane_p = None,
         plane_n = None),

    dict(bag     = '2023-10-19/one_cal_data_2023-10-19-20-36-57.bag',
         topic   = '/lidar/velodyne_front_tilted_points',
         plane_p = np.array((3.020,-1.281,-0.590)),
         plane_n = np.array((-0.8521,0.5171,0.0807)),
         ),
    dict(bag     = '2023-10-19/one_cal_data_2023-10-19-20-36-57.bag',
         topic   = '/lidar/velodyne_front_horiz_points',
         plane_p = None,
         plane_n = None,
         ),
    dict(bag     = '2023-10-19/one_cal_data_2023-10-19-20-36-57.bag',
         topic   = '/lidar/velodyne_back_points',
         plane_p = None,
         plane_n = None),

    dict(bag     = '2023-10-19/one_cal_data_2023-10-19-20-37-04.bag',
         topic   = '/lidar/velodyne_front_tilted_points',
         plane_p = np.array((3.171,-1.382,-0.668)),
         plane_n = np.array((-0.8216,0.4531,0.3461)),
         ),
    dict(bag     = '2023-10-19/one_cal_data_2023-10-19-20-37-04.bag',
         topic   = '/lidar/velodyne_front_horiz_points',
         plane_p = None,
         plane_n = None,
         ),
    dict(bag     = '2023-10-19/one_cal_data_2023-10-19-20-37-04.bag',
         topic   = '/lidar/velodyne_back_points',
         plane_p = None,
         plane_n = None),

    dict(bag     = '2023-10-19/one_cal_data_2023-10-19-20-37-12.bag',
         topic   = '/lidar/velodyne_front_tilted_points',
         plane_p = np.array((3.100,-1.294,-0.593)),
         plane_n = np.array((-0.9635,-0.2279,0.1399)),
         ),
    dict(bag     = '2023-10-19/one_cal_data_2023-10-19-20-37-12.bag',
         topic   = '/lidar/velodyne_front_horiz_points',
         plane_p = None,
         plane_n = None),
    dict(bag     = '2023-10-19/one_cal_data_2023-10-19-20-37-12.bag',
         topic   = '/lidar/velodyne_back_points',
         plane_p = None,
         plane_n = None),


    # 2023-10-19/one_cal_data_2023-10-19-20-37-27.bag
    # 2023-10-19/one_cal_data_2023-10-19-20-37-32.bag
    # 2023-10-19/one_cal_data_2023-10-19-20-37-40.bag
    # 2023-10-19/one_cal_data_2023-10-19-20-37-44.bag
    # 2023-10-19/one_cal_data_2023-10-19-20-37-48.bag
    # 2023-10-19/one_cal_data_2023-10-19-20-37-53.bag
    # 2023-10-19/one_cal_data_2023-10-19-20-38-02.bag


    dict(bag     = '2023-10-19/one_cal_data_2023-10-19-20-38-06.bag',
         topic   = '/lidar/velodyne_front_tilted_points',
         plane_p = None,
         plane_n = None,
         ),
    dict(bag     = '2023-10-19/one_cal_data_2023-10-19-20-38-06.bag',
         topic   = '/lidar/velodyne_front_horiz_points',
         plane_p = np.array((-2.716,4.241,0.560)),
         plane_n = np.array((-0.7384,0.6608,0.1346)),
         ),
    dict(bag     = '2023-10-19/one_cal_data_2023-10-19-20-38-06.bag',
         topic   = '/lidar/velodyne_back_points',
         plane_p = np.array((0.566,-4.260,0.328)),
         plane_n = np.array((-0.7427,0.6681,0.0437)),
         ),
    # 2023-10-19/one_cal_data_2023-10-19-20-38-10.bag
    # 2023-10-19/one_cal_data_2023-10-19-20-38-14.bag
    # 2023-10-19/one_cal_data_2023-10-19-20-38-18.bag
    # 2023-10-19/one_cal_data_2023-10-19-20-38-23.bag
    # 2023-10-19/one_cal_data_2023-10-19-20-39-03.bag

    dict(bag     = '2023-10-19/one_cal_data_2023-10-19-20-39-15.bag',
         topic   = '/lidar/velodyne_front_tilted_points',
         plane_p = None,
         plane_n = None,
         ),
    dict(bag     = '2023-10-19/one_cal_data_2023-10-19-20-39-15.bag',
         topic   = '/lidar/velodyne_front_horiz_points',
         plane_p = None,
         plane_n = None,
         ),
    dict(bag     = '2023-10-19/one_cal_data_2023-10-19-20-39-15.bag',
         topic   = '/lidar/velodyne_back_points',
         plane_p = np.array((2.789,-0.641,0.068)),
         plane_n = np.array((-0.9770,0.2065,0.0534)),
         ),
    # 2023-10-19/one_cal_data_2023-10-19-20-39-32.bag
    # 2023-10-19/one_cal_data_2023-10-19-20-39-39.bag
    # 2023-10-19/one_cal_data_2023-10-19-20-39-50.bag
    # 2023-10-19/one_cal_data_2023-10-19-20-39-57.bag
    # 2023-10-19/one_cal_data_2023-10-19-20-40-02.bag
    # 2023-10-19/one_cal_data_2023-10-19-20-40-09.bag
    # 2023-10-19/one_cal_data_2023-10-19-20-40-13.bag

    dict(bag     = '2023-10-19/one_cal_data_2023-10-19-20-40-17.bag',
         topic   = '/lidar/velodyne_front_tilted_points',
         plane_p = None,
         plane_n = None,
         ),
    dict(bag     = '2023-10-19/one_cal_data_2023-10-19-20-40-17.bag',
         topic   = '/lidar/velodyne_front_horiz_points',
         plane_p = None,
         plane_n = None,
         ),
    dict(bag     = '2023-10-19/one_cal_data_2023-10-19-20-40-17.bag',
         topic   = '/lidar/velodyne_back_points',
         plane_p = np.array((2.229,1.001,-0.094)),
         plane_n = np.array((0.3710,0.6990,0.6114)),
         ),
    # 2023-10-19/one_cal_data_2023-10-19-20-40-22.bag
    # 2023-10-19/one_cal_data_2023-10-19-20-40-28.bag
    # 2023-10-19/one_cal_data_2023-10-19-20-40-56.bag
    # 2023-10-19/one_cal_data_2023-10-19-20-41-00.bag
    # 2023-10-19/one_cal_data_2023-10-19-20-41-03.bag
    # 2023-10-19/one_cal_data_2023-10-19-20-41-07.bag
    # 2023-10-19/one_cal_data_2023-10-19-20-41-11.bag

    dict(bag     = '2023-10-19/one_cal_data_2023-10-19-20-41-18.bag',
         topic   = '/lidar/velodyne_front_tilted_points',
         plane_p = None,
         plane_n = None,
         ),
    dict(bag     = '2023-10-19/one_cal_data_2023-10-19-20-41-18.bag',
         topic   = '/lidar/velodyne_front_horiz_points',
         plane_p = None,
         plane_n = None,
         ),
    dict(bag     = '2023-10-19/one_cal_data_2023-10-19-20-41-18.bag',
         topic   = '/lidar/velodyne_back_points',
         plane_p = None,
         plane_n = None),
    # 2023-10-19/one_cal_data_2023-10-19-20-41-23.bag
    # 2023-10-19/one_cal_data_2023-10-19-20-41-27.bag
    # 2023-10-19/one_cal_data_2023-10-19-20-41-36.bag
    # 2023-10-19/one_cal_data_2023-10-19-20-41-40.bag
    # 2023-10-19/one_cal_data_2023-10-19-20-41-45.bag

    dict(bag     = '2023-10-19/one_cal_data_2023-10-19-20-41-50.bag',
         topic   = '/lidar/velodyne_front_tilted_points',
         plane_p = None,
         plane_n = None,
         ),
    dict(bag     = '2023-10-19/one_cal_data_2023-10-19-20-41-50.bag',
         topic   = '/lidar/velodyne_front_horiz_points',
         plane_p = None,
         plane_n = None,
         ),
    # This one does have a plane, but it's only 3 rings, and easy to miss
    dict(bag     = '2023-10-19/one_cal_data_2023-10-19-20-41-50.bag',
         topic   = '/lidar/velodyne_back_points',
         plane_p = None,
         plane_n = None),
    # 2023-10-19/one_cal_data_2023-10-19-20-41-54.bag
    # 2023-10-19/one_cal_data_2023-10-19-20-41-59.bag
    # 2023-10-19/one_cal_data_2023-10-19-20-42-03.bag
    # 2023-10-19/one_cal_data_2023-10-19-20-42-07.bag

    dict(bag     = '2023-10-19/one_cal_data_2023-10-19-20-42-11.bag',
         topic   = '/lidar/velodyne_front_tilted_points',
         plane_p = None,
         plane_n = None,
         ),
    dict(bag     = '2023-10-19/one_cal_data_2023-10-19-20-42-11.bag',
         topic   = '/lidar/velodyne_front_horiz_points',
         plane_p = np.array((-2.483,-4.519,0.108)),
         plane_n = np.array((0.0300,-0.8140,0.5802)),
         ),
    dict(bag     = '2023-10-19/one_cal_data_2023-10-19-20-42-11.bag',
         topic   = '/lidar/velodyne_back_points',
         plane_p = np.array((0.256,4.393,-0.064)),
         plane_n = np.array((0.1178,0.8078,0.5775)),
         ),

    )


for test in tests + tests_private:

    bag = f"{args.root}/{test['bag']}"
    topic = test['topic']

    vizcmd = fr'''
  x0y0x1y1=(-{max_range} -{max_range} {max_range} {max_range});
  ./lidar-segmentation-test.py --dump \
    {topic} \
    {bag} \
  | awk " $x0y0x1y1[1] < \$1 && \$1 < $x0y0x1y1[3] && $x0y0x1y1[2] < \$2 && \$2 < $x0y0x1y1[4]" \
  | feedgnuplot \
      --style all "with points pt 7 ps 0.5" \
      --style stage1-segment "with vectors" \
      --tuplesize stage1-segment 6 \
      --style label "with labels" \
      --tuplesize label 4 \
      --3d \
      --domain \
      --dataid \
      --square \
      --points \
      --tuplesizeall 3 \
      --autolegend \
      --xlabel x \
      --ylabel y \
      --zlabel z'''

    print(f"Evaluating test. Visualize like this:  {vizcmd}")

    segmentation = \
        clc.lidar_segmentation(bag         = bag,
                               lidar_topic = topic)

    Nplanes_found = len(segmentation['plane_p'])

    if test['plane_n'] is None:
        testutils.confirm_equal(Nplanes_found, 0,
                                msg=f'I expect to find exactly 0 planes')
        continue



    testutils.confirm_equal(Nplanes_found, 1,
                            msg=f'I expect to find exactly 1 plane')
    if Nplanes_found != 1: continue

    # I have just one plane
    plane_p = segmentation['plane_p'][0,:]
    plane_n = segmentation['plane_n'][0,:]

    # Normalize vectors. This shouldn't be needed
    test['plane_n'] /= nps.mag(test['plane_n'])

    testutils.confirm_equal(nps.mag(plane_n), 1.,
                            msg = 'Reported plane normal is a unit vector')

    cos_dth = nps.inner(plane_n,test['plane_n'])
    testutils.confirm_equal(np.abs(cos_dth), 1,
                            eps = np.cos(0.1 * np.pi/180.),
                            msg=f'Plane orientation')

    dp = plane_p - test['plane_p']
    mag_dp_normal  = np.inner(dp, test['plane_n'])
    dp_inplane = dp - mag_dp_normal*test['plane_n']
    mag_dp_inplane = nps.mag(dp_inplane)

    testutils.confirm_equal(mag_dp_normal, 0,
                            eps = 5e-3,
                            msg=f'Plane normal-to-plane location')
    testutils.confirm_equal(mag_dp_inplane, 0,
                            eps = 200e-3,
                            msg=f'Plane in-the-plane location')

testutils.finish()
