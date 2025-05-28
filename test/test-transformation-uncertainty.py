#!/usr/bin/python3
r'''Validate the uncertainty computations

Compare the theoretical covariance reported by the solved with an
empirically-sampled one

'''


import sys
import argparse
import re
import os

def parse_args():

    parser = \
        argparse.ArgumentParser(description = __doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--context',
                        required = True,
                        help = '''.pickle file from fit.py --dump''')
    parser.add_argument('--topics',
                        type=str,
                        nargs='+',
                        help = '''Used for sampled validation in the given
                        --isector. Either TOPIC1 or TOPIC0,TOPIC1. We look at
                        Var( transform(rt_1r, transform(rt_r0, p0)) ). If only
                        TOPIC1 is given, we assume that TOPIC0 is at the
                        reference of coordinates, and thus look at Var(
                        transform(rt_1r, p0) ). May be given multiple
                        space-separated arguments to validate multiple sets of
                        topics''')
    parser.add_argument('--isector',
                        type=int,
                        nargs='+',
                        help='''Used for sampled validation of the given
                        --topics. we evaluate the uncertainty at the center of
                        this sector. My be given multiple space-separated
                        arguments to validate multiple sets of sectors''')
    parser.add_argument('--Nsamples',
                        type=int,
                        default=100,
                        help='''How many random samples to evaluate''')
    parser.add_argument('--verbose',
                        action = 'store_true',
                        help='''Report details about the solve''')
    args = parser.parse_args()

    if args.topics is not None and args.isector is     None or \
       args.topics is     None and args.isector is not None:
        print("Both or neither of --topics and --isector must be given",
              file=sys.stderr)
        sys.exit(1)

    if args.topics is not None:
        args.topics = [T.split(',') for T in args.topics]
        for T in args.topics:
            if len(T) < 1 or len(T) > 2:
                print(f"Every element of --topics must be either TOPIC1 or TOPIC0,TOPIC1. Instead, got {len(T)} topics in '{T}'",
                      file=sys.stderr)
                sys.exit(1)

    return args


args = parse_args()

testdir = os.path.dirname(os.path.realpath(__file__))
# I import the LOCAL clc since that's what I'm testing
sys.path[:0] = f"{testdir}/..",

import pickle
import numpy as np
import numpysane as nps
import mrcal
import clc

import testutils


with open(args.context, "rb") as f:
    context = pickle.load(f)

kwargs_calibrate = context['kwargs_calibrate']
result           = context['result']

rt_lidar0_lidar  = result['rt_lidar0_lidar']
rt_lidar0_camera = result['rt_lidar0_camera']

Nlidars  = len(rt_lidar0_lidar)
Ncameras = len(rt_lidar0_camera)




def get__rt_ref_sensor(i, rt_ref_lidar, rt_ref_camera):
    if i < Nlidars: return rt_ref_lidar [i]
    else:           return rt_ref_camera[i-Nlidars]


def get_plidar0(isector):
    th = (isector + 0.5) * 2.*np.pi/kwargs_calibrate['Nsectors']
    pvehicle = np.array((np.cos(th), np.sin(th), 0 )) * kwargs_calibrate['uncertainty_quantification_range']

    if kwargs_calibrate['rt_vehicle_lidar0'] is None:
        return pvehicle

    return \
        mrcal.transform_point_rt( kwargs_calibrate['rt_vehicle_lidar0'], pvehicle,
                                  inverted = True)


def get_Var_predicted(isensors, isector):

    plidar0 = get_plidar0(isector)
    rt_lidar0_sensor = [get__rt_ref_sensor(i,
                                           rt_lidar0_lidar,
                                           rt_lidar0_camera) for i in isensors]
    p0 = mrcal.transform_point_rt(rt_lidar0_sensor[0], plidar0, inverted=True)


    Var_rt_lidar0_sensor = result['Var_rt_lidar0_sensor']


    # shape (3,6)
    _,dplidar0__drt_lidar0_sensor0,_ = \
        mrcal.transform_point_rt(rt_lidar0_sensor[0], p0,
                                 get_gradients = True)
    # shape (3,6), (3,3)
    _,dp1__drt_lidar0_sensor1,dp1__dplidar0 = \
        mrcal.transform_point_rt(rt_lidar0_sensor[1], plidar0,
                                 inverted      = True,
                                 get_gradients = True)

    # shape (3,6)
    dp1__drt_lidar0_sensor0 = nps.matmult(dp1__dplidar0, dplidar0__drt_lidar0_sensor0)


    if isensors[0] == 0:
        # only sensor1 is probabilistic
        # shape (6,6)
        Var_rt_lidar0_sensor1 = Var_rt_lidar0_sensor[isensors[1]-1,:,
                                                     isensors[1]-1,:]

        return \
            nps.matmult(dp1__drt_lidar0_sensor1,
                        Var_rt_lidar0_sensor1,
                        nps.transpose(dp1__drt_lidar0_sensor1))
    if isensors[1] == 0:
        # only sensor0 is probabilistic
        # shape (6,6)
        Var_rt_lidar0_sensor0 = Var_rt_lidar0_sensor[isensors[0]-1,:,
                                                     isensors[0]-1,:]

        return \
            nps.matmult(dp1__drt_lidar0_sensor0,
                        Var_rt_lidar0_sensor0,
                        nps.transpose(dp1__drt_lidar0_sensor0))

    # both sensors are probabilistic
    # Each has shape (6,6)
    A = Var_rt_lidar0_sensor[isensors[0]-1,:,
                             isensors[0]-1,:]
    B = Var_rt_lidar0_sensor[isensors[1]-1,:,
                             isensors[0]-1,:]
    C = Var_rt_lidar0_sensor[isensors[1]-1,:,
                             isensors[1]-1,:]

    # shape (12,12)
    Var_rt_lidar0_sensor01 = nps.glue( nps.glue(A, nps.transpose(B), axis=-1),
                                    nps.glue(B, C,                axis=-1),
                                    axis = -2 )
    # shape (3,12)
    dp1__drt_lidar0_sensor01 = nps.glue(dp1__drt_lidar0_sensor0, dp1__drt_lidar0_sensor1,
                                     axis=-1)

    return \
        nps.matmult(dp1__drt_lidar0_sensor01,
                    Var_rt_lidar0_sensor01,
                    nps.transpose(dp1__drt_lidar0_sensor01))


def get_sampled_results():
    print('Sampling...')

    samples = [None] * args.Nsamples

    for isample in range(args.Nsamples):
        if (isample+1) % 20 == 0:
            print(f"Sampling {isample+1}/{args.Nsamples}")

        result_sample = \
            clc.fit_from_inputs_dump(result['inputs_dump'],
                                     do_inject_noise = True,
                                     verbose         = args.verbose)

        samples[isample] = (result_sample['rt_lidar0_lidar'],
                            result_sample['rt_lidar0_camera'])

    return samples


def get_Var_observed(isensors, isector, samples):

    plidar0 = get_plidar0(isector)
    rt_lidar0_sensor = [get__rt_ref_sensor(i,
                                           rt_lidar0_lidar,
                                           rt_lidar0_camera) for i in isensors]
    p0 = mrcal.transform_point_rt(rt_lidar0_sensor[0], plidar0, inverted=True)




    p1_sampled = np.zeros((args.Nsamples,3), dtype=float)
    for isample in range(args.Nsamples):
        (rt_lidar0_lidar__sampled, rt_lidar0_camera__sampled) = samples[isample]

        rt_lidar0_sensor__sampled = \
            [get__rt_ref_sensor(i, rt_lidar0_lidar__sampled, rt_lidar0_camera__sampled) \
             for i in isensors]

        p1_sampled[isample] = \
            mrcal.transform_point_rt(rt_lidar0_sensor__sampled[1],
                                     mrcal.transform_point_rt(rt_lidar0_sensor__sampled[0], p0),
                                     inverted      = True)




    p1_sampled_mean = np.mean(p1_sampled, axis=-2)

    return \
        nps.matmult((p1_sampled - p1_sampled_mean).T,
                    (p1_sampled - p1_sampled_mean)) / args.Nsamples


def topic_index(l,t):
    try:
        return l.index(t)
    except ValueError:
        print(f"Requested topic '{t}' not present in the context file '{args.context}; topics: {l}'",
              file=sys.stderr)
        sys.exit(1)
        return None




for isector in range(kwargs_calibrate['Nsectors']):

    isensors = result['isensors_pair_stdev_worst'][isector]
    if not np.any(isensors):
        # both isensors are 0: this stdev_worst isn't valid because this sector
        # wasn't observed by any pair of sensor
        continue
    stdev_worst_observed = \
        np.sqrt( np.max( np.linalg.eig(get_Var_predicted(isensors, isector))[0] ))

    stdev_worst_predicted = result['stdev_worst_per_sector'][isector]

    testutils.confirm_equal(stdev_worst_predicted,
                            stdev_worst_observed ,
                            eps  = 1e-6,
                            msg = f"stdev_worst in sector {isector}")



samples = get_sampled_results()
if args.topics is not None:

    for topics in args.topics:
        isensor = [topic_index(context['kwargs_calibrate']['topics'],t) for t in topics]

        if len(isensor) == 1:
            if isensor[0] == 0:
                print("Validating one sensor, lidar0, which is at the reference coordinate system. Nothing to do",
                      file = sys.stderr)
                sys.exit(1)

            isensors = (isensor[0],0)
        else:
            if isensor[0] == 0 and isensor[1] == 0:
                print("Validating two sensors; both arelidar0, which is at the reference coordinate system. Nothing to do",
                      file = sys.stderr)
                sys.exit(1)
            isensors = isensor

        for isector in args.isector:
            Var_predicted = get_Var_predicted(isensors, isector)
            Var_observed  = get_Var_observed (isensors, isector, samples)

            testutils.confirm_covariances_equal(Var_predicted,
                                                Var_observed ,
                                                what = f"transformation between {topics} in the requested sector {isector}",
                                                eps_eigenvalues       = 0.2, # relative
                                                eps_eigenvectors_deg  = 5.,
                                                check_sqrt_eigenvalue = True)


testutils.finish()
