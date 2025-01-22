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

    parser.add_argument('--topics',
                        type=str,
                        required = True,
                        help = '''Either TOPIC1 or TOPIC0,TOPIC1. We look at
                        Var( transform(rt_1r, transform(rt_r0, p0)) ). If only
                        TOPIC1 is given, we assume that TOPIC0 is at the
                        reference of coordinates, and thus look at Var(
                        transform(rt_1r, p0) )''')
    parser.add_argument('--context',
                        required = True,
                        help = '''.pickle file from fit.py --dump''')
    parser.add_argument('--Nsamples',
                        type=int,
                        default=100,
                        help='''How many random samples to evaluate''')
    parser.add_argument('--isector',
                        type=int,
                        required = True,
                        help='''Required argument: we evaluate the uncertainty
                        at the center of this sector''')
    args = parser.parse_args()

    args.topics = args.topics.split(',')
    if len(args.topics) < 1 or len(args.topics) > 2:
        print(f"--topics must be either TOPIC1 or TOPIC0,TOPIC1. Instead, got {len(args.topics)} topics",
              file=sys.stderr)
        sys.exit(1)
    return args


args = parse_args()

import pickle
import numpy as np
import numpysane as nps
import mrcal
import clc

import testutils


with open(args.context, "rb") as f:
    context = pickle.load(f)

rt_lidar0_lidar  = context['result']['rt_ref_lidar' ]
rt_lidar0_camera = context['result']['rt_ref_camera']

Nlidars  = len(rt_lidar0_lidar)
Ncameras = len(rt_lidar0_camera)

topics_requested = args.topics

def index(l,t):
    try:
        return l.index(t)
    except ValueError:
        print(f"Requested topic '{t}' not present in the context file '{args.context}; topics: {l}'",
              file=sys.stderr)
        sys.exit(1)
        return None
isensor = [index(context['topics'],t) for t in topics_requested]

if len(isensor) == 1:
    if isensor[0] == 0:
        print("Validating one sensor, lidar0, which is at the reference coordinate system. Nothing to do",
              file = sys.stderr)
        sys.exit(1)

    isensor = (isensor[0],0)
else:
    if isensor[0] == 0 and isensor[1] == 0:
        print("Validating two sensors; both arelidar0, which is at the reference coordinate system. Nothing to do",
              file = sys.stderr)
        sys.exit(1)

# sample point, in the vehicle coord system
th = (args.isector + 0.5) * 2.*np.pi/context['Nsectors']
pquery_vehicle = np.array((np.cos(th), np.sin(th), 0 )) * context['uncertainty_quantification_range']

pref = \
    mrcal.transform_point_rt( context['rt_vehicle_lidar0'], np.array(pquery_vehicle),
                              inverted = True)

def get__rt_ref_sensor(i, rt_lidar0_lidar, rt_lidar0_camera):
    if i < Nlidars: return rt_lidar0_lidar [i]
    else:           return rt_lidar0_camera[i-Nlidars]
rt_ref_sensor = [get__rt_ref_sensor(i, rt_lidar0_lidar, rt_lidar0_camera) for i in isensor]

p0 = mrcal.transform_point_rt(rt_ref_sensor[0], pref, inverted=True)

# shape (3,6)
_,dpref__drt_ref_sensor0,_ = \
    mrcal.transform_point_rt(rt_ref_sensor[0], p0,
                             get_gradients = True)
# shape (3,6), (3,3)
_,dp1__drt_ref_sensor1,dp1__dpref = \
    mrcal.transform_point_rt(rt_ref_sensor[1], pref,
                             inverted      = True,
                             get_gradients = True)

# shape (3,6)
dp1__drt_ref_sensor0 = nps.matmult(dp1__dpref, dpref__drt_ref_sensor0)


if isensor[0] == 0:
    # only sensor1 is probabilistic
    # shape (6,6)
    Var_rt_ref_sensor1 = context['result']['Var'][isensor[1]-1,:,
                                                  isensor[1]-1,:]

    # shape (3,3)
    Var_predicted = \
        nps.matmult(dp1__drt_ref_sensor1,
                    Var_rt_ref_sensor1,
                    nps.transpose(dp1__drt_ref_sensor1))
elif isensor[1] == 0:
    # only sensor0 is probabilistic
    # shape (6,6)
    Var_rt_ref_sensor0 = context['result']['Var'][isensor[0]-1,:,
                                                  isensor[0]-1,:]

    # shape (3,3)
    Var_predicted = \
        nps.matmult(dp1__drt_ref_sensor0,
                    Var_rt_ref_sensor0,
                    nps.transpose(dp1__drt_ref_sensor0))
else:
    # both sensors are probabilistic
    # Each has shape (6,6)
    A = context['result']['Var'][isensor[0]-1,:,
                                 isensor[0]-1,:]
    B = context['result']['Var'][isensor[1]-1,:,
                                 isensor[0]-1,:]
    C = context['result']['Var'][isensor[1]-1,:,
                                 isensor[1]-1,:]

    # shape (12,12)
    Var_rt_ref_sensor01 = nps.glue( nps.glue(A, nps.transpose(B), axis=-1),
                                    nps.glue(B, C,                axis=-1),
                                    axis = -2 )
    # shape (3,12)
    dp1__drt_ref_sensor01 = nps.glue(dp1__drt_ref_sensor0, dp1__drt_ref_sensor1,
                                     axis=-1)

    # shape (3,3)
    Var_predicted = \
        nps.matmult(dp1__drt_ref_sensor01,
                    Var_rt_ref_sensor01,
                    nps.transpose(dp1__drt_ref_sensor01))



# Done with Var_predicted. Compute empirical Var_observed

p1_sampled = np.zeros((args.Nsamples,3), dtype=float)
for isample in range(args.Nsamples):
    if (isample+1) % 20 == 0:
        print(f"Sampling {isample+1}/{args.Nsamples}")

    result = clc.fit_from_inputs_dump(context['result']['inputs_dump'],
                                      do_inject_noise = True)

    rt_ref_sensor__sampled = [get__rt_ref_sensor(i, result['rt_ref_lidar'], result['rt_ref_camera']) \
                              for i in isensor]

    p1_sampled[isample] = \
        mrcal.transform_point_rt(rt_ref_sensor__sampled[1],
                                 mrcal.transform_point_rt(rt_ref_sensor__sampled[0], p0),
                                 inverted      = True)




p1_sampled_mean = np.mean(p1_sampled, axis=-2)

Var_observed = nps.matmult((p1_sampled - p1_sampled_mean).T,
                           (p1_sampled - p1_sampled_mean)) / args.Nsamples

testutils.confirm_covariances_equal(Var_predicted,
                                    Var_observed ,
                                    what = f"transformation between {topics_requested}",
                                    eps_eigenvalues       = 0.2, # relative
                                    eps_eigenvectors_deg  = 5.,
                                    check_sqrt_eigenvalue = True)


testutils.finish()
