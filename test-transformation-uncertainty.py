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

    parser.add_argument('--topic',
                        type=str,
                        required = True,
                        help = '''The one topic to validate''')
    parser.add_argument('--context',
                        required = True,
                        help = '''.pickle file from fit.py --dump''')
    parser.add_argument('--Nsamples',
                        type=int,
                        default=100,
                        help='''How many random samples to evaluate''')
    parser.add_argument('--p0',
                        type  = float,
                        nargs = 3,
                        default=(8., 2., 1.),
                        help='''Operating point, in the reference lidar coord
                        system. By default an arbitrary reasonable point is used''')

    args = parser.parse_args()

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

topic_requested = args.topic
try:
    isensor = \
        context['topics'].index(topic_requested)
except:
    print(f"Requested topic '{topic_requested}' not present in the context file '{args.context}; topics: {context['topics']}'",
          file=sys.stderr)
    sys.exit(1)

if isensor == 0:
    print("Validating lidar0, which is at the reference coordinate system. Nothing to do",
          file = sys.stderr)
    sys.exit(1)

# arbitrary sample point, in the reference lidar's coord system
# shape (..., 3)
p0 = np.array(args.p0)

if isensor < Nlidars: rt_lidar0_sensor = rt_lidar0_lidar [isensor]
else:                 rt_lidar0_sensor = rt_lidar0_camera[isensor-Nlidars]
p1 = mrcal.transform_point_rt(rt_lidar0_sensor, p0, inverted=True)

# shape (Nysample,Nxsample,3,6)
_,dp0__drt_sensor_ref,_ = \
    mrcal.transform_point_rt(rt_lidar0_sensor, p1,
                             get_gradients = True)

# shape (6,6)
Var_rt_sensor_ref = context['result']['Var'][isensor-1,:,
                                             isensor-1,:]

# shape (Nysample,Nxsample,3,3)
Var_predicted = \
    nps.matmult(dp0__drt_sensor_ref,
                Var_rt_sensor_ref,
                nps.transpose(dp0__drt_sensor_ref))



p0_sampled = np.zeros((args.Nsamples,3), dtype=float)
for isample in range(args.Nsamples):
    if (isample+1) % 20 == 0:
        print(f"Sampling {isample+1}/{args.Nsamples}")

    result = clc.fit_from_inputs_dump(context['result']['inputs_dump'],
                                      do_inject_noise = True)

    if isensor < Nlidars: rt_lidar0_sensor__sampled = result['rt_ref_lidar' ][isensor]
    else:                 rt_lidar0_sensor__sampled = result['rt_ref_camera'][isensor-Nlidars]

    p0_sampled[isample] = mrcal.transform_point_rt(rt_lidar0_sensor__sampled, p1)



p0_sampled_mean = np.mean(p0_sampled, axis=-2)

Var_observed = nps.matmult((p0_sampled - p0_sampled_mean).T,
                           (p0_sampled - p0_sampled_mean)) / args.Nsamples

testutils.confirm_covariances_equal(Var_predicted,
                                    Var_observed ,
                                    what = f"transformation to reference lidar {context['topics'][0]} of {topic_requested}",
                                    eps_eigenvalues       = 0.2, # relative
                                    eps_eigenvectors_deg  = 5.,
                                    check_sqrt_eigenvalue = True)


testutils.finish()
