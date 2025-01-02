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

    parser.add_argument('--lidar-topic',
                        type=str,
                        required = True,
                        help = '''The one LIDAR topic to validate''')
    parser.add_argument('--context',
                        required = True,
                        help = '''.pickle file from fit.py''')

    args = parser.parse_args()

    return args


args = parse_args()

import pickle
import numpy as np
import numpysane as nps
import mrcal
import clc




with open(args.context, "rb") as f:
    context = pickle.load(f)

rt_lidar0_lidar = context['result']['rt_ref_lidar']

lidar_topic_requested = args.lidar_topic
try:
    ilidar = \
        context['lidar_topic'].index(lidar_topic_requested)
except:
    print(f"Requested topic '{lidar_topic_requested}' not present in the context file '{args.context}; topics: {context['lidar_topic']}'",
          file=sys.stderr)
    sys.exit(1)

if ilidar == 0:
    print("Validating lidar0, which is at the reference coordinate system. Nothing to do",
          file = sys.stderr)
    sys.exit(1)

# arbitrary sample point, in the reference lidar's coord system
# shape (..., 3)
p0 = np.array((8., 2., 1.),)

p1 = mrcal.transform_point_rt(rt_lidar0_lidar[ilidar], p0, inverted=True)

# shape (Nysample,Nxsample,3,6)
_,dp0__drt_lidar_ref,_ = \
    mrcal.transform_point_rt(rt_lidar0_lidar[ilidar], p1,
                             get_gradients = True)

# shape (6,6)
Var_rt_lidar_ref = context['result']['Var'][ilidar-1,:,
                                            ilidar-1,:]

# shape (Nysample,Nxsample,3,3)
Var_predicted = \
    nps.matmult(dp0__drt_lidar_ref,
                Var_rt_lidar_ref,
                nps.transpose(dp0__drt_lidar_ref))



Nsamples = 100
p0_sampled = np.zeros((Nsamples,3), dtype=float)
for i in range(Nsamples):

    result = clc.fit_from_optimization_inputs(context['result']['inputs-dump'],
                                              inject_noise = True)
    p0_sampled[i] = mrcal.transform_point_rt(result['rt_ref_lidar'][ilidar], p1)



p0_sampled_mean = np.mean(p0_sampled, axis=-2)

Var_observed = nps.matmult((p0_sampled - p0_sampled_mean).T,
                           (p0_sampled - p0_sampled_mean)) / Nsamples

l_observed, v_observed  = mrcal.sorted_eig(Var_observed)
l_predicted,v_predicted = mrcal.sorted_eig(Var_predicted)

print(nps.cat(l_observed, l_predicted))
print(np.arccos( nps.inner(v_observed.T, v_predicted.T) ) * 180./np.pi)


import IPython
IPython.embed()
sys.exit()
