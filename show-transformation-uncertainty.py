#!/usr/bin/python3

r'''Visualize transformation uncertainty between a pair of sensors

SYNOPSIS

  $ ./fit.py ... --dump /tmp/clc-context.pickle

  $ ./show-transformation-uncertainty.py                  \
      --bag camera-lidar.bag                              \
      --topic /lidar/vl_points_0,/lidar/vl_points_1       \
      --context /tmp/clc-context.pickle
    [plot pops up to show the aligned points]

Displays uncertainties of transformations between pairs of sensors

'''


import sys
import argparse
import argparse_helpers
import re
import os

def parse_args():

    parser = \
        argparse.ArgumentParser(description = __doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--gridn',
                        type=int,
                        default = 25,
                        help='''How densely we should sample the space. We use a
                        square grid with gridn cells on each side. By default
                        gridn=25''')
    parser.add_argument('--radius',
                        type=float,
                        default = 20,
                        help='''How far we should sample the space. We use a
                        square grid, 2*radius m per side. By default radius=20''')
    parser.add_argument('--ellipsoids',
                        action='store_true',
                        help = '''By default we plot the transformation
                        uncertainty, which is derived from uncertainty
                        ellipsoids. It is sometimes useful to see the ellipsoids
                        themselves, usually for debugging. Pass --ellipsoids to
                        do that''')
    parser.add_argument('--topic',
                        type=str,
                        required = True,
                        help = '''The topics to visualize. This is a
                        comma-separated list of topics''')
    parser.add_argument('--bag',
                        help = '''The one bag we're visualizing. Required if --ellipsoids''')
    parser.add_argument('--after',
                        type=str,
                        help = '''If given, start reading the bags at this time.
                        Could be an integer (s since epoch or ns since epoch), a
                        float (s since the epoch) or a string, to be parsed with
                        dateutil.parser.parse()''')
    parser.add_argument('--context',
                        required = True,
                        help = '''.pickle file from fit.py''')
    parser.add_argument('--threshold',
                        type=float,
                        default = 20.,
                        help = '''Max distance where we cut the plot''')
    parser.add_argument('--cbmax',
                        type=float,
                        help = '''If given, we use this cbmax in the uncertainty plots''')



    args = parser.parse_args()
    args.topic = args.topic.split(',')

    if args.ellipsoids and args.bag is None:
        print("ERROR: --ellipsoids requires --bag", file=sys.stderr)
        sys.exit(1)

    return args


args = parse_args()

import bag_interface
import pickle
import numpysane as nps
import mrcal
import gnuplotlib as gp
import numpy as np

import clc








def transformation_covariance_decomposed( # shape (...,3)
                                          p0,
                                          rt_lidar0_lidar,
                                          rt_lidar0_camera,
                                          isensor,
                                          Var):
    if isensor <= 0:
        raise Exception("Must have isensor>0 because isensor==0 is the reference frame, and has no covariance")

    Nlidars = len(rt_lidar0_lidar)
    if isensor < Nlidars: rt_lidar0_sensor = rt_lidar0_lidar [isensor]
    else:                 rt_lidar0_sensor = rt_lidar0_camera[isensor-Nlidars]

    # shape (...,3)
    p1 = \
        mrcal.transform_point_rt(rt_lidar0_sensor, p0, inverted=True)

    # shape (...,3,6)
    _,dp0__drt_lidar01,_ = \
        mrcal.transform_point_rt(rt_lidar0_sensor, p1,
                                 get_gradients = True)

    # shape (6,6)
    Var_rt_lidar01 = Var[isensor-1,:,
                         isensor-1,:]

    # shape (...,3,3)
    Var_p0 = nps.matmult(dp0__drt_lidar01,
                         Var_rt_lidar01,
                         nps.transpose(dp0__drt_lidar01))

    # shape (...,3) and (...,3,3)
    l,v = mrcal.sorted_eig(Var_p0)

    return l,v


def get_psphere(scale = 1.):


    az = np.linspace(-np.pi, np.pi,40, endpoint = False)
    el = nps.mv(np.linspace(-np.pi/2., np.pi/2., 20), -1,-2)
    caz = np.cos(az)
    saz = np.sin(az)
    cel = np.cos(el)
    sel = np.sin(el)

    # shape (Npoints,3)
    return \
        scale * \
        nps.clump(nps.mv(nps.cat(caz*cel, saz*cel, sel*np.ones(az.shape)),
                         0,-1),
                  n=2)



with open(args.context, "rb") as f:
    context = pickle.load(f)

rt_vehicle_lidar0 = context['kwargs_calibrate'].get('rt_vehicle_lidar0')
if rt_vehicle_lidar0 is not None:
    Rt_vehicle_lidar0 = mrcal.Rt_from_rt(rt_vehicle_lidar0)
else:
    rt_vehicle_lidar0 = mrcal.identity_rt()
    Rt_vehicle_lidar0 = mrcal.identity_Rt()






Var_rt_lidar0_sensor = context['result']['Var_rt_lidar0_sensor']

isensor_solve_from_isensor_requested = [None] * len(args.topic)
for isensor_requested,topic_requested in enumerate(args.topic):
    try:
        i = context['kwargs_calibrate']['topics'].index(topic_requested)
    except:
        print(f"Requested topic '{topic_requested}' not present in the context file '{args.context}'; topics: {context['kwargs_calibrate']['topics']}",
              file=sys.stderr)
        sys.exit(1)

    isensor_solve_from_isensor_requested[isensor_requested] = i

    if i > 0:
        Var_rt_lidar0_sensor_this = Var_rt_lidar0_sensor[i-1,:,i-1,:]

        l,v = mrcal.sorted_eig(Var_rt_lidar0_sensor_this)

        s = v[:,-1] * np.sqrt(l[-1])
        print(f"Topic {topic_requested} rt_lidar0_sensor worst-direction 1-sigma stdev: {s[:3]}rad, {s[3:]}m")


x_sample = np.linspace(-args.radius,args.radius,args.gridn)
y_sample = np.linspace(-args.radius,args.radius,args.gridn)
z_sample = 0

# shape (Nysample, Nxsample,3)
p0_vehicle = \
    nps.mv(nps.cat( *np.meshgrid(x_sample,y_sample),
                    z_sample * np.ones((args.gridn,args.gridn), dtype=float) ),
           0,-1)

p0_lidar0 = mrcal.transform_point_Rt(Rt_vehicle_lidar0, p0_vehicle,
                                     inverted = True)



if args.ellipsoids:
    psphere = get_psphere(scale = 10.) # 10x ellipses to improve legibility

    plot_tuples = \
        clc.pointcloud_plot_tuples(args.bag, args.topic,
                                   context['result']['rt_lidar0_lidar'],
                                   threshold_range   = args.threshold,
                                   Rt_vehicle_lidar0 = Rt_vehicle_lidar0,
                                   start = args.after)

    for isensor_requested,topic_requested in enumerate(args.topic):
        isensor_solve = isensor_solve_from_isensor_requested[isensor_requested]
        if isensor_solve == 0: continue # reference coord system

        l,v_lidar0 = \
            clc.transformation_covariance_decomposed(p0_lidar0,
                                                     context['result']['rt_lidar0_lidar'],
                                                     context['result']['rt_lidar0_camera'],
                                                     isensor_solve,
                                                     Var_rt_lidar0_sensor)
        stdev = np.sqrt(l)

        # v_lidar0 stored each eigenvector in COLUMNS. I transpose to store them in
        # rows instead, to follow numpy's broadcasting rules
        v_lidar0 = nps.transpose(v_lidar0)
        v_vehicle = mrcal.rotate_point_R(Rt_vehicle_lidar0[:3,:],
                                         v_lidar0)

        # shape (Nspherepoints,Nysample,Nxsample,3)
        pellipsoid_vehicle = \
            p0_vehicle + \
            nps.matmult(# shape (Nspherepoints,1,1,1,3) * (Nysample,Nxsample,1,3)
                        nps.dummy(psphere, -2,-2,-2) * nps.dummy(stdev, axis=-2),
                        v_vehicle)[...,0,:]

        plot_tuples.append( ( nps.clump(pellipsoid_vehicle, n=3),
                              dict( tuplesize = -3,
                                    _with     = f'dots lc rgb "{clc.color_sequence_rgb()[isensor_solve%len(clc.color_sequence_rgb())]}"',
                                    legend = f'1-sigma uncertainty for {topic_requested}')), )

    clc.plot(*plot_tuples,
         _3d = True,
         square = True,
         xlabel = 'x (vehicle)',
         ylabel = 'y (vehicle)',
         zlabel = 'z (vehicle)',
         wait = True)

    sys.exit()




for isensor_requested,topic_requested in enumerate(args.topic):
    isensor_solve = isensor_solve_from_isensor_requested[isensor_requested]
    if isensor_solve == 0: continue # reference coord system

    # These apply to ALL the sensors, not just the ones being requested
    sensor_forward_vectors_plot_tuples = \
        clc.sensor_forward_vectors_plot_tuples(mrcal.compose_rt(rt_vehicle_lidar0, context['result']['rt_lidar0_lidar']),
                                               mrcal.compose_rt(rt_vehicle_lidar0, context['result']['rt_lidar0_camera']),
                                               context['kwargs_calibrate']['topics'],
                                               isensor = isensor_solve)


    l,v_lidar0 = \
        clc.transformation_covariance_decomposed(p0_lidar0,
                                                 context['result']['rt_lidar0_lidar'],
                                                 context['result']['rt_lidar0_camera'],
                                                 isensor_solve,
                                                 Var_rt_lidar0_sensor)
    stdev = np.sqrt(l)

    # v_lidar0 stored each eigenvector in COLUMNS. I transpose to store them in
    # rows instead, to follow numpy's broadcasting rules
    v_lidar0 = nps.transpose(v_lidar0)
    v_vehicle = mrcal.rotate_point_R(Rt_vehicle_lidar0[:3,:],
                                     v_lidar0)

    # The eigenvalues/vectors are sorted. The worst-case one is biggest, and
    # stored last
    uncertainty_1sigma = stdev    [...,-1]
    eigv_worst         = v_vehicle[...,-1]

    cos_vertical = np.abs(eigv_worst[...,2])
    thdeg_vertical = np.arccos(np.clip(cos_vertical,-1,1)) * 180./np.pi

    using = f'({x_sample[0]} + $1*({x_sample[-1]-x_sample[0]})/{args.gridn-1}):({y_sample[0]} + $2*({y_sample[-1]-y_sample[0]})/{args.gridn-1}):3'

    clc.plot((uncertainty_1sigma,
          dict(tuplesize = 3,
               _with = 'image',
               using = using),
          ),
         *sensor_forward_vectors_plot_tuples,
         cbmin = 0,
         cbmax = args.cbmax,
         square = True,
         wait = True,
         xlabel = 'x (vehicle)',
         ylabel = 'y (vehicle)',
         title = f'Worst-case 1-sigma transform uncertainty for {topic_requested} (top-down view)',
         ascii = 1, # needed for the "using" scale
         _set  = ('xrange [:] noextend', 'yrange [:] noextend'),
         hardcopy=f'/tmp/uncertainty-1sigma-isensor={isensor_solve}.gp')

    clc.plot((thdeg_vertical,
          dict(tuplesize = 3,
               _with = 'image',
               using = using),
          ),
         *sensor_forward_vectors_plot_tuples,
         cbmin = 0,
         cbmax = 30,
         square = True,
         wait = True,
         xlabel = 'x (vehicle)',
         ylabel = 'y (vehicle)',
         title = f'Worst-case transform uncertainty for {topic_requested} (top-down view): angle off vertical (deg)',
         ascii = 1, # needed for the "using" scale
         _set  = ('xrange [:] noextend', 'yrange [:] noextend'),
         hardcopy=f'/tmp/uncertainty-direction-1sigma-isensor={isensor_solve}.gp')
