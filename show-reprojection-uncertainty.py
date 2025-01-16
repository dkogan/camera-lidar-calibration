#!/usr/bin/python3
r'''Display a set of LIDAR point clouds in the aligned coordinate system

SYNOPSIS

  $ ./show-aligned-lidar-pointclouds.py                   \
      --bag camera-lidar.bag                              \
      --lidar-topic /lidar/vl_points_0,/lidar/vl_points_1 \
      /tmp/lidar[01]-mounted.cameramodel
    [plot pops up to show the aligned points]

Displays aligned point clouds. Useful for debugging

'''


import sys
import argparse
import re
import os

import numpy as np

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
    parser.add_argument('--rt-vehicle-lidar0',
                        type=float,
                        nargs=6,
                        help='''The vehicle-lidar0 transform. The solve is
                        always done in lidar0 coordinates, but we may want to
                        operate in a different "vehicle" frame. This argument
                        specifies the relationship between those frames. If
                        omitted, we assume an identity transform: the vehicle
                        frame is the lidar0 frame''')
    parser.add_argument('--ellipsoids',
                        action='store_true',
                        help = '''By default we plot the reprojection
                        uncertainty, which is derived from uncertainty
                        ellipsoids. It is sometimes useful to see the ellipsoids
                        themselves, usually for debugging. Pass --ellipsoids to
                        do that''')
    parser.add_argument('--lidar-topic',
                        type=str,
                        required = True,
                        help = '''The LIDAR topics to visualize. This is a
                        comma-separated list of topics''')
    parser.add_argument('--bag',
                        required = True,
                        help = '''The one bag we're visualizing''')
    parser.add_argument('--context',
                        required = True,
                        help = '''.pickle file from fit.py''')
    parser.add_argument('--threshold',
                        type=float,
                        default = 20.,
                        help = '''Max distance where we cut the plot''')


    args = parser.parse_args()
    args.lidar_topic = args.lidar_topic.split(',')

    if args.rt_vehicle_lidar0 is not None:
       args.rt_vehicle_lidar0 = np.array(args.rt_vehicle_lidar0, dtype=float)

    return args


args = parse_args()

import bag_interface
import pickle
import numpysane as nps
import mrcal
import gnuplotlib as gp

import clc



with open(args.context, "rb") as f:
    context = pickle.load(f)


ilidar_in_solve_from_ilidar = [None] * len(args.lidar_topic)
for ilidar in range(len(args.lidar_topic)):
    lidar_topic_requested = args.lidar_topic[ilidar]
    try:
        ilidar_in_solve_from_ilidar[ilidar] = \
            context['lidar_topic'].index(lidar_topic_requested)
    except:
        print(f"Requested topic '{lidar_topic_requested}' not present in the context file '{args.context}'; topics: {context['lidar_topic']}",
              file=sys.stderr)
        sys.exit(1)


x_sample = np.linspace(-args.radius,args.radius,args.gridn)
y_sample = np.linspace(-args.radius,args.radius,args.gridn)
z_sample = 0

### point coordinates in the vehicle frame

# Each has shape (Ny_sample,Nx_sample)
px0, py0 = np.meshgrid(x_sample,y_sample)
pz0 = z_sample * np.ones(px0.shape, dtype=float)
# shape (Nysample, Nxsample,3)
p0 = nps.mv(nps.cat(px0,py0,pz0),
            0,-1)

# I want p0 in the lidar0 frame, so I transform
if args.rt_vehicle_lidar0 is not None:
    mrcal.transform_point_rt(args.rt_vehicle_lidar0, p0,
                             inverted = True,
                             out      = p0)

if args.ellipsoids:

    az = np.linspace(-np.pi, np.pi,40, endpoint = False)
    el = nps.mv(np.linspace(-np.pi/2., np.pi/2., 20), -1,-2)
    caz = np.cos(az)
    saz = np.sin(az)
    cel = np.cos(el)
    sel = np.sin(el)

    # shape (Npoints,3)
    psphere = nps.clump(nps.mv(nps.cat(caz*cel, saz*cel, sel*np.ones(az.shape)),
                               0,-1),
                        n=2)

    # 1. is "to scale". Higher numbers improve legibility
    ellipse_scale = 10.

    data_tuples = clc.get_pointcloud_plot_tuples(args.bag, args.lidar_topic, args.threshold,
                                                 ilidar_in_solve_from_ilidar = None)


    for ilidar,topic in enumerate(args.lidar_topic):
        ilidar_solve = ilidar_in_solve_from_ilidar[ilidar]
        if ilidar_solve == 0: continue # reference coord system

        l,v = \
            clc.reprojection_covariance_decomposed(p0,
                                                   context['result']['rt_ref_lidar'],
                                                   ilidar_solve,
                                                   context['result']['Var'])
        stdev = np.sqrt(l)

        # shape (Nspherepoints,Nysample,Nxsample,3)
        pellipsoid = \
            p0 + \
            ellipse_scale * \
            nps.matmult(# shape (Nspherepoints,1,1,1,3) * (Nysample,Nxsample,1,3)
                        nps.dummy(psphere, -2,-2,-2) * nps.dummy(stdev, axis=-2),
                        nps.transpose(v))[...,0,:]

        data_tuples.append( ( nps.clump(pellipsoid, n=3),
                              dict( tuplesize = -3,
                                    _with     = f'dots lc rgb "{clc.color_sequence_rgb[ilidar_solve%len(clc.color_sequence_rgb)]}"',
                                    legend = f'1-sigma uncertainty for {topic}')), )

    clc.plot(*data_tuples,
         _3d = True,
         square = True,
         xlabel = 'x',
         ylabel = 'y',
         zlabel = 'z',
         wait = True)

    sys.exit()




# These apply to ALL the sensors, not just the ones being requested
lidars_origin  = context['result']['rt_ref_lidar'][:,3:]
lidars_forward = mrcal.rotate_point_r(context['result']['rt_ref_lidar'][:,:3], np.array((1.,0,0)))
lidars_forward_xy = np.array(lidars_forward[...,:2])
# to avoid /0 for straight-up vectors
mag_lidars_forward_xy = nps.mag(lidars_forward_xy)
i = mag_lidars_forward_xy>0
lidars_forward_xy[i,:] /= nps.dummy(mag_lidars_forward_xy[i], axis=-1)
lidars_forward_xy[~i,:] = 0
lidar_forward_arrow_length = 4.
def data_tuples_lidar_forward_vectors(ilidar_solve):
    return \
    (
      # LIDAR positions AND their forward vectors
      (nps.glue( lidars_origin [...,:2],
                 lidars_forward_xy * lidar_forward_arrow_length,
                 axis = -1 ),
       dict(_with = 'vectors lw 2 lc "black"',
            tuplesize = -4) ),

      # # JUST the LIDAR positions
      # ( lidars_origin [...,:2],
      #   dict(_with = 'points pt 2 lc "black"',
      #        tuplesize = -2) ),

      ( lidars_origin[np.arange(len(context['lidar_topic'])) != ilidar_solve,0],
        lidars_origin[np.arange(len(context['lidar_topic'])) != ilidar_solve,1],
        np.array(context['lidar_topic'])[np.arange(len(context['lidar_topic'])) != ilidar_solve],
        dict(_with = 'labels textcolor "black"',
             tuplesize = 3)),
      ( lidars_origin[np.array((ilidar_solve,),),0],
        lidars_origin[np.array((ilidar_solve,),),1],
        np.array(context['lidar_topic'])[np.array((ilidar_solve,),)],
        dict(_with = 'labels textcolor "red"',
             tuplesize = 3))
     )



for ilidar,topic in enumerate(args.lidar_topic):
    ilidar_solve = ilidar_in_solve_from_ilidar[ilidar]
    if ilidar_solve == 0: continue # reference coord system

    l,v = \
        clc.reprojection_covariance_decomposed(p0,
                                               context['result']['rt_ref_lidar'],
                                               ilidar_solve,
                                               context['result']['Var'])
    stdev = np.sqrt(l)


    # shape (Nysample,Nxsample)

    iworst  = np.argmax(stdev,axis=-1, keepdims=True)
    uncertainty_1sigma = np.take_along_axis(stdev,iworst, -1)[...,0]

    eigv_worst = np.take_along_axis(v,nps.dummy(iworst,-1), -1)[...,0]
    cos_vertical = np.abs(eigv_worst[...,2])
    thdeg_vertical = np.arccos(np.clip(cos_vertical,-1,1)) * 180./np.pi

    using = f'({x_sample[0]} + $1*({x_sample[-1]-x_sample[0]})/{args.gridn-1}):({y_sample[0]} + $2*({y_sample[-1]-y_sample[0]})/{args.gridn-1}):3'

    clc.plot((uncertainty_1sigma,
          dict(tuplesize = 3,
               _with = 'image',
               using = using),
          ),
         *data_tuples_lidar_forward_vectors(ilidar_solve),
         cbmin = 0,
         square = True,
         wait = True,
         xlabel = 'x',
         ylabel = 'y',
         title = f'Worst-case 1-sigma transform uncertainty for {topic} (top-down view)',
         ascii = 1, # needed for the "using" scale
         _set  = ('xrange [:] noextend', 'yrange [:] noextend'),
         hardcopy=f'/tmp/uncertainty-1sigma-ilidar={ilidar_solve}.gp')

    clc.plot((thdeg_vertical,
          dict(tuplesize = 3,
               _with = 'image',
               using = using),
          ),
         *data_tuples_lidar_forward_vectors(ilidar_solve),
         cbmin = 0,
         cbmax = 30,
         square = True,
         wait = True,
         xlabel = 'x',
         ylabel = 'y',
         title = f'Worst-case transform uncertainty for {topic} (top-down view): angle off vertical (deg)',
         ascii = 1, # needed for the "using" scale
         _set  = ('xrange [:] noextend', 'yrange [:] noextend'),
         hardcopy=f'/tmp/uncertainty-direction-1sigma-ilidar={ilidar_solve}.gp')
