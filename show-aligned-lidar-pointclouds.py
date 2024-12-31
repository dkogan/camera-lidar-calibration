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

    parser.add_argument('--lidar-topic',
                        type=str,
                        required = True,
                        help = '''The LIDAR topics to visualize. This is a
                        comma-separated list of topics''')
    parser.add_argument('--bag',
                        help = '''The one bag we're visualizing''')
    parser.add_argument('--context',
                        help = '''.pickle file from fit.py''')
    parser.add_argument('--threshold',
                        type=float,
                        default = 20.,
                        help = '''Max distance where we cut the plot''')
    parser.add_argument('lidar-models',
                        nargs   = '*',
                        help = '''The .cameramodel for the lidars in question.
                        Must correspond to the set in --lidar-topic. Exclusive
                        with --context. Exactly one of (lidar-models,--context)
                        must be given''')


    args = parser.parse_args()

    args.lidar_topic = args.lidar_topic.split(',')
    args.lidar_models = getattr(args, 'lidar-models')
    if len(args.lidar_models) == 0: args.lidar_models = None

    if (args.context is not None and args.lidar_models is not None) or \
       (args.context is     None and args.lidar_models is     None):
        print("Exactly one of (lidar-models,--context) must be given", file=sys.stderr)
        sys.exit(1)

    if args.lidar_models is not None:
        if len(args.lidar_models) != len(args.lidar_topic):
            print(f"MUST have been given a matching number of lidar models and topics. Got {len(args.lidar_models)=} and {len(args.lidar_topic)=} instead",
                  file=sys.stderr)
            sys.exit(1)

    return args


args = parse_args()

import bag_interface
import pickle
import numpysane as nps
import mrcal
import gnuplotlib as gp



if args.context is None:
    ilidar_in_solve_from_ilidar = None
    context                     = None
else:
    with open(args.context, "rb") as f:
        context = pickle.load(f)


    ilidar_in_solve_from_ilidar = [None] * len(args.lidar_topic)
    for ilidar in range(len(args.lidar_topic)):
        lidar_topic_requested = args.lidar_topic[ilidar]
        try:
            ilidar_in_solve_from_ilidar[ilidar] = \
                context['lidar_topic'].index(lidar_topic_requested)
        except:
            print(f"Requested topic '{lidar_topic_requested}' not present in the covariance file '{args.context}'",
                  file=sys.stderr)
            sys.exit(1)



if args.lidar_models is None:
    # one transform for each lidar in the solve
    rt_lidar0_lidar = context['rt_ref_lidar']
else:
    # one transform for each --lidar-topic
    lidar_models = [mrcal.cameramodel(f) for f in args.lidar_models]
    rt_lidar0_lidar = [m.extrinsics_rt_toref() for m in lidar_models]







def plot(*args,
         hardcopy = None,
         **kwargs):
    r'''Wrapper for gp.plot(), but printing out where the hardcopy went'''
    gp.plot(*args, **kwargs,
            hardcopy = hardcopy)
    if hardcopy is not None:
        print(f"Wrote '{hardcopy}'")



try:
    pointcloud_msgs = \
        [ next(bag_interface.messages(args.bag, (topic,))) \
          for topic in args.lidar_topic ]
except:
    raise Exception(f"Bag '{args.bag}' doesn't have at least one message for each of {args.lidar_topic}")

# Package into a numpy array
pointclouds = [ msg['array']['xyz'].astype(float) \
                for msg in pointcloud_msgs ]

# Throw out everything that's too far, in the LIDAR's own frame
pointclouds = [ p[ nps.mag(p) < args.threshold ] for p in pointclouds ]

# Transform to ref coords
if context is None:
    pointclouds = [ mrcal.transform_point_rt(rt_lidar0_lidar[i],
                                             p) for i,p in enumerate(pointclouds) ]
else:
    pointclouds = [ mrcal.transform_point_rt(rt_lidar0_lidar[ ilidar_in_solve_from_ilidar[i] ],
                                             p) for i,p in enumerate(pointclouds) ]

# from "gnuplot -e 'show linetype'"
color_sequence_rgb = (
    "#9400d3",
    "#009e73",
    "#56b4e9",
    "#e69f00",
    "#f0e442",
    "#0072b2",
    "#e51e10",
    "#000000"
)

data_tuples = [ ( p, dict( tuplesize = -3,
                           legend    = args.lidar_topic[i],
                           _with     = f'points pt 7 ps 1 lc rgb "{color_sequence_rgb[i%len(color_sequence_rgb)]}"')) \
                for i,p in enumerate(pointclouds) ]


if context is None:
    plot(*data_tuples,
         _3d = True,
         square = True,
         xlabel = 'x',
         ylabel = 'y',
         zlabel = 'z',
         wait = True)
    sys.exit()


x_sample = np.linspace(-20,20,25)
y_sample = np.linspace(-20,20,25)
z_sample = 1

Nxsample = len(x_sample)
Nysample = len(y_sample)

### point coordinates in the lidar0 frame. This is the solve reference:
### ilidar_in_solve_from_ilidar[ilidar]==0
# Each has shape (Ny_sample,Nx_sample)
px0, py0 = np.meshgrid(x_sample,y_sample)
pz0 = z_sample * np.ones(px0.shape, dtype=float)
# shape (Nysample, Nxsample,3)
p0 = nps.mv(nps.cat(px0,py0,pz0),
            0,-1)

az = np.linspace(-np.pi, np.pi,40, endpoint = False)
el = nps.mv(np.linspace(-np.pi/2., np.pi/2., 20), -1,-2)
caz = np.cos(az)
saz = np.sin(az)
cel = np.cos(el)
sel = np.sin(el)


do_plot_ellipsoids               = False
do_plot_worst_eigenvalue_heatmap = True


if do_plot_ellipsoids:
    # shape (Npoints,3)
    psphere = nps.clump(nps.mv(nps.cat(caz*cel, saz*cel, sel*np.ones(az.shape)),
                               0,-1),
                        n=2)

    # 1. is "to scale". Higher numbers improve legibility
    ellipse_scale = 10.



print("WARNING: the uncertainty propagation should be cameras AND lidars")

lidars_origin  = rt_lidar0_lidar[:,3:]
lidars_forward = mrcal.rotate_point_r(rt_lidar0_lidar[:,:3], np.array((1.,0,0)))

lidars_forward_xy = np.array(lidars_forward[...,:2])
# to avoid /0 for straight-up vectors
mag_lidars_forward_xy = nps.mag(lidars_forward_xy)
i = mag_lidars_forward_xy>0
lidars_forward_xy[i,:] /= nps.dummy(mag_lidars_forward_xy[i], axis=-1)
lidars_forward_xy[~i,:] = 0
lidar_forward_arrow_length = 4.
i = np.array([ilidar_in_solve_from_ilidar[i] for i in range(len(args.lidar_topic))],
             dtype=np.int32)
data_tuples_lidar_forward_vectors = \
    (
      # LIDAR positions AND their forward vectors
      (nps.glue( lidars_origin [i,:2],
                 lidars_forward_xy[i] * lidar_forward_arrow_length,
                 axis = -1 ),
       dict(_with = 'vectors lw 2 lc "black"',
            tuplesize = -4) ),

      # # JUST the LIDAR positions
      # ( lidars_origin [i,:2],
      #   dict(_with = 'points pt 2 lc "black"',
      #        tuplesize = -2) ),

      ( lidars_origin[i,0],
        lidars_origin[i,1],
        np.array(args.lidar_topic),
        dict(_with = 'labels textcolor "red"',
             tuplesize = 3))
     )

for ilidar in range(len(args.lidar_topic)):

    topic = args.lidar_topic[ilidar]
    ilidar = ilidar_in_solve_from_ilidar[ilidar]

    if ilidar == 0: continue # reference coord system

    # shape (Nysample,Nxsample,3)
    p1 = \
        mrcal.transform_point_rt(rt_lidar0_lidar[ilidar], p0, inverted=True)

    # shape (Nysample,Nxsample,3,6)
    _,dp0__drt_lidar01,_ = \
        mrcal.transform_point_rt(rt_lidar0_lidar[ilidar], p1,
                                 get_gradients = True)

    # shape (6,6)
    Var_rt_lidar01 = context['Var'][ilidar-1,:,
                                    ilidar-1,:]

    # shape (Nysample,Nxsample,3,3)
    Var_p0 = nps.matmult(dp0__drt_lidar01,
                         Var_rt_lidar01,
                         nps.transpose(dp0__drt_lidar01))

    # shape (Nysample,Nxsample,3) and (Nysample,Nxsample,3,3)
    l,v = np.linalg.eig(Var_p0)
    stdev = np.sqrt(l)

    if do_plot_ellipsoids:
        # shape (Nspherepoints,Nysample,Nxsample,3)
        pellipsoid = \
            p0 + \
            ellipse_scale * \
            nps.matmult(# shape (Nspherepoints,1,1,1,3) * (Nysample,Nxsample,1,3)
                        nps.dummy(psphere, -2,-2,-2) * nps.dummy(stdev, axis=-2),
                        nps.transpose(v))[...,0,:]

        data_tuples.append( ( nps.clump(pellipsoid, n=3),
                              dict( tuplesize = -3,
                                    _with     = f'dots lc rgb "{color_sequence_rgb[ilidar%len(color_sequence_rgb)]}"',
                                    legend = f'1-sigma uncertainty for {topic}')), )


    if do_plot_worst_eigenvalue_heatmap:
        # shape (Nysample,Nxsample)

        iworst  = np.argmax(stdev,axis=-1, keepdims=True)
        uncertainty_1sigma = np.take_along_axis(stdev,iworst, -1)[...,0]

        eigv_worst = np.take_along_axis(v,nps.dummy(iworst,-1), -1)[...,0]
        cos_vertical = np.abs(eigv_worst[...,2])
        thdeg_vertical = np.arccos(np.clip(cos_vertical,-1,1)) * 180./np.pi

        using = f'({x_sample[0]} + $1*({x_sample[-1]-x_sample[0]})/{Nxsample-1}):({y_sample[0]} + $2*({y_sample[-1]-y_sample[0]})/{Nysample-1}):3'

        plot((uncertainty_1sigma,
              dict(tuplesize = 3,
                   _with = 'image',
                   using = using),
              ),
             *data_tuples_lidar_forward_vectors,
             cbmin = 0,
             square = True,
             wait = True,
             xlabel = 'x',
             ylabel = 'y',
             title = f'Worst-case 1-sigma transform uncertainty for {topic} (top-down view)',
             ascii = 1, # needed for the "using" scale
             _set  = ('xrange [:] noextend', 'yrange [:] noextend'),
             hardcopy=f'/tmp/uncertainty-1sigma-ilidar={ilidar}.gp')

        plot((thdeg_vertical,
              dict(tuplesize = 3,
                   _with = 'image',
                   using = using),
              ),
             *data_tuples_lidar_forward_vectors,
             cbmin = 0,
             cbmax = 30,
             square = True,
             wait = True,
             xlabel = 'x',
             ylabel = 'y',
             title = f'Worst-case transform uncertainty for {topic} (top-down view): angle off vertical (deg)',
             ascii = 1, # needed for the "using" scale
             _set  = ('xrange [:] noextend', 'yrange [:] noextend'),
             hardcopy=f'/tmp/uncertainty-direction-1sigma-ilidar={ilidar}.gp')


if do_plot_ellipsoids:
    plot(*data_tuples,
         _3d = True,
         square = True,
         xlabel = 'x',
         ylabel = 'y',
         zlabel = 'z',
         wait = True)
    sys.exit()
