#!/usr/bin/python3
r'''Display a set of LIDAR point clouds in an aligned coordinate system

SYNOPSIS

  $ ./show-aligned-lidar-pointclouds.py                   \
      --rt-lidar-ref 0,0,0,0,0,0                          \
      --rt-lidar-ref 0.1,0,0.2,1,2,3                      \
      --bag camera-lidar.bag                              \
      --lidar-topic /lidar/vl_points_0,/lidar/vl_points_1
    [plot pops up to show the aligned results]

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

    parser.add_argument('--threshold',
                        type=float,
                        default = 20.,
                        help = '''Max distance where we cut the plot''')

    parser.add_argument('--rt-lidar-ref',
                        type = str,
                        action = 'append',
                        help = '''Transforms for each LIDAR. Each transform is
                        given as r,r,r,t,t,t. Exactly as many --rt-lidar-ref
                        arguments must be given as LIDAR topics. This is
                        exclusive with --rt-ref-lidar''')
    parser.add_argument('--rt-ref-lidar',
                        type = str,
                        action = 'append',
                        help = '''Transforms for each LIDAR. Each transform is
                        given as r,r,r,t,t,t. Exactly as many --rt-ref-lidar
                        arguments must be given as LIDAR topics. This is
                        exclusive with --rt-lidar-ref''')

    parser.add_argument('--lidar-topic',
                        type=str,
                        required = True,
                        help = '''The LIDAR topic to visualize. This is a
                        comma-separated list of topics''')

    parser.add_argument('bag',
                        help = '''The one bag we're visualizing''')


    args = parser.parse_args()

    args.lidar_topic = args.lidar_topic.split(',')

    if args.rt_lidar_ref is None and \
       args.rt_ref_lidar is None:
        print("Exactly one of --rt-lidar-ref or --rt-ref-lidar must be given; got neither",
              file=sys.stderr)
        sys.exit(1)
    if args.rt_lidar_ref is not None and \
       args.rt_ref_lidar is not None:
        print("Exactly one of --rt-lidar-ref or --rt-ref-lidar must be given; got both",
              file=sys.stderr)
        sys.exit(1)

    def parse_rt(rt):
        s = rt.split(',')
        if len(s) != 6:
            print(f"Each --rt-... MUST be a comma-separated list of exactly 6 values: '{rt}' has the wrong number of values",
                  file=sys.stderr)
            sys.exit(1)
        try:
            s = [float(x) for x in s]
        except:
            print(f"Each --rt-... MUST be a comma-separated list of exactly 6 values: '{rt}' not parseable as a list of floats",
                  file=sys.stderr)
            sys.exit(1)
        return s

    Nlidar = len(args.lidar_topic)
    if args.rt_lidar_ref is not None:
        if len(args.rt_lidar_ref) != Nlidar:
            print(f"MUST have been given a matching number of --rt-lidar-ref and topics. Got {len(args.rt_lidar_ref)} and {Nlidar} respectively instead",
                  file=sys.stderr)
            sys.exit(1)

        args.rt_lidar_ref = np.array([parse_rt(rt) for rt in args.rt_lidar_ref])


    else:
        if len(args.rt_ref_lidar) != Nlidar:
            print(f"MUST have been given a matching number of --rt-ref-lidar and topics. Got {len(args.rt_ref_lidar)} and {Nlidar} respectively instead",
                  file=sys.stderr)
            sys.exit(1)

        args.rt_ref_lidar = np.array([parse_rt(rt) for rt in args.rt_ref_lidar])

    return args


args = parse_args()

import sys
import numpysane as nps
import mrcal
import gnuplotlib as gp

import calibration_data_import

try:
    pointcloud_msgs = \
        [ next(calibration_data_import.bag_messages_generator(args.bag, (topic,))) \
          for topic in args.lidar_topic ]
except:
    raise Exception(f"Bag '{args.bag}' doesn't have at least one message for each of {args.lidar_topic}")

# Package into a numpy array
pointclouds = [ msg['array']['xyz'].astype(float) \
                for msg in pointcloud_msgs ]

# Throw out everything that's too far, in the LIDAR's own frame
pointclouds = [ p[ nps.mag(p) < args.threshold ] for p in pointclouds ]

# Transform
if args.rt_lidar_ref is not None:
    args.rt_ref_lidar = mrcal.invert_rt(args.rt_lidar_ref)
pointclouds = [ mrcal.transform_point_rt(args.rt_ref_lidar[i],p) for i,p in enumerate(pointclouds) ]


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




Ncameras = 0
Nlidars  = 3
istate_camera_pose_0 = 0
Nstate_camera_pose   = 6 * Ncameras
istate_lidar_pose_0  = istate_camera_pose_0 + Nstate_camera_pose
Nstate_lidar_pose    = 6 * (Nlidars-1) # lidar0 is the reference coord system

import pickle
with open("/tmp/Var_state_poses.pickle", "rb") as f:
    Var_state_poses = pickle.load(f)


x_sample = np.linspace(-20,20,25)
y_sample = np.linspace(-20,20,25)
z_sample = 1

Nxsample = len(x_sample)
Nysample = len(y_sample)

# Each has shape (Ny_sample,Nx_sample)
px0, py0 = np.meshgrid(x_sample,y_sample)
pz0 = z_sample * np.ones(px0.shape, dtype=float)
# shape (Nysample, Nxsample,3)
p0 = nps.mv(nps.cat(px0,py0,pz0),
            0,-1)
if Ncameras:
    raise Exception(f"Hard-coded {Ncameras=} is wrong: I'm assuming ilidar = isensor below")
if len(args.rt_ref_lidar) != Nlidars:
    raise Exception(f"Hard-coded {Nlidars=} is wrong")
if nps.norm2(args.rt_ref_lidar[0]):
    raise Exception("The first rt_ref_lidar must be 0")

az = np.linspace(-np.pi, np.pi,20, endpoint = False)
el = nps.mv(np.linspace(-np.pi/2., np.pi/2., 10), -1,-2)
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




def plot(*args,
         hardcopy = None,
         **kwargs):
    r'''Wrapper for gp.plot(), but printing out where the hardcopy went'''
    gp.plot(*args, **kwargs,
            hardcopy = hardcopy)
    if hardcopy is not None:
        print(f"Wrote '{hardcopy}'")



print("WARNING: the uncertainty propagation should be cameras AND lidars")
rt_lidar0_lidar = np.array(args.rt_ref_lidar, dtype=float)

lidars_origin  = rt_lidar0_lidar[:,3:]
lidars_forward = mrcal.rotate_point_r(rt_lidar0_lidar[:,:3], np.array((1.,0,0)))

lidars_forward_xy = np.array(lidars_forward[...,:2])
# to avoid /0 for straight-up vectors
mag_lidars_forward_xy = nps.mag(lidars_forward_xy)
i = mag_lidars_forward_xy>0
lidars_forward_xy[i,:] /= nps.dummy(mag_lidars_forward_xy[i], axis=-1)
lidars_forward_xy[~i,:] = 0
lidar_forward_arrow_length = 4.
data_tuples_lidar_forward_vectors = \
    ( (nps.glue( lidars_origin [...,:2],
                 lidars_forward_xy * lidar_forward_arrow_length,
                 axis = -1 ),
       dict(_with = 'vectors lw 2 lc "black"',
            tuplesize = -4) ),

      ( lidars_origin[...,0],
        lidars_origin[...,1],
        np.array([f"Lidar {i}" for i in range(Nlidars)]),
        dict(_with = 'labels textcolor "red"',
             tuplesize = 3))
     )



for ilidar,rt_ref_lidar in enumerate(args.rt_ref_lidar):
    if ilidar == 0: continue

    # shape (Nysample,Nxsample,3)
    p1 = \
        mrcal.transform_point_rt(rt_ref_lidar, p0,
                                 inverted = True)

    rt_lidar_ref = mrcal.invert_rt(rt_ref_lidar)

    # shape (Nysample,Nxsample,3,6)
    _,dp0__drt_lidar_ref,_ = \
        mrcal.transform_point_rt(rt_lidar_ref, p1,
                                 inverted      = True,
                                 get_gradients = True)
    # shape (6,6)
    Var_rt_lidar_ref = Var_state_poses[ilidar-1,:,ilidar-1,:]

    # shape (Nysample,Nxsample,3,3)
    Var_p0 = nps.matmult(dp0__drt_lidar_ref,
                         Var_rt_lidar_ref,
                         nps.transpose(dp0__drt_lidar_ref))

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
                                    legend = f'1-sigma uncertainty for lidar {ilidar}')), )

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
             title = f'Worst-case 1-sigma transform uncertainty for lidar {ilidar} (top-down view)',
             ascii = 1, # needed for the "using" scale
             _set  = ('xrange [:] noextend', 'yrange [:] noextend'),
             hardcopy=f'/tmp/uncertainty-1sigma-{ilidar=}.gp')

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
             title = f'Worst-case transform uncertainty for lidar {ilidar} (top-down view): angle off vertical (deg)',
             ascii = 1, # needed for the "using" scale
             _set  = ('xrange [:] noextend', 'yrange [:] noextend'),
             hardcopy=f'/tmp/uncertainty-direction-1sigma-{ilidar=}.gp')

if do_plot_ellipsoids:
    plot(*data_tuples,
         _3d = True,
         square = True,
         wait = True,
         xlabel = 'x',
         ylabel = 'y',
         zlabel = 'z',
         hardcopy='/tmp/ellipsoids.gp')
