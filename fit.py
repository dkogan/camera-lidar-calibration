#!/usr/bin/python3

r'''Calibrate a set of cameras and LIDARs into a common coordinate system

SYNOPSIS

  $ lidars=(/lidar/vl_points_0)
  $ cameras=(/front/multisense/{{left,right}/image_mono_throttle,aux/image_color_throttle})
  $ sensors=($lidars $cameras)

  $ ./fit.py \
      --topics ${(j:,:)sensors} \
      --bag 'camera-lidar-*.bag'      \
      intrinsics/{left,right,aux}_camera/camera-0-OPENCV8.cameramodel

  [ The tool chugs for a bit, and in the end produces diagnostics and the aligned ]
  [ models                                                                        ]

This tool computes a geometry-only calibration. It is assumed that the camera
intrinsics have already been computed. The results are computed in the
coordinate system of the first LIDAR. All the sensors must overlap each other
transitively: every sensor doesn't need to overlap every other sensor, but there
must be an overlapping path between each pair of sensors.

The data comes from a set of ROS bags. Each bag is assumed to have captured a
single frame (one set of images, LIDAR revolutions) of a stationary scene

'''


import sys
import argparse
import clc.argparse_helpers
import re
import os

import clc
lidar_segmentation_parameters = clc.lidar_segmentation_parameters()


def parse_args():

    parser = \
        argparse.ArgumentParser(description = __doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--topics',
                        type=str,
                        required = True,
                        help = '''Which lidar(s) and camera(s) we're talking to.
                        This is a comma-separated list of topics. Any Nlidars >=
                        1 and Ncameras >= 0 is supported''')

    parser.add_argument('--bag',
                        type=str,
                        required = True,
                        help = '''Glob for the rosbag that contains the lidar
                        and camera data. This can match multiple files''')

    parser.add_argument('--exclude-bag',
                        type=str,
                        action = 'append',
                        help = '''Bags to exclude from the processing. These are
                        a regex match against the bag paths''')
    parser.add_argument('--decimation-period',
                        type=float,
                        help = '''If given, we expect ONE bag, and rather than
                        taking the first message from each bag, we take all the
                        messages from THIS bag, spaced out with a period given
                        by this argument, in seconds''')
    parser.add_argument('--after',
                        type=str,
                        help = '''If given, start reading the bags at this time.
                        Could be an integer (s since epoch or ns since epoch), a
                        float (s since the epoch) or a string, to be parsed with
                        dateutil.parser.parse()''')
    parser.add_argument('--before',
                        type=str,
                        help = '''If given, stop reading the bags at this time.
                        Could be an integer (s since epoch or ns since epoch), a
                        float (s since the epoch) or a string, to be parsed with
                        dateutil.parser.parse()''')
    parser.add_argument('--exclude-time-period',
                        type=str,
                        action='append',
                        help = '''If given, a comma-separated pair of time
                        periods to be excluded from the data. Each could be an
                        integer (s since epoch or ns since epoch), a float (s
                        since the epoch) or a string, to be parsed with
                        dateutil.parser.parse(). May be given multiple times''')
    parser.add_argument('--dump',
                        type=str,
                        help = '''Write solver diagnostics into the given
                        .pickle file''')

    parser.add_argument('--Nsectors',
                        type=int,
                        default=36,
                        help='''Used in the uncertainty quantification. We
                        report the uncertainty in radial sectors around the
                        vehicle origin. If omitted, we use 36 sectors''')
    parser.add_argument('--threshold-valid-lidar-range',
                        type=float,
                        default=1.0,
                        help='''Used in the uncertainty quantification. Lidar
                        returns closer than this are classified as "occluded".
                        This is used to determine the lidar field-of-view in the
                        uncertainty reporting. If omitted, we set this to 1.0m''')
    parser.add_argument('--threshold-valid-lidar-Npoints',
                        type=int,
                        default=100,
                        help='''Used in the uncertainty quantification. We
                        require at least this many unoccluded lidar returns in a
                        sector to classify it as "visible". If omitted, we require
                        100 returns''')
    parser.add_argument('--uncertainty-quantification-range',
                        type=int,
                        default=10,
                        help='''Used in the uncertainty quantification. We
                        report the uncertainty in radial sectors around the
                        vehicle origin. In each sector we look at a point this
                        distance away from the origin. If omitted, we look 10m
                        ahead''')
    parser.add_argument('--max-time-spread-s',
                        type=float,
                        help='''The maximum time spread of observations in a
                        snapshot. Any snapshot that contains sensor observations
                        with a bigger time differences than this are thrown out;
                        the WHOLE snapshot''')
    parser.add_argument('--rt-vehicle-lidar0',
                        type=clc.argparse_helpers.comma_separated_list_of_floats,
                        help='''Used in the uncertainty quantification. The
                        vehicle-lidar0 transform. The solve is always done in
                        lidar0 coordinates, but we the uncertainty
                        quantification operates in a different "vehicle" frame.
                        This argument specifies the relationship between those
                        frames. If omitted, we assume an identity transform: the
                        vehicle frame is the lidar0 frame''')
    parser.add_argument('--verbose',
                        action = 'store_true',
                        help='''Report details about the solve''')
    parser.add_argument('models',
                        type = str,
                        nargs='*',
                        help='''Camera models for the optical calibration. Only
                        the intrinsics are used. The number of models given must
                        match the number of camera --topics arguments EXACTLY''')

    for param,metadata in lidar_segmentation_parameters.items():
        # Here I have a --dump that's different from the 'dump' in the
        # segmentation parameters. I ignore the latter for now
        if param == 'dump': continue

        if metadata['pyparse'] == 'p':
            # special case for boolean
            parser.add_argument(f'--{param.replace("_","-")}',
                                action  = 'store_true',
                                help = metadata['doc'])
        else:
            parser.add_argument(f'--{param.replace("_","-")}',
                                type    = type(metadata['default']),
                                default = metadata['default'],
                                help = metadata['doc'])

    args = parser.parse_args()

    import glob
    import re
    args.bag = args.bag.rstrip('/') # to not confuse os.path.splitext()
    bags = sorted(glob.glob(args.bag))

    if args.exclude_bag is not None:
        for ex in args.exclude_bag:
            bags = [b for b in bags if not re.search(ex, b)]

    args.bag = bags

    args.topics = args.topics.split(',')

    if args.decimation_period is not None:
        if args.decimation_period <= 0:
            print("--decimation-period given, and it must some >0 number of seconds", file=sys.stderr)
            sys.exit(1)
        if len(args.bag) > 1:
            print("--decimation-period given, so we MUST have gotten exactly one bag", file=sys.stderr)
            sys.exit(1)

    if args.exclude_time_period is None:
        args.exclude_time_period = []
    else:
        try:
            args.exclude_time_period = [ t0t1.split(',') for t0t1 in args.exclude_time_period]
        except:
            print("each --exclude-time-period should be a comma-separated timestamp", file=sys.stderr)
            sys.exit(1)

    return args


args = parse_args()



import numpy as np
import numpysane as nps
import gnuplotlib as gp
import io
import pickle
import mrcal

# ingest the lidar segmentation parameters from the arguments
ctx = dict()
for k in lidar_segmentation_parameters.keys():
    # Here I have a --dump that's different from the 'dump' in the
    # segmentation parameters. I ignore the latter for now
    if k == 'dump': continue
    ctx[k] = getattr(args, k)


if args.rt_vehicle_lidar0 is not None:
    args.rt_vehicle_lidar0 = np.array(args.rt_vehicle_lidar0)



if args.models:

    def open_model(f):
        try: return mrcal.cameramodel(f)
        except:
            print(f"Couldn't open '{f}' as a camera model",
                  file=sys.stderr)
            sys.exit(1)
    models = [open_model(f) for f in args.models]

    # I assume each model used the same calibration object
    # shape (Ncameras, Nh,Nw,3)
    p_board_local__all = \
        [mrcal.ref_calibration_object(optimization_inputs =
                                      m.optimization_inputs()) \
         for m in models]
    def is_different(x,y):
        try:    return nps.norm2((x-y).ravel()) > 1e-12
        except: return True
    if any(is_different(p_board_local__all[0][...,:2],
                        p_board_local__all[i][...,:2]) \
           for i in range(1,len(models))):
        print("Each model should have been made with the same chessboard, but some are different. I use this calibration-time chessboard for the camera-lidar calibration",
              file = sys.stderr)
        sys.exit(1)

    # shape (Nh,Nw,3)
    p_board_local = p_board_local__all[0]
    p_board_local[...,2] = 0 # assume flat. calobject_warp may differ between samples

else:
    p_board_local = None





if len(args.models) > 0:
    m = mrcal.cameramodel(args.models[0])
    o = m.optimization_inputs()
    H,W = o['observations_board'].shape[-3:-1]
    ctx['object_spacing']  = o['calibration_object_spacing']
    ctx['object_width_n']  = W
    ctx['object_height_n'] = H

kwargs_calibrate = dict(bags                               = args.bag,
                        topics                             = args.topics,
                        decimation_period_s                = args.decimation_period,
                        max_time_spread_s                  = args.max_time_spread_s,
                        exclude_time_periods               = args.exclude_time_period,
                        start                              = args.after,
                        stop                               = args.before,
                        models                             = args.models,
                        check_gradient                     = False,
                        verbose                            = args.verbose,
                        Nsectors                           = args.Nsectors,
                        threshold_valid_lidar_range        = args.threshold_valid_lidar_range,
                        threshold_valid_lidar_Npoints      = args.threshold_valid_lidar_Npoints,
                        uncertainty_quantification_range   = args.uncertainty_quantification_range,
                        rt_vehicle_lidar0                  = args.rt_vehicle_lidar0,
                        **ctx)

result = clc.calibrate(do_dump_inputs = args.dump is not None,
                       **kwargs_calibrate)

if 'rt_vehicle_lidar' in result:
    rt_ref_lidar  = result['rt_vehicle_lidar']
    rt_ref_camera = result['rt_vehicle_camera']
else:
    rt_ref_lidar  = result['rt_lidar0_lidar']
    rt_ref_camera = result['rt_lidar0_camera']

D = '/tmp'
for ilidar in range(len(rt_ref_lidar)):
    # dummy lidar "cameramodel". The intrinsics are made-up, but the extrinsics
    # are true, and can be visualized with the usual tools
    filename = f"lidar{ilidar}-mounted.cameramodel"
    path = f"{D}/{filename}"
    model = mrcal.cameramodel( intrinsics = ('LENSMODEL_PINHOLE',
                                             np.array((1.,1.,0.,0.))),
                               imagersize = (1,1),
                               extrinsics_rt_toref = rt_ref_lidar[ilidar] )
    model.write(path,
                note = "Intrinsics are made-up and nonsensical")

    symlink =  f"{D}/sensor{ilidar}-mounted.cameramodel"
    try:    os.unlink(symlink)
    except: pass
    os.symlink(filename, symlink)

    print(f"Wrote '{path}' and a symlink '{symlink}'")

for imodel in range(len(args.models)):
    models[imodel].extrinsics_rt_toref(rt_ref_camera[imodel])
    d,f = os.path.split(args.models[imodel])
    r,e = os.path.splitext(f)
    filename = f"{r}-mounted{e}"
    path     = f"{D}/{filename}"
    models[imodel].write(path)

    symlink =  f"{D}/sensor{len(rt_ref_lidar) + imodel}-mounted.cameramodel"
    try:    os.unlink(symlink)
    except: pass
    os.symlink(filename, symlink)

    print(f"Wrote '{path}' and a symlink '{symlink}'")



context = \
    dict(result           = result,
         kwargs_calibrate = kwargs_calibrate)
if args.dump is not None:
    with open(args.dump, 'wb') as f:
        pickle.dump( context,
                     f )
    print(f"Wrote '{args.dump}'")



sensor_forward_vectors_plot_tuples = \
    clc.sensor_forward_vectors_plot_tuples(rt_ref_lidar,
                                           rt_ref_camera,
                                           args.topics)



# shape (Nsensors, Nsectors)
isvisible_per_sensor_per_sector = result['isvisible_per_sensor_per_sector']

angular_width_ratio = 0.9
Nsensors = isvisible_per_sensor_per_sector.shape[0]
dth = np.pi*2./args.Nsectors
th = np.arange(args.Nsectors)*dth + dth/2 * (1.-angular_width_ratio)
plotradius = nps.transpose(np.arange(Nsensors) + 10)
ones = np.ones( (args.Nsectors,) )

filename = '/tmp/observability.pdf'
clc.plot( (th,                # angle
          plotradius*ones,    # radius
          ones*dth*angular_width_ratio, # angular width of slice
          ones*0.9,           # depth of slice
          isvisible_per_sensor_per_sector,
          dict(_with = 'sectors palette fill solid',
               tuplesize = 5)),
         *sensor_forward_vectors_plot_tuples,
         _xrange = (-10-Nsensors,10+Nsensors),
         _yrange = (-10-Nsensors,10+Nsensors),
         square = True,
         unset = 'colorbox',
         title = 'Observability map of each sensor',
         hardcopy = filename,
        )

stdev_worst_per_sector = result['stdev_worst_per_sector']
i = stdev_worst_per_sector != 0
filename = '/tmp/uncertainty.pdf'
clc.plot( (th[i],                # angle
          10.*ones[i],           # radius
          ones[i]*dth*0.9,       # angular width of slice
          ones[i]*0.9,           # depth of slice
          stdev_worst_per_sector[i],
          dict(tuplesize = 5,
               _with = 'sectors palette fill solid')),
         *sensor_forward_vectors_plot_tuples,
         _xrange = (-11,11),
         _yrange = (-11,11),
         square = True,
         title = f'Worst-case uncertainty at {result["uncertainty_quantification_range"]}m. Put the board in high-uncertainty regions',
         hardcopy = filename,
        )

observations_per_sector = result['observations_per_sector']
filename = '/tmp/observations_per_sector.pdf'
clc.plot( (th,                # angle
          10.*ones,           # radius
          ones*dth*0.9,       # angular width of slice
          ones*0.9,           # depth of slice
          observations_per_sector,
          dict(tuplesize = 5,
               _with = 'sectors palette fill solid')),
         *sensor_forward_vectors_plot_tuples,
         _xrange = (-11,11),
         _yrange = (-11,11),
         square = True,
         title = 'Observations per sector',
         hardcopy = filename,
        )

