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

  ....
  clc.c(3362) fit(): Finished full solve
  clc.c(3387) fit(): RMS fit error: 0.43 normalized units
  clc.c(3404) fit(): RMS fit error (camera): 0.71 pixels
  clc.c(3410) fit(): RMS fit error (lidar): 0.013 m
  clc.c(3415) fit(): norm2(error_regularization)/norm2(error): 0.00
  clc.c(2695) plot_residuals(): Wrote '/tmp/residuals.gp'
  clc.c(2727) plot_residuals(): Wrote '/tmp/residuals-histogram-lidar.gp'
  clc.c(3020) plot_geometry(): Wrote '/tmp/geometry.gp'
  clc.c(3020) plot_geometry(): Wrote '/tmp/geometry-onlyaxes.gp'

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
    parser.add_argument('--object-width-n',
                        type=int,
                        help='''Used if any cameras are present. How many
                        points the calibration board has per horizontal side. If
                        omitted, we try to use the calibration-time board in the
                        models''')
    parser.add_argument('--object-height-n',
                        type=int,
                        help='''Used if any cameras are present. How many
                        points the calibration board has per vertical side. If
                        omitted, we try to use the calibration-time board in the
                        models''')
    parser.add_argument('--object-spacing',
                        type=float,
                        help='''Used if any cameras are present. Width of each
                        square in the calibration board, in meters. If omitted,
                        we try to use the calibration-time board in the models''')
    parser.add_argument('--verbose',
                        action = 'store_true',
                        help='''Report details about the solve''')
    parser.add_argument('models',
                        type = str,
                        nargs='*',
                        help='''Camera models for the optical calibration. Only
                        the intrinsics are used. The number of models given must
                        match the number of camera --topics arguments EXACTLY''')

    args = parser.parse_args()

    import glob
    import re
    args.bag = args.bag.rstrip('/') # to not confuse os.path.splitext()
    bags = sorted(glob.glob(args.bag))

    if len(bags) == 0:
        print(f"No bags matched the glob '{args.bag}'",
              file=sys.stderr)
        sys.exit(1)

    if args.exclude_bag is not None:
        for ex in args.exclude_bag:
            bags = [b for b in bags if not re.search(ex, b)]

    args.bag = bags
    if len(bags) == 0:
        print(f"After applying --exclude-bag, no bags matched the glob '{args.bag}'",
              file=sys.stderr)
        sys.exit(1)

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

import clc
lidar_segmentation_parameters = clc.lidar_segmentation_parameters()

# ingest the lidar segmentation parameters from the arguments
ctx = dict()
for k in lidar_segmentation_parameters.keys():
    # Here I have a --dump that's different from the 'dump' in the
    # segmentation parameters. I ignore the latter for now
    if k == 'dump': continue
    ctx[k] = getattr(args, k)


if args.rt_vehicle_lidar0 is not None:
    args.rt_vehicle_lidar0 = np.array(args.rt_vehicle_lidar0)



if len(args.models) > 0:
    def open_model(f):
        try: return mrcal.cameramodel(f)
        except:
            print(f"Couldn't open '{f}' as a camera model",
                  file=sys.stderr)
            sys.exit(1)
    models = [open_model(f) for f in args.models]



    if all( x is not None for x in (args.object_width_n,
                                    args.object_height_n,
                                    args.object_spacing)):
        # The user gave us the board parameters
        ctx['object_width_n']  = args.object_width_n
        ctx['object_height_n'] = args.object_height_n
        ctx['object_spacing']  = args.object_spacing

    else:
        # The user did NOT give us the board parameters. Get them from the
        # calibration-time board. They all must match

        def hw_from_model(m):
            return m.optimization_inputs()['observations_board'].shape[1:3]

        ctx['object_height_n'],ctx['object_width_n']  = hw_from_model(models[0])
        ctx['object_spacing']  = models[0].optimization_inputs()['calibration_object_spacing']

        for m in models[1:]:
            hw = hw_from_model(m)
            if ctx['object_height_n'] != hw[0] or \
               ctx['object_width_n']  != hw[1] or \
               ctx['object_spacing']  != m.optimization_inputs()['calibration_object_spacing']:
                print("The calibration-time boards used for the cameras do NOT match: pass --object-width-n and --object-height-n and --object-spacing",
                  file = sys.stderr)
                sys.exit(1)

        if args.object_width_n  is not None and ctx['object_width_n']  != args.object_width_n:
            print("--object_width_n given, but others aren't, and the calibration-time board doesn't match. Pass --object-width-n AND --object-height-n AND --object-spacing",
                  file=sys.stderr)
            sys.exit(1)
        if args.object_height_n is not None and ctx['object_height_n'] != args.object_height_n:
            print("--object_height_n given, but others aren't, and the calibration-time board doesn't match. Pass --object-width-n AND --object-height-n AND --object-spacing",
                  file=sys.stderr)
            sys.exit(1)
        if args.object_spacing  is not None and ctx['object_spacing']  != args.object_spacing:
            print("--object_spacing given, but others aren't, and the calibration-time board doesn't match. Pass --object-width-n AND --object-height-n AND --object-spacing",
                  file=sys.stderr)
            sys.exit(1)




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

# Reorder the topics by lidar and then camera. This is the sensor order in the
# solve. The downstream tools (show-transformation-uncertainty.py for instance)
# can then use the topic list directly
args.topics = [args.topics[i] for i in \
               result['itopics_lidar'] + result['itopics_camera']]
del result['itopics_lidar']
del result['itopics_camera']
kwargs_calibrate['topics'] = args.topics

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

    filename = f"camera{imodel}-mounted.cameramodel"
    path = f"{D}/{filename}"
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

