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
import argparse_helpers
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
    parser.add_argument('--rt-vehicle-lidar0',
                        type=argparse_helpers.comma_separated_list_of_floats,
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

    args = parser.parse_args()

    import glob
    import re
    args.bag = args.bag.rstrip('/') # to not confuse os.path.splitext()
    bags = sorted(glob.glob(args.bag))

    if args.exclude_bag is not None:
        for ex in args.exclude_bag:
            bags = [b for b in bags if not re.search(ex, b)]

    if len(bags) < 3:
        print(f"--bag '{args.bag}' must match at least 3 files. Instead this matched {len(bags)} files",
              file=sys.stderr)
        sys.exit(1)
    args.bag = bags

    args.topics = args.topics.split(',')

    return args


args = parse_args()



import numpy as np
import numpysane as nps
import gnuplotlib as gp
import io
import pickle
import mrcal
import clc


if args.rt_vehicle_lidar0 is None:
    args.rt_vehicle_lidar0 = mrcal.identity_rt()
else: args.rt_vehicle_lidar0 = np.array(args.rt_vehicle_lidar0)



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
    kwargs_calibration_object = \
        dict(object_spacing  = o['calibration_object_spacing'],
             object_width_n  = W,
             object_height_n = H)
else:
    kwargs_calibration_object = dict()

kwargs_calibrate = dict(bags            = args.bag,
                        topics          = args.topics,
                        models          = args.models,
                        check_gradient  = False,
                        Npoints_per_segment                = 15,
                        threshold_min_Nsegments_in_cluster = 4,
                        **kwargs_calibration_object)
result = clc.calibrate(do_dump_inputs = args.dump is not None,
                       **kwargs_calibrate)


for imodel in range(len(args.models)):
    models[imodel].extrinsics_rt_toref(result['rt_ref_camera'][imodel])
    root,extension = os.path.splitext(args.models[imodel])
    filename = f"{root}-mounted{extension}"
    models[imodel].write(filename)
    print(f"Wrote '{filename}'")

for ilidar,rt_ref_lidar in enumerate(result['rt_ref_lidar']):
    # dummy lidar "cameramodel". The intrinsics are made-up, but the extrinsics
    # are true, and can be visualized with the usual tools
    filename = f"/tmp/lidar{ilidar}-mounted.cameramodel"
    model = mrcal.cameramodel( intrinsics = ('LENSMODEL_PINHOLE',
                                             np.array((1.,1.,0.,0.))),
                               imagersize = (1,1),
                               extrinsics_rt_toref = rt_ref_lidar )
    model.write(filename,
                note = "Intrinsics are made-up and nonsensical")
    print(f"Wrote '{filename}'")



context = \
    dict(result                           = result,
         topics                           = args.topics,
         Nsectors                         = args.Nsectors,
         threshold_valid_lidar_range      = args.threshold_valid_lidar_range,
         threshold_valid_lidar_Npoints    = args.threshold_valid_lidar_Npoints,
         uncertainty_quantification_range = args.uncertainty_quantification_range,
         rt_vehicle_lidar0                = args.rt_vehicle_lidar0,
         kwargs_calibrate                 = kwargs_calibrate)
if args.dump is not None:
    with open(args.dump, 'wb') as f:
        pickle.dump( context,
                     f )
    print(f"Wrote '{args.dump}'")




statistics = clc.post_solve_statistics(bag                              = args.bag[0],
                                       topics                           = args.topics,
                                       Nsectors                         = args.Nsectors,
                                       threshold_valid_lidar_range      = args.threshold_valid_lidar_range,
                                       threshold_valid_lidar_Npoints    = args.threshold_valid_lidar_Npoints,
                                       uncertainty_quantification_range = args.uncertainty_quantification_range,
                                       rt_vehicle_lidar0                = args.rt_vehicle_lidar0,
                                       models                           = args.models,
                                       **result)




data_tuples_sensor_forward_vectors = \
    clc.get_data_tuples_sensor_forward_vectors(context['result']['rt_ref_lidar' ],
                                               context['result']['rt_ref_camera'],
                                               context['topics'])



# shape (Nsensors, Nsectors)
isvisible_per_sensor_per_sector = statistics['isvisible_per_sensor_per_sector']

Nsensors = isvisible_per_sensor_per_sector.shape[0]
dth = np.pi*2./args.Nsectors
th = np.arange(args.Nsectors)*dth + dth/2.
plotradius = nps.transpose(np.arange(Nsensors) + 10)
ones = np.ones( (args.Nsectors,) )

filename = '/tmp/observability.pdf'
gp.plot( (th,                 # angle
          plotradius*ones,    # radius
          ones*dth*0.9,       # angular width of slice
          ones*0.9,           # depth of slice
          isvisible_per_sensor_per_sector,
          dict(_with = 'sectors palette fill solid',
               tuplesize = 5)),

         *data_tuples_sensor_forward_vectors,

         _xrange = (-10-Nsensors,10+Nsensors),
         _yrange = (-10-Nsensors,10+Nsensors),
         square = True,
         unset = 'colorbox',
         title = 'Observability map of each sensor',
         hardcopy = filename,
        )
print(f"Wrote '{filename}'")

stdev_worst = statistics['stdev_worst']
i = stdev_worst != 0
filename = '/tmp/uncertainty.pdf'
gp.plot( (th[i],                 # angle
          10.*ones[i],           # radius
          ones[i]*dth*0.9,       # angular width of slice
          ones[i]*0.9,           # depth of slice
          stdev_worst[i],
          dict(tuplesize = 5,
               _with = 'sectors palette fill solid')),
         *data_tuples_sensor_forward_vectors,
         _xrange = (-11,11),
         _yrange = (-11,11),
         square = True,
         title = 'Worst-case uncertainty. Put the board in high-uncertainty regions',
         hardcopy = filename,
        )
print(f"Wrote '{filename}'")
