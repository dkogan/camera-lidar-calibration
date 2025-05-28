#!/usr/bin/python3

import sys
import numpy as np
import numpysane as nps
import os

import _clc
import mrcal
import bag_interface

def lidar_segmentation(*,
                       bag, lidar_topic,
                       # used if bag,lidar_topic are None
                       points = None,
                       rings  = None,
                       # if integer: s since epoch or ns since epoch
                       # if float:   s since epoch
                       # if str:     try to parse as an integer or float OR with dateutil.parser.parse()
                       start  = None,
                       **kwargs):

    if bag is not None:
        if not os.path.exists(bag):
            raise Exception(f"Bag path '{bag}' does not exist")

        array = next(bag_interface.messages(bag, (lidar_topic,),
                                            start = start))['array']

        points = array['xyz']
        rings  = array['ring']

    ipoint, plane_pn = \
        _clc.lidar_segmentation(points = points,
                                rings  = rings,
                                **kwargs)
    Nplanes = len(ipoint)

    return \
        dict( points  = [points[ipoint[i]] for i in range(Nplanes)],
              plane_p = plane_pn[:,:3],
              plane_n = plane_pn[:,3:] )


def _lidar_points(msg):
    if msg is None: return None
    array = msg['array']
    return \
        (array['xyz' ],
         array['ring'])
def _images(msg):
    if msg is None: return None
    return msg['array']


def _is_message_pointcloud(msg):

    if msg is None:
        # I don't know if this was supposed to be a point cloud. I arbitrarily
        # say "no"
        return False

    dtype_names = msg['array'].dtype.names

    if dtype_names is None:
        return False

    return 'xyz' in dtype_names


def _sorted_sensor_snapshots(bags, topics,
                             *,
                             decimation_period = None,
                             start             = None,
                             stop              = None,
                             max_time_spread_s = None,
                             exclude_time_periods = [],
                             verbose           = False):
    for bag in bags:
        if not os.path.exists(bag):
            raise Exception(f"Bag path '{bag}' does not exist")

    if decimation_period is None:
        # Each bag is a snapshot in time. We take the first set of messages (one
        # per topic) from each bag
        messages_bags = \
            [bag_interface.first_message_from_each_topic(bag, topics,
                                                         start = start,
                                                         stop  = stop,
                                                         max_time_spread_s = max_time_spread_s,
                                                         verbose = verbose) \
             for bag in bags]

        # each snapshot in messages_bags maybe None (if no usable data in a bag
        # at all) or [x,x,x,x] where each x is the data for each topic. Too many
        # of the x may be None
        require_at_least_N_topics = 2
        def is_snapshot_usable(msgs):
            if msgs is None: return False
            Nstored = sum(0 if msg is None else 1 for msg in msgs)
            return Nstored >= require_at_least_N_topics

        mask          = [is_snapshot_usable(m) for m in messages_bags]
        messages_bags = [m for i,m in enumerate(messages_bags) \
                         if mask[i]]
        bags_selected = [b for i,b in enumerate(bags) \
                         if mask[i]]
        if verbose:
            for isnapshot,bag in enumerate(bags_selected):
                print(f"{isnapshot=}: {bag=}")
    else:
        # We have one long bag. I look at each time segment decimation_period
        # long, and take the first set of messages (one per topic) from such
        # segment
        if len(bags) != 1:
            raise Exception("decimation_period is not None, so I expect exactly one bag to have been given")
        bag = bags[0]
        # slurp an iterator into a list. This wastes memory, but saves me some
        # coding time today
        messages_bags = \
            list( bag_interface. \
                  first_message_from_each_topic_in_time_segments(bag, topics,
                                                                 start    = start,
                                                                 stop     = stop,
                                                                 period_s = decimation_period,
                                                                 require_at_least_N_topics = 2,
                                                                 max_time_spread_s = max_time_spread_s,
                                                                 exclude_time_periods = exclude_time_periods,
                                                                 verbose = verbose) )

    # I need to figure out which topic corresponds to a lidar and which to a
    # camera. I can get this information from the data, but if any bag is
    # missing any particular topic, I cannot figure this out from that bag.
    itopics_lidar  = set()
    itopics_camera = set()

    Ntopics_identified = 0
    for messages in messages_bags:
        if Ntopics_identified == len(topics):
            break

        # messages is a list of length len(topics)
        for i,msg in enumerate(messages):
            if msg is None:
                continue
            if i in itopics_lidar or i in itopics_camera:
                continue
            if _is_message_pointcloud(msg): itopics_lidar .add(i)
            else:                           itopics_camera.add(i)
            Ntopics_identified += 1
    if Ntopics_identified != len(topics):
        itopics_missing = \
            set(range(len(topics))).difference(itopics_lidar).difference(itopics_camera)
        raise Exception(f"Some topics have no data in any of the bags: {[topics[i] for i in itopics_missing]}")

    itopics_lidar  = sorted(itopics_lidar)
    itopics_camera = sorted(itopics_camera)

    return \
        tuple( ( tuple(_lidar_points(messages[i]) for i in itopics_lidar),
                 tuple(_images      (messages[i]) for i in itopics_camera) ) \
               for messages in messages_bags )


def calibrate(*,
              bags, topics,
              decimation_period = None,
              start             = None,
              stop              = None,
              max_time_spread_s = None,
              verbose           = False,
              exclude_time_periods = [],
              **kwargs):

    r'''Re-runs a previously-dumped calibration

SYNOPSIS

    result = clc.calibrate(bags   = bags,
                           topics = topics,
                           ...)

    with open('clc.dump', 'wb') as f:
        f.write(result['inputs_dump'])

    ....

    with open('clc.dump', "rb") as f:
        dump = f.read()

    result = clc.fit_from_inputs_dump(dump)

CLC has a lot of complexity, and it's very useful to be able to rerun previous
computations to find and fix issues. Solve state can be dumped using the
"buf_inputs_dump" argument in the C clc() function, or it can be obtained in the
binary clc.calibrate(....)['inputs_dump'] in the Python API.

The dump can be loaded and re-solved by calling clc_fit_from_inputs_dump() in
the C API or clc.fit_from_inputs_dump() in the Python API.

This dump/replay infrastructure exercises the solve only. The LIDAR point
segmentation is NOT a part of this tooling: the dump contains already-segmented
points.

These functions load the previous solve state, and re-run the fit from it. Some
small tweaks to the solve are available:

- The do_fit_seed argument allows the seed solve to be re-run, or the dumped
  seed output to be used instead

- The do_inject_noise argument allows noise to be added to the input. This is
  useful for validation of the uncertainty-quantification routine

These two arguments work together. The full logic:

  if(!do_fit_seed && !do_inject_noise) {
    fit from the previous fit_seed() result; do NOT fit_seed()
    Useful to experiment with the fit() routine.
  }

  if(!do_fit_seed &&  do_inject_noise) {
    fit from the previous fit() result; do NOT fit_seed()

    Useful to test the uncertainty logic. Many noise samples will be taken, with
    a separate fit() for each. Fitting from the dumped fit() result makes each
    solve converge very quickly, since we will start very close to the optimum
  }

  if(do_fit_seed) {
    fit_seed() && fit()
    Useful to experiment with fit_seed() and the full fit pipeline
  }

The other arguments are described below

ARGUMENTS

- inputs_dump: a Python "bytes" object, containing the binary dump of the solve
  inputs. This comes from either a .pickle file from "fit.py --dump" or from the
  buf_inputs_dump argument to the clc_...() C functions

- isnapshot_exclude: optional iterable of integers. Each snapshot in this list
  will NOT be a part of the solve

- fit_seed_position_err_threshold
- fit_seed_cos_angle_err_threshold

  Optional values to customize the behavior of fit_seed()

- do_inject_noise: optional boolean, defaulting to False. If True, we inject
  some expected noise into the inputs. See above for details.

- do_fit_seed: optional boolean, defaulting to False. If True, we re-run
  fit_seed(). By default we use the dumped result of fit_seed() instead. See
  above for details.

- verbose: optional boolean, defaulting to False. If True, verbose output about
  the solve is produced on stdout

- do_skip_plots: optional boolean, defaulting to True. If True, we do NOT
  produce plots of the results

RETURNED VALUE

A dict describing the result. The items are:

- rt_lidar0_lidar:      A (Nlidars,6) numpy array containing rt transforms
                        mapping points in each lidar frame to the frame of
                        lidar0

- rt_lidar0_camera:     A (Ncameras,6) numpy array containing rt transforms
                        mapping points in each camera frame to the frame of
                        lidar0

- Var_rt_lidar0_sensor: A (Nsensors_optimized, 6, Nsensors_optimized, 6)
                        symmetric numpy array representing the 1-sigma
                        uncertainty of the solution due to the expected noise in
                        the inputs. Nsensors_optimized counts the number of
                        sensors in the optimization problem: Nsensors_optimized
                        = Nlidars + Ncameras - 1. Lidar0 is always at the
                        reference frame, and thus is not a part of the
                        optimization.

'''

    return _clc.calibrate( _sorted_sensor_snapshots(bags, topics,
                                                    decimation_period = decimation_period,
                                                    start             = start,
                                                    stop              = stop,
                                                    max_time_spread_s = max_time_spread_s,
                                                    exclude_time_periods = exclude_time_periods,
                                                    verbose           = verbose),
                           verbose = verbose,
                           **kwargs)




def color_sequence_rgb():
    r'''Returns the default color sequence for gnuplot objects

SYNOPSIS

    clc.color_sequence_rgb()
    ===> ('#9400d3', '#009e73', '#56b4e9', '#e69f00', '#f0e442', '#0072b2', '#e51e10', '#000000')


    import gnuplotlib as gp
    colors = clc.color_sequence_rgb()
    gp.plot( equation = ( 'x   notitle axis x1y1',
                          'x*x notitle axis x1y2'),
             _set     = (f'ytics  textcolor rgb "{colors[0]}"',
                         f'ylabel "line" textcolor "{colors[0]}"',
                         f'y2tics textcolor rgb "{colors[1]}"',
                         f'y2label "parabola" textcolor "{colors[1]}"',
                         ),
             wait = 1)

    [ plot pops up showing a line against the left y axis and a parabola against  ]
    [ the right y axis; the color of the line matches the left-y-axis labels; the ]
    [ the color of the parabola matches the right-y-axis labels                   ]

Gnuplot uses a specific sequence of colors for each data object being plotted.
This function returns this sequence in an iterable, each color represented by a
'#RRGGBB' string. This is useful if we want to match the color of the plotted
data to the color of some other objects in the plot

This is obtained by running

  $ gnuplot -e 'show linetype'"

Any named colors reported here are defined in gnuplot/src/tables.c.

ARGUMENTS

None

RETURNED VALUE

An iterable of strings for each color, in order.

    '''

    return ("#9400d3",
            "#009e73",
            "#56b4e9",
            "#e69f00",
            "#f0e442",
            "#0072b2",
            "#e51e10",
            "#000000")


def plot(*args,
         hardcopy = None,
         **kwargs):
    r'''Wrapper for gnuplotlib.plot(), reporting the hardcopy output to the console

SYNOPSIS

    clc.plot(....,
             hardcopy = '/tmp/plot.svg')

    ===> Wrote '/tmp/plot.svg'

This is a thin wrapper to gnuplotlib.plot(). If we're writing a hardcopy,
clc.plot() will report the hardcopy name to the console.

ARGUMENTS

All args and kwargs forwarded to gnuplotlib.plot() unmodified

RETURNED VALUE

Whatever gnuplotlib.plot() returned

    '''

    import gnuplotlib as gp
    gp.plot(*args, **kwargs,
            hardcopy = hardcopy)
    if hardcopy is not None:
        print(f"Wrote '{hardcopy}'")


def pointcloud_plot_tuples(bag, lidar_topics,
                           rt_lidar0_lidar,
                           *,
                           threshold_range     = None,
                           isensor_from_itopic = None,
                           Rt_vehicle_lidar0   = None,
                           start               = None):

    r'''Helper function for visualizing LIDAR data

SYNOPSIS

    import gnuplotlib as gp

    data_tuples = \
        clc.pointcloud_plot_tuples(bag, lidar_topics,
                                   rt_lidar0_lidar,
                                   threshold_range = threshold_range)

    plotkwargs = dict(....)

    # Can add to the data_tuples here, to plot more stuff

    gp.plot(*data_tuples, **plotkwargs)

CLC usually keeps track of the raw LIDAR data, the geometric transforms between
the LIDARs and between the base LIDAR and the vehicle. This function contains
all the common logic applying the various transforms to the raw data, to make
them available for plotting.

ARGUMENTS

- bag: a string containing a path to the rosbag with the data we're plotting.
  Uses the "rosbags" library to interpret this, so generally ROS1 and ROS2 are
  supported

- lidar_topics: an iterable of strings for the ROSs topics we're visualizing

- rt_lidar0_lidar: a (Nlidars,6) numpy array. Each row i is a (6,) array
  containing an rt transform FROM the coord system of LIDAR i to the coord
  system of LIDAR 0

- threshold_range: optional distance, in m. If given, all points further from
  the sensor than this threshold are culled in the visualization

- isensor_from_itopic: optional map to/from integers. If given, associates topic
  lidar_topics[itopic] with transform rt_lidar0_lidar[
  isensor_from_itopic[ilidar] ]. If omitted or None, an identity mapping is
  assumed: lidar_topics[itopic] goes with rt_lidar0_lidar[itopic]

- Rt_vehicle_lidar0: optional (4,3) numpy array (an Rt transform), mapping
  points in the lidar0 coord system to the vehicle coord system.

- start: optional timestamp. If given, we visualize the first message for each
  topic in the bag AFTER the given timestamp. If omitted, we take the first
  message.

RETURNED VALUE

Plot tuples passable to gnuplotlib.plot() as in the SYNOPSIS above.

    '''
    try:
        pointcloud_msgs = \
            [ next(bag_interface.messages(bag, (topic,),
                                          start = start)) \
              for topic in lidar_topics ]
    except:
        raise Exception(f"Bag '{bag}' doesn't have at least one message for each of {lidar_topics} in the requested time span")

    for i,msg in enumerate(pointcloud_msgs):
        if not _is_message_pointcloud(msg):
            raise Exception(f"Topic {lidar_topics[i]} is not a pointcloud type")

    # Package into a numpy array
    pointclouds = [ msg['array']['xyz'].astype(float) \
                    for msg in pointcloud_msgs ]

    # Throw out everything that's too far, in the LIDAR's own frame
    if threshold_range is not None:
        pointclouds = [ p[ nps.mag(p) < threshold_range ] for p in pointclouds ]

    if isensor_from_itopic is not None:
        pointclouds = \
            [ mrcal.transform_point_rt(rt_lidar0_lidar[ isensor_from_itopic[i] ],
                                       p) for i,p in enumerate(pointclouds) ]
    else:
        pointclouds = \
            [ mrcal.transform_point_rt(rt_lidar0_lidar[i],
                                       p) for i,p in enumerate(pointclouds) ]

    if Rt_vehicle_lidar0 is not None:
        pointclouds = [mrcal.transform_point_Rt(Rt_vehicle_lidar0, p) \
                       for p in pointclouds]

    data_tuples = [ ( p, dict( tuplesize = -3,
                               legend    = lidar_topics[i],
                               _with     = f'points pt 7 ps 1 lc rgb "{color_sequence_rgb()[i%len(color_sequence_rgb())]}"')) \
                    for i,p in enumerate(pointclouds) ]

    return data_tuples


def sensor_forward_vectors_plot_tuples(rt_ref_lidar,
                                       rt_ref_camer1a,
                                       topics,
                                       *,
                                       isensor = None):

    r'''Helper function for visualizing sensor poses

SYNOPSIS

    import gnuplotlib as gp

    result = clc.calibrate(...)

    rt_ref_lidar  = result['rt_lidar0_lidar']
    rt_ref_camera = result['rt_lidar0_camera']

    sensor_forward_vectors_plot_tuples = \
        clc.sensor_forward_vectors_plot_tuples(rt_ref_lidar,
                                               rt_ref_camera,
                                               topics)

    gp.plot( *sensor_forward_vectors_plot_tuples,
             ... )

This function provides the gnuplotlib directives to display the "forward"
direction at the sensor location of all the sensors in a 2D plot in the
horizontal xy plane of the lidar0 frame. Each sensor is labelled. If given, the
sensor "isensor" is highlighted in red.

ARGUMENTS


- rt_ref_lidar: a (Nlidars,6) numpy array. Each row i is a (6,) array containing
  an rt transform FROM the coord system of LIDAR i to the reference coord system
  (lidar0)

- rt_ref_camera: a (Ncameras,6) numpy array. Each row i is a (6,) array
  containing an rt transform FROM the coord system of CAMERA i to the reference
  coord system (lidar0)

- topics: a list of strings, used to label each sensor. The number of strings
  must match the number of sensors: len(rt_ref_lidar) + len(rt_ref_camera)

- isensor: optional integer, indicating which sensor should be highlighted. If
  given, this sensor's direction and label are shown in red

RETURNED VALUE

The gnuplotlib tuples, passable to gnuplotlib.plot()
    '''

    if len(rt_ref_lidar)+len(rt_ref_camera) != len(topics):
        raise Exception("Mismatched transform/topic counts")

    # These apply to ALL the sensors, not just the ones being requested
    lidars_origin   = rt_ref_lidar [:,3:]
    cameras_origin  = rt_ref_camera[:,3:]
    lidars_forward  = mrcal.rotate_point_r(rt_ref_lidar [:,:3], np.array((1.,0,0 )))
    cameras_forward = mrcal.rotate_point_r(rt_ref_camera[:,:3], np.array(( 0,0,1.)))

    sensors_origin  = nps.glue(lidars_origin,  cameras_origin,  axis=-2)
    sensors_forward = nps.glue(lidars_forward, cameras_forward, axis=-2)

    sensors_forward_xy = np.array(sensors_forward[...,:2])
    # to avoid /0 for straight-up vectors
    mag_sensors_forward_xy = nps.mag(sensors_forward_xy)
    i = mag_sensors_forward_xy>0
    sensors_forward_xy[i,:] /= nps.dummy(mag_sensors_forward_xy[i], axis=-1)
    sensors_forward_xy[~i,:] = 0
    sensor_forward_arrow_length = 4.

    with_labels = np.array(['labels textcolor "black"'] * len(sensors_origin))
    if isensor is not None:
        with_labels[isensor] = 'labels textcolor "red"'

    return \
        (
          # sensor positions AND their forward vectors
          (nps.glue( sensors_origin [...,:2],
                     sensors_forward_xy * sensor_forward_arrow_length,
                     axis = -1 ),
           dict(_with = 'vectors lw 2 lc "black"',
                tuplesize = -4) ),

          ( nps.dummy(sensors_origin[...,0], -1),
            nps.dummy(sensors_origin[...,1], -1),
            nps.dummy(np.array(topics),-1),
            dict(_with = with_labels,
                 tuplesize = 3)),
         )




lidar_segmentation_default_context = _clc.lidar_segmentation_default_context
lidar_segmentation_parameters      = _clc.lidar_segmentation_parameters
fit_from_inputs_dump               = _clc.fit_from_inputs_dump
