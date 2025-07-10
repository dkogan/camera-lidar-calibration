#!/usr/bin/python3

import sys
import numpy as np
import numpysane as nps
import os

import clc._clc as _clc
import mrcal
import clc.bag_interface

def lidar_segmentation(*,
                       bag, lidar_topic,
                       # used if bag,lidar_topic are None
                       points = None,
                       rings  = None,
                       # if integer: s since epoch or ns since epoch
                       # if float:   s since epoch
                       # if str:     try to parse as an integer or float OR with dateutil.parser.parse()
                       start  = None,
                       bag_context = None,
                       **lidar_segmentation_parameters):

    r'''Find the calibration plane in a LIDAR point cloud

SYNOPSIS

    import gnuplotlib as gp

    result = clc.lidar_segmentation(bag = BAG,
                                    lidar_topic = TOPIC,
                                    ...)

    gp.plot( result['points'],
             tuplesize = -3,
             _with     = 'points',
             _3d       = True,
             square    = True )

    # a 3D plot pops up showing the points in the calibration object

LIDAR point cloud segmentation is a significant part of the whole clc pipeline.
This function allows this component to be exercised in isolation, all by itself,
to make it possible to effectively test it.

The input to the segmentation routine is a point cloud resulting from one full
rotation of the LIDAR. The LIDAR unit contains some number of lasers rigidly
mounted onto a spindle, rotating around the z axis of the LIDAR coordinate
system. Thus for each laser, the elevation = atan( z, mag(x,y) ) will be
constant for all points produced by that laser. Furthermore, the spacing of az =
atan2(y,x) is generally even and consistent for all lasers.

In the LIDAR data ingested by clc, each laser is identied by a single integer
"ring". The "ring" values are sequential, from 0 to Nrings-1.

This lidar_segmentation() function can ingest either

- A "bag" of data and a "lidar_topic". The first full 360deg set of points is
  pulled out of this bag, and used for the segmentation

- Numpy arrays "points" and "rings". Both of these have length = Npoints. These
  are assumed to represent a 360deg set of points, and are used for the
  segmentation

Whether the points are read from the "bag" or the "poitns" array, invalid points
will have p = (0,0,0), and will be ignored

ARGUMENTS

- bag: optional string pointing to the bag we're reading. If given and not None,
  the the "lidar_topic" must also be given and not None, while "rings" and
  "points" MUST be omitted or None

- lidar_topic: optional string identifying the specific sensor we're reading.
  Required and used if "bag" is also given. MUST be omitted or None if "points"
  and "rings" are given

- "points": optional numpy array of shape (Npoints,3). If given, this is the
  point cloud we're segmenting. If given, the "rings" must be given also, while
  "bag" and "lidar_topic" must not be.

- "rings" optional numpy array of shape (Npoints,); identifies the laser that
  captured each corresponding point in "points". If given, "points" is required,
  while "bag" and "lidar_topic" must be omitted or None

- start: optional timestamp, indicating the start point in the bag. Used only if
  "bag" is given and not None. This timestamp is compared against
  msg['time_ns']: the time when the message was recorded, NOT when the sensor
  produced it. This timestamp is interpreted differently, based on its type. If
  it's an integer: we interpret it as "sec since the epoch" or "nanosec since
  epoch", depending on the numerical value. If it's a float: "sec since epoch".
  If a string: we parse it as an integer or float OR as a freeform string using
  dateutil.parser.parse().

- bag_context: optional iterable of bag paths. The rosbags that contain lidar
  data for OTHER snapshots in this data collection session. Used to detect
  static objects in the scene, and to throw them away before segmenting the
  board

- **lidar_segmentation_parameters: optional parameters controlling the operation
  of the segmentation routine. Any parameters that are omitted will take their
  default values. The available parameters, a description of their operation and
  their default values are given in clc.h in the
  CLC_LIDAR_SEGMENTATION_LIST_CONTEXT macro.

RETURNED VALUE

A dict describing the result. The key/values are:

- "points": a list of length Nplanes. Each element is a numpy array of shape
  (Npoints[iplane],3). These are the points determined to lie on a plane
  suitable for calibrating

- "plane_p": a numpy array of shape (Nplane,3). For each plane, contains a point
  lying somewhere on the plane

- "plane_n": a numpy array of shape (Nplane,3). For each plane, contains a unit
  normal vector FROM the LIDAR unit TO the plane.

This dict is returned even if no planes were found: the same results will be
returned, with Nplanes=0. If an error occurred during the computation, and
exception will be raised.

    '''

    if bag is not None:
        if not os.path.exists(bag):
            raise Exception(f"Bag path '{bag}' does not exist")

        try:
            array = next(clc.bag_interface.messages(bag, (lidar_topic,),
                                                    start = start))['array']
        except StopIteration:
            raise Exception(f"{bag=} does not contain {lidar_topic=} past {start=}")

        if bag_context is not None:
            range_mode = \
                lidar_scene_range_mode(bag_context,
                                       start       = start,
                                       lidar_topic = lidar_topic)
        else:
            range_mode = None

        points = mask_out_static_scene(array['xyz'], array['ring'],
                                       range_mode = range_mode)
        rings  = array['ring']


    ipoint, plane_pn = \
        _clc.lidar_segmentation(points = points,
                                rings  = rings,
                                **lidar_segmentation_parameters)
    Nplanes = len(ipoint)

    return \
        dict( points  = [points[ipoint[i]] for i in range(Nplanes)],
              plane_p = plane_pn[:,:3],
              plane_n = plane_pn[:,3:] )






static_mask_quantum = .2

def iaz_from_point(points, Npoints_per_rotation):
    iaz = \
        np.round( (np.pi + np.arctan2(points[:,1],
                                      points[:,0])) / (2.*np.pi) * Npoints_per_rotation ).astype(int)
    iaz[ iaz < 0                     ] = 0
    iaz[ iaz >= Npoints_per_rotation ] = 0
    return iaz


def lidar_scene_range_mode(bags,
                           start,
                           lidar_topic):

    if start is not None:
        raise Exception("This only works with one-snapshot-per-bag for now")

    # Hard-coded here for now. Computed for real in clc_lidar_preprocess()
    Npoints_per_rotation = 2048
    Nrings               = 128 # upper bound
    Nsnapshots           = len(bags)

    ranges = np.zeros( (Nrings,Npoints_per_rotation,  Nsnapshots),
                       dtype = np.float32 )

    for isnapshot,bag in enumerate(bags):
        array = next(clc.bag_interface.messages(bag, (lidar_topic,),
                                                start = start))['array']

        points = array['xyz']
        rings  = array['ring']

        r      = nps.mag(points)
        ivalid = r > 0

        points = points[ivalid]
        rings  = rings [ivalid]
        r      = r     [ivalid]

        iaz = iaz_from_point(points, Npoints_per_rotation)

        # reshape and reindex an array of shape (Nrings*Npoints_per_rotation)
        ranges[array['ring'],iaz,isnapshot] = r

    # I now have ranges. I compute the non-zero mode range per pixel. Any
    # uncertain mode is reported as 0
    # shape (Nrings,Npoints_per_rotation)
    return \
        _clc._mode_over_lastdim_ignoring0(ranges,
                                          quantum                  = static_mask_quantum,
                                          report_mode_if_N_atleast = Nsnapshots//2)


def mask_out_static_scene(# shape (Npoints,3)
                          points,
                          # shape (Npoints,)
                          rings,
                          *,
                          # shape (Nrings,Npoints_per_rotation)
                          range_mode = None):
    if range_mode is None:
        return points

    # Areas with uncertain range_mode are given as 0. We treat this as valid
    # anyway: at worst we'll ignore points very close to the lidar. That is ok
    Npoints_per_rotation = range_mode.shape[-1]

    # shape (Npoints,)
    iaz = iaz_from_point(points, Npoints_per_rotation)
    r   = nps.mag(points)

    range_err = r - range_mode.ravel()[ rings*Npoints_per_rotation + iaz]

    istatic = np.abs(range_err) < static_mask_quantum*2

    points[istatic] *= 0
    return points



def _lidar_points(msg,
                  *,
                  range_mode = None):
    if msg is None: return None
    array = msg['array']
    return \
        (mask_out_static_scene(array['xyz' ], array['ring'],
                               range_mode = range_mode),
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
                             decimation_period_s  = None,
                             start                = None,
                             stop                 = None,
                             max_time_spread_s    = None,
                             exclude_time_periods = [],
                             verbose              = False):
    for bag in bags:
        if not os.path.exists(bag):
            raise Exception(f"Bag path '{bag}' does not exist")

    if decimation_period_s is None:
        # Each bag is a snapshot in time. We take the first set of messages (one
        # per topic) from each bag
        messages_bags = \
            [clc.bag_interface.first_message_from_each_topic(bag, topics,
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
        # We have one long bag. I look at each time segment decimation_period_s
        # long, and take the first set of messages (one per topic) from such
        # segment
        if len(bags) != 1:
            raise Exception("decimation_period_s is not None, so I expect exactly one bag to have been given")
        bag = bags[0]
        # slurp an iterator into a list. This wastes memory, but saves me some
        # coding time today
        messages_bags = \
            list( clc.bag_interface. \
                  first_message_from_each_topic_in_time_segments(bag, topics,
                                                                 start    = start,
                                                                 stop     = stop,
                                                                 period_s = decimation_period_s,
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

    if verbose:
        Nlidars = len(itopics_lidar)
        for i,itopic in enumerate(itopics_lidar):
            print(f"LIDAR  {i} (sensor {i}): {topics[itopic]}")
        for i,itopic in enumerate(itopics_camera):
            print(f"camera {i} (sensor {i+Nlidars}): {topics[itopic]}")

    range_mode = \
        [ lidar_scene_range_mode(bags,
                                 start       = start,
                                 lidar_topic = topics[itopic_lidar]) \
          for itopic_lidar in itopics_lidar ]


    return \
        ( tuple( ( tuple(_lidar_points(messages[i], range_mode = range_mode[ilidar]) for ilidar,i in enumerate(itopics_lidar)),
                   tuple(_images      (messages[i]) for i in itopics_camera) ) \
                 for messages in messages_bags ),
          itopics_lidar,
          itopics_camera )



def calibrate(*,
              bags, topics,
              decimation_period_s  = None,
              start                = None,
              stop                 = None,
              max_time_spread_s    = None,
              verbose              = False,
              exclude_time_periods = [],
              **kwargs):

    r'''Invoke the full clc calibration routine

SYNOPSIS

    result = clc.calibrate(bags                = BAGS,
                           topics              = TOPICS,
                           decimation_period_s = 2,
                           max_time_spread_s   = 0.1)

    mrcal.show_geometry( nps.glue( mrcal.invert_rt(result['rt_lidar0_lidar']),
                                   mrcal.invert_rt(result['rt_lidar0_camera']),
                                   axis = -2 ),
                         cameranames = TOPICS )

    # A plot pops up, showing the solved geometry

This function runs the full clc pipeline, including the LIDAR point
segmentation, the chessboard detection in the camera images, and the joint
solve.

The input comes from "rosbags", with the README describing the input data
details:

  A rosbag may contain multiple data streams with a "topic" string identifying
  each one. The data stream for any given topic is a series of messages of
  identical data type. clc reads lidar scans (msgtype
  'sensor_msgs/msg/PointCloud2') and images (msgtype 'sensor_msgs/msg/Image').

  We want to get a set of time-synchronized "snapshots" from the data, reporting
  observations of a moving calibration object by a set of stationary sensors.
  Each snapshot should report observations from a single instant in time.

  There are two ways to capture such data:

  - Move the chessboard between stationary poses; capture a small rosbag from
    each sensor at each stationary pose. Each bag provides one snapshot. This
    works well, but takes more work from the people capturing the data.
    Therefore, most people prefer the next method

  - Move the chessboard; slowly. Continuously capture the data into a single
    bag. Subdivide the bag into time periods of length =decimation_period_s=.
    Each decimation period produces one snapshot. This method has risks of
    motion blur and synchronization issues, so the motions need to be slow, and
    the tooling needs to enforce tight timings, and it is highly desireable to
    have an outlier rejection method.

  The tooling supports both methods. The functions and tools that accept a
  "decimation period" will use the one-snapshot-per-bag scheme if the decimation
  period is omitted, and the one-big-bag scheme if the decimation period is
  given.

We can more finely control what data we use by setting the start, stop and
exclude_time_periods arguments (see the docstring for
clc.bag_interface.first_message_from_each_topic_in_time_segments() for details).

ARGUMENTS

A number of the arguments control the gathering of the input data, and are
specific to the clc Python API

- bags: an iterable of strings pointing to paths on disk of the bag(s) with the
  input data. If all the input is in one bag (indicated by decimation_period_s
  is None), then this should be a length-1 iterable

- topics: an iterable of strings, for the sensors being calibrated. These topics
  should contain LIDAR point clouds or camera images.

- decimation_period_s: optional decimation period size. If omitted or None, we
  use one bag per snapshot. If given, we subdivide the time in the ONE bag we're
  given, to produce a snapshot from each period of the given size.

- start, stop: optional timestamps indicating where we should start/stop reading
  the bag. Given as s since the epoch, ns since the epoch or a string parsed by
  dateutil.parser.parse(). By default, we start at the beginning of the bag, and
  finish at the end.

- max_time_spread_s: optional limit for the sync difference between all the
  messages in a snapshot. We want all the data in a snapshot to have been
  captured at one instant in time. This isn't possible, so we settle for the
  time spread to be better than some limit, given here. By default, we don't
  enforce this at all, and simply take the first set of messages in each
  bag/decimation period

- exclude_time_periods: optional iterable of time periods to exclude. Each
  element of the iterable is a (t0,t1) time intervals. Where each of the
  timestamps is given as s since the epoch, ns since the epoch or as a string
  parsed by dateutil.parser.parse().

The rest of the arguments control the operation of clc, and are directly passed
to the CLC C API

- models: an iterable of filenames pointing to the mrcal cameramodels
  representing the cameras. Only the intrinsics are used. Exactly as many models
  must be given as there are camera topics given. Omit if we have no cameras
  (this is a LIDAR-only solve)

- Nsectors
- threshold_valid_lidar_range
- threshold_valid_lidar_Npoints
- uncertainty_quantification_range
  Parameters for sector-based diagnostics. All are optional. See the README

- seed: optional dict representing the solution used to initialize the solve.
  Useful for difficult solves where the internal seeding algorithm is
  insufficient. Normally this would be omitted, and clc will compute the seed by
  itself. If given, must have keys ("rt_lidar0_lidar","rt_lidar0_camera") or
  ("rt_vehicle_lidar","rt_vehicle_camera") to localize the geometry. The camera
  geometry is optional if we have no cameras. The vehicle-referenced geometry
  requires rt_vehicle_lidar0 to be given. The output of this function is a dict
  that can be passed back into the seed

- object_height_n
- object_width_n
- object_spacing
  optional parameters describing the chessboard used for the camera calibration.
  These are required if we have cameras

- fit_seed_position_err_threshold
- fit_seed_cos_angle_err_threshold
  Optional validation parameters used by the internal seeding routine. If the
  seed computes geometry whose translation or orientation errors are beyond
  these bounds, that snapshot is discarded. Reasonable defaults are used if
  omitted.

- rt_vehicle_lidar0: Optional transform: a numpy array of shape (6,). If given,
  we use this to align the sensor suite to the vehicle. We then produce more
  output using the vehicle frame. If omitted, we don't know where the vehicle
  is, and we don't produce the extra diagnostics

- isnapshot_for_isvisible: optional integer identifying which snapshot to use
  for visibility reporting. By default we use the first one

- check_gradient: optional boolean, defaulting to False. If True, we run the
  gradient checker instead of the full solve

- check_gradient__use_distance_to_plane: optional boolean, defaulting to False.
  If True, we run the gradient checker of the distance-to-plane LIDAR error
  metric instead of the full solve

- do_dump_inputs: optional boolean, defaulting to False. If True, we produce and
  return a binary dump that can be used to replay this solve with
  fit_from_inputs_dump(). Turning this on produces ADDITIONAL output, and does
  not affect the other results

- verbose: optional boolean, defaulting to False. If True, verbose output about
  the solve is produced on stdout

RETURNED VALUE

A dict describing the result. The items are:


- rt_lidar0_lidar: A (Nlidars,6) numpy array containing rt transforms mapping
  points in each lidar frame to the frame of lidar0

- rt_lidar0_camera: A (Ncameras,6) numpy array containing rt transforms mapping
  points in each camera frame to the frame of lidar0

- Var_rt_lidar0_sensor: A (Nsensors_optimized, 6, Nsensors_optimized, 6)
  symmetric numpy array representing the 1-sigma uncertainty of the solution due
  to the expected noise in the inputs. Nsensors_optimized counts the number of
  sensors in the optimization problem: Nsensors_optimized = Nlidars + Ncameras -
  1. Lidar0 is always at the reference frame, and thus is not a part of the
  optimization.

- rt_vehicle_lidar
- rt_vehicle_camera
  Similar to rt_lidar0_lidar,rt_lidar0_camera, but referenced to the vehicle.
  Reported only if rt_vehicle_lidar0 is given

- observations_per_sector
- isvisible_per_sensor_per_sector
- stdev_worst_per_sector
- isensors_pair_stdev_worst
- isector_of_last_snapshot
  Sector-based diagnostics. See the README

- inputs_dump: if do_dump_inputs: this is "bytes" that can by used to re-run
  this solve with fit_from_inputs_dump()

- Nsectors
- threshold_valid_lidar_range
- threshold_valid_lidar_Npoints
- uncertainty_quantification_range
  Parameters used in the computation. If given to calibrate(), these are exactly
  what was passed in. If omitted, these are the defaults

    '''

    (snapshots, itopics_lidar, itopics_camera) = \
        _sorted_sensor_snapshots(bags, topics,
                                 decimation_period_s  = decimation_period_s,
                                 start                = start,
                                 stop                 = stop,
                                 max_time_spread_s    = max_time_spread_s,
                                 exclude_time_periods = exclude_time_periods,
                                 verbose              = verbose)
    result = _clc.calibrate( snapshots,
                             verbose = verbose,
                             **kwargs)
    result['itopics_lidar']  = itopics_lidar
    result['itopics_camera'] = itopics_camera
    return result


def color_sequence_rgb():
    r'''Return the default color sequence for gnuplot objects

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

    r'''Helper function for visualizing LIDAR data in a common frame

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
            [ next(clc.bag_interface.messages(bag, (topic,),
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
                                       rt_ref_camera,
                                       topics,
                                       *,
                                       isensor = None):

    r'''Helper function for visualizing sensor poses in geometric plots

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
