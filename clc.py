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


def lidar_points(msg):
    if msg is None: return None
    array = msg['array']
    return \
        (array['xyz' ],
         array['ring'])
def images(msg):
    if msg is None: return None
    return msg['array']


def is_message_pointcloud(msg):

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
                             stop              = None,):
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
                                                         require_at_least_N_topics = 2) \
             for bag in bags]
    else:
        # We have one long bag. I look at each time segment decimation_period
        # long, and take the first set of messages (one per topic) from such
        # segment
        if len(bags) != 1:
            raise Exception("decimation_period is not None, so I expect exactly one bag to have been given")
        bag = bags[0]
        messages_bags = \
            bag_interface. \
            first_message_from_each_topic_in_time_segments(bag, topics,
                                                           start    = start,
                                                           stop     = stop,
                                                           period_s = decimation_period,
                                                           require_at_least_N_topics = 2)


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
            if is_message_pointcloud(msg): itopics_lidar .add(i)
            else:                          itopics_camera.add(i)
            Ntopics_identified += 1
    if Ntopics_identified != len(topics):
        itopics_missing = \
            set(range(len(topics))).difference(itopics_lidar).difference(itopics_camera)
        raise Exception(f"Some topics have no data in any of the bags: {[topics[i] for i in itopics_missing]}")

    itopics_lidar  = sorted(itopics_lidar)
    itopics_camera = sorted(itopics_camera)

    return \
        tuple( ( tuple(lidar_points(messages[i]) for i in itopics_lidar),
                 tuple(images      (messages[i]) for i in itopics_camera) ) \
               for messages in messages_bags )


def calibrate(*,
              bags, topics,
              decimation_period = None,
              start             = None,
              stop              = None,
              **kwargs):
    return _clc.calibrate( _sorted_sensor_snapshots(bags, topics,
                                                    decimation_period = decimation_period,
                                                    start             = start,
                                                    stop              = stop),
                           **kwargs)




# from "gnuplot -e 'show linetype'"
# Named colors are defined in gnuplot/src/tables.c
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

def plot(*args,
         hardcopy = None,
         **kwargs):
    r'''Wrapper for gp.plot(), but printing out where the hardcopy went'''

    import gnuplotlib as gp
    gp.plot(*args, **kwargs,
            hardcopy = hardcopy)
    if hardcopy is not None:
        print(f"Wrote '{hardcopy}'")


def get_pointcloud_plot_tuples(bag, lidar_topic, threshold,
                               rt_lidar0_lidar,
                               *,
                               isensor_solve_from_isensor_requested = None,
                               Rt_vehicle_lidar0                    = None,
                               start                                = None):

    try:
        pointcloud_msgs = \
            [ next(bag_interface.messages(bag, (topic,),
                                          start = start)) \
              for topic in lidar_topic ]
    except:
        raise Exception(f"Bag '{bag}' doesn't have at least one message for each of {lidar_topic} in the requested time span")

    for i,msg in enumerate(pointcloud_msgs):
        if not is_message_pointcloud(msg):
            raise Exception(f"Topic {lidar_topic[i]} is not a pointcloud type")

    # Package into a numpy array
    pointclouds = [ msg['array']['xyz'].astype(float) \
                    for msg in pointcloud_msgs ]

    # Throw out everything that's too far, in the LIDAR's own frame
    pointclouds = [ p[ nps.mag(p) < threshold ] for p in pointclouds ]

    if isensor_solve_from_isensor_requested is not None:
        pointclouds = \
            [ mrcal.transform_point_rt(rt_lidar0_lidar[ isensor_solve_from_isensor_requested[i] ],
                                       p) for i,p in enumerate(pointclouds) ]
    else:
        pointclouds = \
            [ mrcal.transform_point_rt(rt_lidar0_lidar[i],
                                       p) for i,p in enumerate(pointclouds) ]

    if Rt_vehicle_lidar0 is not None:
        pointclouds = [mrcal.transform_point_Rt(Rt_vehicle_lidar0, p) \
                       for p in pointclouds]

    data_tuples = [ ( p, dict( tuplesize = -3,
                               legend    = lidar_topic[i],
                               _with     = f'points pt 7 ps 1 lc rgb "{color_sequence_rgb[i%len(color_sequence_rgb)]}"')) \
                    for i,p in enumerate(pointclouds) ]

    return data_tuples


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


def get_data_tuples_sensor_forward_vectors(rt_ref_lidar,
                                           rt_ref_camera,
                                           topics,
                                           *,
                                           isensor = None):

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
fit_from_inputs_dump               = _clc.fit_from_inputs_dump
