#!/usr/bin/python3

import sys
import numpy as np
import numpysane as nps
import os

import bag_interface
import _clc

def lidar_segmentation(*,
                       bag, lidar_topic,
                       # used if bag,lidar_topic are None
                       points = None,
                       rings  = None,
                       **kwargs):

    if bag is not None:
        if not os.path.exists(bag):
            raise Exception(f"Bag path '{bag}' does not exist")

        array = next(bag_interface.messages(bag, (lidar_topic,) ))['array']

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


def calibrate(*,
              bags, lidar_topic, camera_topic,
              check_gradient__use_distance_to_plane = False,
              check_gradient                        = False,
              **kwargs):

    def images(msg):
        if msg is None: return None
        return msg['array']


    def sensor_snapshot(bag):
        if not os.path.exists(bag):
            raise Exception(f"Bag path '{bag}' does not exist")

        messages = \
            bag_interface. \
            first_message_from_each_topic(bag,
                                          lidar_topic + camera_topic)
        Nlidar = len(lidar_topic)
        return \
            ( tuple(lidar_points(msg) for msg in messages[:Nlidar]),
              tuple(images      (msg) for msg in messages[Nlidar:]) )

    if not (check_gradient__use_distance_to_plane or \
            check_gradient ):
        for i,bag in enumerate(bags):
            print(f"Bag {i: 3d} {bag}")
        for i,topic in enumerate(lidar_topic):
            print(f"LIDAR topic {i: 2d} {topic}")
        for i,topic in enumerate(camera_topic):
            print(f"Camera topic {i: 2d} {topic}")

    return _clc.calibrate( tuple(sensor_snapshot(bag) for bag in bags),
                           check_gradient__use_distance_to_plane = check_gradient__use_distance_to_plane,
                           check_gradient                        = check_gradient,
                           **kwargs)

def post_solve_statistics(*,
                          bag,
                          lidar_topic,
                          **kwargs):

    if not os.path.exists(bag):
        raise Exception(f"Bag path '{bag}' does not exist")

    messages = \
        bag_interface. \
        first_message_from_each_topic(bag, lidar_topic)

    kwargs = dict(kwargs)
    del kwargs['inputs_dump']
    return _clc.post_solve_statistics(lidar_scans = tuple(lidar_points(msg) for msg in messages),
                                      **kwargs)




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

def plot(*args,
         hardcopy = None,
         **kwargs):
    r'''Wrapper for gp.plot(), but printing out where the hardcopy went'''
    gp.plot(*args, **kwargs,
            hardcopy = hardcopy)
    if hardcopy is not None:
        print(f"Wrote '{hardcopy}'")


def get_pointcloud_plot_tuples(bag, lidar_topic, threshold,
                               *,
                               ilidar_in_solve_from_ilidar = None):

    try:
        pointcloud_msgs = \
            [ next(bag_interface.messages(bag, (topic,))) \
              for topic in lidar_topic ]
    except:
        raise Exception(f"Bag '{bag}' doesn't have at least one message for each of {lidar_topic}")

    # Package into a numpy array
    pointclouds = [ msg['array']['xyz'].astype(float) \
                    for msg in pointcloud_msgs ]

    # Throw out everything that's too far, in the LIDAR's own frame
    pointclouds = [ p[ nps.mag(p) < threshold ] for p in pointclouds ]

    if ilidar_in_solve_from_ilidar is not None:
        pointclouds = \
            [ mrcal.transform_point_rt(rt_lidar0_lidar[ ilidar_in_solve_from_ilidar[i] ],
                                       p) for i,p in enumerate(pointclouds) ]
    else:
        pointclouds = \
            [ mrcal.transform_point_rt(rt_lidar0_lidar[i],
                                       p) for i,p in enumerate(pointclouds) ]

    data_tuples = [ ( p, dict( tuplesize = -3,
                               legend    = args.lidar_topic[i],
                               _with     = f'points pt 7 ps 1 lc rgb "{clc.color_sequence_rgb[i%len(clc.color_sequence_rgb)]}"')) \
                    for i,p in enumerate(pointclouds) ]

    return data_tuples




lidar_segmentation_default_context = _clc.lidar_segmentation_default_context
fit_from_optimization_inputs       = _clc.fit_from_optimization_inputs
