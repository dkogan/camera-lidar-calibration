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

        array = next(bag_interface.bag_messages_generator(bag, (lidar_topic,) ))['array']

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


def calibrate(*,
              bags, lidar_topic, camera_topic,
              check_gradient__use_distance_to_plane = False,
              check_gradient                        = False,
              **kwargs):

    def lidar_points(msg):
        array = msg['array']
        return \
            (array['xyz' ],
             array['ring'])

    def images(msg):
        raise

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

lidar_segmentation_default_context = _clc.lidar_segmentation_default_context
