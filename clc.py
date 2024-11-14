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
              bags, lidar_topics):

    def lidar_points(bag, lidar_topic):
        array = next(bag_interface.bag_messages_generator(bag, (lidar_topic,) ))['array']

        points = array['xyz']
        rings  = array['ring']

        return (points,rings)

    def lidar_points_all_topics(bag):
        return tuple( lidar_points(bag, lidar_topic) for lidar_topic in lidar_topics )

    def sensor_snapshot(bag):
        if not os.path.exists(bag):
            raise Exception(f"Bag path '{bag}' does not exist")
        return (None, lidar_points_all_topics(bag))

    for i,bag in enumerate(bags):
        print(f"Bag {i: 3d} {bag}")
    for i,topic in enumerate(lidar_topics):
        print(f"Topic {i: 2d} {topic}")

    return _clc.calibrate( tuple(sensor_snapshot(bag) for bag in bags) )

lidar_segmentation_default_context = _clc.lidar_segmentation_default_context
