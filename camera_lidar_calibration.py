#!/usr/bin/python3

import sys
import numpy as np
import numpysane as nps
import os

import bag_interface
import _camera_lidar_calibration

def point_segmentation(bag, lidar_topic,
                       **kwargs):

    if not os.path.exists(bag):
        raise Exception(f"Bag path '{bag}' does not exist")

    array = next(bag_interface.bag_messages_generator(bag, (lidar_topic,) ))['array']

    points = array['xyz']
    ring   = array['ring']

    rings = np.unique(ring) # unique and sorted

    # I need to sort by ring and then by th
    th = np.arctan2( points[:,1], points[:,0] )
    def points_from_rings():
        for iring in rings:
            idx = ring==iring
            yield points[idx][ np.argsort(th[idx]) ]

    points_sorted = nps.glue( *points_from_rings(),
                              axis = -2 )

    Npoints = np.array([np.count_nonzero(ring==iring) for iring in rings],
                       dtype = np.int32)

    ipoint, plane_pn = \
        _camera_lidar_calibration.point_segmentation(points  = points_sorted,
                                                     Npoints = Npoints,
                                                     Nrings  = len(Npoints),
                                                     **kwargs)
    Nplanes = len(ipoint)

    return \
        dict( points  = [points_sorted[ipoint[i]] for i in range(Nplanes)],
              plane_p = plane_pn[:,:3],
              plane_n = plane_pn[:,3:] )

default_context = _camera_lidar_calibration.default_context
