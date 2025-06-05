#!/usr/bin/python3

r'''Format the output calibration geometry for ROS

SYNOPSIS

  $ lidars=(/lidar/velodyne_0/points \
            /lidar/velodyne_1/points \
            /lidar/velodyne_2/points);


  $ ./fit.py                   \
    --topics ${(j:,:)lidars}   \
    --bag $BAG

  ....
  Wrote '/tmp/lidar0-mounted.cameramodel' and a symlink '/tmp/sensor0-mounted.cameramodel'
  Wrote '/tmp/lidar1-mounted.cameramodel' and a symlink '/tmp/sensor1-mounted.cameramodel'
  Wrote '/tmp/lidar2-mounted.cameramodel' and a symlink '/tmp/sensor2-mounted.cameramodel'


  $ ./format-geometry-for-ros.py \
    --topics ${(j:,:)lidars}     \
    $BAG

  stamp:
      nsec: 1749145667549590272
  transforms:
      velodyne_0:
          parent_frame: base_link
          translation:
              x:  ...
              y:  ...
              z:  ...
          rotation:
              x: ...
              y: ...
              z: ...
              w: ...
      velodyne_1:
  .....

The solve computes everything in the frame of lidar0. To communicate the
solution to ROS, I want to base everything off the "base_link" frame. I read the
"/tf_static" topic to get transform between the lidar0 and the base_link. I then
use this transform to shift the solution, and output the solution in some yaml
thing that's palatable to ROS.

There's some funkyness in that each sensor is identified both by its "topic" and
by its "name". The name is what appears in /tf_static and that's what I use in
the output. I can't find any consistent way to map topics to/from names. So I
assume that each topic is "/xxx/xxx/xxx/NAME/xxx/xxx/xxx". I.e. there's lots of
unknowable cruft, with the sensor name in the middle somewhere. I use the names
in /tf_static to infer which topic component has the name, and I then use that.

'''

import sys
import argparse
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

    parser.add_argument('bag',
                        type=str,
                        help = '''The bag we read the /tf_static transform from''')

    args = parser.parse_args()
    args.topics = args.topics.split(',')
    return args


args = parse_args()



import time
import pathlib
import numpy as np
import numpysane as nps
import mrcal
from rosbags.highlevel import AnyReader

header = \
fr'''stamp:
    nsec: {int(time.time() * 1e9)}
transforms:'''

element_fmt = \
    r'''    {name}:
        parent_frame: base_link
        translation:
            x:  {x}
            y:  {y}
            z:  {z}
        rotation:
            x: {qx}
            y: {qy}
            z: {qz}
            w: {qw}'''


models_result = [f'/tmp/sensor{i}-mounted.cameramodel' for i in range(len(args.topics)) ]





rt_ref_sensor = [mrcal.cameramodel(m).extrinsics_rt_toref() for m in models_result]


def rt_from_rosqt(q, t):
    rt = np.array( (0,0,0, t.x, t.y, t.z),
                   dtype=float)

    half_theta = np.arccos(q.w)
    theta      = half_theta * 2.0

    if abs(theta) > 1.0e-9:
        s = theta / np.sin(half_theta)
        rt[0] = q.x * s
        rt[1] = q.y * s
        rt[2] = q.z * s

    return rt


def rosqt_from_rt(rt):
    t = rt[3:]

    theta = nps.mag(rt[:3])

    if np.abs(theta) > 1.0e-9:
        s = np.sin(theta / 2.0) / theta
    else:
        s = 0
    q = np.array( (np.cos(theta / 2.0),
                   rt[0] * s,
                   rt[1] * s,
                   rt[2] * s),
                  dtype=float)
    return q,t


topic0_elements_list = args.topics[0].split('/')
topic0_elements_set  = set(topic0_elements_list)


with AnyReader((pathlib.Path(args.bag),)) as reader:

    connections = [c for c in reader.connections if c.topic == '/tf_static']

    for conn, timestamp, rawdata in reader.messages(connections=connections):
        msg = reader.deserialize(rawdata, conn.msgtype)

        for transform in msg.transforms:
            if transform.header.frame_id == 'base_link' and \
               transform.child_frame_id  in topic0_elements_set:

                iname = topic0_elements_list.index(transform.child_frame_id)

                t = transform.transform.translation
                q = transform.transform.rotation

                rt_base_lidar0 = rt_from_rosqt(q,t)

                print(header)

                for isensor in range(len(args.topics)):

                    rt_base_sensor = mrcal.compose_rt( rt_base_lidar0, rt_ref_sensor[isensor] )
                    q,t = rosqt_from_rt( rt_base_sensor )

                    print(element_fmt.format(name = args.topics[isensor].split('/')[iname],
                                             x    = t[0],
                                             y    = t[1],
                                             z    = t[2],
                                             qw   = q[0],
                                             qx   = q[1],
                                             qy   = q[2],
                                             qz   = q[3],
                                             ))

                sys.exit(0)

print(f"No '/tf_static' messages relating 'base_link' and any of {topic0_elements_set}")
