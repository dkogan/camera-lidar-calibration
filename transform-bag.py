#!/usr/bin/python3

r"""Modify tf tree in a ros bag

SYNOPSIS

  $ transform-bag.py \
      joint.bag                                    \
      --outdir /tmp                                \
      --reference /topic_1                         \
      --topics /topic_2 /topic_3                   \
      --transforms "[0.1,0.0,0.0,0.1,0,0]" "[0.2,0,0,0,2,0]"
    ...

"""


import sys
import argparse
import re
import os


def parse_args():

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--outdir",
        default="/tmp",
        help="""The output directory to store the split bags.
                        If omitted, use the current directory""",
    )
    parser.add_argument(
        "-r",
        "--reference",
        type=str,
        help="""The reference topic that all transforms are expressed relative to""",
    )

    parser.add_argument("bag", type=str, help="""The rosbag that should be processed""")

    parser.add_argument(
        "-t",
        "--topics",
        nargs="+",
        type=str,
        help="""The LIDAR topics that should be transformed in a space delimited list""",
    )
    parser.add_argument(
        "-x",
        "--transforms",
        nargs="+",
        type=str,
        help="""The transforms for each topic to be transformed in format
        [Roll,Pitch,Yaw,X,Y,Z], use spaces to separate multiple transforms"""
    )

    args = parser.parse_args()

    # validate input arguments
    if len(args.topics) != len(args.transforms):
        raise Exception("Number of topics must be equal to number of transforms")
    transforms = []
    for transform in args.transforms:
        transforms.append(
            [float(t) for t in ast.literal_eval(transform)]
        )
    for transform in transforms:
        if len(transform) != 6:
            raise Exception(
                f"Transform {transform} is malformed; it must have six entries in "
                + "the form [r,p,y,x,y,z]"
            )
    args.transforms = transforms
    return args

args = parse_args()



import ast
import rosbags
import rosbags.rosbag2
from pathlib import Path
import numpy as np

try:
    import transforms3d
except Exception as e:
    print(f"This tool requires ROS2 to be installed and configured:\n\n{e}",
          file=sys.stderr)
    sys.exit(1)


typestore = rosbags.typesys.get_typestore(rosbags.typesys.Stores.ROS2_HUMBLE)




def get_msgs(bag, topic, limit=None):
    msgs = []
    with rosbags.rosbag2.Reader(args.bag) as reader:
        connections = [
            c for c in reader.connections if c.topic == topic
        ]
        for i, (connection, time_ns, rawdata) in enumerate(
                reader.messages(
                    connections=connections
                )):
            if limit and i >= limit:
                break
            msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
            msgs.append(
                {
                    "rawdata": rawdata,
                    "msg": msg
                }
            )
    return msgs


def rpyxyz_to_affine(roll, pitch, yaw, x, y, z):
    return np.linalg.inv(
        transforms3d.affines.compose(
            [x, y, z],
            transforms3d.euler.euler2mat(roll, pitch, yaw),
            [1.0, 1.0, 1.0]
        )
    )


def ros_transform_to_affine(transform):
    return transforms3d.affines.compose(
        [
            transform.transform.translation.x,
            transform.transform.translation.y,
            transform.transform.translation.z,
        ],
        transforms3d.quaternions.quat2mat(
            [
                transform.transform.rotation.w,
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z
            ]
        ),
        [1.0, 1.0, 1.0],
    )


def affine_to_ros_transform(affine, ros_transform):
    q = transforms3d.quaternions.mat2quat(affine[:3, :3])
    ros_transform.transform.rotation.w = q[0]
    ros_transform.transform.rotation.x = q[1]
    ros_transform.transform.rotation.y = q[2]
    ros_transform.transform.rotation.z = q[3]
    ros_transform.transform.translation.x = affine[0, 3]
    ros_transform.transform.translation.y = affine[1, 3]
    ros_transform.transform.translation.z = affine[2, 3]



# get reference topic
reference_msg = get_msgs(args.bag, args.reference, limit=1)
if not reference_msg:
    raise Exception(
        f"Topic {args.reference} was not found in {args.bag}"
    )
reference_msg = reference_msg[0]["msg"]

# get reference frame of reference lidar
tf_static_msgs = get_msgs(args.bag, "/tf_static")
reference_transform = None
for tf_static_msg in tf_static_msgs:
    for transform in tf_static_msg["msg"].transforms:
        if transform.child_frame_id == reference_msg.header.frame_id:
            reference_transform = transform
            break
if reference_transform is None:
    raise Exception("Could not find transform for frame {args.reference}")
reference_transform = ros_transform_to_affine(reference_transform)

# get other topics' reference/parent frames
topic_parent_frames = []
for topic in args.topics:
    msg = get_msgs(args.bag, topic, limit=1)
    if not topic:
        raise Exception(
            f"Topic {topic} was not found in {args.bag}"
        )
    msg = msg[0]["msg"]
    topic_parent_frames.append(msg.header.frame_id)

topic_transforms = []
for transform in args.transforms:
    topic_transforms.append(
        rpyxyz_to_affine(
            transform[0], transform[1], transform[2],
            transform[3], transform[4], transform[5],
        )
    )

connections = {}
with rosbags.rosbag2.Reader(args.bag) as reader:
    with rosbags.rosbag2.Writer(
        Path(args.outdir) / (str(Path(f"{args.bag}").stem) + ".bag"),
        version=8
    ) as writer:

        # Version 8 is needed because the structure of offered_qos_profiles
        # is not compatible with ROS2 HUMBLE for ros2 play
        for connection, time_ns, rawdata in reader.messages(
                connections=reader.connections):
            if connection.topic not in connections:
                qos = connection.ext.offered_qos_profiles
                connections[connection.topic] = writer.add_connection(
                    connection.topic,
                    connection.msgtype,
                    typestore=typestore,
                    offered_qos_profiles=qos,
                )
            if connection.topic == "/tf_static":
                msg = typestore.deserialize_cdr(
                    rawdata, connection.msgtype
                )
                for transform in msg.transforms:
                    if transform.child_frame_id in topic_parent_frames:
                        index = topic_parent_frames.index(transform.child_frame_id)
                        topic_transform = topic_transforms[index]
                        new_transform = reference_transform.dot(topic_transform)
                        affine_to_ros_transform(
                            new_transform,
                            transform
                        )
                        print("\nGenerated new transform:")
                        print(transform)
                cdr_bytes = typestore.serialize_cdr(
                    msg, connection.msgtype
                )
                writer.write(connections[connection.topic], time_ns, cdr_bytes)
            else:
                writer.write(connections[connection.topic], time_ns, rawdata)
