#!/usr/bin/python3

r"""Split a rosbag into separate one-synced-dataset-at-a-time bags

SYNOPSIS

  $ split-bag-into-synchronized-stationary-chunks.py \
      --period 10                                    \
      --outdir /tmp                                  \
      joint.bag                                      \
      '/lidar/*'

    Reading topics ['/lidar/left/velodyne_points', '/lidar/right/velodyne_points']
    Writing '/tmp/00000.0.bag'
    Writing '/tmp/00010.0.bag'
    Writing '/tmp/00020.0.bag'
    ...

The main camera-lidar calibration tool assumes a set of stationary observations
from all the sensors, each stationary set appearing in a separate bag. To
process datasets captured as one-big-bag-for-all-the-data, this tool can be used
to split the bag. This tool splits time into chunks of size --period. The first
message of each topic in a period is written into the bag for that period. Each
bag for a period contains exactly one message for EACH requested topic.

At this time, no data analysis is being performed to find messages observing a
STATIONARY scene. This would be an excellent extension to this tool

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
        default=".",
        help="""The output directory to store the split bags.
                        If omitted, use the current directory""",
    )
    parser.add_argument(
        "--period",
        default=1.0,
        type=float,
        help="""The period for the split bag output. We write
                out one bag for each time interval of this length. By
                default we use a period of 1sec""",
    )

    parser.add_argument(
        "--timestamp-file",
        default=None,
        dest="timestamp_file",
        help="""Optionally pass a text file containing timestamps in seconds
                with one timestamp on each line, where one timestamp corresponds
                to a stationary board position""",
    )

    parser.add_argument("bag", type=str, help="""The rosbag that should be split""")

    parser.add_argument(
        "lidar_topic",
        metavar="lidar-topic",
        type=str,
        help="""The LIDAR topic glob pattern we're looking at""",
    )

    args = parser.parse_args()
    args.period_ns = args.period * 1e9
    return args


args = parse_args()



import fnmatch
import rosbags
import rosbags.rosbag2
import bag_interface
from typing import TypeVar
import importlib
import numpy as np


typestore = rosbags.typesys.get_typestore(rosbags.typesys.Stores.ROS2_HUMBLE)


def write(
    msgs_now_from_topic,
    topics,
    tf_msg,
    tf_msg_qs,
    tf_static_msgs,
    tf_static_qos,
    outdir=".",
):

    if len(msgs_now_from_topic) != len(topics):
        return

    from rosbags.rosbag2.metadata import dump_qos_v8, dump_qos_v9

    time_ns = msgs_now_from_topic[topics[0]]["time_ns"]
    time_s = time_ns / 1e9
    bagfile = f"{outdir}/{time_s:07.6f}.bag"
    print(f"Writing '{bagfile}'")

    with rosbags.rosbag2.Writer(
        bagfile, version=8
    ) as writer:
        # Version 8 is needed because the structure of offered_qos_profiles in V9 is not
        # compatible with ROS2 HUMBLE for ros2 play
        for topic in topics:
            msg = msgs_now_from_topic[topic]
            connection = writer.add_connection(
                topic, msg["msgtype"], typestore=typestore,
                offered_qos_profiles=msg["qos"],
            )
            writer.write(connection, msg["time_ns"], msg["rawdata"])
        connection = writer.add_connection(
            "/tf", "tf2_msgs/msg/TFMessage", typestore=typestore,
            offered_qos_profiles=tf_qos
        )
        writer.write(connection, time_ns, tf_msg)
        connection = writer.add_connection(
            "/tf_static", "tf2_msgs/msg/TFMessage", typestore=typestore,
            offered_qos_profiles=tf_static_qos
        )
        for tf_static_msg in tf_static_msgs:
            writer.write(connection, time_ns, tf_static_msg)


# T = TypeVar('T')
NATIVE_CLASSES: dict[str, type] = {}


def to_native(msg: object) -> object:
    """Convert rosbags message to native message.

    Args:
        msg: Rosbags message.

    Returns:
        Native message.

    """
    msgtype: str = msg.__msgtype__  # type: ignore[attr-defined]
    if msgtype not in NATIVE_CLASSES:
        pkg, name = msgtype.rsplit('/', 1)
        NATIVE_CLASSES[msgtype] = (
            getattr(importlib.import_module(pkg.replace('/', '.')), name)
        )
    fields = {}

    for name, field in msg.__dataclass_fields__.items():  # type: ignore[attr-defined]
        if 'ClassVar' in field.type:
            continue
        value = getattr(msg, name)
        if isinstance(value, list):
            value = [to_native(x) for x in value]
        elif isinstance(value, np.ndarray):
            value = value.tolist()
        elif '__msg__' in field.type:
            value = to_native(value)
        fields[name] = value
    return NATIVE_CLASSES[msgtype](**fields)




topics_all = bag_interface.topics(args.bag)
topics = fnmatch.filter(topics_all, args.lidar_topic)
print(f"Reading topics {topics}")

timestamps_ns = []
if args.timestamp_file:
    with open(args.timestamp_file, "r") as f:
        for line in f:
            timestamps_ns.append(float(line.strip()) * 1.0e9)
    timestamps_ns.sort()
timestamps_ns = iter(timestamps_ns)

msg_dict = dict()
t_0 = None
t_threshold = None
t_prev = -1

# High-level structure from the "rosbag2" sample:
#   https://ternaris.gitlab.io/rosbags/topics/rosbag2.html
tf_msg = None
tf_msg_qos = None
tf_static_msgs = []
tf_static_qos = None
with rosbags.rosbag2.Reader(args.bag) as reader:
    connections = [
        c for c in reader.connections if c.topic in ["/tf", "/tf_static"]
    ]
    for connection, time_ns, rawdata in reader.messages(
            connections=connections):
        msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
        if not tf_msg and connection.topic == "/tf":
            tf_msg = rawdata
            tf_qos = connection.ext.offered_qos_profiles
        if connection.topic == "/tf_static":
            tf_static_msgs.append(rawdata)
            tf_static_qos = connection.ext.offered_qos_profiles
        if tf_msg:
            continue

# For each topic I take the first event in each period. I assume that the whole
# sequence of events is monotonically increasing, and that all the topics
# cross each period threshold together
print(topics)
for msg in bag_interface.messages(args.bag, topics):
    t = msg["time_ns"]
    if not t_0:
        t_0 = t
        if args.timestamp_file:
            try:
                t_threshold = next(timestamps_ns)
            except StopIteration:
                break
        else:
            t_threshold = t_0 + args.period_ns
    if t_prev > t:
        raise Exception(
            "Messages were found out of order; this implementation assumes strictly "
            + "monotonically increasing timestamps"
        )
    if t > t_threshold:
        topic = msg["topic"]
        if topic not in msg_dict:
            msg_dict[topic] = msg
        if set(topics).issubset(msg_dict):
            write(
                msg_dict,
                topics,
                tf_msg,
                tf_qos,
                tf_static_msgs,
                tf_static_qos,
                outdir=args.outdir,
            )
            msg_dict = dict()
            if args.timestamp_file:
                try:
                    t_threshold = next(timestamps_ns)
                except StopIteration:
                    break
            else:
                t_threshold = t_threshold + args.period_ns
    t_prev = t
