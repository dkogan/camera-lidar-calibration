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

import argparse


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

    parser.add_argument("bag", type=str, help="""The rosbag that should be split""")

    parser.add_argument(
        "lidar-topic",
        type=str,
        help="""The LIDAR topic glob pattern we're looking at""",
    )

    args = parser.parse_args()

    args.lidar_topic = getattr(args, "lidar-topic")
    args.period_ns = args.period * 1e9

    return args


args = parse_args()


import fnmatch
import rosbags.rosbag2
import bag_interface


def write(msgs_now_from_topic, outdir="."):

    if len(msgs_now_from_topic) != len(topics):
        return

    # Get the relative timestamp for the purposes of output naming
    time_ns = msgs_now_from_topic[topics[0]]["time_ns"]
    i_period = time_ns // args.period_ns
    time_ns = i_period * args.period_ns
    if not hasattr(write, "time_ns0"):
        write.time_ns0 = time_ns
    t = (time_ns - write.time_ns0) / 1e9

    bagfile = f"{outdir}/{t:07.1f}.bag"

    print(f"Writing '{bagfile}'")

    with rosbags.rosbag2.Writer(
        bagfile, version=rosbags.rosbag2.Writer.VERSION_LATEST
    ) as writer:
        for topic in topics:
            msg = msgs_now_from_topic[topic]
            connection = writer.add_connection(
                topic, msg["msgtype"], typestore=bag_interface.typestore
            )
            writer.write(connection, msg["time_ns"], msg["rawdata"])


topics_all = bag_interface.topics(args.bag)
topics = fnmatch.filter(topics_all, args.lidar_topic)

print(f"Reading topics {topics}")

msgs_now_from_topic = dict()
i_period0 = -1

# For each topic I take the first event in each period. I assume that the whole
# sequence of events is monotonic, and that all the topics cross each period
# threshold together
for msg in bag_interface.bag_messages_generator(args.bag, topics):

    i_period = msg["time_ns"] // args.period_ns
    if i_period0 > i_period:
        raise Exception(
            "This implementation assumes a strictly monotonic sequence of i_period"
        )

    if i_period0 < i_period:
        write(msgs_now_from_topic, outdir=args.outdir)

        msgs_now_from_topic = dict()
        i_period0 = i_period

    topic = msg["topic"]
    if not topic in msgs_now_from_topic:
        msgs_now_from_topic[topic] = msg

write(msgs_now_from_topic, outdir=args.outdir)
