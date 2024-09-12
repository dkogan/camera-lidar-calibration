#!/usr/bin/python3

import sys
import numpy as np

import pathlib
import rosbags.highlevel.anyreader
import rosbags.typesys


def bag_messages_generator(bag, topics):

    dtype_cache = dict()

    # Read-only stuff for reading rosbags. Used by bag_messages_generator() only. I
    # want to evaluate this stuff only once
    name_from_type = { np.float32: 'FLOAT32',
                       np.float64: 'FLOAT64',
                       np.int16:   'INT16',
                       np.int32:   'INT32',
                       np.int8:    'INT8',
                       np.uint16:  'UINT16',
                       np.uint32:  'UINT32',
                       np.uint8:   'UINT8' }
    # These are what I always see, so I hard-code it. I make sure it's right in
    # bag_messages_generator(). If it's ever wrong, I will need to parse it at
    # runtime
    types = { 1: np.int8,
              2: np.uint8,
              3: np.int16,
              4: np.uint16,
              5: np.int32,
              6: np.uint32,
              7: np.float32,
              8: np.float64 }


    def dtype_from_msg(msg, key_cache):
        nonlocal dtype_cache
        if key_cache in dtype_cache: return dtype_cache[key_cache]

        dtype_dict = dict()

        # I want to be able to look at xyz points as a dense units, so I
        # special-case that
        if \
           types[7] is np.float32 and \
           msg.fields[0].name == 'x' and msg.fields[0].offset == 0 and msg.fields[0].datatype == 7 and \
           msg.fields[1].name == 'y' and msg.fields[1].offset == 4 and msg.fields[1].datatype == 7 and \
           msg.fields[2].name == 'z' and msg.fields[2].offset == 8 and msg.fields[2].datatype == 7:
            dtype_dict['xyz'] = ( (np.float32, (3,)), 0)

        for i,f in enumerate(msg.fields):
            if i < 3 and 'xyz' in dtype_dict:
                # special-case xyz type handled above
                continue

            T = types[f.datatype]
            datatype_lookup = getattr(f,name_from_type[T],-1)
            if datatype_lookup != f.datatype:
                raise Exception(f"Couldn't confirm type id, or mismatch: {f=} {types=}")
            if f.count == 1:
                dtype_dict[f.name] = (T, f.offset)
            else:
                dtype_dict[f.name] = ((T, (f.count,)), f.offset)

        dtype = np.dtype(dtype_dict)

        # This is a hack. I don't know how to detect this reliably, so I work
        # off of the LIDAR data formats that I have seen:
        #
        #   itemsize = 22: requires no alignment or padding
        #   itemsize = 28: requires padding up to the next multiple of 16: until 32
        #   itemsize = 34: requires padding up to the next multiple of 16: until 48
        #
        # I add padding-to-16 if it looks like it needs it. Might be the right
        # thing to do...
        if dtype.itemsize > msg.point_step:
            raise Exception(f"Unexpected data layout: itemsize > point_step ({dtype.itemsize}>{msg.point_step}) in {msg=}")
        if dtype.itemsize != msg.point_step:
            if msg.point_step%16 != 0:
                raise Exception(f"Unexpected data layout: itemsize<point_step, so I assume we need to pad to 16-byte boundaries (because that's what I've seen before), but this data has {msg.point_step=}, which isn't divisible by 16 in {msg=}")

            # pad to multiple of 16
            ilast_before_x16 = msg.point_step-1
            dtype_dict['_padding'] = (np.uint8, ilast_before_x16)
            dtype = np.dtype(dtype_dict)

        if not (msg.point_step == dtype.itemsize and \
                msg.width * msg.point_step == msg.row_step and \
                not msg.is_bigendian and \
                msg.data.dtype == np.uint8 and \
                msg.data.size == msg.row_step * msg.height):
            raise Exception(f"Unexpected data layout: {msg=}")

        dtype_cache[key_cache] = dtype

        return dtype

    def connection_from_topic(connections, topic):
        connections = [ c for c in connections \
                        if c.topic == topic ]

        if len(connections) == 0:
            raise Exception(f"Topic '{topic}' not found in '{bag}'; the available topics are {globals()['topics'](bag)}")
        if len(connections) > 1:
            raise Exception(f"Multiple connections for topic '{topic}' found in '{bag}'; I expect exactly one")
        return connections[0]

    # High-level structure from the "rosbag2" sample:
    #   https://ternaris.gitlab.io/rosbags/topics/rosbag2.html
    with rosbags.highlevel.anyreader.AnyReader( (pathlib.Path(bag),) ) as reader:

        # I expect exactly one matching connection for each topic given
        connections = [ connection_from_topic(reader.connections, topic) \
                        for topic in topics ]

        for connection, time_ns, rawdata in \
                reader.messages( connections = connections ):

            qos = connection.ext.offered_qos_profiles
            msg   = reader.deserialize(rawdata, connection.msgtype)
            dtype = dtype_from_msg(msg, connection.msgtype)
            data  = np.frombuffer(msg.data, dtype = dtype)

            time_header_ns = msg.header.stamp.sec*1000000000 + msg.header.stamp.nanosec

            yield dict( time_ns        = time_ns,
                        time_header_ns = time_header_ns,
                        frame_id       = msg.header.frame_id,
                        topic          = connection.topic,
                        msgtype        = connection.msgtype,
                        array          = data,
                        rawdata        = rawdata,
                        msg            = msg,
                        qos            = qos,
                        )

def topics(bag):
    with rosbags.highlevel.anyreader.AnyReader( (pathlib.Path(bag),) ) as reader:
        return [c.topic for c in reader.connections]
