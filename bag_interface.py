#!/usr/bin/python3

import sys
import numpy as np

import rosbags.rosbag2
import rosbags.typesys

# Used by readers and writers. Imported by some other modules
typestore = rosbags.typesys.get_typestore(rosbags.typesys.Stores.LATEST)

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

        # This is a hack. I don't know how to detect this reliably. So
        # far I've seen two different LIDAR data formats:
        #
        #   itemsize = 22: requires no alignment or padding
        #   itemsize = 34: requires padding up to the next multiple of 16: until 48
        #
        # I support those cases explicitly. Other cases may not work
        has_padding = False
        if dtype.itemsize == 34:
            has_padding = True
            dtype_dict['_padding'] = (np.uint8,47)
            dtype = np.dtype(dtype_dict)

        if not (msg.point_step == dtype.itemsize and \
                msg.width * msg.point_step == msg.row_step and \
                msg.is_dense and \
                not msg.is_bigendian and \
                msg.data.dtype == np.uint8 and \
                msg.data.size == msg.row_step * msg.height):
            raise Exception(f"Unexpected data layout: {msg=}")

        dtype_cache[key_cache] = dtype

        return dtype


    # High-level structure from the "rosbag2" sample:
    #   https://ternaris.gitlab.io/rosbags/topics/rosbag2.html
    with rosbags.rosbag2.Reader(bag) as reader:

        connections = [ c for c in reader.connections \
                        if c.topic in topics ]

        for connection, time_ns, rawdata in \
                reader.messages( connections = connections ):

            msg   = typestore.deserialize_cdr(rawdata, connection.msgtype)
            dtype = dtype_from_msg(msg, connection.msgtype)
            data  = np.frombuffer(msg.data, dtype = dtype)

            time_header_ns = msg.header.stamp.sec*1000000000 + msg.header.stamp.nanosec

            yield dict( time_ns        = time_ns,
                        time_header_ns = time_header_ns,
                        frame_id       = msg.header.frame_id,
                        topic          = connection.topic,
                        msgtype        = connection.msgtype,
                        array          = data,
                        rawdata        = rawdata )

def topics(bag):
    with rosbags.rosbag2.Reader(bag) as reader:
        return [c.topic for c in reader.connections]
