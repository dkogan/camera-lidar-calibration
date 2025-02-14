#!/usr/bin/python3

import sys
import numpy as np

import pathlib
import rosbags.highlevel.anyreader
import rosbags.typesys
import re
import dateutil


def parse_timestamp_to_ns_since_epoch(t):
    if t is None: return None

    if isinstance(t, str):
        # parse the string numerically, if possible
        if re.match(r"[0-9]+$", t):
            t = int(t)
        elif re.match(r"-?[0-9.eE]+$", t):
            t = float(t)

    if isinstance(t, int):
        if t < 0x80000000:
            # in range; assume this is in seconds
            return int(t * 1e9)
        return t
    if isinstance(t, float):
        return int(t * 1e9)

    return int(dateutil.parser.parse(t).timestamp() * 1e9)


def _time_spread_s(msgs):
    try:
        tmin = min(m['time_header_ns'] for m in msgs if m is not None)
        tmax = max(m['time_header_ns'] for m in msgs if m is not None)
    except:
        return None
    return (tmax-tmin)/1e9


def messages(bag, topics,
             *,
             # if integer: s since epoch or ns since epoch
             # if float:   s since epoch
             # if str:     try to parse as an integer or float OR with dateutil.parser.parse()
             start = None,
             stop  = None,
             ignore_unknown_message_types = False):

    dtype_cache = dict()

    # Read-only stuff for reading rosbags. Used by messages() only. I
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
    # messages(). If it's ever wrong, I will need to parse it at
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

            # msg.is_dense isn't reliable it looks like, so I don't check this.
            # In the 2023-09-26 dataset is_dense is True, but extra padding is
            # clearly required
            #
            # if msg.is_dense:
            #     raise Exception(f"Unexpected data layout: itemsize != point_step ({dtype.itemsize}!={msg.point_step}) but {msg.is_dense=} in {msg=}")

            # pad to multiple of 16
            ilast_before_x16 = msg.point_step-1
            dtype_dict['_padding'] = (np.uint8, ilast_before_x16)
            dtype = np.dtype(dtype_dict)

        if not (msg.point_step == dtype.itemsize and \
                not msg.is_bigendian and \
                msg.data.dtype == np.uint8):
            raise Exception(f"Unexpected data layout: {msg=}")

        if msg.row_step == 0:
            if not msg.is_dense:
                raise Exception(f"{msg.row_step=} but {msg.is_dense}; these shouldn't go together")
        else:
            if not (msg.width * msg.point_step == msg.row_step and \
                    msg.data.size == msg.row_step * msg.height):
                raise Exception(f"Unexpected data layout: {msg=}")


        dtype_cache[key_cache] = dtype

        return dtype

    def connection_from_topic(connections, topic):
        connections = [ c for c in connections \
                        if c.topic == topic ]

        if len(connections) == 0:
            return None
        if len(connections) > 1:
            raise Exception(f"Multiple connections for topic '{topic}' found in '{bag}'; I expect exactly one")
        return connections[0]

    messages.re_report_message = None

    # High-level structure from the "rosbag2" sample:
    #   https://ternaris.gitlab.io/rosbags/topics/rosbag2.html
    with rosbags.highlevel.anyreader.AnyReader( (pathlib.Path(bag),) ) as reader:

        # I expect exactly one matching connection for each topic given
        connections = [ connection_from_topic(reader.connections, topic) \
                        for topic in topics ]
        connections = [ c for c in connections if c is not None ]

        if len(connections) == 0: return

        for connection, time_ns, rawdata in \
                reader.messages( connections = connections,
                                 start = parse_timestamp_to_ns_since_epoch(start),
                                 stop  = parse_timestamp_to_ns_since_epoch(stop)):

            # This is GLOBAL, and WILL BREAK if I have multiple such iterators
            # going at the same time. It's not clear how to fix that
            while messages.re_report_message is not None:
                _msg = messages.re_report_message
                messages.re_report_message = None
                yield _msg

            try:
                qos = connection.ext.offered_qos_profiles
            except:
                # Sometimes we don't have this, like when looking at ros1 data
                qos = None

            msg = reader.deserialize(rawdata, connection.msgtype)

            if re.search(r'\bsensor_msgs__msg__PointCloud2\b', str(type(msg))):
                dtype = dtype_from_msg(msg, connection.msgtype)
                data  = np.frombuffer(msg.data, dtype = dtype)

                # 2023-11-01 dataset contains data that is almost completely
                # comprised of duplicated points. The only difference in these
                # duplicates is the 'ret' field. This is always 0 or 1. Mostly these
                # alternate, with a small period of all 1. I have no idea what this
                # means, so I only take ret==0; this suppresses the duplicates
                if 'ret' in dtype.fields:
                    data = data[data['ret'] == 0]

            elif re.search(r'\bsensor_msgs__msg__Image\b', str(type(msg))):
                data  = msg.data
                shape = data.shape

                if msg.encoding == 'mono8':
                    if not ( len(shape) == 1 and \
                             shape[0] == msg.width*msg.height and \
                             data.dtype == np.uint8 and \
                             msg.width == msg.step ):
                        raise Exception("rosbags.usertypes.sensor_msgs__msg__Image data should contain a flattened, dense byte array")
                    data = data.reshape(msg.height,msg.width)
                elif msg.encoding == 'bgr8':
                    if not ( len(shape) == 1 and \
                             shape[0] == msg.width*msg.height*3 and \
                             data.dtype == np.uint8 and \
                             msg.width*3 == msg.step ):
                        raise Exception("rosbags.usertypes.sensor_msgs__msg__Image data should contain a flattened, dense byte array")
                    data = data.reshape(msg.height,msg.width,3)
                else:
                    raise Exception(f"Unknown {msg.encoding=}")
            else:
                if ignore_unknown_message_types:
                    continue
                raise Exception(f"Unknown message type {type(msg)=}")

            time_header_ns = msg.header.stamp.sec*1000000000 + msg.header.stamp.nanosec

            yield dict( time_ns        = time_ns,
                        time_header_ns = time_header_ns,
                        frame_id       = msg.header.frame_id,
                        topic          = connection.topic,
                        msgtype        = connection.msgtype,
                        array          = data,
                        rawdata        = rawdata,
                        qos            = qos )


# Returns None if we reached the end, and no data is available
# Returns [None,None,....] if no data is available, but there's more data in the
# bag/iterator
def first_message_from_each_topic(bag, # the bag file OR an existing message iterator
                                  topics,
                                  *,
                                  # if integer: s since epoch or ns since epoch
                                  # if float:   s since epoch
                                  # if str:     try to parse as an integer or float OR with dateutil.parser.parse()
                                  start             = None,
                                  stop              = None,
                                  max_time_spread_s = None,
                                  verbose           = False):

    def out_culled_if_too_spread(out, max_time_spread_s):
        if max_time_spread_s is None or \
           not any(out):
            return out

        # Here I look at the data-acquisition time, since that most
        # closely describes the data being consistent
        dt = _time_spread_s(out)
        if dt > max_time_spread_s:

            timestamps = ['-' if m is None else m['time_header_ns'] for m in out]
            if verbose:
                print(f"Not reporting snapshot in time interval {[start,stop]} because the time_header_ns is too spread-out. {dt=:.2f}s. {timestamps=}")
            return [None] * len(topics)
        return out



    out = [None] * len(topics)
    idx = dict()
    for i,t in enumerate(topics): idx[t] = i
    Nstored = 0

    if isinstance(bag, str):
        message_iterator = messages(bag, topics,
                                    start = start,
                                    stop  = stop)
    else:
        message_iterator = bag
    start = parse_timestamp_to_ns_since_epoch(start)
    stop  = parse_timestamp_to_ns_since_epoch(stop)
    end_of_file = False
    for msg in message_iterator:
        if start is not None and msg['time_ns'] < start:
            continue
        if stop is not None and msg['time_ns'] >= stop:
            # This message is past the stop time, so we're done. If we reuse
            # this iterator with the current stop time being the new start time,
            # then I want to return this message then. I save it in the iterator
            # for that purpose

            # This is GLOBAL, and WILL BREAK if I have iterators going at the
            # same time. It's not clear how to fix that
            if not isinstance(bag, str):
                messages.re_report_message = msg
            break

        i = idx[msg['topic']]
        if out[i] is None:
            out[i] = msg
            Nstored += 1
            if Nstored == len(topics):
                return out_culled_if_too_spread(out, max_time_spread_s)
    else:
        end_of_file = True

    if end_of_file and Nstored == 0:
        return None

    return out_culled_if_too_spread(out, max_time_spread_s)


def first_message_from_each_topic_in_time_segments(bag, topics,
                                                   *,
                                                   period_s, # in seconds
                                                   # if integer: s since epoch or ns since epoch
                                                   # if float:   s since epoch
                                                   # if str:     try to parse as an integer or float OR with dateutil.parser.parse()
                                                   start = None,
                                                   stop  = None,
                                                   require_at_least_N_topics = 1,
                                                   verbose = False,
                                                   max_time_spread_s = None):

    message_iterator = messages(bag, topics,
                                start = start)

    start = parse_timestamp_to_ns_since_epoch(start)
    stop  = parse_timestamp_to_ns_since_epoch(stop)
    d     = info(bag)

    t1 = start if start is not None else d['t0']

    msgs = []
    while True:
        t0 = t1
        t1 = t0 + int(period_s*1e9)
        if stop is not None and t0 > stop:
            break

        msgs_now = \
            first_message_from_each_topic(message_iterator, topics,
                                          start = t0,
                                          stop  = t1,
                                          max_time_spread_s = max_time_spread_s,
                                          verbose = verbose)

        if msgs_now is None:
            # End of file
            break

        Nstored = sum(0 if msg is None else 1 for msg in msgs_now)
        if Nstored < require_at_least_N_topics:
            # There's more data available in the file, but THIS time segment
            # doesn't have enough
            continue

        msgs.append(msgs_now)

        if verbose:
            isnapshot = len(msgs)-1
            print(f"{isnapshot=}: at time_ns = {['-' if m is None else m['time_ns'] for m in msgs_now]} (spread={_time_spread_s(msgs_now):.2f}s) '{bag}'")

    return msgs


def topics(bag):
    with rosbags.highlevel.anyreader.AnyReader( (pathlib.Path(bag),) ) as reader:
        return [c.topic for c in reader.connections]

def info(bag):
    with rosbags.highlevel.anyreader.AnyReader( (pathlib.Path(bag),) ) as reader:
        return dict( topics = [c.topic for c in reader.connections],
                     t0     = reader.start_time,
                     t1     = reader.end_time )

def print_info(bag):

    d = info(bag)
    t0 = d['t0']
    t1 = d['t1']

    print(f"Bag '{bag}':")

    print(f"  {t0=} {t1=} duration={(t1-t0)/1e9:.1f} seconds")
    for topic in d['topics']:
        print(f"  {topic}")
