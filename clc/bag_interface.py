#!/usr/bin/python3

import sys
import numpy as np

import pathlib
import rosbags.highlevel.anyreader
import rosbags.typesys
import re
import dateutil


def _parse_timestamp_to_ns_since_epoch(t):
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


def messages(bag, topics,
             *,
             start = None,
             stop  = None,
             ignore_unknown_message_types = False):

    r'''Iterator to read messages from a rosbag

SYNOPSIS

    import clc

    bag = '/path/to/bag'
    for msg in clc.bag_interface.messages(bag, ('/lidar1/points', '/camera1/image')):

        print(msg['time_header_ns'])
        # timestamp when this message was produced by the hardware, in ns since the epoch

        print(msg['time_ns'])
        # timestamp when this message was recorded by the logging tool, in ns since the epoch

        print(msg['frame_id'])
        # The geometric frame of this sensor

        if msg['msgtype'] == 'sensor_msgs/msg/PointCloud2':
            # This is a single revolution of the lidar

            # 3D points in the LIDAR frame. Numpy array of shape (Npoints,3)
            points = msg['array']['xyz']

        elif msg['msgtype'] == 'sensor_msgs/msg/Image':
            # A single image from a camera

            # Numpy array of shape (H,W) for grayscale images or (H,W,3) for
            # color images
            image = msg['array']

As of today, clc uses ROS bags as its input data storage format. A rosbag may
contain multiple data streams with a "topic" string identifying each one. The
data stream for any given topic is a series of messages of identical data type.
messages() is an iterator that produces messages "msg" one at a time. clc is
primarily interested in lidar scans (msg['msgtype'] =
'sensor_msgs/msg/PointCloud2') and images (msg['msgtype'] =
'sensor_msgs/msg/Image').

Each message yielded by this iterator is a dict with keys:

- time_ns:        timestamp when this message was recorded by the logging tool

- time_header_ns: timestamp when this messages was produced by the hardware. If
                  unavailable, same as time_ns

- frame_id:       The name of the geometric frame of this sensor. None if
                  unavailable

- topic:          string identifying this topic

- msgtype:        string identifying the type of this message

- array:          parsed numpy array representing this data. This depends on the
                  specific msgtype and the hardware and its driver. At this time
                  I only parse LIDAR and camera data. Other messages will
                  produce None. See below for details

- rawdata:        Unparsed bytes representing this message. Will never be None.

- qos:            ros IPC quality-of-service data

Each LIDAR message contains readings for a single revolution of the lidar. The
data is returned in a numpy array msg['array']. This has a compound dtype,
containing much metadata; the exact set varies depending on the LIDAR model and
its driver. msg['array']['xyz'] will always be a point cloud of shape
(Npoints,3). These are represented in the LIDAR's local coordinate system: xyz
is (forward,right,down). Invalid points will usually have p = (0,0,0) for that
point. Each point will usually also have intensity information in
msg['array']['intensity'] and "ring" information in msg['array']['ring']. The
"ring" identifies the specific laser that was used to capture this point. Since
the lasers are rigidly mounted on a spindle rotating around the z axis, the
elevation = atan( z, mag(x,y) ) of every point will be constant for each ring.

Each camera image is stored in msg['array']. This is a 2D array of shape (H,W)
for grayscale images or a 3D array of shape (H,W,3) for color images.

clc does NOT actually use ROS, and it is NOT required to be installed; instead
it uses the "rosbags" library to read the data.

ARGUMENTS

- bag: required string. The path on disk to the bag containing the data we're
  reading. Everything supported by "rosbags" is valid here. Specifically, both
  ros1 and ros2 data should work

- topics: required iterable for the topics we're interested in. If we're
  interested in a single topic, this should still be an iterable: something like
  (topic,)

- start: optional timestamp, indicating the start point in the bag. If None or
  omitted, we start at the beginning. These timestamps are compared against
  msg['time_ns']: the time when the message was recorded, NOT when the sensor
  produced it. This timestamp is interpreted differently, based on its type. If
  it's an integer: we interpret it as "sec since the epoch" or "nanosec since
  epoch", depending on the numerical value. If it's a float: "sec since epoch".
  If a string: we parse it as an integer or float OR as a freeform string using
  dateutil.parser.parse().

- stop: optional timestamp, indicating the end point in the bag. If None or
  omitted, we read the bag to the end. The details are the same as with the
  "start" timestamp.

- ignore_unknown_message_types: optional boolean, defaulting to False. If
  ignore_unknown_message_types: we will not yield any messages that aren't LIDAR
  scans or images. Else: we will return all messages for the requested topics.
  Unknown messages may have msg['array'] is None

RETURNED VALUE

An iterator, producing a dict for each message.

    '''

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

    messages.re_report_message = None

    # High-level structure from the "rosbag2" sample:
    #   https://ternaris.gitlab.io/rosbags/topics/rosbag2.html
    with rosbags.highlevel.anyreader.AnyReader( (pathlib.Path(bag),) ) as reader:

        # I expect exactly one matching connection for each topic given
        connections = [ c \
                        for topic in topics \
                        for c in reader.connections \
                        if c.topic == topic ]
        connections = [ c for c in connections if c is not None ]

        if len(connections) == 0: return

        for connection, time_ns, rawdata in \
                reader.messages( connections = connections,
                                 start = _parse_timestamp_to_ns_since_epoch(start),
                                 stop  = _parse_timestamp_to_ns_since_epoch(stop)):

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


                ######### making a copy. This is necessary to make the array
                ######### writeable, which is needed for
                ######### mask_out_static_scene(). I can make a copy in
                ######### mask_out_static_scene() instead, but this causes
                ######### issues downstream, where the stride of points and
                ######### rings must be exactly the same
                data  = np.array( np.frombuffer(msg.data, dtype = dtype) )

                # 2023-11-01 dataset contains data that is almost completely
                # comprised of duplicated points. The only difference in these
                # duplicates is the 'ret' field. This is always 0 or 1. Mostly these
                # alternate, with a small period of all 1. I have no idea what this
                # means, so I only take ret==0; this suppresses the duplicates
                if 'ret' in dtype.fields:
                    data = data[data['ret'] == 0]


                # Some datasets in the test suite were taken indoors. The floor
                # and ceiling are confusing the lidar segmentation right now. So
                # I explicitly throw them out; just for now hopefully
                def env_true(n):
                    import os
                    v = os.environ.get(n, False)
                    try:    v = int(v)
                    except: pass
                    return bool(v)

                # Deegan
                if env_true('CLC_CULL_FLOOR_CEILING_1'):
                    import mrcal
                    rt_world_lidar = np.array(( 0,.225,0,
                                                0,0,0,),
                                              dtype=float)
                    p = data['xyz']
                    p = mrcal.transform_point_rt(rt_world_lidar, p.astype(float))
                    i = (p[...,2] > -0.3) * (p[...,2] < 2)
                    data = data[i]
                # Casey Majhor
                elif env_true('CLC_CULL_FLOOR_CEILING_2'):
                    import mrcal
                    rt_world_lidar = np.array(( 0,-.11,0,
                                                0,0,0,),
                                              dtype=float)
                    p = data['xyz']
                    p = mrcal.transform_point_rt(rt_world_lidar, p.astype(float))
                    i = (p[...,2] > -1.5) * (p[...,2] < 1.5)
                    data = data[i]




            elif re.search(r'\bsensor_msgs__msg__Image\b', str(type(msg))):
                data  = msg.data
                shape = data.shape

                if msg.encoding == 'mono8':
                    if not ( len(shape) == 1 and \
                             shape[0] == msg.step*msg.height and \
                             data.dtype == np.uint8 and \
                             msg.width == msg.step ):
                        raise Exception("rosbags.usertypes.sensor_msgs__msg__Image data should contain a flattened, dense byte array")
                    data = data.reshape(msg.height,msg.width)
                elif msg.encoding == 'mono16':
                    if not ( len(shape) == 1 and \
                             shape[0] == msg.step*msg.height and \
                             data.dtype == np.uint8 and \
                             msg.width*2 == msg.step ):
                        raise Exception("rosbags.usertypes.sensor_msgs__msg__Image data should contain a flattened, dense byte array")
                    data = data.view(np.uint16).reshape(msg.height,msg.width)
                elif msg.encoding == 'bgr8':
                    if not ( len(shape) == 1 and \
                             shape[0] == msg.step*msg.height and \
                             data.dtype == np.uint8 and \
                             msg.width*3 == msg.step ):
                        raise Exception("rosbags.usertypes.sensor_msgs__msg__Image data should contain a flattened, dense byte array")
                    data = data.reshape(msg.height,msg.width,3)
                else:
                    raise Exception(f"Unknown {msg.encoding=}")
            elif re.search(r'\bsensor_msgs__msg__CompressedImage\b', str(type(msg))):
                if msg.format == 'rgb8; jpeg compressed bgr8':
                    import cv2
                    data = cv2.imdecode(msg.data, cv2.IMREAD_COLOR)
                elif msg.format == 'mono8; png compressed ':
                    import cv2
                    data = cv2.imdecode(msg.data,
                                        cv2.IMREAD_GRAYSCALE + cv2.IMREAD_ANYDEPTH)
                elif msg.format == 'mono16; png compressed ':
                    import cv2
                    data = cv2.imdecode(msg.data,
                                        cv2.IMREAD_GRAYSCALE + cv2.IMREAD_ANYDEPTH)
                elif msg.format == 'bgr8; png compressed bgr8':
                    import cv2
                    data = cv2.imdecode(msg.data,
                                        cv2.IMREAD_COLOR)
                else:
                    raise Exception(f"Have {type(msg)=}; don't know how to interpret {msg.format=}")
            else:
                if ignore_unknown_message_types:
                    continue
                data = None

            try:    time_header_ns = msg.header.stamp.sec*1000000000 + msg.header.stamp.nanosec
            except: time_header_ns = time_ns

            try:    frame_id = msg.header.frame_id
            except: frame_id = None

            yield dict( time_ns        = time_ns,
                        time_header_ns = time_header_ns,
                        frame_id       = frame_id,
                        topic          = connection.topic,
                        msgtype        = connection.msgtype,
                        array          = data,
                        msg            = msg,
                        qos            = qos )


def first_message_from_each_topic(bag,
                                  topics,
                                  *,
                                  start             = None,
                                  stop              = None,
                                  max_time_spread_s = None,
                                  verbose           = False):
    r'''Report the first message in the bag for each topic

SYNOPSIS

    import clc

    bag      = '/path/to/bag'
    topics   = ('/lidar1/points', '/camera1/image')
    messages = clc.bag_interface.first_message_from_each_topic(bag, topics)

clc tries to geometrically align data from multiple sensors. Each sensor
produces data in a ros "topic", and the data we're ingesting comes from one or
more "rosbags".

We want to get a set of time-synchronized "snapshots" from the data, reporting
observations of a moving calibration object by a set of stationary sensors. Each
snapshot should report observations from a single instant in time.

One strategy for capturing such input data is to record a small rosbag for each
stationary configuration of sensors and the calibration object. In this
scenario, any set of messages in each bag is a good representation of a
stationary sensor "snapshot", and we can use the first_message_from_each_topic()
function to read off such a snapshot.

Another strategy is to record one bag that contains ALL the data, including the
transitions between stationary poses. We would then need slow motions and tight
timings and outlier rejection to get consistent, stationary snapshots. in that
scenario we can use first_message_from_each_topic() with start,stop specifying
moving time windows and a tight max_time_spread_s to reduce synchronization
errors. This is largely what first_message_from_each_topic_in_time_segments()
does.

For details about rosbags and how we read them, and what each "message"
contains, see the docstring for clc.bag_interface.messages().

This function calls messages(ignore_unknown_message_types = True), so there's an
expectation that this will be used for LIDAR and/or camera data only.

ARGUMENTS

- bag: a required string (a path on disk pointing to a readable rosbag) or an
  existing iterator returned by clc.bag_interface.messages().

- topics: required iterable for the topics we're interested in. If we're
  interested in a single topic, this should still be an iterable: something like
  (topic,)

- start: optional timestamp, indicating the start point in the bag. If None or
  omitted, we start at the beginning. These timestamps are compared against
  msg['time_ns']: the time when the message was recorded, NOT when the sensor
  produced it. This timestamp is interpreted differently, based on its type. If
  it's an integer: we interpret it as "sec since the epoch" or "nanosec since
  epoch", depending on the numerical value. If it's a float: "sec since epoch".
  If a string: we parse it as an integer or float OR as a freeform string using
  dateutil.parser.parse().

- stop: optional timestamp, indicating the end point in the bag. If None or
  omitted, we read the bag to the end. The details are the same as with the
  "start" timestamp.

- max_time_spread_s: optional value for the worst-possible acceptable spread
  between the earliest and latest message in the returned set. If omitted or
  None, no checks are performed on the message timestamps. If max_time_spread_s
  is given, the returned set WILL satisfy this requirement, but the returned
  messages for some of the topics may be None, if the requirement could not be
  satisfied. The algorithm used to satisfy this requirement is heuristic and NOT
  optimal, but should be sufficient to return usable data

- verbose: optional boolean, defaulting to False. Report diagnostics. At this
  time, this does nothing

RETURNED VALUE

Under nominal conditions, a list of messages, corresponding to each topic in
topics. Some/all of the messages in the list may be None if no valid data for
that topic was found.

If more data is available in the bag, past the given stop time, the list will be
returned, possibly with all-None entries.

If we have reached the end of the bag, and no more data is available, we return
None instead of a list

    '''

    if max_time_spread_s is None:
        max_time_spread_ns = None
    else:
        max_time_spread_ns = max_time_spread_s * 1e9

    idx = dict()
    for i,t in enumerate(topics): idx[t] = i


    out = [None] * len(topics)
    Nstored = 0
    time_max,time_min = None,None

    def accumulate(msg, itopic):
        nonlocal out
        nonlocal Nstored
        nonlocal time_max,time_min


        if max_time_spread_ns is None:
            if out[itopic] is None:
                out[itopic] = msg
                Nstored += 1
            return

        time = msg['time_header_ns']

        # We're trying to meet a worst-case-time-error requirement in
        # max_time_spread_ns. The data will come in more-or-less in order of
        # increasing time, but not 100% so. I try to be greedy, and hope that
        # eventually I'll get a usable set. This algorithm isn't perfect (it
        # isn't guaranteed to find a good set), and it isn't perfectly fast (I
        # recompute O(N) the min/max periodically instead of using a heap).
        # Should be sufficient
        if Nstored == 0:
            # this is the first message in the set:
            out[itopic] = msg
            Nstored += 1

            time_max = time
            time_min = time

        elif time_min <= time and time <= time_max:
            # new message is within existing time bounds
            if out[itopic] is None:
                # we dont already have a message for this topic
                out[itopic] = msg
                Nstored += 1
            else:
                # we DO already have a message for this topic. I take the newer
                # one of the two. Because future messages for other topics will
                # presumably sit even further in the future, and the newer
                # message will be more compatible with those
                if time <= out[itopic]['time_header_ns']:
                    # Existing message is newer. Ignore new message
                    return

                # Existing message is older. Toss existing message. THIS message
                # is neither min nor max. But the existing message I'm removing
                # might be min (can't be max because THIS message isn't max)
                out[itopic] = msg
                if Nstored == 1:
                    time_max = time
                    time_min = time
                    return

                time_min = min(m['time_header_ns'] for m in out if m is not None)
                return

        elif time > time_max:
            # new message is the new time-max

            if out[itopic] is not None:
                # we DO already have a message for this topic. I take the newer
                # one of the two. Because future messages for other topics will
                # presumably sit even further in the future, and the newer
                # message will be more compatible with those
                if time < out[itopic]['time_header_ns']:
                    # Existing message is newer. Ignore new message
                    return

                # Existing message is older. Toss existing message
                if Nstored == 1:
                    # It was the only one; just replace it
                    out[itopic] = msg
                    time_max = time
                    time_min = time
                    return

                out[itopic] = None
                Nstored -= 1

                time_min = min(m['time_header_ns'] for m in out if m is not None)

                # THIS message is about to become the new max, so I don't need
                # to set the max here; it's about to be overwritten

            # Don't already have a message for this topic. Either I didn't have
            # it, or I just removed what I had
            itopic_sorted = np.argsort(np.array([m['time_header_ns'] if m is not None else 1.e308 for m in out ], dtype=float))
            i = 0
            while time - time_min > max_time_spread_ns:
                # New message breaks the sync requirement. We expect future
                # messages to be even further out in the future, so this
                # will get worse. I throw out the youngest message, to
                # hopefully be able to use a future message for THAT topic
                Nstored -= 1
                out[itopic_sorted[i]] = None
                if Nstored == 0:
                    break

                time_min = out[itopic_sorted[i+1]]['time_header_ns']
                i += 1

            # The current min is compatible with the new message. Store the
            # new message
            if Nstored == 0: # the new message is the only message
                time_min = time
            out[itopic] = msg
            Nstored += 1
            time_max = time

        else:
            # New message is the new time-min. I expect these to mostly
            # arrive in ascending order, so I just consider this a fluke,
            # and move on
            return







    if isinstance(bag, str):
        message_iterator = messages(bag, topics,
                                    start = start,
                                    stop  = stop,
                                    ignore_unknown_message_types = True)
    else:
        message_iterator = bag
    start = _parse_timestamp_to_ns_since_epoch(start)
    stop  = _parse_timestamp_to_ns_since_epoch(stop)
    end_of_file = False
    for msg in message_iterator:
        if start is not None and msg['time_header_ns'] < start:
            continue
        if stop is not None and msg['time_header_ns'] >= stop:
            # This message is past the stop time, so we're done. If we reuse
            # this iterator with the current stop time being the new start time,
            # then I want to return this message then. I save it in the iterator
            # for that purpose

            # This is GLOBAL, and WILL BREAK if I have iterators going at the
            # same time. It's not clear how to fix that
            if not isinstance(bag, str):
                messages.re_report_message = msg
            break

        accumulate(msg,
                   itopic = idx[msg['topic']])

        if Nstored == len(out):
            break
    else:
        end_of_file = True

    if end_of_file and Nstored == 0:
        return None

    return out


def first_message_from_each_topic_in_time_segments(bag, topics,
                                                   *,
                                                   period_s,
                                                   start                     = None,
                                                   stop                      = None,
                                                   require_at_least_N_topics = 2,
                                                   verbose                   = False,
                                                   exclude_time_periods      = [],
                                                   max_time_spread_s         = None):

    r'''Report sets of messages for each topic over time

SYNOPSIS

    import clc

    bag      = '/path/to/bag'
    topics   = ('/lidar1/points', '/camera1/image')

    for messages in clc.bag_interface. \
                  first_message_from_each_topic_in_time_segments(bag, topics,
                                                                 start    = start,
                                                                 stop     = stop,
                                                                 period_s = 2,
                                                                 max_time_spread_s = 0.1)

        # "messages" is a list of messages (one per topic) in this 2-sec-long
        # time interval; the messages are guaranteed within 0.1s of each other

        for topic,msg in zip(topics,messages):
            if msg is None: continue
            # Have "msg" corresopnding to "topic"
            ....

clc tries to geometrically align data from multiple sensors. Each sensor
produces data in a ros "topic", and the data we're ingesting comes from one or
more "rosbags".

We want to get a set of time-synchronized "snapshots" from the data, reporting
observations of a moving calibration object by a set of stationary sensors. Each
snapshot should report observations from a single instant in time.

One strategy for capturing the input data is to record a small rosbag for each
stationary configuration of sensors and the calibration object. In this
scenario, any set of messages in each bag is a good representation of a
stationary sensor "snapshot", and we can use the first_message_from_each_topic()
function to read off such a snapshot.

Another strategy is to record one bag that contains ALL the data, including the
transitions between stationary poses. We would then need slow motions and tight
timings and outlier rejection to get consistent, stationary snapshots. in that
scenario we can use first_message_from_each_topic() with start,stop specifying
moving time windows and a tight max_time_spread_s to reduce synchronization
errors. This is largely what first_message_from_each_topic_in_time_segments()
does.

For details about rosbags and how we read them, and what each "message"
contains, see the docstring for clc.bag_interface.messages().

This function calls messages(ignore_unknown_message_types = True), so there's an
expectation that this will be used for LIDAR and/or camera data only.

This function creates a message iterator with messages() using the given
[start,stop] interval. Then it gets sets of messages from this iterator for each
window of period_s within that [start,stop] interval using a sequence of
first_message_from_each_topic() calls

ARGUMENTS

- bag: a required string (a path on disk pointing to a readable rosbag)

- topics: required iterable for the topics we're interested in. If we're
  interested in a single topic, this should still be an iterable: something like
  (topic,)

- period_s: required; the duration, in seconds of the time interval for each
  reported set of messages

- start: optional timestamp, indicating the start point in the bag. If None or
  omitted, we start at the beginning. These timestamps are compared against
  msg['time_ns']: the time when the message was recorded, NOT when the sensor
  produced it. This timestamp is interpreted differently, based on its type. If
  it's an integer: we interpret it as "sec since the epoch" or "nanosec since
  epoch", depending on the numerical value. If it's a float: "sec since epoch".
  If a string: we parse it as an integer or float OR as a freeform string using
  dateutil.parser.parse().

- stop: optional timestamp, indicating the end point in the bag. If None or
  omitted, we read the bag to the end. The details are the same as with the
  "start" timestamp.

- require_at_least_N_topics: optional integer. We only report messages sets if
  the set has valid data for at least this many topics. If a set doesn't have
  enough valid messages, we return nothing for this interval, and move on to the
  next one. If omitted, defaults to 2.

- exclude_time_periods: optional list of time periods to exclude from being
  reported by this function. Each element of the list is a (time_start,time_end)
  iterable describing that interval.

- max_time_spread_s: optional value for the worst-possible acceptable spread
  between the earliest and latest message in the returned set. If omitted or
  None, no checks are performed on the message timestamps. If max_time_spread_s
  is given, the returned set WILL satisfy this requirement, but the returned
  messages for some of the topics may be None, if the requirement could not be
  satisfied. The algorithm used to satisfy this requirement is heuristic and NOT
  optimal, but should be sufficient to return usable data

- verbose: optional boolean, defaulting to False. Report diagnostics.

RETURNED VALUE

An iterator. Each item returned by the iterator is a list of messages,
corresponding to each topic in topics. Some/all of the messages in the list may
be None if no valid data for that topic was found. Each set of messages respects
require_at_least_N_topics and max_time_spread_s.
    '''


    message_iterator = messages(bag, topics,
                                start = start)

    start = _parse_timestamp_to_ns_since_epoch(start)
    stop  = _parse_timestamp_to_ns_since_epoch(stop)
    d     = info(bag)

    exclude_time_periods = [ (_parse_timestamp_to_ns_since_epoch(t0),
                              _parse_timestamp_to_ns_since_epoch(t1)) \
                            for t0,t1 in exclude_time_periods ]

    t1 = start if start is not None else d['t0']

    N = 0

    def excluded(t0,t1, exclude_time_periods ):
        for t0_excluded,t1_excluded in exclude_time_periods:
            if t0_excluded > t0 and t0_excluded < t1 or \
               t0 > t0_excluded and t0 < t1_excluded:
                return True
        return False

    while True:
        t0 = t1
        t1 = t0 + int(period_s*1e9)

        if excluded( t0,t1, exclude_time_periods):
            continue

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

        if verbose:
            isnapshot = N

            tmin = min(m['time_header_ns'] for m in msgs_now if m is not None)
            tmax = max(m['time_header_ns'] for m in msgs_now if m is not None)
            spread_s = (tmax-tmin)/1e9

            print(f"{isnapshot=}: at time_header_ns = {['-' if m is None else m['time_header_ns'] for m in msgs_now]} (spread={spread_s:.2f}s) '{bag}'")

        N += 1
        yield msgs_now



def topics(bag):
    r'''Report the topics present in a ros bag

SYNOPSIS

    clc.bag_interface.topics(bag)
    ===> ('/lidar0/points', '/lidar1/points', '/tf_static')

ROS bags contain a set of "topics" to describe the different data streams. This
function reports which topics are present in a given bag

ARGUMENTS

- bag: a string pointing to the bag we're reading. This is either a file or a
  directory on disk. The "rosbags" library is used to read the data, so the ROS1
  and the various ROS2 formats are supported

RETURNED VALUE

A tuple of strings, describing the topics
    '''
    with rosbags.highlevel.anyreader.AnyReader( (pathlib.Path(bag),) ) as reader:
        return [c.topic for c in reader.connections]


def info(bag):
    r'''Report metadata about a ros bag

SYNOPSIS

    clc.bag_interface.info(bag)

    ===>
    {'t0': 1748372407601618910,
     't1': 1748372828435060696,
     'topics': ['/lidar0/points',
                '/lidar1/points',
                '/tf_static']}

This function reports various metadata about a ros bag. This is a dict of
key/value pairs. Currently this reports a topic list and the timestamps for the
start/end of the data in the bag. More may be added in the future.

ARGUMENTS

- bag: a string pointing to the bag we're reading. This is either a file or a
  directory on disk. The "rosbags" library is used to read the data, so the ROS1
  and the various ROS2 formats are supported

RETURNED VALUE

A dict with keys:

- t0, t1: timestamps for the start,end of the data in the bag. An integer, in ns
  since the unix epoch

- topics: a list of strings for the available topics

    '''

    with rosbags.highlevel.anyreader.AnyReader( (pathlib.Path(bag),) ) as reader:
        return dict( topics = [c.topic for c in reader.connections],
                     t0     = reader.start_time,
                     t1     = reader.end_time )
