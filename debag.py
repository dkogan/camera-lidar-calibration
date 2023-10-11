#!/usr/bin/python3
# Software License Agreement (BSD License)
#
# Copyright (c) 2008, Willow Garage, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of Willow Garage, Inc. nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Revision $Id$


# This is a hacked "rostopic echo" to produce .vnl with binary data in separate
# files
#
# The repo with patches is here:
#   https://nora.jpl.nasa.gov/common/python3-rostopic/-/tree/patch-queue/master
#
# It is a fork off the Debian ROS comm package:
#   https://salsa.debian.org/science-team/ros-ros-comm.git
#
# The patches are in our patch-queue/master branch

import os
import sys
import traceback
import genpy

import rosgraph
import rospy

import numpy as np
import mrcal
import rosbag

long = int

class ROSTopicException(Exception):
    """
    Base exception class of rostopic-related errors
    """
    pass

def _isstring_type(t):
    valid_types = [str]
    try:
        valid_types.append(unicode)
    except NameError:
        pass
    return t in valid_types

# code adapted from rqt_plot
def msgevalgen(pattern):
    """
    Generates a function that returns the relevant field(s) (aka 'subtopic(s)') of a Message object
    :param pattern: subtopic, e.g. /x[2:]/y[:-1]/z, ``str``
    :returns: function that converts a message into the desired value, ``fn(Message) -> value``
    """
    evals = []  # list of (field_name, slice_object) pairs
    fields = [f for f in pattern.split('/') if f]
    for f in fields:
        if '[' in f:
            field_name, rest = f.split('[', 1)
            if not rest.endswith(']'):
                print("missing closing ']' in slice spec '%s'" % f, file=sys.stderr)
                return None
            rest = rest[:-1]  # slice content, removing closing bracket
            try:
                array_index_or_slice_object = _get_array_index_or_slice_object(rest)
            except AssertionError as e:
                print("field '%s' has invalid slice argument '%s': %s"
                      % (field_name, rest, str(e)), file=sys.stderr)
                return None
            evals.append((field_name, array_index_or_slice_object))
        else:
            evals.append((f, None))

    def msgeval(msg, evals):
        for i, (field_name, slice_object) in enumerate(evals):
            try: # access field first
                msg = getattr(msg, field_name)
            except AttributeError:
                print("no field named %s in %s" % (field_name, pattern), file=sys.stderr)
                return None

            if slice_object is not None: # access slice
                try:
                    msg = msg.__getitem__(slice_object)
                except IndexError as e:
                    print("%s: %s" % (str(e), pattern), file=sys.stderr)
                    return None

                # if a list is returned here (i.e. not only a single element accessed),
                # we need to recursively call msg_eval() with the rest of evals
                # in order to handle nested slices
                if isinstance(msg, list):
                    rest = evals[i + 1:]
                    return [msgeval(m, rest) for m in msg]
        return msg

    return (lambda msg: msgeval(msg, evals)) if evals else None


def _get_array_index_or_slice_object(index_string):
    assert index_string != '', 'empty array index'
    index_string_parts = index_string.split(':')
    if len(index_string_parts) == 1:
        try:
            array_index = int(index_string_parts[0])
        except ValueError:
            assert False, "non-integer array index step '%s'" % index_string_parts[0]
        return array_index

    slice_args = [None, None, None]
    if index_string_parts[0] != '':
        try:
            slice_args[0] = int(index_string_parts[0])
        except ValueError:
            assert False, "non-integer slice start '%s'" % index_string_parts[0]
    if index_string_parts[1] != '':
        try:
            slice_args[1] = int(index_string_parts[1])
        except ValueError:
            assert False, "non-integer slice stop '%s'" % index_string_parts[1]
    if len(index_string_parts) > 2 and index_string_parts[2] != '':
            try:
                slice_args[2] = int(index_string_parts[2])
            except ValueError:
                assert False, "non-integer slice step '%s'" % index_string_parts[2]
    if len(index_string_parts) > 3:
        assert False, 'too many slice arguments'
    return slice(*slice_args)

def _get_nested_attribute(msg, nested_attributes):
    value = msg
    for attr in nested_attributes.split('/'):
        value = getattr(value, attr)
    return value

def _str_plot_fields(val, f, field_filter):
    """
    get vnl representation of fields used by _str_plot
    :returns: list of fields as a vnl string, ``str``
    """
    s = _sub_str_plot_fields(val, f, field_filter)
    if s is not None:
        return "time "+s
    else:
        return 'time '

def _sub_str_plot_fields(val, f, field_filter):
    """recursive helper function for _str_plot_fields"""
    # vnl
    type_ = type(val)
    if type_ in (bool, int, long, float) or \
           isinstance(val, genpy.TVal):
        return f
    # duck-type check for messages
    elif hasattr(val, "_slot_types"):
        if field_filter is not None:
            if type(val).__name__ == '_sensor_msgs__Image':
                fields = [ 'field.header.seq',
                           'field.header.stamp',
                           'field.header.frame_id',
                           'image' ]
                return ' '.join(fields)
            elif type(val).__name__ == '_sensor_msgs__PointCloud2':
                fields = [ 'field.header.seq',
                           'field.header.stamp',
                           'field.header.frame_id',
                           'points' ]
                return ' '.join(fields)

            else:
                fields = list(field_filter(val))

        else:
            fields = val.__slots__
        sub = (_sub_str_plot_fields(_convert_getattr(val, a, t), f+"."+a, field_filter) for a,t in zip(val.__slots__, val._slot_types) if a in fields)
        sub = [s for s in sub if s is not None]
        if sub:
            return ' '.join([s for s in sub])
    elif _isstring_type(type_):
        return f
    elif type_ in (list, tuple):
        if len(val) == 0:
            return None
        val0 = val[0]
        type0 = type(val0)
        # no arrays of arrays
        if type0 in (bool, int, long, float) or \
               isinstance(val0, genpy.TVal):
            return ' '.join(["%s%s"%(f,x) for x in range(0,len(val))])
        elif _isstring_type(type0):
            
            return ' '.join(["%s%s"%(f,x) for x in range(0,len(val))])
        elif hasattr(val0, "_slot_types"):
            labels = ["%s%s"%(f,x) for x in range(0,len(val))]
            sub = [s for s in [_sub_str_plot_fields(v, sf, field_filter) for v,sf in zip(val, labels)] if s]
            if sub:
                return ' '.join([s for s in sub])
    return None


def _str_plot(val, time_offset=None, current_time=None, field_filter=None, type_information=None, fixed_numeric_width=None, output_directory = None):
    """
    Convert value to matlab/octave-friendly vnl string representation.

    :param val: message
    :param current_time: current :class:`genpy.Time` to use if message does not contain its own timestamp.
    :param time_offset: (optional) for time printed for message, print as offset against this :class:`genpy.Time`
    :param field_filter: filter the fields that are stringified for Messages, ``fn(Message)->iter(str)``
    :returns: comma-separated list of field values in val, ``str``
    """
        
    s = _sub_str_plot(val, time_offset, field_filter, output_directory)
    if s is None:
        s = ''

    if time_offset is not None:
        time_offset = time_offset.to_nsec()
    else:
        time_offset = 0            
        
    if current_time is not None:
        return "%s %s"%(current_time.to_nsec()-time_offset, s)
    elif getattr(val, "_has_header", False):
        return "%s %s"%(val.header.stamp.to_nsec()-time_offset, s)
    else:
        return "%s %s"%(rospy.get_rostime().to_nsec()-time_offset, s)

def _sub_str_plot(val, time_offset, field_filter, output_directory):
    """Helper routine for _str_plot."""
    # vnl
    type_ = type(val)
    
    if type_ == bool:
        return '1' if val else '0'
    elif type_ in (int, long, float) or \
           isinstance(val, genpy.TVal):
        if time_offset is not None and isinstance(val, genpy.Time):
            return str(val-time_offset)
        else:
            return str(val)    
    elif hasattr(val, "_slot_types"):
        if field_filter is not None:

            if type(val).__name__ == '_sensor_msgs__Image':

                if output_directory is None:
                    raise Exception("Need valid --output-directory to write out the image vnl")

                if val.encoding == 'mono8':
                    if val.step != val.width:
                        raise Exception(f"Got _sensor_msgs__Image.encoding == mono8. Expecting dense storage, but step != width: {val.step} != {val.width}")
                    if len(val.data) != val.width*val.height:
                        raise Exception(f"Got _sensor_msgs__Image.encoding == mono8. Expecting dense storage, but len(data) != width*height: {len(val.data)} != {val.width}*{val.height}")

                    image = \
                        np.frombuffer(val.data,
                                      dtype = np.uint8).reshape((val.height, val.width),)
                    directory      = f"{output_directory}/{val.header.frame_id}"
                    os.makedirs(directory, exist_ok = True)

                    try:    i_image = _sub_str_plot.i_image
                    except: i_image = 0
                    filename = f"{directory}/image{i_image:05d}.png"
                    mrcal.save_image(filename, image)
                    _sub_str_plot.i_image = i_image + 1

                    fields = ['header', 'image']

                    return \
                        _sub_str_plot(_convert_getattr(val, 'header', 'std_msgs/Header'), time_offset, field_filter, output_directory) + \
                        ' ' + filename

                else:
                    raise Exception(f"I only support mono8 images for now. Got {val.encoding=}")
            elif type(val).__name__ == '_sensor_msgs__PointCloud2':

                if output_directory is None:
                    raise Exception("Need valid --output-directory to write out the lidar points vnl")

                # I have
                #
                #   $ rosmsg info sensor_msgs/PointCloud2
                #   std_msgs/Header header
                #     uint32 seq
                #     time stamp
                #     string frame_id
                #   uint32 height
                #   uint32 width
                #   sensor_msgs/PointField[] fields
                #     uint8 INT8=1
                #     uint8 UINT8=2
                #     uint8 INT16=3
                #     uint8 UINT16=4
                #     uint8 INT32=5
                #     uint8 UINT32=6
                #     uint8 FLOAT32=7
                #     uint8 FLOAT64=8
                #     string name
                #     uint32 offset
                #     uint8 datatype
                #     uint32 count
                #   bool is_bigendian
                #   uint32 point_step
                #   uint32 row_step
                #   uint8[] data
                #   bool is_dense
                #
                # I assume the above type IDs are always the ones being used. No
                # obvious way to pull them out of the "val" object
                types = { 1: np.int8,
                          2: np.uint8,
                          3: np.int16,
                          4: np.uint16,
                          5: np.int32,
                          6: np.uint32,
                          7: np.float32,
                          8: np.float64 }

                fmts = { np.int8:    '%d',
                         np.uint8:   '%d',
                         np.int16:   '%d',
                         np.uint16:  '%d',
                         np.int32:   '%d',
                         np.uint32:  '%d',
                         np.float32: '%.8f',
                         np.float64: '%.8f' }


                dtype_dict = dict()
                for f in val.fields:
                    if f.count != 1:
                        raise Exception("Only scalar types supported. I don't know how to specify both an offset AND a count in a dtype")
                    dtype_dict[f.name] = (types[f.datatype], f.offset)

                dtype_dict['pad'] = (np.uint8,47)
                dtype = np.dtype(dtype_dict, align=True)

                points = \
                    np.frombuffer(val.data, dtype = dtype)

                directory      = f"{output_directory}/{val.header.frame_id}"
                os.makedirs(directory, exist_ok = True)

                try:    i_points = _sub_str_plot.i_points
                except: i_points = 0
                filename = f"{directory}/points{i_points:05d}.vnl"

                np.savetxt(filename, points,
                           header = ' '.join(dtype.names),
                           fmt    = [fmts[types[f.datatype]] for f in val.fields] + ["%d",])


                _sub_str_plot.i_points = i_points + 1

                fields = ['header', 'points']

                return \
                    _sub_str_plot(_convert_getattr(val, 'header', 'std_msgs/Header'), time_offset, field_filter, output_directory) + \
                    ' ' + filename

            else:
                fields = list(field_filter(val))
        else:
            fields = val.__slots__            

        sub = (_sub_str_plot(_convert_getattr(val, f, t), time_offset, field_filter, output_directory) for f,t in zip(val.__slots__, val._slot_types) if f in fields)
        sub = [s for s in sub if s is not None]
        if sub:
            return ' '.join(sub)
    elif _isstring_type(type_):
        return val
    elif type_ in (list, tuple):
        if len(val) == 0:
            return None
        val0 = val[0]
        # no arrays of arrays
        type0 = type(val0)
        if type0 == bool:
            return ' '.join([('1' if v else '0') for v in val])
        elif type0 in (int, long, float) or \
               isinstance(val0, genpy.TVal):
            return ' '.join([str(v) for v in val])
        elif _isstring_type(type0):
            return ' '.join([v for v in val])            
        elif hasattr(val0, "_slot_types"):
            sub = [s for s in [_sub_str_plot(v, time_offset, field_filter, output_directory) for v in val] if s is not None]
            if sub:
                return ' '.join([s for s in sub])
    return None

# copied from roslib.message
def _convert_getattr(val, f, t):
    """
    Convert atttribute types on the fly, if necessary.  This is mainly
    to convert uint8[] fields back to an array type.
    """
    attr = getattr(val, f)
    if _isstring_type(type(attr)) and 'uint8[' in t:
        return [ord(x) for x in attr]
    else:
        return attr

class CallbackEcho(object):
    """
    Callback instance that can print callback data in a variety of
    formats. Used for all variants of rostopic echo
    """

    def __init__(self, topic, filter_fn=None,
                 echo_all_topics=False,
                 offset_time=False, count=None,
                 field_filter_fn=None, fixed_numeric_width=None,
                 output_directory = None):
        """
        :param filter_fn: function that evaluates to ``True`` if message is to be echo'd, ``fn(topic, msg)``
        :param echo_all_topics: (optional) if ``True``, echo all messages in bag, ``bool``
        :param offset_time: (optional) if ``True``, display time as offset from current time, ``bool``
        :param count: number of messages to echo, ``None`` for infinite, ``int``
        :param field_filter_fn: filter the fields that are stringified for Messages, ``fn(Message)->iter(str)``
        :param fixed_numeric_width: fixed width for numeric values, ``None`` for automatic, ``int``
        """
        if topic and topic[-1] == '/':
            topic = topic[:-1]
        self.topic = topic
        self.output_directory = output_directory
        self.filter_fn = filter_fn
        self.fixed_numeric_width = fixed_numeric_width

        self.echo_all_topics = echo_all_topics
        self.offset_time = offset_time

        # done tracks when we've exceeded the count
        self.done = False
        self.max_count = count
        self.count = 0

        self.field_filter=field_filter_fn

        # first tracks whether or not we've printed anything yet. Need this for printing plot fields.
        self.first = True

        # cache
        self.last_topic = None
        self.last_msg_eval = None

    def callback(self, data, callback_args, current_time=None):
        """
        Callback to pass to rospy.Subscriber or to call
        manually. rospy.Subscriber constructor must also pass in the
        topic name as an additional arg
        :param data: Message
        :param topic: topic name, ``str``
        :param current_time: override calculation of current time, :class:`genpy.Time`
        """
        topic = callback_args['topic']
        type_information = callback_args.get('type_information', None)
        if self.filter_fn is not None and not self.filter_fn(data):
            return

        if self.max_count is not None and self.count >= self.max_count:
            self.done = True
            return
        
        try:
            msg_eval = None
            if topic == self.topic:
                pass
            elif self.topic.startswith(topic + '/'):
                # self.topic is actually a reference to topic field, generate msgeval
                if topic == self.last_topic:
                    # use cached eval
                    msg_eval = self.last_msg_eval
                else:
                    # generate msg_eval and cache
                    self.last_msg_eval = msg_eval = msgevalgen(self.topic[len(topic):])
                    self.last_topic = topic
            elif not self.echo_all_topics:
                return

            if msg_eval is not None:
                data = msg_eval(data)
                
            # data can be None if msg_eval returns None
            if data is not None:
                # NOTE: we do all prints using direct writes to sys.stdout, which works better with piping
                
                self.count += 1
                
                # print fields header for plot
                if self.first:
                    sys.stdout.write("# "+_str_plot_fields(data, 'field', self.field_filter)+'\n')
                    self.first = False

                if self.offset_time:
                    sys.stdout.write(_str_plot(data, time_offset=rospy.get_rostime(),
                                               current_time=current_time, field_filter=self.field_filter,
                                               type_information=type_information, fixed_numeric_width=self.fixed_numeric_width,
                                               output_directory=self.output_directory) + '\n')
                else:
                    sys.stdout.write(_str_plot(data,
                                               current_time=current_time, field_filter=self.field_filter,
                                               type_information=type_information, fixed_numeric_width=self.fixed_numeric_width,
                                               output_directory=self.output_directory) + '\n')

                # we have to flush in order before piping to work
                sys.stdout.flush()
            # #2778 : have to check count after incr to set done flag
            if self.max_count is not None and self.count >= self.max_count:
                self.done = True

        except IOError:
            self.done = True
        except:
            # set done flag so we exit
            self.done = True
            traceback.print_exc()

def _rostopic_echo(topic, callback_echo, bag_file, echo_all_topics=False):
    """
    Print new messages on topic to screen.

    :param topic: topic name, ``str``
    :param bag_file: name of bag file to echo messages from or ``None``, ``str``
    """
    # we have to init a node regardless and bag echoing can print timestamps

    # initialize rospy time due to potential timestamp printing
    rospy.rostime.set_rostime_initialized(True)
    if not os.path.exists(bag_file):
        raise ROSTopicException("bag file [%s] does not exist"%bag_file)
    first = True

    with rosbag.Bag(bag_file) as b:
        for t, msg, timestamp in b.read_messages():
        # bag files can have relative paths in them, this respects any
            # dynamic renaming
            if t[0] != '/':
                t = rosgraph.names.script_resolve_name('rostopic', t)
            callback_echo.callback(msg, {'topic': t}, current_time=timestamp)
            # done is set if there is a max echo count
            if callback_echo.done:
                break



def debag(bag,
          topic,
          *,
          fixed_numeric_width = None,
          filter_expr         = None,
          nostr               = False,
          noarr               = False,
          all_topics          = False,
          msg_count           = None,
          offset_time         = False,
          output_directory    = "/tmp"):
    r'''The main function in this module

    The input is a file on disk. Output is text on standard output and possibly
    more files on disk
    '''

    def create_field_filter(echo_nostr, echo_noarr):
        def field_filter(val):
            fields = val.__slots__
            field_types = val._slot_types
            for f, t in zip(val.__slots__, val._slot_types):
                if echo_noarr and '[' in t:
                    continue
                elif echo_nostr and 'string' in t:
                    continue
                yield f
        return field_filter

    def expr_eval(expr):
        def eval_fn(m):
            return eval(expr)
        return eval_fn



    filter_fn = None
    if filter_expr:
        filter_fn = expr_eval(filter_expr)

    field_filter_fn = create_field_filter(nostr, noarr)

    callback_echo = CallbackEcho(topic,
                                 filter_fn=filter_fn,
                                 echo_all_topics=all_topics,
                                 offset_time=offset_time, count=msg_count,
                                 field_filter_fn=field_filter_fn,
                                 fixed_numeric_width=fixed_numeric_width,
                                 output_directory=output_directory)
    _rostopic_echo(topic, callback_echo, bag_file=bag)


if __name__ == "__main__":

    args = sys.argv[1:]
    from optparse import OptionParser
    parser = OptionParser(usage="usage: %prog echo [options] /topic", prog=sys.argv[0])
    parser.add_option("-b", "--bag",
                      dest="bag",
                      help="echo messages from .bag file", metavar="BAGFILE")
    parser.add_option("-w",
                      dest="fixed_numeric_width", default=None, metavar="NUM_WIDTH",
                      help="fixed width for numeric values")
    parser.add_option("--filter", 
                      dest="filter_expr", default=None,
                      metavar="FILTER-EXPRESSION",
                      help="Python expression to filter messages that are printed. Expression can use Python builtins as well as m (the message) and topic (the topic name).")
    parser.add_option("--nostr", 
                      dest="nostr", default=False,
                      action="store_true",
                      help="exclude string fields")
    parser.add_option("--noarr",
                      dest="noarr", default=False,
                      action="store_true",
                      help="exclude arrays")
    parser.add_option("-a", "--all",
                      dest="all_topics", default=False,
                      action="store_true",
                      help="display all message in bag, only valid with -b option")
    parser.add_option("-n", 
                      dest="msg_count", default=None, metavar="COUNT",
                      help="number of messages to echo")
    parser.add_option("--offset",
                      dest="offset_time", default=False,
                      action="store_true",
                      help="display time as offsets from current time (in seconds)")
    parser.add_option("--output-directory",
                      help="""The output directory to store the extracted data.
                      Currently this is used (and required) only for images and
                      LIDAR scans""")

    (options, args) = parser.parse_args(args)
    if not options.bag:
        parser.error("The bag is required")
    if len(args) > 1:
        parser.error("you may only specify one input topic")
    if options.offset_time:
        parser.error("offset time option is not valid with bag files")
    if options.all_topics:
        topic = ''
    else:
        if len(args) == 0:
            parser.error("topic must be specified")        
        topic = rosgraph.names.script_resolve_name('rostopic', args[0])
        # suppressing output to keep it clean

    try:
        options.msg_count = int(options.msg_count) if options.msg_count else None
    except ValueError:
        parser.error("COUNT must be an integer")

    try:
        options.fixed_numeric_width = int(options.fixed_numeric_width) if options.fixed_numeric_width else None
        if options.fixed_numeric_width is not None and options.fixed_numeric_width < 2:
            parser.error("Fixed width for numeric values must be at least 2")
    except ValueError:
        parser.error("NUM_WIDTH must be an integer")

    debag(options.bag,
          topic,
          fixed_numeric_width = options.fixed_numeric_width,
          filter_expr         = options.filter_expr,
          nostr               = options.nostr,
          noarr               = options.noarr,
          all_topics          = options.all_topics,
          msg_count           = options.msg_count,
          offset_time         = options.offset_time,
          output_directory    = options.output_directory)
