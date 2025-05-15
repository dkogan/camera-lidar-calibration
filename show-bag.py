#!/usr/bin/python3

r'''Display the LIDAR scans from a rosbag

SYNOPSIS

  $ ./show-bag.py             \
      'camera-lidar*.bag'

  Bag 'camera-lidar-0.bag':
    /points0
    /points1
  Bag 'camera-lidar-1.bag':
    /points0
    /points1
    /image0
  ....


  $ ./show-bag.py             \
      --period 2              \
      --set "view 60,30,2"    \
      --set "xrange [-20:20]" \
      --set "yrange [-20:20]" \
      --set "zrange [-2:2]"   \
      --topic /points0 \
      'camera-lidar*.bag'

    [A plot pops up showing the LIDAR scans from the bags, updating every 2 sec]


  $ ./show-bag.py     \
      --topic /image0 \
      'camera-lidar*.bag'

    [A plot pops up showing the images in the bags. Updated with the next bag
    when the user closes the plot]

This is a display tool to show the contents of a ros bag. This is roughly
similar to the "rosbag" and "ros2 bag" and "rviz" tools, but uses the "rosbags"
Python package, and does NOT require ROS to be installed

'''


import sys
import argparse
import re
import os


def parse_args():

    parser = \
        argparse.ArgumentParser(description = __doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--maxrange',
                        type=float,
                        help = '''Applies to LIDAR data. If given, cut off the
                        points at this range''')
    parser.add_argument('--with',
                        default = 'dots',
                        help = '''Applies to LIDAR data. If given, uses the
                        requested style to plot the lidar points. The default is
                        "dots". If there aren't many points to show, this can be
                        illegible, and "points" works better''')
    parser.add_argument('--no-intensity',
                        action = 'store_true',
                        help = '''Applies to LIDAR data. By default we
                        color-code each point by intensity. With --no-intensity,
                        we plot all the points with the same color. Improves
                        legibility in some cases''')
    parser.add_argument('--ring',
                        type=int,
                        help = '''Applies to LIDAR data. If given, show ONLY
                        data from this ring. Otherwise, display all of them''')
    parser.add_argument('--xy',
                        action='store_true',
                        help = '''Applies to LIDAR data. If given, I make a 2D
                        plot, ignoring the z axis''')
    parser.add_argument('--extract-images',
                        action='store_true',
                        help = '''Applies to camera data. If given, I write the
                        images to the directory given in this argument. Exactly
                        one bag and one topic is expected. I write to the
                        directory in --outdir ('/tmp' by default) and I write at
                        most --extract-images-count (all images by default)''')
    parser.add_argument('--extract-images-count',
                        type=int,
                        help = '''If --extract-images is given I write this many
                        images at most. By default, I write ALL the images''')
    parser.add_argument('--outdir',
                        default = '/tmp',
                        help = '''The directory --extract-.... will write to.
                        The default is "/tmp/"''')
    parser.add_argument('--period',
                        type=float,
                        default = 0,
                        help = '''How much time to wait between moving to the
                        next bag; if <= 0 (the default), we wait until each
                        window is manually closed''')
    parser.add_argument('--hardcopy',
                        type=str,
                        help='''Write the output to disk, instead of an interactive plot''')
    parser.add_argument('--terminal',
                        type=str,
                        help=r'''gnuplotlib terminal. The default is good almost always, so most people don't
                        need this option''')
    parser.add_argument('--set',
                        type=str,
                        action='append',
                        help='''Extra 'set' directives to gnuplotlib. Can be given multiple times''')
    parser.add_argument('--unset',
                        type=str,
                        action='append',
                        help='''Extra 'unset' directives to gnuplotlib. Can be given multiple times''')

    parser.add_argument('--topic',
                        type=str,
                        help = '''The topic we're visualizing. Can select LIDAR
                        or camera data. If omitted, we report the topics present
                        in the bag, and we exit. If --timeline, this is a
                        ,-separated list of topics to visualize''')
    parser.add_argument('--decimation-period',
                        type=float,
                        help = '''If given, we expect ONE bag, and rather than
                        taking the first message from each bag, we take all the
                        messages from THIS bag, spaced out with a period given
                        by this argument, in seconds''')
    parser.add_argument('--after',
                        type=str,
                        help = '''If given, start reading the bags at this time.
                        Could be an integer (s since epoch or ns since epoch), a
                        float (s since the epoch) or a string, to be parsed with
                        dateutil.parser.parse()''')
    parser.add_argument('--before',
                        type=str,
                        help = '''If given, stop reading the bags at this time.
                        Could be an integer (s since epoch or ns since epoch), a
                        float (s since the epoch) or a string, to be parsed with
                        dateutil.parser.parse()''')
    parser.add_argument('--timeline',
                        type = float,
                        help = '''If given, we plot time message timeline from
                        the ONE give bag. Multiple bags not allowed. If no
                        --topic, we report ALL the topics. Takes one argument:
                        the duration (in seconds) of the requested plot. If <=
                        0, we plot the whole bag''')
    parser.add_argument('--time-header-ns',
                        action = 'store_true',
                        help = '''If given, we use the time_header_ns for
                        --timeline. This is the time the data was SENT, not the
                        time it was recorded. All the log replay code uses
                        time_ns''')
    parser.add_argument('bags',
                        type=str,
                        nargs='+',
                        help = '''Glob(s) for the rosbags that contain the lidar
                        data. Each of these will be sorted alphanumerically''')


    args = parser.parse_args()

    if args.timeline is not None:
        if len(args.bags) > 1:
            print("--timeline works only with ONE bag", file=sys.stderr)
            sys.exit(1)

    if args.extract_images:
        if len(args.bags) > 1:
            print("--extract-images expects exactly ONE bag", file=sys.stderr)
            sys.exit(1)

    if args.decimation_period is not None:
        if args.decimation_period <= 0:
            print("--decimation-period given, and it must some >0 number of seconds", file=sys.stderr)
            sys.exit(1)
        if len(args.bags) > 1:
            print("--decimation-period given, so we MUST have gotten exactly one bag", file=sys.stderr)
            sys.exit(1)

    return args


args = parse_args()



def bags():
    import glob
    for bag_glob in args.bags:
        bags = sorted(glob.glob(bag_glob))
        if len(bags) == 0:
            print(f"No files matched the glob '{bag_glob}'", file=sys.stderr)
            sys.exit(1)

        for bag in bags:
            yield(bag)


import bag_interface

if args.timeline is None and args.topic is None:
    for bag in bags():
        bag_interface.print_info(bag)
    sys.exit()



import numpy as np
import numpysane as nps
import gnuplotlib as gp
import time



if args.timeline is not None:
    bag = args.bags[0]

    if args.topic is None:
        topics = bag_interface.topics(bag)
    else:
        topics = args.topic.split(',')

    if args.extract_images:
        if len(topics) > 1:
            print("--extract-images expects exactly ONE topic", file=sys.stderr)
            sys.exit(1)


    timestamps = []

    index = dict()
    for i,t in enumerate(topics):
        index[t] = i

    t0       = None
    duration = args.timeline
    messages = bag_interface.messages(bag, topics,
                                      start = args.after,
                                      stop  = args.before,
                                      ignore_unknown_message_types = True)

    time_key = 'time_header_ns' if args.time_header_ns else 'time_ns'

    while True:
        try:
            msg = next(messages)
            time_ns = msg[time_key]
            topic   = msg['topic']
        except StopIteration:
            break

        t = time_ns/1e9
        if t0 is None:
            t0 = t
        else:
            if duration > 0 and t - t0 > duration:
                break

        timestamps.append( (t,index[topic]) )

    ytics = ','.join( [ f'"{t}" {i}' for i,t in enumerate(topics)])
    timestamps = np.array(timestamps)
    t0 = bag_interface.info(bag)['t0']/1e9
    timestamps[:,0] -= t0
    gp.plot( timestamps,
             tuplesize = -2,
             _with     = 'points',
             ymin = -0.5,
             ymax = len(topics) - 0.5,
             _set = f'ytics ({ytics})',
             xlabel = f'Time relative to {t0=:.3f}; in seconds',
             wait = True)

    sys.exit(0)




def show_lidar(bag, p,
               _with = 'dots',
               no_intensity = False):
    kwargs = dict( _set     = args.set,
                   _unset   = args.unset,
                   terminal = args.terminal,
                   hardcopy = args.hardcopy)

    xyz       = p['xyz']
    intensity = p['intensity']

    if args.ring is not None:
        i = p['ring'] == args.ring
        xyz       = xyz[i]
        intensity = intensity[i]

    if args.maxrange is not None:
        i = nps.norm2(xyz) <= args.maxrange * args.maxrange
        xyz       = xyz[i]
        intensity = intensity[i]

    if not args.xy:
        data_tuple = (xyz[:,0], xyz[:,1], xyz[:,2])
    else:
        data_tuple = (xyz[:,0], xyz[:,1])

    if no_intensity:
        kwargs['_with']     = _with
    else:
        data_tuple = data_tuple + (intensity,)
        kwargs['_with']     = f'{_with} palette'
        kwargs['cblabel']   = 'intensity'

    if len(xyz) == 0:
        raise Exception("No data to plot")

    gp.plot(*data_tuple,
            tuplesize = len(data_tuple),
            square = True,
            _3d    = not args.xy,
            title  = f"{bag=} {args.topic=}",
            xlabel = 'x',
            ylabel = 'y',
            wait   = args.period <= 0,
            **kwargs)

    if args.period > 0:
        time.sleep(args.period)



def show_image(bag, p):
    kwargs = dict( _set   = args.set,
                   _unset = args.unset,
                   terminal = args.terminal,
                   hardcopy = args.hardcopy)

    gp.add_plot_option(kwargs,
                       title = bag,
                       overwrite = False)

    gp.add_plot_option(kwargs,
                       set = ('xrange [:] noextend',
                              'yrange [:] noextend reverse',),
                       unset = 'colorbox')

    if p.ndim == 2:
        # grayscale
        gp.add_plot_option(kwargs,
                           set = 'palette gray')

        gp.plot(p,
                _with     = 'image',
                tuplesize = 3,
                square    = True,
                wait      = args.period <= 0,
                **kwargs)
    else:
        # color
        gp.plot(p[...,2], p[...,1], p[...,0],
                _with     = 'rgbimage',
                tuplesize = 5,
                square    = True,
                wait      = args.period <= 0,
                **kwargs)

    if args.period > 0:
        time.sleep(args.period)



Nimages_written = 0

# we have just one topic
topic = args.topic

for bag in bags():

    if args.decimation_period is not None:
        msg_iterator = bag_interface. \
            first_message_from_each_topic_in_time_segments(bag, (topic,),
                                                           period_s = args.decimation_period,
                                                           start = args.after,
                                                           stop  = args.before)
    else:
        msg_iterator = ( bag_interface. \
                         first_message_from_each_topic(bag, (topic,),
                                                       start = args.after,
                                                       stop  = args.before), )


    for msg in msg_iterator:

        itopic = 0 # we have just one topic
        p = msg[itopic]['array']

        def has_xyz(p):
            try:
                if 'xyz' in p.dtype.names:
                    return True
            except:
                return False

        def is_image(p):
            return \
                p.dtype == np.uint8 and \
                (p.ndim == 2 or (p.ndim==3 and p.shape[-1] == 3))



        if args.extract_images:
            if is_image(p):
                filename = f"{args.outdir}/image{Nimages_written:05d}.jpg"

                import mrcal
                mrcal.save_image(filename, p)
                print(f"Wrote '{filename}'")

                Nimages_written += 1

                if args.extract_images_count is not None and \
                   Nimages_written >= args.extract.images.count:
                    sys.exit(0)

            continue


        if has_xyz(p):
            show_lidar(bag, p,
                       _with = getattr(args, "with"),
                       no_intensity = args.no_intensity)
        elif is_image(p):
            show_image(bag, p)
        else:
            print(f"Cannot interpret message from {topic}",
                  file=sys.stderr)
            sys.exit(1)
