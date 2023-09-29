#!/usr/bin/python3

r'''Do thing

SYNOPSIS

  $ xxx
'''


import sys
import argparse
import re
import os


def parse_args():

    parser = \
        argparse.ArgumentParser(description = __doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--camera-frame',
                        type=str,
                        required = True,
                        help = '''The frame of the camera we're looking at''')

    parser.add_argument('--lidar-topic',
                        type=str,
                        required = True,
                        help = '''Which lidar we're talking to''')

    parser.add_argument('--bag',
                        type=str,
                        required = True,
                        help = '''The rosbag that contains the lidar and camera data''')

    parser.add_argument('--t0',
                        type=float,
                        required = True,
                        help = '''Reference time value. Usually obtained with a
                        shell command like t0=$(<
                        multisense-metadata-front-left.vnl vnl-filter --eval
                        '{print field.header.stamp; exit;}'). Must match the t0
                        used when computing the image-timestamps.vnl table
                        passed in the "timestamp" argument''')

    parser.add_argument('--viz',
                        action='store_true',
                        help = '''Visualize the points''')

    parser.add_argument('model',
                        type = str,
                        help='''Camera model from the optical calibration.
                        Assumed to have been compute by the bag in --bag''')

    parser.add_argument('timestamps',
                        type = str,
                        help='''vnl with timestamps for each image. Columns "t imagepath"''')

    parser.add_argument('reference-geometry',
                        type = str,
                        help='''json with the reference geometry of the cameras,
                        lidar units''')


    return parser.parse_args()


args = parse_args()



import numpy as np
import numpysane as nps
import gnuplotlib as gp
import vnlog
import json
import pcl
import scipy.optimize
import subprocess
import io

sys.path[:0] = '/home/dima/projects/mrcal',
import mrcal







def find_stationary_frame(t, rt_rf):

    rt_rf0 = rt_rf[:-1]
    rt_rf1 = rt_rf[1:]
    rt_f0f1 = \
        mrcal.compose_rt( mrcal.invert_rt(rt_rf0),
                          rt_rf1 )

    # differences between successive poses. The domain is (tmid01, tmid12, tmid23, ....)
    dr   = nps.mag(rt_f0f1[..., :3])
    dxyz = nps.mag(rt_f0f1[..., 3:])
    dt   = t[1:] - t[:-1]

    # worst-case of two neighbor successive poses. The domain is (t1,t2,t3,...)
    dr   = np.max(nps.cat(dr[1:],   dr[:-1]),   axis=-2)
    dxyz = np.max(nps.cat(dxyz[1:], dxyz[:-1]), axis=-2)
    dt   = np.max(nps.cat(dt[1:],   dt[:-1]),   axis=-2)


    threshold_dxyz   = 3e-3
    threshold_dr_deg = 0.2
    threshold_dt     = 1.5

    if False:
        # I look at chunks of time where dr,dxyz are consistently low
        gp.plot((t[1:-1], dxyz,            dict(legend = "diff(translation)")),
                (t[1:-1], dr*180./np.pi, dict(legend = "diff(rotation)", y2=1)),
                ymin = 0,
                y2min = 0,
                xlabel  = "Time (s)",
                ylabel  = 'Frame-frame shift in position (m)',
                y2label = 'Frame-frame shift in orientation (deg)',
                equation = (f"{threshold_dxyz} title \"threshold_dxyz\"",
                            f"{threshold_dr_deg} title \"threshold_dr_deg\" axis x1y2",))

    # indexes t[1:-1]
    idx = np.nonzero((dxyz < threshold_dxyz)*(dr < threshold_dr_deg * np.pi/180.))[0]

    # Accept only points that are not within a large time jump
    idx = idx[ dt[idx] < threshold_dt ]

    # index on t, not t[1:-1]
    idx += 1

    return idx


def estimate__rt_lidar_camera(lidar_frame,
                              camera_frame):

    reference_geometry_filename = getattr(args,'reference-geometry')
    with open(reference_geometry_filename) as f:
        extrinsics_estimate_dict = json.load(f)


    def extrinsics_from_reference(what):
        try:
            d = extrinsics_estimate_dict['extrinsics'][what]
        except KeyError:
            print(f"'{what}' not found in '{reference_geometry_filename}'. Giving up",
                  file=sys.stderr)
            sys.exit(1)

        return \
            np.array(( d['angle_axis_x'], d['angle_axis_y'], d['angle_axis_z'],
                       d['x'],            d['y'],            d['z'] ))

    rt_lidar_ref  = extrinsics_from_reference(lidar_frame)
    rt_camera_ref = extrinsics_from_reference(camera_frame)

    return mrcal.compose_rt( rt_lidar_ref,
                             mrcal.invert_rt(rt_camera_ref) )

def load_lidar_points(filename):

    points,list_keys,dict_key_index = vnlog.slurp(filename)

    points_xyz = points[:, (dict_key_index['x'],
                            dict_key_index['y'],
                            dict_key_index['z'])]
    ring       = points[:, dict_key_index['ring']].astype(int)

    r = nps.mag(points_xyz)
    idx = r < 5.
    points_xyz = points_xyz[idx]
    ring       = ring      [idx]
    return points_xyz, ring

def cluster_points(cloud):

    tree = cloud.make_kdtree()

    ec = cloud.make_EuclideanClusterExtraction()
    ec.set_ClusterTolerance(0.4)
    ec.set_MinClusterSize(100)
    ec.set_MaxClusterSize(25000)
    ec.set_SearchMethod(tree)
    return ec.Extract()

def find_plane(points):

    seg = pcl.PointCloud(points.astype(np.float32)).make_segmenter_normals(ksearch=50)

    seg.set_optimize_coefficients(True)
    seg.set_model_type(pcl.SACMODEL_NORMAL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    seg.set_max_iterations(100)
    seg.set_distance_threshold(0.2)
    seg.set_normal_distance_weight(.1)

    idx_plane, coefficients = seg.segment()

    return idx_plane

def longest_run_of_0(x):
    r'''Returns the start and end (inclusive) of the largest contiguous run of 0

If no 0 values are present, returns (None,None). If multiple "longest" sequences
are found, the first one is returned (we use np.argmax internally)

    '''

    # The start and end of each run, inclusive
    if len(x) == 0:
        return None,None
    if len(x) == 1:
        if not x[0]:
            return 0,0
    if np.all(x):
        return None,None

    i_start_run = np.nonzero(np.diff(x.astype(int)) == -1)[0] + 1
    if not x[0]:
        i_start_run = nps.glue(0, i_start_run,
                               axis = -1)
    i_end_run   = np.nonzero(np.diff(x.astype(int)) ==  1)[0]
    if not x[-1]:
        i_end_run = nps.glue(i_end_run, len(x)-1,
                             axis = -1)

    N = i_end_run - i_start_run

    i = np.argmax(N)
    return i_start_run[i],i_end_run[i]

def find_chessboard_in_plane_fit(points_plane,
                                 rings_plane,
                                 p_center__estimate,
                                 n__estimate):

    mask_plane_keep = np.zeros( (len(points_plane),), dtype=bool)

    # For each ring I find the longest contiguous section on my plane
    th_plane = np.arctan2(points_plane[:,1],
                          points_plane[:,0])

    rings_plane_min = np.min(rings_plane)
    rings_plane_max = np.max(rings_plane)

    Nrings = rings_plane_max+1 - rings_plane_min

    for ring_plane in range(rings_plane_min,rings_plane_max+1):
        # shape (Npoints_plane,)
        mask_ring = rings_plane == ring_plane

        # Throw out all points that are too far from where we expect the
        # chessboard to be

        # shape (Npoints_ring,)
        p_ring = points_plane[mask_ring]
        if len(p_ring) == 0:
            continue

        distance_threshold = 1.0
        offplane_threshold = 0.5
        # shape (Npoints_ring,)
        mask_near_estimate = \
            ( np.abs(nps.inner(points_plane[mask_ring] - p_center__estimate,
                               n__estimate)) < offplane_threshold ) * \
            (nps.norm2(points_plane[mask_ring] - p_center__estimate) < distance_threshold*distance_threshold)

        # shape (Npoints_ring,); indexes xxx_plane
        idx_ring = np.nonzero(mask_ring)[0]
        idx_ring = idx_ring[mask_near_estimate]

        if len(idx_ring) == 0:
            continue

        th_ring = th_plane[idx_ring]

        # This is now invalid, so I get rid of it. Use idx_ring
        del mask_ring

        # I examined the data to confirm that the points come regularly at an
        # even interval of:
        dth = 0.2 * np.pi/180.
        # Validated by making this plot, observing that with this dth I get
        # integers with this plot:
        #   gp.plot(np.diff(np.sort(th_ring/dth)))
        # So I make my dth, and any gap of > 1*dth means there was a gap in the
        # plane scan. I look for the biggest interval with no gaps
        idx_sort = np.argsort(th_ring)
        diff_ring_plane_gap = np.diff(th_ring[idx_sort]/dth) > 1.5

        # I look for the largest run of False in diff_ring_plane_gap
        # These are inclusive indices into diff(th)
        if len(diff_ring_plane_gap) == 0:
            continue
        if np.all(diff_ring_plane_gap):
            continue

        i0,i1 = longest_run_of_0(diff_ring_plane_gap)

        # I want to index th, not diff(th). Still inclusive indices
        i1 += 1

        # If the selected segment is too short, I throw it out as noise
        len_segment = \
            nps.mag(points_plane[idx_ring[idx_sort[i1]]] - points_plane[idx_ring[idx_sort[i0]]])
        if len_segment < 0.85:
            continue

        ########## TODO: only contiguous chunks of rings should be accepted

        # indexes th_ring
        idx_longest_run = idx_sort[i0:i1+1]

        mask_plane_keep[idx_ring[idx_longest_run]] = True

    return mask_plane_keep



def find_chessboard_in_view(rt_lidar_board__estimate,
                            lidar_points_vnl,
                            ref_chessboard):
    # shape (N,3)
    p__estimate = \
        nps.clump( \
            mrcal.transform_point_rt(rt_lidar_board__estimate,
                                     ref_chessboard),
            n = 2 )

    # The center point and normal vector where we expect the chessboard to be
    # observed. In the lidar frame. Assuming our geometry is correct
    p_center__estimate = np.mean(p__estimate, axis=-2)
    n__estimate = mrcal.rotate_point_r(rt_lidar_board__estimate[:3], np.array((0,0,1.),))



    points, ring = load_lidar_points(lidar_points_vnl)

    if False:
        # azimuth, elevation we can use for visualization and studies
        th  = np.arctan2( points[:,1], points[:,0] )
        phi = np.arctan2( nps.mag(points[:,:2]), points[:,2] )


    cloud = pcl.PointCloud(points.astype(np.float32))

    p_accepted = None

    i_cluster = -1
    i_cluster_accepted = None
    for idx_cluster in cluster_points(cloud):

        i_cluster += 1

        points_cluster = points[idx_cluster]
        ring_cluster   = ring[idx_cluster]

        idx_plane = find_plane(points_cluster)
        if len(idx_plane) == 0:
            continue

        points_plane = points_cluster[idx_plane]
        rings_plane  = ring_cluster[idx_plane]

        mask_plane_keep = \
            find_chessboard_in_plane_fit(points_plane,
                                         rings_plane,
                                         p_center__estimate,
                                         n__estimate)

        if args.viz:

            mask_cluster = np.zeros( (len(points),), dtype=bool)
            mask_cluster[idx_cluster] = True

            mask_plane = np.zeros( (len(points_cluster),), dtype=bool)
            mask_plane[idx_plane] = True

            hardcopy = f'/tmp/tst{i_cluster}.gp'
            plot_tuples = \
                [
                  ( points_cluster[~mask_plane],
                    dict(_with  = 'points pt 1 ps 1',
                         legend = 'In cluster, not in plane') ),
                  ( points_plane[~mask_plane_keep],
                    dict(_with  = 'points pt 2 ps 1',
                         legend = 'In cluster, in plane, does not match estimate') ),
                  ( points_plane[mask_plane_keep],
                    dict(_with  = 'points pt 7 ps 2 lc "red"',
                         legend = 'In cluster, in plane, matches estimate. Using these') ),
                  (p__estimate,
                   dict(_with  = 'points pt 3 ps 1',
                        legend = 'Assuming old calibration')),
                ]
            if False:
                plot_tuples = \
                    [
                        ( points[ ~mask_cluster ],
                          dict(_with  = 'dots',
                               legend = 'Not in cluster') ),
                        *plot_tuples
                    ]

            gp.plot(
                *plot_tuples,
                cbmin     = 0,
                cbmax     = 5,
                tuplesize = -3,
                xlabel = 'x',
                ylabel = 'y',
                zlabel = 'z',
                title = f"Cluster {i_cluster}",
                _3d       = True,
                square    = True,
                wait      = True,
                hardcopy  = hardcopy,
                )
            print(f"Wrote '{hardcopy}'")

        if not np.any(mask_plane_keep):
            continue

        # Found an acceptable set of points on the chessboard in this cluster!

        if p_accepted is not None:
            raise Exception("More than one cluster found that observes a board")

        p_accepted = points_cluster[ mask_plane][mask_plane_keep]
        i_cluster_accepted = i_cluster








    if p_accepted is None:
        raise Exception("No chessboard found in view")

    print(f"Accepted cluster={i_cluster_accepted}")
    return p_accepted


def fit_camera_lidar(joint_observations,
                     rt_camera_lidar__seed = mrcal.identity_rt()):
    r'''Align the LIDAR and camera geometry

A plane in camera coordinates: all pcam where

  nt pcam = d

In lidar coords:

  pcam = Rcl xlidar + tcl
  nt Rcl xlidar + nt tcl = d
  (nt Rcl) xlidar = d - nt tcl

Lidar measurements are distances along a vector vlidar. xlidar = dlidar vlidar.
I have a measurement dlidar. Along the lidar vector vlidar, the distance should
satisfy

  (nt Rcl) vlidar dlidar = d - nt tcl

So dlidar = (d - nt tcl) / ( nt Rcl vlidar )

And the error is

  dlidar - dlidar_observed

    '''

    Nmeasurements = sum(len(o['dlidar']) for o in joint_observations)

    def cost(rt_camera_lidar, *,

             # simplified computation for seeding
             use_distance_to_plane = False):

        Rt_cl = mrcal.Rt_from_rt(rt_camera_lidar)
        R_cl = Rt_cl[:3,:]
        t_cl = Rt_cl[ 3,:]

        x     = np.zeros((Nmeasurements,), dtype=float)
        imeas = 0

        for o in joint_observations:
            Nmeas_here = len(o['dlidar'])

            if use_distance_to_plane:
                x[imeas:imeas+Nmeas_here] = \
                    nps.inner(o['ncam'],
                              mrcal.transform_point_Rt(Rt_cl,o['plidar'])) \
                    - o['dcam']
            else:
                dp = ( o['dcam'] - nps.inner(o['ncam'],t_cl)) / \
                    nps.inner(o['vlidar'], \
                              nps.inner(o['ncam'],R_cl.T))
                x[imeas:imeas+Nmeas_here] = dp - o['dlidar']

            imeas += Nmeas_here

        return x


    res = scipy.optimize.least_squares(cost,

                                       rt_camera_lidar__seed,
                                       method  = 'dogbox',
                                       verbose = 2,
                                       kwargs = dict(use_distance_to_plane = True))

    rt_camera_lidar__presolve = res.x
    res = scipy.optimize.least_squares(cost,

                                       rt_camera_lidar__presolve,
                                       method  = 'dogbox',
                                       verbose = 2,
                                       kwargs = dict(use_distance_to_plane = False))

    rt_camera_lidar = res.x

    import IPython
    IPython.embed()
    sys.exit()


    return rt_camera_lidar




model = mrcal.cameramodel(args.model)

optimization_inputs = model.optimization_inputs()

imagepaths      = optimization_inputs['imagepaths']
rt_camera_board = optimization_inputs['frames_rt_toref']

if len(imagepaths) != len(rt_camera_board) or \
   optimization_inputs['extrinsics_rt_fromref'].size > 0:
    raise Exception("I'm assuming a monocular camera calibration, with the camera at the origin")

# Read the timestamps, and get a t array to timestamp each board pose
if True:
    timestamps = \
        np.loadtxt(args.timestamps,
                   dtype = np.dtype([('t',float), ('imagepath','S200')]))
    t_from_imagepath = dict()
    for t,imagepath in timestamps:
        t_from_imagepath[imagepath.decode()] = t
    t = np.zeros(imagepaths.shape)
    for i in range(len(imagepaths)):
        t[i] = t_from_imagepath[imagepaths[i]]

if np.any(np.diff(t) <= 0):
    # The timestamps aren't sorted. This is probably OK, but more logic maybe is
    # needed, and this case needs more testing. Here's the code I had to deal
    # with it. It might be enough

    # idx   = t.argsort()
    # t     = t[idx]
    # rt_camera_board = rt_camera_board[idx]

    raise Exception("Images in the optical calibration set are not in order")


iobservation_stationary = find_stationary_frame(t, rt_camera_board)

if len(iobservation_stationary) == 0:
    raise Exception("No stationary image frames found")


joint_observations = list()

lidar_frame = None
for i in iobservation_stationary:

    t_stationary = t[i]

    tmargin = 0.1

    filter_string = \
        f"m.header.stamp.to_sec() > {args.t0/1e9+t_stationary-tmargin} and " + \
        f"m.header.stamp.to_sec() < {args.t0/1e9+t_stationary+tmargin}"

    cmd = ( 'rostopic', 'echo',
            f'--filter={filter_string}',
            '-p',
            '-b', args.bag,
            '-n', '1',
            '--output-directory', '/tmp',
            args.lidar_topic )
    metadata_string = \
        subprocess.check_output( cmd )
    lidar_metadata = list(vnlog.vnlog(io.StringIO(metadata_string.decode())))
    if len(lidar_metadata) == 0:
        raise Exception(f"Couldn't find lidar scan near stationary image frame. Command: {cmd}")
    if len(lidar_metadata) != 1:
        raise Exception(f"Found multiple lidar scans near stationary image frame. I asked for exactly 1. Command: {cmd}")

    lidar_metadata = lidar_metadata[0]

    lidar_points_filename = lidar_metadata['points']

    if lidar_frame is None:
        lidar_frame = lidar_metadata['field.header.frame_id']
        rt_lidar_camera__estimate = \
            estimate__rt_lidar_camera(lidar_frame,
                                      args.camera_frame)

    elif lidar_frame != lidar_metadata['field.header.frame_id']:
        raise Exception(f"LIDAR points aren't all from in the same frame. Saw {lidar_frame} and {lidar_metadata['field.header.frame_id']}")

    rt_lidar_board__estimate = \
        mrcal.compose_rt(rt_lidar_camera__estimate,
                         rt_camera_board[i])

    try:
        plidar = \
            find_chessboard_in_view(rt_lidar_board__estimate,
                                    lidar_points_filename,
                                    mrcal.ref_calibration_object(optimization_inputs =
                                                                 optimization_inputs))
    except:
        print(f"No board observation found for stationary observation {i} (t={t_stationary})")
        continue


    # I have a chessboard pose. I represent its plane as all x where nt x = d. n
    # is the normal to the plane from the origin. d is the distance to the plane
    # along this normal.
    #
    # I find n,d in the camera coordinate system
    Rt_camera_board = mrcal.Rt_from_rt(rt_camera_board[i])
    R_camera_board = Rt_camera_board[:3,:]
    t_camera_board = Rt_camera_board[ 3,:]
    # The normal is [0,0,1] in board coords. Here I convert it to camera
    # coordinates
    ncam = R_camera_board[:,2]
    # The point [0,0,0] in board coords is on the plane. I convert to camera
    # coordinates
    dcam = nps.inner(t_camera_board, ncam)
    if dcam < 0:
        dcam *= -1
        ncam *= -1


    # I convert the lidar points to direction, magnitude
    dlidar = nps.mag(plidar)
    vlidar = plidar / nps.dummy(dlidar,-1)

    joint_observations.append(dict(plidar = plidar,
                                   dlidar = dlidar,
                                   vlidar = vlidar,
                                   ncam   = ncam,
                                   dcam   = dcam))

if len(joint_observations) < 3:
    print(f"I need at least 3 joint camera/lidar observations (the set of all plane normals must span R^3). Got only {len(joint_observations)}",
          file=sys.stderr)
    sys.exit(1)


rt_camera_lidar = fit_camera_lidar(joint_observations,
                                   rt_camera_lidar__seed = mrcal.invert_rt(rt_lidar_camera__estimate))

# Done. I want to plot the whole thing
data_tuples, plot_options = \
    mrcal.show_geometry((model,
                         mrcal.invert_rt(rt_camera_lidar)),
                        cameranames = ('camera', 'lidar'),
                        axis_scale       = 1.0,
                        return_plot_args = True)


points = [ mrcal.transform_point_rt(rt_camera_lidar, o['plidar']) \
           for o in joint_observations]

gp.plot(*data_tuples,
        *[ (points[i], dict(_with     = 'points',
                            legend    = f"Points from frame {i}",
                            tuplesize = -3)) \
           for i in range(len(points)) ],
        **plot_options,
        hardcopy = '/tmp/tst.gp')
