#!/usr/bin/python3

r'''Calibrate a set of cameras and LIDARs into a common coordinate system

SYNOPSIS

  $ lidars=(/lidar/vl_points_0)
  $ cameras=(/front/multisense/{{left,right}/image_mono_throttle,aux/image_color_throttle})

  $ ./fit.py \
      --lidar-topic  ${(j:,:)lidars}  \
      --camera-topic ${(j:,:)cameras} \
      --bag 'camera-lidar-*.bag'      \
      --viz \
      intrinsics/{left,right,aux}_camera/camera-0-OPENCV8.cameramodel

  [ The tool chugs for a bit, and in the end produces diagnostics and the aligned ]
  [ models                                                                        ]

This tool computes a geometry-only calibration. It is assumed that the camera
intrinsics have already been computed. The results are computed in the
coordinate system of the first LIDAR. All the sensors must overlap each other
transitively: every sensor doesn't need to overlap every other sensor, but there
must be an overlapping path between each pair of sensors.

The data comes from a set of ROS bags. Each bag is assumed to have captured a
single frame (one set of images, LIDAR revolutions) of a stationary scene

'''


import sys
import argparse
import re
import os


def parse_args():

    parser = \
        argparse.ArgumentParser(description = __doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--lidar-topic',
                        type=str,
                        required = True,
                        help = '''Which lidar(s) we're talking to. This is a
                        comma-separated list of topics. Any number of lidars >=
                        1 is supported''')

    parser.add_argument('--camera-topic',
                        type=str,
                        required = True,
                        help = '''The topic that contains the images. This is a
                        comma-separated list of topics. Any number of cameras >=
                        1 is supported. The number of camera topics must match
                        the number of given models EXACTLY''')

    parser.add_argument('--bag',
                        type=str,
                        required = True,
                        help = '''Glob for the rosbag that contains the lidar
                        and camera data. This can match multiple files''')

    parser.add_argument('--viz',
                        action='store_true',
                        help = '''Visualize the LIDAR point cloud as we search
                        for the chessboafd''')

    parser.add_argument('--viz-show-point-cloud-context',
                        action='store_true',
                        help = '''If given, display ALL the points in the scene
                        to make it easier to orient ourselves''')

    parser.add_argument('--viz-show-only-accepted',
                        action='store_true',
                        help = '''If given, only plot the frames where a board
                        was found''')

    parser.add_argument('--cache',
                        default = '/tmp/lidar-camera-calibration-session.pickle',
                        help = '''The filename we use to store the results of
                        the slow computation. We ALWAYS write to this. We read
                        from this ONLY if --read-cache''')

    parser.add_argument('--read-cache',
                        action='store_true',
                        help = '''If given, we don't run the slow computation,
                        but read it from the file given in --cache''')

    parser.add_argument('models',
                        type = str,
                        nargs='+',
                        help='''Camera model for the optical calibration. Only
                        the intrinsics are used. The number of models given must
                        match the number of --camera-topic EXACTLY''')

    args = parser.parse_args()

    import glob
    f = glob.glob(args.bag)

    if len(f) < 3:
        print(f"--bag '{args.bag}' must match at least 3 files. Instead this matched {len(f)} files",
              file=sys.stderr)
        sys.exit(1)
    args.bag = f

    args.lidar_topic  = args.lidar_topic.split(',')
    args.camera_topic = args.camera_topic.split(',')

    if len(args.models) != len(args.camera_topic):
        print(f"The number of models given must match the number of --camera-topic EXACTLY",
              file=sys.stderr)
        sys.exit(1)

    return args


args = parse_args()



import numpy as np
import numpysane as nps
import gnuplotlib as gp
import vnlog
import json
import pcl
import scipy.optimize
import io
import cv2
import pickle

sys.path[:0] = '/home/dima/projects/mrcal',
import mrcal
import mrcal.calibration
import mrcal.utils

import mrgingham
if not hasattr(mrgingham, "find_board"):
    print("mrginham too old. Need at least 1.24",
          file=sys.stderr)
    sys.exit(1)

import debag


# from mrcal/scales.h:
SCALE_ROTATION_CAMERA    = (0.1 * np.pi/180.0)
SCALE_TRANSLATION_CAMERA = 1.0
SCALE_ROTATION_FRAME     = (15.0 * np.pi/180.0)
SCALE_TRANSLATION_FRAME  = 1.0
SCALE_POSITION_POINT     = SCALE_TRANSLATION_FRAME
SCALE_CALOBJECT_WARP     = 0.01
SCALE_DISTORTION         = 1.0

SCALE_RT_REF_BOARD = np.array((SCALE_ROTATION_FRAME,     SCALE_ROTATION_FRAME,    SCALE_ROTATION_FRAME,
                               SCALE_TRANSLATION_FRAME,  SCALE_TRANSLATION_FRAME, SCALE_TRANSLATION_FRAME,))
SCALE_RT_CAMERA_REF= np.array((SCALE_ROTATION_CAMERA,    SCALE_ROTATION_CAMERA,   SCALE_ROTATION_CAMERA,
                               SCALE_TRANSLATION_CAMERA, SCALE_TRANSLATION_CAMERA,SCALE_TRANSLATION_FRAME,))
SCALE_RT_LIDAR_REF = SCALE_RT_CAMERA_REF

SCALE_MEASUREMENT_PX                = 0.15   # expected noise levels
SCALE_MEASUREMENT_M                 = 0.015  # expected noise levels
SCALE_MEASUREMENT_REGULARIZATION_r  = 100.   # rad
SCALE_MEASUREMENT_REGULARIZATION_t  = 10000. # meters
SCALE_MEASUREMENT_REGULARIZATION_rt = np.array((SCALE_MEASUREMENT_REGULARIZATION_r,
                                                SCALE_MEASUREMENT_REGULARIZATION_r,
                                                SCALE_MEASUREMENT_REGULARIZATION_r,
                                                SCALE_MEASUREMENT_REGULARIZATION_t,
                                                SCALE_MEASUREMENT_REGULARIZATION_t,
                                                SCALE_MEASUREMENT_REGULARIZATION_t))




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

def cluster_points(cloud,
                   *,
                   cluster_tolerance = 0.4,
                   min_cluster_size  = 100,
                   max_cluster_size  = 25000):

    tree = cloud.make_kdtree()

    ec = cloud.make_EuclideanClusterExtraction()
    ec.set_ClusterTolerance(cluster_tolerance)
    ec.set_MinClusterSize(min_cluster_size)
    ec.set_MaxClusterSize(max_cluster_size)
    ec.set_SearchMethod(tree)
    return ec.Extract()

def find_plane(points,
               *,
               ksearch                = 50,
               max_iterations         = 100,
               distance_threshold     = 0.2,
               normal_distance_weight = 0.1):

    seg = pcl.PointCloud(points.astype(np.float32)).make_segmenter_normals(ksearch=
                                                                           ksearch)

    seg.set_optimize_coefficients(True)
    seg.set_model_type(pcl.SACMODEL_NORMAL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    seg.set_max_iterations(max_iterations)
    seg.set_distance_threshold(distance_threshold)
    seg.set_normal_distance_weight(normal_distance_weight)

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

def rotation_any_v_to_z(v):
    r'''Return any rotation matrix that maps the given unit vector v to [0,0,1]'''
    z = v/nps.mag(v)
    if np.abs(z[0]) < .9:
        x = np.array((1,0,0.))
    else:
        x = np.array((0,1,0.))
    x -= nps.inner(x,z)*z
    x /= nps.mag(x)
    y = np.cross(z,x)
    return nps.cat(x,y,z)

def cloud_to_plane_fit(p):
    r'''Fits a plane to some points and returns a transformed point cloud

    The returned point cloud has mean-0 and normal (0,0,1)

    p.shape is (N,3)
    '''
    p = p - np.mean(p, axis=-2)
    l,v = mrcal.utils._sorted_eig(nps.matmult(p.T,p))
    n = v[:,0]

    # I have the normal n to the plane

    R_board_world = rotation_any_v_to_z(n)
    return mrcal.rotate_point_R(R_board_world, p)

def distance_between_furthest_pair_of_points(p):

    r'''Given a set of N points, returns the largest distance between a pair of them

    Fits a plane to operate in 2D, computes the convex hull, then exhausively
    computes each pairwise distance, and returns the largest

    '''
    import scipy.spatial
    import scipy.spatial.distance

    # shape (N,2)
    p = cloud_to_plane_fit(p)[:,:2]

    hull = scipy.spatial.ConvexHull(p)

    return np.max(scipy.spatial.distance.pdist( p[hull.vertices,:] ))

def find_chessboard_in_plane_fit(points_plane,
                                 rings_plane,
                                 p_center__estimate,
                                 n__estimate,
                                 *,
                                 debug = False):

    # For each ring I find the longest contiguous section on my plane
    th_plane = np.arctan2(points_plane[:,1],
                          points_plane[:,0])

    rings_plane_min = np.min(rings_plane)
    rings_plane_max = np.max(rings_plane)

    Nrings = rings_plane_max+1 - rings_plane_min

    mask_ring_accepted = np.zeros((Nrings,), dtype=bool)

    mask_plane_keep_per_ring = [None] * Nrings


    for iring_plane in range(Nrings):
        ring_plane = iring_plane + rings_plane_min

        # shape (Npoints_plane,)
        mask_ring = rings_plane == ring_plane

        # Throw out all points that are too far from where we expect the
        # chessboard to be

        # shape (Npoints_ring,)
        points_ring = points_plane[mask_ring]
        if len(points_ring) < 20:
            continue

        # shape (Npoints_ring,); indexes_plane
        idx_ring = np.nonzero(mask_ring)[0]
        # This is about to become invalid, so I get rid of it. Use idx_ring
        del mask_ring

        if p_center__estimate is not None and \
           n__estimate is not None:
            distance_threshold = 1.0
            offplane_threshold = 0.5
            # shape (Npoints_ring,)
            mask_near_estimate = \
                ( np.abs(nps.inner(points_ring - p_center__estimate,
                                   n__estimate)) < offplane_threshold ) * \
                (nps.norm2(points_ring - p_center__estimate) < distance_threshold*distance_threshold)

            idx_ring = idx_ring[mask_near_estimate]
            if len(idx_ring) == 0:
                continue

        th_ring = th_plane[idx_ring]
        idx_sort = np.argsort(th_ring)

        # Below is logic to look for long continuous scans. Any missing points
        # indicate that we're in a noisy area that maybe isn't in the plane.
        # This was sometimes too strong a filter, and I was disabling it because
        # it was sometimes throwing out far too much data that was truly on the
        # board. I'm enabling it because it works now
        if False:
            i0 = 0
            i1 = len(idx_sort)-1
        else:
            # I examined the data to confirm that the points come regularly at an
            # even interval of:
            dth = 0.2 * np.pi/180.
            # Validated by making this plot, observing that with this dth I get
            # integers with this plot:
            #   gp.plot(np.diff(np.sort(th_ring/dth)))
            # So I make my dth, and any gap of > 1*dth means there was a gap in the
            # plane scan. I look for the biggest interval with no BIG gaps. I
            # allow small gaps (hence 3.5 and not 1.5)
            diff_ring_plane_gap = np.diff(th_ring[idx_sort]/dth) > 3.5

            # I look for the largest run of False in diff_ring_plane_gap
            # These are inclusive indices into diff(th)
            if len(diff_ring_plane_gap) == 0:
                continue
            if np.all(diff_ring_plane_gap):
                continue

            i0,i1 = longest_run_of_0(diff_ring_plane_gap)

            # I want to index th, not diff(th). Still inclusive indices
            i1 += 1

        # indexes th_ring
        idx_keep = idx_sort[i0:i1+1]

        # If the selected segment is too short, I throw it out as noise
        len_segment = \
            nps.mag(points_plane[idx_ring[idx_keep[-1]]] - \
                    points_plane[idx_ring[idx_keep[ 0]]])
        if len_segment < 0.85:
            continue
        if len_segment > np.sqrt(2):
            continue

        mask_ring_accepted[iring_plane] = 1

        mask_plane_keep_per_ring[iring_plane] = np.zeros( (len(points_plane),), dtype=bool)
        mask_plane_keep_per_ring[iring_plane][idx_ring[idx_keep]] = True

    if debug:
        import IPython
        IPython.embed()

    # I want at least 4 contiguous rings to have data on my plane
    iring_hasdata_start,iring_hasdata_end = longest_run_of_0(~mask_ring_accepted)
    if iring_hasdata_start is None or iring_hasdata_end is None:
        return None
    if iring_hasdata_end-iring_hasdata_start+1 < 4:
        return None

    # Join all the masks of the ring I'm keeping
    # Start with mask_plane_keep_per_ring[iring_hasdata_start], and add to it
    for iring_plane in range(iring_hasdata_start+1,iring_hasdata_end+1):
        mask_plane_keep_per_ring[iring_hasdata_start] |= \
            mask_plane_keep_per_ring[iring_plane]
    mask_plane_keep = mask_plane_keep_per_ring[iring_hasdata_start]

    # The longest distance between points in the set cannot be longer than the
    # corner-corner distance of the chessboard
    if distance_between_furthest_pair_of_points(points_plane[mask_plane_keep]) > np.sqrt(2) + 0.1:
        return None

    return mask_plane_keep



def find_chessboard_in_view(rt_lidar_board__estimate,
                            lidar_points_vnl,
                            p_board_local,
                            *,
                            # identifying string
                            what):

    if rt_lidar_board__estimate is not None:
        # shape (N,3)
        p__estimate = \
            nps.clump( \
                mrcal.transform_point_rt(rt_lidar_board__estimate,
                                         p_board_local),
                n = 2 )

        # The center point and normal vector where we expect the chessboard to be
        # observed. In the lidar frame. Assuming our geometry is correct
        p_center__estimate = np.mean(p__estimate, axis=-2)
        n__estimate = mrcal.rotate_point_r(rt_lidar_board__estimate[:3], np.array((0,0,1.),))
    else:
        p__estimate        = None
        p_center__estimate = None
        n__estimate        = None


    points, ring = load_lidar_points(lidar_points_vnl)

    if False:
        # azimuth, elevation we can use for visualization and studies
        th  = np.arctan2( points[:,1], points[:,0] )
        phi = np.arctan2( nps.mag(points[:,:2]), points[:,2] )


    cloud = pcl.PointCloud(points.astype(np.float32))

    p_accepted = None
    p_accepted_multiple = False

    i_cluster             = -1
    i_cluster_accepted    = None
    i_subcluster_accepted = None
    for idx_cluster in cluster_points(cloud,
                                      cluster_tolerance = 0.5):

        i_cluster += 1
        i_subcluster = -1

        points_cluster = points[idx_cluster]
        ring_cluster   = ring[idx_cluster]

        mask_plane = None
        while True:

            i_subcluster += 1

            # Remove the previous plane found in this cluster, and go again.
            # This is required if a cluster contains multiple planes
            if mask_plane is not None:

                points_cluster = points_cluster[~mask_plane]
                ring_cluster   = ring_cluster  [~mask_plane]

                print(f"{len(points_cluster)} points remaining")

                if len(points_cluster) < 20:
                    break


            print(f"looking for plane within {len(points_cluster)} points")
            idx_plane = find_plane(points_cluster,
                                   distance_threshold     = 0.05,
                                   ksearch                = 500,
                                   normal_distance_weight = 0.1)
            if len(idx_plane) == 0:
                break

            mask_plane = np.zeros( (len(points_cluster),), dtype=bool)
            mask_plane[idx_plane] = True

            points_plane = points_cluster[idx_plane]
            rings_plane  = ring_cluster[idx_plane]

            mask_plane_keep = \
                find_chessboard_in_plane_fit(points_plane,
                                             rings_plane,
                                             p_center__estimate,
                                             n__estimate,
                                             # debug = (i_cluster==1),
                                             )
            if mask_plane_keep is None:
                mask_plane_keep = np.zeros( (len(points_plane),), dtype=bool)

            if args.viz and \
               (not args.viz_show_only_accepted or np.any(mask_plane_keep)):

                if np.any(mask_plane_keep): any_accepted = "-SOMEACCEPTED"
                else:                       any_accepted = ""
                hardcopy = f'/tmp/lidar-{what}-{i_cluster}-{i_subcluster}{any_accepted}.gp'

                plot_tuples = \
                    [
                      ( points_cluster[~mask_plane],
                        dict(_with  = 'points pt 1 ps 1',
                             legend = 'In cluster, not in plane') ),
                      ( points_plane[~mask_plane_keep],
                        dict(_with  = 'points pt 2 ps 1',
                             legend = 'In cluster, in plane, rejected by find_chessboard_in_plane_fit()') ),
                      ( points_plane[mask_plane_keep],
                        dict(_with  = 'points pt 7 ps 2 lc "red"',
                             legend = 'ACCEPTED') ),
                    ]
                if p__estimate is not None:
                    plot_tuples.append( (p__estimate,
                                         dict(_with  = 'points pt 3 ps 1',
                                              legend = 'Assuming old calibration')),
                                       )

                plot_options = \
                    dict(cbmin     = 0,
                         cbmax     = 5,
                         tuplesize = -3,
                         xlabel = 'x',
                         ylabel = 'y',
                         zlabel = 'z',
                         title = f"Cluster {i_cluster}",
                         _3d       = True,
                         square    = True,
                         wait      = True)


                if args.viz_show_point_cloud_context:
                    mask_cluster = np.zeros( (len(points),), dtype=bool)
                    mask_cluster[idx_cluster] = True
                    plot_tuples = \
                        [
                            ( points[ ~mask_cluster ],
                              dict(_with  = 'dots',
                                   legend = 'Not in cluster') ),
                            *plot_tuples
                        ]

                if hardcopy is not None:
                    plot_options['hardcopy'] = hardcopy
                gp.plot( *plot_tuples, **plot_options)
                if hardcopy is not None:
                    print(f"Wrote '{hardcopy}'")

            if not np.any(mask_plane_keep):
                continue

            # Found an acceptable set of points on the chessboard in this cluster!

            if p_accepted is not None:
                p_accepted_multiple = True
                # This is an error. I don't fail immediately because I want to
                # generate all the plots
            else:
                p_accepted = points_cluster[ mask_plane][mask_plane_keep]
                i_cluster_accepted    = i_cluster
                i_subcluster_accepted = i_subcluster







    if p_accepted_multiple:
        raise Exception("More than one cluster found that observes a board")
    if p_accepted is None:
        raise Exception("No chessboard found in view")

    print(f"Accepted cluster={i_cluster_accepted} subcluster={i_subcluster_accepted}")
    return p_accepted

def observations_camera(joint_observations):
    iobservation = 0
    for iboard in range(len(joint_observations)):
        q_observed_all = joint_observations[iboard][0]
        for icamera in range(len(q_observed_all)):
            q_observed = q_observed_all[icamera]
            if q_observed is None:
                continue

            yield (q_observed,iboard,icamera,iobservation)

            iobservation += 1

def observations_lidar(joint_observations):
    iobservation = 0
    for iboard in range(len(joint_observations)):
        plidar_all = joint_observations[iboard][1]
        for ilidar in range(len(plidar_all)):
            plidar = plidar_all[ilidar]
            if plidar is None:
                continue

            yield (plidar,iboard,ilidar,iobservation)

            iobservation += 1

def fit_estimate( joint_observations,
                  Nboards, Ncameras, Nlidars,
                  Nmeas_camera_observation,
                  Nmeas_camera_observation_all,
                  Nmeas_lidar_observation_all ):

    r'''Simplified fit() used to produce a seed for fit() to refine

    Same arguments as fit()'''

    # joint_observations is
    # [ obs0, obs1, obs2, ... ] where each observation corresponds to a board pose
    # Each obs is (q_observed, p_lidar)
    # q_observed is a list of board corners; one per camera; some could be None
    # p_lidar is a list of lidar points on the board; one per lidar; some could be None


    def get__Rt_camera_board(q_observed, iboard, icamera, what):

        observation_qxqyw = np.ones( (len(q_observed),3), dtype=float)
        observation_qxqyw[:,:2] = q_observed

        Rt_camera_board = \
            mrcal.calibration._estimate_camera_pose_from_fixed_point_observations( \
                                *models[icamera].intrinsics(),
                                observation_qxqyw = observation_qxqyw,
                                points_ref = nps.clump(p_board_local, n=2),
                                what = what)
        if Rt_camera_board[3,2] <= 0:
            print("Chessboard is behind the camera")
            return None

        if False:
            # diagnostics
            q_perfect = mrcal.project(mrcal.transform_point_Rt(Rt_camera_board,
                                                               nps.clump(p_board_local,n=2)),
                                      *models[icamera].intrinsics())

            rms_error = np.sqrt(np.mean(nps.norm2(q_perfect - q_observed)))
            print(f"RMS error: {rms_error}")
            gp.plot(q_perfect,
                    tuplesize = -2,
                    _with = 'linespoints pt 2 ps 2 lw 2',
                    rgbimage = image_filename,
                    square = True,
                    yinv = True,
                wait = True)

        Rt_camera_board_cache[iboard,icamera] = Rt_camera_board

        return Rt_camera_board

    def pcenter_camera(q_observed, iboard, icamera, what,
                       out = None):
        return \
            mrcal.transform_point_Rt(get__Rt_camera_board(q_observed,
                                                          iboard,
                                                          icamera,
                                                          what = what),
                                     p_center_board,
                                     out = out)

    def pcenter_lidar(plidar, ilidar, what,
                      out = None):
        return \
            np.mean(plidar, axis=-2,
                    out = out)




    # The estimate of the center of the board, in board coords. This doesn't
    # need to be precise. If the board has an even number of corners, I just
    # take the nearest one'''
    Nh,Nw = p_board_local.shape[:2]
    p_center_board = p_board_local[Nh//2,Nw//2,:]



    # results go here
    Rt_lidar0_camera = np.zeros((Ncameras, 4,3), dtype=float)
    Rt_lidar0_lidar  = np.zeros((Nlidars-1,4,3), dtype=float)
    Rt_lidar0_board  = np.zeros((Nboards,  4,3), dtype=float)

    Rt_camera_board_cache = np.zeros((Nboards,Ncameras,4,3),
                                     dtype = float)


    Nsensors = Ncameras + Nlidars

    def pairwise_index(a,b,N = Nsensors):
        # Conceptually I have an (N,N) symmetric matrix with a 0 diagonal. I
        # store only the upper triangle: a 1D array of (N*(N-1)/2) values. This
        # function returns the linear index into this array
        #
        # If a > b: a + b*N - sum(1..b+1) = a + b*N - (b+1)*(b+2)/2
        if a>b: return a + b*N - (b+1)*(b+2)//2
        else:   return b + a*N - (a+1)*(a+2)//2

    def pairwise_N(N = Nsensors):
        # I have an (N,N) symmetric matrix with a 0 diagonal. I store only the
        # upper triangle: (N*(N-1)/2) values. This function returns the size of
        # the linear array
        return N*(N-1)//2

    def connectivity_matrices():
        r'''Returns a connectivity matrix of sensor observations

        Returns a symmetric (Nsensor,Nsensor) matrix of integers, where each
        entry contains the number of frames containing overlapping observations
        for that pair of sensors.

        The sensors are the lidars,cameras; in order

        '''

        def observation_sets():
            for iboard in range(len(joint_observations)):
                q_observed_all,plidar_all = joint_observations[iboard]
                cameras = [(i,q_observed_all[i]) for i in range(len(q_observed_all)) \
                           if q_observed_all[i] is not None]
                lidars  = [(i,plidar_all[i]) for i in range(len(plidar_all)) \
                           if plidar_all [i] is not None]

                yield (cameras,lidars,iboard)


        shared_observation_counts = np.zeros( (pairwise_N(),), dtype=int )

        # I preallocate too many. I will grow the buffer as I need to. The
        # currently needed buffer size is in shared_observation_counts
        #
        # shared_observation_pcenter[...,0,:] is from isensor0 and
        # shared_observation_pcenter[...,1,:] is from isensor1
        # where isensor0 < isensor1
        shared_observation_pcenter = [ np.zeros((16,2,3), dtype=float) \
                                       for i in range(pairwise_N()) ]

        def get_pcloud_next(idx):
            i = shared_observation_counts[idx]
            Nbuffer = shared_observation_pcenter[idx].shape[0]
            if i >= Nbuffer:
                # need to grow buffer
                x = np.zeros((Nbuffer*2,*shared_observation_pcenter[idx].shape[1:]),
                             dtype=float)
                x[:Nbuffer] = shared_observation_pcenter[idx]
                shared_observation_pcenter[idx] = x
            return shared_observation_pcenter[idx][i]


        for cameras,lidars,iboard in observation_sets():

            for ic0 in range(len(cameras)-1):
                icamera0,q_observed0 = cameras[ic0]

                pcenter_camera0 = \
                    pcenter_camera(q_observed0,
                                   iboard,
                                   icamera0,
                                   what = f"{iboard=},icamera={icamera0}")

                for ic1 in range(ic0+1,len(cameras)):
                    icamera1,q_observed1 = cameras[ic1]

                    idx = pairwise_index(Nlidars+icamera0,
                                         Nlidars+icamera1)

                    # shape (2,3)
                    pcloud_next = get_pcloud_next(idx)
                    pcloud_next[0] = pcenter_camera0
                    pcenter_camera(q_observed1,
                                   iboard,
                                   icamera1,
                                   what = f"{iboard=},icamera={icamera1}",
                                   out = pcloud_next[1])

                    shared_observation_counts[idx] += 1



            for il0 in range(len(lidars)-1):
                ilidar0,plidar0 = lidars[il0]

                pcenter_lidar0 = \
                    pcenter_lidar(plidar0,
                                  ilidar0,
                                  what = f"{iboard=},ilidar={ilidar0}")

                for il1 in range(il0+1,len(lidars)):
                    ilidar1,plidar1 = lidars[il1]

                    idx = pairwise_index(ilidar1,
                                         ilidar0)

                    # shape (2,3)
                    pcloud_next = get_pcloud_next(idx)
                    pcloud_next[0] = pcenter_lidar0
                    pcenter_lidar(plidar1,
                                  ilidar1,
                                  what = f"{iboard=},ilidar={ilidar1}",
                                  out = pcloud_next[1])

                    shared_observation_counts[idx] += 1


            for ic in range(len(cameras)):
                icamera,q_observed = cameras[ic]

                pcenter_camera0 = \
                    pcenter_camera(q_observed,
                                   iboard,
                                   icamera,
                                   what = f"{iboard=},icamera={icamera}")

                for il in range(len(lidars)):
                    ilidar,plidar = lidars[il]

                    idx = pairwise_index(ilidar,
                                         Nlidars+icamera)

                    # shape (2,3)
                    pcloud_next = get_pcloud_next(idx)
                    # isensor(camera) > isensor(lidar) always, so I store the
                    # camera into pcloud_next[1] and the lidar into
                    # pcloud_next[0]
                    pcloud_next[1] = pcenter_camera0
                    pcenter_lidar(plidar,
                                  ilidar,
                                  what = f"{iboard=},ilidar={ilidar}",
                                  out = pcloud_next[0])

                    shared_observation_counts[idx] += 1


        return shared_observation_counts, shared_observation_pcenter

    def align_point_clouds(isensor0,isensor1):

        # shape (N,2,3)
        pclouds = shared_observation_pcenter[pairwise_index(isensor1,isensor0)]

        if isensor1 > isensor0:
            pcloud0 = pclouds[...,0,:]
            pcloud1 = pclouds[...,1,:]
        else:
            pcloud0 = pclouds[...,1,:]
            pcloud1 = pclouds[...,0,:]

        Rt01 = \
            mrcal.align_procrustes_points_Rt01(pcloud0, pcloud1)

        # Errors are reported this way (Rt01=0) in the bleeding-edge mrcal only.
        # So I also check for N
        if len(pclouds) < 3 or not np.any(Rt01):
            raise Exception(f"Insufficient overlap between sensors {isensor0} and {isensor1}")

        return Rt01

    def found_best_path_to_node(isensor1, isensor0):
        '''A shortest path was found'''
        if isensor1 == 0:
            # This is the reference sensor. Nothing to do
            return

        Rt01 = align_point_clouds(isensor0,isensor1)

        if isensor1 >= Nlidars:
            icamera1 = isensor1 - Nlidars

            if isensor0 >= Nlidars:
                icamera0 = isensor0 - Nlidars

                # from a camera to a camera
                if not np.any(Rt_lidar0_camera[icamera0]):
                    raise Exception(f"Computing pose of camera {icamera1} from camera {icamera0}, but the pose of camera {icamera0} is not initialized")
                Rt_lidar0_camera[icamera1] = mrcal.compose_Rt(Rt_lidar0_camera[icamera0],
                                                              Rt01)

            else:
                ilidar0 = isensor0

                # from a lidar to a camera

                if ilidar0 == 0:
                    # from the reference
                    Rt_lidar0_camera[icamera1] = Rt01
                else:
                    if not np.any(Rt_lidar0_lidar[ilidar0-1]):
                        raise Exception(f"Computing pose of camera {icamera1} from lidar {ilidar0}, but the pose of lidar {ilidar0} is not initialized")
                    Rt_lidar0_camera[icamera1] = mrcal.compose_Rt(Rt_lidar0_lidar[ilidar0-1],
                                                                  Rt01)
        else:
            ilidar1  = isensor1
            # ilidar1 == 0 will not happen; checked above
            if isensor0 >= Nlidars:
                icamera0 = isensor0 - Nlidars

                # from a camera to a lidar
                if not np.any(Rt_lidar0_camera[icamera0]):
                    raise Exception(f"Computing pose of lidar {ilidar1} from camera {icamera0}, but the pose of camera {icamera0} is not initialized")
                Rt_lidar0_lidar[ilidar1-1] = mrcal.compose_Rt(Rt_lidar0_camera[icamera0],
                                                              Rt01)

            else:
                ilidar0 = isensor0

                # from a lidar to a lidar

                if ilidar0 == 0:
                    # from the reference
                    Rt_lidar0_lidar[ilidar1-1] = Rt01
                else:
                    if not np.any(Rt_lidar0_lidar[ilidar0-1]):
                        raise Exception(f"Computing pose of lidar {ilidar1} from lidar {ilidar0}, but the pose of lidar {ilidar0} is not initialized")
                    Rt_lidar0_lidar[ilidar1-1] = mrcal.compose_Rt(Rt_lidar0_lidar[ilidar0-1],
                                                                  Rt01)


    def cost_edge(isensor0, isensor1):
        # I want to MINIMIZE cost, so I MAXIMIZE the shared frames count and
        # MINIMIZE the hop count. Furthermore, I really want to minimize the
        # number of hops, so that's worth many shared frames.
        num_shared_frames = shared_observation_counts[pairwise_index(isensor0,isensor1)]
        cost = 100000 - num_shared_frames
        assert(cost > 0) # dijkstra's algorithm requires this to be true
        return cost

    def neighbors(isensor0):
        for isensor1 in range(Nsensors):
            if isensor1 == isensor0 or \
               shared_observation_counts[pairwise_index(isensor1,isensor0)] == 0:
                continue
            yield isensor1

    shared_observation_counts, shared_observation_pcenter = connectivity_matrices()

    mrcal.calibration._traverse_sensor_connections \
        ( Nsensors,
          neighbors,
          cost_edge,
          found_best_path_to_node )

    for i in range(len(Rt_lidar0_camera)):
        if not np.any(Rt_lidar0_camera[i]):
            raise Exception(f"ERROR: Don't have complete observations overlap: camera {i} not connected")

    for i in range(len(Rt_lidar0_lidar)):
        if not np.any(Rt_lidar0_lidar[i]):
            raise Exception(f"ERROR: Don't have complete observations overlap: lidar {i+1} not connected")


    for iboard in range(len(joint_observations)):
        q_observed_all = joint_observations[iboard][0]

        icamera_first = \
            next((i for i in range(len(q_observed_all)) if q_observed_all[i] is not None),
                 None)
        if icamera_first is not None:
            # We have some camera observation. I arbitrarily use the first one

            Rt_lidar0_board[iboard] = \
                mrcal.compose_Rt(Rt_lidar0_camera[icamera_first],
                                 Rt_camera_board_cache[iboard,icamera_first])

        else:
            # This board is observed only by LIDARs
            plidar_all = joint_observations[iboard][1]
            ilidar_first = \
                next((i for i in range(len(plidar_all)) if plidar_all[i] is not None),
                     None)
            if ilidar_first is not None:
                raise Exception(f"Getting here is a bug: no camera or lidar observations for {iboard=}")

            plidar = plidar_all[ilidar_first]
            plidar_mean = np.mean(plidar, axis=-2)
            p = plidar - plidar_mean
            n = mrcal.utils._sorted_eig(nps.matmult(nps.transpose(p),p))[1][:,0]
            # I have the normal to the board, in lidar coordinates. Compute an
            # arbitrary rotation that matches this normal. This is unique only
            # up to yaw
            Rt_board_lidar = np.zeros((4,3), dtype=float)
            Rt_board_lidar[:3,:] = rotation_any_v_to_z(n)
            # I want p_center_board to map to plidar_mean: R_board_lidar
            # plidar_mean + t_board_lidar = p_center_lidar
            Rt_board_lidar[3,:] = p_center_lidar - mrcal.rotate_point_R(Rt_board_lidar[:3,:],plidar_mean)

            Rt_lidar0_board[iboard] = \
                mrcal.compose_Rt(Rt_lidar0_lidar[ilidar_first],
                                 mrcal.invert_Rt(Rt_board_lidar))

    return \
        dict(rt_ref_board  = \
                 nps.atleast_dims(mrcal.rt_from_Rt(Rt_lidar0_board),
                                  -2),
             rt_camera_ref = \
                 nps.atleast_dims(mrcal.rt_from_Rt(mrcal.invert_Rt(Rt_lidar0_camera)),
                                  -2),
             rt_lidar_ref  = \
                 nps.atleast_dims(nps.glue(mrcal.identity_rt(),
                                           mrcal.rt_from_Rt(mrcal.invert_Rt(Rt_lidar0_lidar)),
                                           axis = -2),
                                  -2))


def fit( joint_observations,
         Nboards, Ncameras, Nlidars,
         Nmeas_camera_observation,
         Nmeas_camera_observation_all,
         Nmeas_lidar_observation_all ):

    r'''Align the LIDAR and camera geometry

    '''

    # joint_observations is
    # [ obs0, obs1, obs2, ... ] where each observation corresponds to a board pose
    # Each obs is (q_observed, p_lidar)
    # q_observed is a list of board corners; one per camera; some could be None
    # p_lidar is a list of lidar points on the board; one per lidar; some could be None

    # the measurement vector is [camera errors, lidar errors, regularization]
    #
    # The regularization lightly pulls every element of rt_ref_board towards
    # zero. This is necessary because LIDAR-only observations of the board have
    # only 3 DOF: the board is free to translate and yaw in its plane. I can
    # accomplish the same thing with a different T_ref_board representation for
    # LIDAR-only observations: (n,d). That disparate representation would take
    # more typing, so I don't do that just yet
    Nmeas_regularization = 6*Nboards
    Nmeasurements = \
        Nmeas_camera_observation_all + \
        Nmeas_lidar_observation_all  + \
        Nmeas_regularization
    imeas_camera_0         = 0
    imeas_lidar_0          = imeas_camera_0 + Nmeas_camera_observation_all
    imeas_regularization_0 = imeas_lidar_0 + Nmeas_lidar_observation_all

    # I have some number of cameras and some number of lidars. They each
    # observe a chessboard that moves around. At each instant in time the
    # chessboard has a constant pose. The optimization vector contains:
    # - pose of the chessboard in the reference frame
    # - pose of cameras in the reference frame
    # - pose of lidars  in the reference frame
    istate_board_pose_0  = 0
    Nstate_board_pose    = 6 * Nboards
    istate_camera_pose_0 = istate_board_pose_0 + Nstate_board_pose
    Nstate_camera_pose   = 6 * Ncameras
    istate_lidar_pose_0  = istate_camera_pose_0 + Nstate_camera_pose
    Nstate_lidar_pose    = 6 * (Nlidars-1) # lidar0 is the reference coord system

    Nstate = \
        Nstate_board_pose + \
        Nstate_camera_pose + \
        Nstate_lidar_pose

    # The reference coordinate system is defined by the coord system of the
    # first lidar
    def pack_state(# shape (Nboards, 6)
                   rt_ref_board,
                   # shape (Ncameras, 6)
                   rt_camera_ref,
                   # shape (Nlidars, 6)
                   rt_lidar_ref,):
        if np.any(rt_lidar_ref != 0):
            raise Exception("lidar0 is the reference coordinate system so it MUST have the identity transform")
        return nps.glue( (rt_ref_board     / SCALE_RT_REF_BOARD) .ravel(),
                         (rt_camera_ref    / SCALE_RT_CAMERA_REF).ravel(),
                         (rt_lidar_ref[1:] / SCALE_RT_LIDAR_REF) .ravel(),
                         axis = -1)
    def unpack_state(b):
        return                                                                               \
            dict(rt_ref_board = \
                 SCALE_RT_REF_BOARD * \
                 ( b[istate_board_pose_0:                                                          \
                     istate_board_pose_0+Nstate_board_pose].reshape(Nstate_board_pose//6,6)), \

                 rt_camera_ref = \
                 SCALE_RT_CAMERA_REF * \
                 ( b[istate_camera_pose_0:                                                           \
                    istate_camera_pose_0+Nstate_camera_pose].reshape(Nstate_camera_pose//6,6)),    \

                 rt_lidar_ref = \
                 nps.glue( mrcal.identity_rt(),
                           SCALE_RT_LIDAR_REF * \
                           ( b[istate_lidar_pose_0:                                                           \
                               istate_lidar_pose_0+Nstate_lidar_pose].reshape(Nstate_lidar_pose//6,6)),
                           axis = -2 ) )

    def cost(b, *,

             # simplified computation for seeding
             use_distance_to_plane = False):

        x     = np.zeros((Nmeasurements,), dtype=float)
        imeas = 0

        state = unpack_state(b)

        for iboard in range(len(joint_observations)):
            q_observed_all = joint_observations[iboard][0]
            for icamera in range(len(q_observed_all)):
                q_observed = q_observed_all[icamera]

                if q_observed is None:
                    continue

                rt_ref_board  = state['rt_ref_board'] [iboard]
                rt_camera_ref = state['rt_camera_ref'][icamera]

                Rt_ref_board  = mrcal.Rt_from_rt(rt_ref_board)
                Rt_camera_ref = mrcal.Rt_from_rt(rt_camera_ref)

                Rt_camera_board = mrcal.compose_Rt( Rt_camera_ref,
                                                    Rt_ref_board )
                q = mrcal.project( mrcal.transform_point_Rt(Rt_camera_board,
                                                            p_board_local),
                                   *models[icamera].intrinsics() )
                q = nps.clump(q,n=2)
                x[imeas:imeas+Nmeas_camera_observation] = \
                    (q - q_observed).ravel() / SCALE_MEASUREMENT_PX

                imeas += Nmeas_camera_observation


        for iboard in range(len(joint_observations)):
            plidar_all = joint_observations[iboard][1]
            for ilidar in range(len(plidar_all)):
                plidar = plidar_all[ilidar]

                if plidar is None:
                    continue

                dlidar = nps.mag(plidar)
                vlidar = plidar / nps.dummy(dlidar,-1)


                rt_ref_board = state['rt_ref_board'][iboard]
                rt_lidar_ref = state['rt_lidar_ref'][ilidar]

                Rt_lidar_ref = mrcal.Rt_from_rt(rt_lidar_ref)
                Rt_ref_lidar = mrcal.invert_Rt (Rt_lidar_ref)
                Rt_ref_board = mrcal.Rt_from_rt(rt_ref_board)
                Rt_board_ref = mrcal.invert_Rt (Rt_ref_board)

                Nmeas_here = len(plidar)

                # The pose of the board is Rt_ref_board. The board is z=0 in the
                # board coords so the normal to the plane is nref = Rrb[:,2]. I
                # want to define the board as
                #
                #   all x where d = inner(nref,xref) = inner(nref, Rrb xy0 + trl)
                #
                # d = inner(Rrb[:,2], Rrb xy0 + trb) =
                #   = inner(Rrb[:,2], Rrb[:,0]x + Rrb[:,1]y + trb)
                #   = inner(Rrb[:,2], trb)
                if use_distance_to_plane:
                    # Simplified error: look at perpendicular distance off the
                    # plane. inner(x,nref) - dref
                    nref = Rt_ref_board[:3, 2]
                    dref = nps.inner(nref, Rt_ref_board[3, :])
                    x[imeas:imeas+Nmeas_here] = \
                        ( nps.inner(nref,
                                    mrcal.transform_point_Rt(Rt_ref_lidar,plidar)) \
                          - dref ) / SCALE_MEASUREMENT_M
                else:
                    # More complex, but truer error
                    #
                    # A plane is zboard = 0
                    # A lidar point plidar = vlidar dlidar
                    #
                    # pboard = Rbl plidar + tbl
                    # 0 = zboard = pboard[2] = inner(Rbl[2,:],plidar) + tbl[2]
                    # -> inner(Rbl[2,:],vlidar)*dlidar = -tbl[2]
                    # -> dlidar = -tbl[2] / inner(Rbl[2,:],vlidar)
                    #
                    # And the error is
                    #
                    #   dlidar - dlidar_observed
                    Rt_board_lidar = mrcal.compose_Rt( Rt_board_ref,
                                                       Rt_ref_lidar )
                    dlidar_predicted = -Rt_board_lidar[3,2] / nps.inner(Rt_board_lidar[2,:],vlidar)
                    x[imeas:imeas+Nmeas_here] = \
                        (dlidar_predicted - dlidar) / SCALE_MEASUREMENT_M
                imeas += Nmeas_here

        # Regularization
        Nmeas_here = Nboards*6

        x[imeas:imeas+Nmeas_here] = \
            (state['rt_ref_board'] / SCALE_MEASUREMENT_REGULARIZATION_rt).ravel()
        imeas += Nmeas_here

        for iboard in range(len(joint_observations)):
                rt_ref_board = state['rt_ref_board'][iboard]


        if imeas != Nmeasurements:
            raise Exception(f"cost() wrote an unexpected number of measurements: {imeas=}, {Nmeasurements=}")

        return x


    seed = pack_state(**fit_estimate( joint_observations,
                                      Nboards, Ncameras, Nlidars,
                                      Nmeas_camera_observation,
                                      Nmeas_camera_observation_all,
                                      Nmeas_lidar_observation_all ))

    # Docs say:
    # * 0 (default) : work silently.
    # * 1 : display a termination report.
    # * 2 : display progress during iterations (not supported by 'lm'
    verbose = 2
    res = scipy.optimize.least_squares(cost,
                                       seed,
                                       method  = 'dogbox',
                                       verbose = verbose,
                                       kwargs = dict(use_distance_to_plane = True))

    res = scipy.optimize.least_squares(cost,
                                       res.x,
                                       method  = 'dogbox',
                                       verbose = verbose,
                                       kwargs = dict(use_distance_to_plane = False))
    b = res.x

    state = unpack_state(b)


    if True:
        x = cost(b, use_distance_to_plane = False)
        x_camera = \
            x[imeas_camera_0:
              imeas_camera_0+Nmeas_camera_observation_all]
        x_lidar  = \
            x[imeas_lidar_0:
              imeas_lidar_0+Nmeas_lidar_observation_all]
        x_regularization = \
            x[imeas_regularization_0:
              imeas_regularization_0+Nmeas_regularization]
        print(f"RMS fit error: {np.sqrt(np.mean(x*x)):.2f} normalized units")
        print(f"RMS fit error (camera): {np.sqrt(np.mean(x_camera*x_camera))*SCALE_MEASUREMENT_PX:.3f} pixels")
        print(f"RMS fit error (lidar): {(np.sqrt(np.mean(x_lidar *x_lidar ))*SCALE_MEASUREMENT_M):.3f} m")
        print(f"norm2(error_regularization)/norm2(error): {nps.norm2(x_regularization)/nps.norm2(x):.3f} m")

        filename = '/tmp/residuals.gp'
        gp.plot((imeas_camera_0 + np.arange(Nmeas_camera_observation_all),
                 x_camera*SCALE_MEASUREMENT_PX,
                 dict(legend = "Camera residuals")),
                (imeas_lidar_0 + np.arange(Nmeas_lidar_observation_all),
                 x_lidar*SCALE_MEASUREMENT_M,
                 dict(legend = "LIDAR residuals",
                      y2     = True)),
                (imeas_regularization_0 + np.arange(Nmeas_regularization),
                 x_regularization*SCALE_MEASUREMENT_PX,
                 dict(legend = "Regularization residuals; plotted in pixels on the left y axis")),
                _with = 'points',
                ylabel  = 'Camera fit residual (pixels)',
                y2label = 'LIDAR fit residual (m)',
                ymin    = 0,
                y2min   = 0,
                hardcopy = filename)
        print(f"Wrote '{filename}'")

        filename = '/tmp/residuals-histogram.gp'
        gp.plot( (x_camera*SCALE_MEASUREMENT_PX,
                  dict(histogram=True,
                       binwidth = SCALE_MEASUREMENT_PX/10,
                       xrange   = (-3*SCALE_MEASUREMENT_PX,3*SCALE_MEASUREMENT_PX),
                       xlabel   = "Camera residual (px)",
                       ylabel   = "frequency")),
                 (x_lidar*SCALE_MEASUREMENT_M,
                  dict(histogram=True,
                       binwidth = SCALE_MEASUREMENT_M/10,
                       xrange   = (-3*SCALE_MEASUREMENT_M,3*SCALE_MEASUREMENT_M),
                       xlabel   = "LIDAR residual (m)",
                       ylabel   = "frequency")),
                 multiplot='title "LIDAR-camera calibration residuals" layout 2,1',
                 hardcopy = filename)
        print(f"Wrote '{filename}'")

    return state

def read_first_message_in_bag(bag, topic,
                              filter = None):

    pipe = io.StringIO()
    debag.debag(bag = bag,
                topic = topic,
                filter_expr         = filter,
                msg_count           = 1,
                output_directory    = "/tmp",
                output_pipe         = pipe)
    pipe.seek(0)
    return list(vnlog.vnlog(pipe))

def chessboard_corners(bag, camera_topic):
    metadata = read_first_message_in_bag(bag, camera_topic)

    image_filename = metadata[0]['image']
    image = mrcal.load_image(image_filename,
                             bits_per_pixel = 8,
                             channels       = 1)

    if not hasattr(chessboard_corners, 'clahe'):
        chessboard_corners.clahe = cv2.createCLAHE()
        chessboard_corners.clahe.setClipLimit(8)
    image = chessboard_corners.clahe.apply(image)
    cv2.blur(image, (3,3), dst=image)

    q_observed = mrgingham.find_board(image, gridn=14)
    if q_observed is None:
        print(f"Couldn't find chessboard in '{image_filename}'")
        return None
    return q_observed


def get_lidar_observation(bag, lidar_topic,
                          *,
                          what):
    lidar_metadata = read_first_message_in_bag(bag, lidar_topic)
    if len(lidar_metadata) == 0:
        raise Exception(f"Couldn't find lidar scan")
    if len(lidar_metadata) != 1:
        raise Exception(f"Found multiple lidar scans. I asked for exactly 1")
    lidar_metadata = lidar_metadata[0]

    lidar_points_filename = lidar_metadata['points']
    try:
        return \
            find_chessboard_in_view(None,
                                    lidar_points_filename,
                                    p_board_local,
                                    what = what)
    except Exception as e:
        print(f"No unambiguous board observation found for observation at {what=}: {e}")
        return None

def get_joint_observation(bag):
    r'''Compute ONE lidar observation and/or ONE camera observation

    from a bag of ostensibly-stationary data'''

    what = os.path.splitext(os.path.basename(bag))[0]

    Ncameras = len(args.camera_topic)
    q_observed = \
        [ chessboard_corners(bag,
                             args.camera_topic[icamera]) \
          for icamera in range(Ncameras) ]

    Nlidars = len(args.lidar_topic)
    p_lidar = \
        [ get_lidar_observation(bag,
                                args.lidar_topic[ilidar],
                                what = what) \
          for ilidar in range(Nlidars)]

    if all(x is None for x in q_observed) and \
       all(x is None for x in p_lidar):
        return None

    return q_observed,p_lidar




def open_model(f):
    try: return mrcal.cameramodel(f)
    except:
        print(f"Couldn't open '{f}' as a camera model",
              file=sys.stderr)
        sys.exit(1)
models = [open_model(f) for f in args.models]

# I assume each model used the same calibration object
# shape (Ncameras, Nh,Nw,3)
p_board_local__all = \
    [mrcal.ref_calibration_object(optimization_inputs =
                                  m.optimization_inputs()) \
     for m in models]
def is_different(x,y):
    try:    return nps.norm2((x-y).ravel()) > 1e-12
    except: return True
if any(is_different(p_board_local__all[0][...,:2],
                    p_board_local__all[i][...,:2]) \
       for i in range(1,len(models))):
    print("Each model should have been made with the same chessboard, but some are different. I use this calibration-time chessboard for the camera-lidar calibration",
          file = sys.stderr)
    sys.exit(1)

# shape (Nh,Nw,3)
p_board_local = p_board_local__all[0]
p_board_local[...,2] = 0 # assume flat. calobject_warp may differ between samples






if args.read_cache:
    with open(args.cache, "rb") as f:
        ( models,
          p_board_local,
          joint_observations,
          Nboards,
          Ncameras,
          Nlidars,
          Nobservations_camera,
          Nmeas_camera_observation,
          Nmeas_camera_observation_all,
          Nobservations_lidar,
          Nmeas_lidar_observation_all ) = pickle.load(f)

else:

    joint_observations = [get_joint_observation(bag) for bag in args.bag ]

    # Any boards observed by a single sensor aren't useful, and I get rid of
    # them
    def num_sensors_observed(o):
        return \
            sum(0 if x is None else 1 for qp in o for x in qp)
    joint_observations = [o for o in joint_observations \
                          if o is not None and num_sensors_observed(o) > 1]


    # joint_observations is now
    # [ obs0, obs1, obs2, ... ] where each observation corresponds to a board pose
    # Each obs is (q_observed, p_lidar)
    # q_observed is a list of board corners; one per camera; some could be None
    # p_lidar is a list of lidar points on the board; one per lidar; some could be None
    Nboards = len(joint_observations)
    print(f"Have {Nboards} joint observations")

    Ncameras = len(args.camera_topic)
    Nlidars  = len(args.lidar_topic)

    Nobservations_camera = sum(0 if x is None else 1 \
                               for o in joint_observations \
                               for x in o[0])

    Nmeas_camera_observation = p_board_local.shape[-3]*p_board_local.shape[-2]*2
    Nmeas_camera_observation_all = Nobservations_camera * Nmeas_camera_observation

    Nobservations_lidar  = \
        sum(0 if x is None else 1 \
            for o in joint_observations \
            for x in o[1])
    Nmeas_lidar_observation_all = \
        sum(0 if x is None else len(x) \
            for o in joint_observations \
            for x in o[1])

    with open(args.cache, "wb") as f:
        pickle.dump( ( models,
                       p_board_local,
                       joint_observations,
                       Nboards,
                       Ncameras,
                       Nlidars,
                       Nobservations_camera,
                       Nmeas_camera_observation,
                       Nmeas_camera_observation_all,
                       Nobservations_lidar,
                       Nmeas_lidar_observation_all),
                     f)


for icamera in range(Ncameras):
    NcameraObservations_this = sum(0 if o[0][icamera] is None else 1 for o in joint_observations)
    if NcameraObservations_this == 0:
        print(f"I need at least 1 observation of each camera. Got only {len(NcameraObservations_this)} for camera {icamera} from {args.camera_topic[icamera]}",
              file=sys.stderr)
        sys.exit(1)
for ilidar in range(Nlidars):
    NlidarObservations_this = sum(0 if o[1][ilidar] is None else 1 for o in joint_observations)
    if NlidarObservations_this < 3:
        print(f"I need at least 3 observations of each lidar to unambiguously set the translation (the set of all plane normals must span R^3). Got only {NlidarObservations_this} for lidar {ilidar} from {args.lidar_topic[ilidar]}",
              file=sys.stderr)
        sys.exit(1)


solved_state = \
    fit( joint_observations,
         Nboards, Ncameras, Nlidars,
         Nmeas_camera_observation,
         Nmeas_camera_observation_all,
         Nmeas_lidar_observation_all )


rt_ref_board  = solved_state['rt_ref_board']
rt_camera_ref = solved_state['rt_camera_ref']
rt_lidar_ref  = solved_state['rt_lidar_ref']


for imodel in range(len(args.models)):
    models[imodel].extrinsics_rt_fromref(rt_camera_ref[imodel])
    root,extension = os.path.splitext(args.models[imodel])
    filename = f"{root}-mounted{extension}"
    models[imodel].write(filename)
    print(f"Wrote '{filename}'")

# Done. Plot the whole thing
filename = "/tmp/mounted.gp"
data_tuples, plot_options = \
    mrcal.show_geometry((*models,
                         rt_lidar_ref),
                        cameranames = (*args.models, 'lidar'),
                        show_calobjects  = None,
                        axis_scale       = 1.0,
                        return_plot_args = True)

points_camera_observations = \
    [ mrcal.transform_point_rt(rt_ref_board[iboard],
                               nps.clump(p_board_local,n=2) ) \
      for (q_observed,iboard,icamera,iobservation) in observations_camera(joint_observations) ]
points_lidar_observations = \
    [ mrcal.transform_point_rt(mrcal.invert_rt(rt_lidar_ref[ilidar]),
                               plidar) \
            for (plidar,iboard,ilidar,iobservation) in observations_lidar(joint_observations) ]

gp.plot(*data_tuples,
        *[ (points_camera_observations[i],
            dict(_with     = 'lines',
                 legend    = f"Points from camera observation {i}",
                 tuplesize = -3)) \
           for i in range(len(points_camera_observations)) ],
        *[ (points_lidar_observations[i],
            dict(_with     = 'points',
                 legend    = f"Points from lidar observation {i}",
                 tuplesize = -3)) \
           for i in range(len(points_lidar_observations)) ],
        **plot_options,
        hardcopy = filename)
print(f"Wrote '{filename}'")

for iobservation in range(len(joint_observations)):
    (q_observed, p_lidar) = joint_observations[iobservation]
    for ilidar in range(Nlidars):
        if p_lidar[ilidar] is None: continue
        for icamera in range(Ncameras):
            if q_observed[icamera] is None: continue

        rt_camera_lidar = mrcal.compose_rt(rt_camera_ref[icamera],
                                           mrcal.invert_rt(rt_lidar_ref[ilidar]))
        p = mrcal.transform_point_rt(rt_camera_lidar, p_lidar[ilidar])
        q_from_lidar = mrcal.project(p, *models[icamera].intrinsics())

        filename = f"/tmp/reprojected-observation{iobservation}-camera{icamera}-lidar{ilidar}.gp"
        gp.plot( (q_observed[icamera],
                  dict(tuplesize = -2,
                       _with     = 'linespoints',
                       legend    = 'Chessboard corners from the image')),
                 (q_from_lidar,
                  dict(tuplesize = -2,
                       _with     = 'points',
                       legend    = 'Reprojected LIDAR points')),
                 square  = True,
                 _xrange = (0,960),
                 _yrange = (600,0),
                 hardcopy = filename)
        print(f"Wrote '{filename}'")
