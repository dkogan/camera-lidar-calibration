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
                        help = '''The frame of the camera we're looking at. Used
                        to index the --reference-geometry file''')

    parser.add_argument('--lidar-topic',
                        type=str,
                        required = True,
                        help = '''Which lidar we're talking to''')

    parser.add_argument('--image-topic',
                        type=str,
                        required = False,
                        help = '''The topic that contains the images. Used and
                        required if no --optical-calibration-from-bag''')

    parser.add_argument('--bag',
                        type=str,
                        required = True,
                        help = '''Globs for the rosbag that contains the lidar
                        and camera data. If --optical-calibration-from-bag, then
                        ALL the data comes from the bag, and the glob must match
                        exactly one file. Otherwise multiple bags can be given
                        in this argument''')

    parser.add_argument('--t0',
                        type=float,
                        help = '''Reference time value. Usually obtained with a
                        shell command like t0=$(<
                        multisense-metadata-front-left.vnl vnl-filter --eval
                        '{print field.header.stamp; exit;}'). Must match the t0
                        used when computing the image-timestamps.vnl table
                        passed in the "timestamp" argument. Used and required if
                        --optical-calibration-from-bag''')

    parser.add_argument('--reference-geometry',
                        type = str,
                        help='''json with the reference geometry of the cameras,
                        lidar units''')

    parser.add_argument('--optical-calibration-from-bag',
                        action='store_true',
                        help = '''If given, we asusme the calibration data used
                        to computed the optical model appears in the --bag; and
                        thus the rt_camera_board pose is already computed''')

    parser.add_argument('--timestamp-vnl',
                        type = str,
                        help='''vnl with timestamps for each image. Columns "t
                        imagepath". Required if --optical-calibration-from-bag''')

    parser.add_argument('--viz',
                        action='store_true',
                        help = '''Visualize the LIDAR point cloud as we search
                        for the chessboafd''')

    parser.add_argument('--viz-show-point-cloud-context',
                        action='store_true',
                        help = '''If given, display ALL the points in the scene
                        to make it easier to orient ourselves''')

    parser.add_argument('model',
                        type = str,
                        help='''Camera model from the optical calibration. If
                        --optical-calibration-from-bag we grab the
                        alreadycomputed rt_camera_board from the
                        optimization_inputs. Otherwise we estimate it from the
                        intrinsics in this model''')

    args = parser.parse_args()

    if args.optical_calibration_from_bag:
        if args.timestamp_vnl is None:
            print("--optical-calibration-from-bag requires --timestamp-vnl, but this wasn't given",
                  file=sys.stderr)
        if args.t0 is None:
            print("--optical-calibration-from-bag requires --t0, but this wasn't given",
                  file=sys.stderr)
        sys.exit(1)

    if args.image_topic is None and not args.optical_calibration_from_bag:
        print("No --optical-calibration-from-bag requires --image-topic, but this wasn't given",
              file=sys.stderr)
        sys.exit(1)

    import glob
    f = glob.glob(args.bag)

    if len(f) == 0:
        print(f"'{args.bag} matched no files",
              file=sys.stderr)
        sys.exit(1)
    if args.optical_calibration_from_bag:
        if len(f) != 1:
            print(f"'--optical-calibration-from-bag, so '{args.bag}' must match exactly one file. Instead this matched {len(f)} files",
                  file=sys.stderr)
            sys.exit(1)

        # --optical-calibration-from-bag. args.bag is a scalar of the one bag
        args.bag = f[0]

    else:
        if len(f) < 3:
            print(f"'no --optical-calibration-from-bag, so '{args.bag}' must match at least 3 files. Instead this matched {len(f)} files",
                  file=sys.stderr)
            sys.exit(1)

        # No --optical-calibration-from-bag. args.bag is a list of all the matched bags
        args.bag = f

    return args


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

import mrgingham
if not hasattr(mrgingham, "find_board"):
    print("mrginham too old. Need at least 1.24",
          file=sys.stderr)
    sys.exit(1)






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

def find_stationary_image_poses(optimization_inputs,
                                timestamps_vnl):
    imagepaths      = optimization_inputs['imagepaths']
    rt_camera_board = optimization_inputs['frames_rt_toref']

    if len(imagepaths) != len(rt_camera_board) or \
       optimization_inputs['extrinsics_rt_fromref'].size > 0:
        raise Exception("I'm assuming a monocular camera calibration, with the camera at the origin")

    # Read the timestamps, and get a t array to timestamp each board pose
    timestamps = \
        np.loadtxt(timestamps_vnl,
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

    return [ (t[i],rt_camera_board[i]) for i in iobservation_stationary]

def estimate__rt_lidar_camera(lidar_frame,
                              camera_frame,
                              reference_geometry):

    with open(reference_geometry) as f:
        extrinsics_estimate_dict = json.load(f)


    def extrinsics_from_reference(what):
        try:
            d = extrinsics_estimate_dict['extrinsics'][what]
        except KeyError:
            print(f"'{what}' not found in '{reference_geometry}'. Giving up",
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

        # shape (Npoints_ring,); indexes xxx_plane
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
                            ref_chessboard,
                            *,
                            # identifying string
                            what):

    if rt_lidar_board__estimate is not None:
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

            if args.viz:

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

    x = cost(rt_camera_lidar)
    print(f"RMS fit error: {np.sqrt(np.mean(x*x)):.2f}m")

    return rt_camera_lidar

def slurp_rostopic_echo(bag, topic,
                        *rostopic_args,
                        filter = None):
    cmd = ('rostopic', 'echo',) + \
          ((f'--filter={filter}',) if filter is not None else ()) + \
          ( '-p',
            '-b', bag,
            *rostopic_args,
            '--output-directory', '/tmp',
            topic )

    if True:
        print(f"command: {' '.join(cmd)}")
    metadata_string = subprocess.check_output( cmd )
    return list(vnlog.vnlog(io.StringIO(metadata_string.decode())))

def estimate__rt_lidar_board(lidar_frame, rt_camera_board):
    if args.reference_geometry is None:
        return None

    global rt_lidar_camera__estimate
    if not hasattr(estimate__rt_lidar_board,'lidar_frame'):
        estimate__rt_lidar_board.lidar_frame = lidar_frame
        rt_lidar_camera__estimate = \
            estimate__rt_lidar_camera(lidar_frame,
                                      args.camera_frame,
                                      args.reference_geometry)

    elif estimate__rt_lidar_board.lidar_frame != lidar_frame:
        raise Exception(f"LIDAR points aren't all from in the same frame. Saw {estimate__rt_lidar_board.lidar_frame} and {lidar_frame}")

    return \
        mrcal.compose_rt(rt_lidar_camera__estimate,
                         rt_camera_board)

def joint_observation__from__t_pose(t,rt_camera_board):

    r'''Used if --optical-calibration-from-bag'''

    tmargin = 0.1

    filter_string = \
        f"m.header.stamp.to_sec() > {args.t0/1e9+t-tmargin} and " + \
        f"m.header.stamp.to_sec() < {args.t0/1e9+t+tmargin}"

    Rt_camera_board = mrcal.Rt_from_rt(rt_camera_board)
    what            = t
    bag             = args.bag

    return \
        joint_observation__common(Rt_camera_board,rt_camera_board,
                                  what          = t,
                                  bag           = args.bag,
                                  filter_string = filter_string)


def joint_observation__common(Rt_camera_board,rt_camera_board,
                              *,
                              what,
                              bag,
                              filter_string):

    lidar_metadata = slurp_rostopic_echo(bag, args.lidar_topic,
                                         '-n', '1',
                                         filter = filter_string)
    if len(lidar_metadata) == 0:
        raise Exception(f"Couldn't find lidar scan")
    if len(lidar_metadata) != 1:
        raise Exception(f"Found multiple lidar scans. I asked for exactly 1")
    lidar_metadata = lidar_metadata[0]

    rt_lidar_board__estimate = estimate__rt_lidar_board(lidar_metadata['field.header.frame_id'],
                                                        rt_camera_board)

    lidar_points_filename = lidar_metadata['points']
    try:
        plidar = \
            find_chessboard_in_view(rt_lidar_board__estimate,
                                    lidar_points_filename,
                                    p_chessboard_ref,
                                    what = what)
    except Exception as e:
        print(f"No unambiguous board observation found for observation at {what=}: {e}")
        return None


    # I have a chessboard pose. I represent its plane as all x where nt x = d. n
    # is the normal to the plane from the origin. d is the distance to the plane
    # along this normal.
    #
    # I find n,d in the camera coordinate system
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

    print(f"SUCCESS! Found lidar scan of board")
    return dict(plidar = plidar,
                dlidar = dlidar,
                vlidar = vlidar,
                ncam   = ncam,
                dcam   = dcam)

def Rt_camera_board__from__bag(bag):
    import mrcal.calibration

    metadata = slurp_rostopic_echo(bag,
                                   args.image_topic,
                                   '-n', '1',)

    image_filename = metadata[0]['image']
    image = mrcal.load_image(image_filename,
                             bits_per_pixel = 8,
                             channels       = 1)

    if not 'clahe' in globals():
        import cv2
        clahe = cv2.createCLAHE()
        clahe.setClipLimit(8)
    image = clahe.apply(image)
    cv2.blur(image, (3,3), dst=image)

    q_observed = mrgingham.find_board(image, gridn=14)
    if q_observed is None:
        print(f"Couldn't find chessboard in '{image_filename}'")
        return None

    observation_qxqyw = np.ones( (len(q_observed),3), dtype=float)
    observation_qxqyw[:,:2] = q_observed

    Rt_camera_board = \
        mrcal.calibration._estimate_camera_pose_from_fixed_point_observations( \
                            *model.intrinsics(),
                            observation_qxqyw,
                            nps.clump(p_chessboard_ref, n=2),
                            image_filename)
    if Rt_camera_board[3,2] <= 0:
        print("Chessboard is behind the camera")
        return None

    if False:
        # diagnostics
        q_perfect = mrcal.project(mrcal.transform_point_Rt(Rt_camera_board,
                                                           nps.clump(p_chessboard_ref,n=2)),
                                  *model.intrinsics())

        rms_error = np.sqrt(np.mean(nps.norm2(q_perfect - q_observed)))
        print(f"RMS error: {rms_error}")
        gp.plot(q_perfect,
                tuplesize = -2,
                _with = 'linespoints pt 2 ps 2 lw 2',
                rgbimage = image_filename,
                square = True,
                yinv = True,
                wait = True)

    print(f"SUCCESS! Found Rt_camera_board in {image_filename}")
    return Rt_camera_board


def joint_observation__from__bag(bag):
    Rt_camera_board = Rt_camera_board__from__bag(bag)
    if Rt_camera_board is None:
        return None
    rt_camera_board = mrcal.rt_from_Rt(Rt_camera_board)

    what = os.path.splitext(os.path.basename(bag))[0]
    return \
        joint_observation__common(Rt_camera_board, rt_camera_board,
                                  what          = what,
                                  bag           = bag,
                                  filter_string = None)



model = mrcal.cameramodel(args.model)
# shape (Nh,Nw,3)
p_chessboard_ref = mrcal.ref_calibration_object(optimization_inputs =
                                                model.optimization_inputs())

rt_lidar_camera__estimate = None








if args.optical_calibration_from_bag:
    t_pose = find_stationary_image_poses(model.optimization_inputs(),
                                         args.timestamp_vnl)

    joint_observations = [joint_observation__from__t_pose(t,rt_camera_board) \
                          for t,rt_camera_board in t_pose ]
else:
    joint_observations = [joint_observation__from__bag(bag) \
                          for bag in args.bag ]

joint_observations = [o for o in joint_observations if o is not None]

if len(joint_observations) < 3:
    print(f"I need at least 3 joint camera/lidar observations (the set of all plane normals must span R^3). Got only {len(joint_observations)}",
          file=sys.stderr)
    sys.exit(1)


rt_camera_lidar = fit_camera_lidar(joint_observations,
                                   rt_camera_lidar__seed = \
                                   mrcal.invert_rt(rt_lidar_camera__estimate) \
                                   if rt_lidar_camera__estimate is not None \
                                   else mrcal.identity_rt())

# Done. I want to plot the whole thing
data_tuples, plot_options = \
    mrcal.show_geometry((model,
                         mrcal.invert_rt(rt_camera_lidar)),
                        cameranames = ('camera', 'lidar'),
                        show_calobjects  = None,
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
