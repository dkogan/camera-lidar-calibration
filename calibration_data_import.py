#!/usr/bin/python3

r'''Utilities to extract chessboard data from ROS bags

Used by the higher-level calibration routines
'''

import sys
import numpy as np
import numpysane as nps
import gnuplotlib as gp

sys.path[:0] = '/home/dima/projects/mrcal',
import mrcal

import vnlog
import pcl
import io
import cv2

import debag

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

def load_lidar_points(filename):

    points,list_keys,dict_key_index = vnlog.slurp(filename)

    points_xyz = points[:, (dict_key_index['x'],
                            dict_key_index['y'],
                            dict_key_index['z'])]
    ring       = points[:, dict_key_index['ring']].astype(int)

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
    r'''Returns the start and end (exclusive) of the largest contiguous run of 0

If no 0 values are present, returns (None,None). If multiple "longest" sequences
are found, the first one is returned (we use np.argmax internally)

Note that this returns a python-style range: the returned upper bound is one
past the last element

    '''

    # The start and end of each run, inclusive
    if len(x) == 0:
        return None,None
    if len(x) == 1:
        if not x[0]:
            return 0,1
    if np.all(x):
        return None,None

    d = np.diff(x.astype(int))
    i_start_run = np.nonzero(d == -1)[0] + 1
    if not x[0]:
        i_start_run = nps.glue(0, i_start_run,
                               axis = -1)
    # end of run. inclusive
    i_end_run   = np.nonzero(d ==  1)[0]
    if not x[-1]:
        i_end_run = nps.glue(i_end_run, len(x)-1,
                             axis = -1)

    i = np.argmax(i_end_run - i_start_run)
    return i_start_run[i],i_end_run[i]+1

def cloud_to_plane_fit(p):
    r'''Fits a plane to some points and returns a transformed point cloud

    The returned point cloud has mean-0 and normal (0,0,1)

    p.shape is (N,3)
    '''
    p = p - np.mean(p, axis=-2)
    l,v = mrcal.sorted_eig(nps.matmult(p.T,p))
    n = v[:,0]

    # I have the normal n to the plane

    R_board_world = mrcal.R_aligned_to_vector(n)
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

def find_chessboard_in_plane_fit(points, ring, th,
                                 idx_plane,
                                 p_center__estimate,
                                 n__estimate,
                                 *,
                                 # for diagnostics
                                 i_cluster    = None,
                                 i_subcluster = None):

    points_plane = points[idx_plane]
    rings_plane  = ring  [idx_plane]
    th_plane     = th    [idx_plane]

    # I examined the data to confirm that the points come regularly at an
    # even interval of:
    dth = 0.2 * np.pi/180.
    # Validated by making this plot, observing that with this dth I get
    # integers with this plot:
    #   gp.plot(np.diff(th_plane/dth))

    expected_board_size = 1.0 # 1m across

    rings_plane_min = rings_plane[ 0]
    rings_plane_max = rings_plane[-1]

    Nrings = rings_plane_max+1 - rings_plane_min

    mask_ring_accepted = np.zeros((Nrings,), dtype=bool)

    mask_plane_keep_per_ring = [None] * Nrings


    # For each ring I find the longest contiguous section on my plane
    for iring in range(Nrings):
        # shape (Npoints_plane,); indexes_plane
        idx_ring = np.nonzero(rings_plane ==
                              iring + rings_plane_min)[0]
        if len(idx_ring) < 20:
            continue

        # Throw out all points that are too far from where we expect the
        # chessboard to be
        if p_center__estimate is not None and \
           n__estimate is not None:
            distance_threshold = 1.0
            offplane_threshold = 0.5

            # shape (Npoints_ring,)
            points_ring_off_center = points_plane[idx_ring] - p_center__estimate

            # shape (Npoints_ring,)
            mask_near_estimate = \
                (np.abs(nps.inner(points_ring_off_center,
                                  n__estimate)) < offplane_threshold ) * \
                (nps.norm2(points_ring_off_center) < distance_threshold*distance_threshold)

            idx_ring = idx_ring[mask_near_estimate]
            if len(idx_ring) == 0:
                continue

        th_ring = th_plane[idx_ring]

        # the data is pre-sorted, so th_ring is sorted as well

        # Below is logic to look for long continuous scans. Any missing points
        # indicate that we're in a noisy area that maybe isn't in the plane.

        # Any gap of > 1*dth means there was a gap in the plane scan. I look
        # for the biggest interval with no BIG gaps. I allow small gaps
        # (hence 3.5 and not 1.5)
        large_diff_ring_plane_gap = np.diff(th_ring/dth) > 3.5

        # I look for the largest run of False in large_diff_ring_plane_gap
        # These are inclusive indices into diff(th)
        if len(large_diff_ring_plane_gap) == 0:
            continue
        if np.all(large_diff_ring_plane_gap):
            continue

        # i0,i1 are python-style ranges indexing diff(th_ring)
        i0,i1 = longest_run_of_0(large_diff_ring_plane_gap)
        # I convert them to index idx_ring. i0 means "there was a large jump
        # between i0-1,i0". So the segment starts at i0. i1 means "there was
        # a large jump between i1,i1+1". So the segment ends at i1, and to
        # get a python-style range I use i1+1
        i1 += 1
        idx_ring = idx_ring[i0:i1]

        # If the selected segment is too short, I throw it out as noise
        len_segment = \
            nps.mag(points_plane[idx_ring[-1]] - \
                    points_plane[idx_ring[ 0]])
        if len_segment < 0.85*expected_board_size or \
           len_segment > np.sqrt(2)*expected_board_size:
            continue

        # I look at a few LIDAR returns past the edges. The board should be in
        # front of everything, not behind. So these adjacent LIDAR cannot have a
        # shorter range
        NscansAtEdge = 3
        max_range_ahead_allowed = 0.2

        i0 = idx_plane[idx_ring[ 0]] # first point index in this segment
        i1 = idx_plane[idx_ring[-1]] # last  point index in this segment

        def scan_indices_off_edge(i0, N):

            # I keep those indices. Initially I keep all of them
            mask_keep = np.ones((abs(N),), dtype=bool)

            if N < 0:
                # looking BEFORE i. I now have a candidate set of indices
                i = np.arange(i0 + N, i0)

            else:
                # looking AFTER i. I now have a candidate set of indices
                i = np.arange(i0+1, i0 + N+1)

            # Throw away out-of-bounds ones
            mask_keep[i <            0] = 0
            mask_keep[i >= len(points)] = 0

            # Throw away any that are on a different ring
            mask_keep[ ring[i0] != ring[i] ] = 0

            # Throw away any that are more than N azimuth points away. This
            # can happen if we have missing returns
            mask_keep[ np.abs(th[i0] - th[i])  > (abs(N)+0.5)*dth ] = 0

            return i[mask_keep]


        i = scan_indices_off_edge(i0, -NscansAtEdge)
        if i.size:
            range0  = nps.mag(points[i0])
            _range  = nps.mag(points[i ])
            if np.any(range0 - _range > max_range_ahead_allowed):
                continue

        i = scan_indices_off_edge(i1, NscansAtEdge)
        if i.size:
            range0  = nps.mag(points[i1])
            _range  = nps.mag(points[i ])
            if np.any(range0 - _range > max_range_ahead_allowed):
                continue



        mask_ring_accepted[iring] = 1

        mask_plane_keep_per_ring[iring] = np.zeros( (len(points_plane),), dtype=bool)
        mask_plane_keep_per_ring[iring][idx_ring] = True

    # I want at least 4 contiguous rings to have data on my plane
    iring_hasdata_start,iring_hasdata_end = longest_run_of_0(~mask_ring_accepted)
    if iring_hasdata_start is None or iring_hasdata_end is None:
        return None
    if iring_hasdata_end-iring_hasdata_start < 4:
        return None

    # Join all the masks of the ring I'm keeping
    # Start with mask_plane_keep_per_ring[iring_hasdata_start], and add to it
    for iring in range(iring_hasdata_start+1,iring_hasdata_end):
        mask_plane_keep_per_ring[iring_hasdata_start] |= \
            mask_plane_keep_per_ring[iring]
    mask_plane_keep = mask_plane_keep_per_ring[iring_hasdata_start]

    # The longest distance between points in the set cannot be longer than the
    # corner-corner distance of the chessboard. Checking this is useful to catch
    # skewed scans
    p = points_plane[mask_plane_keep]
    if distance_between_furthest_pair_of_points(p) > (np.sqrt(2) + 0.1)*expected_board_size:
        return None

    # The angle of the plane off the lidar plane should be > some threshold. cos(th) = inner(normal,z)
    pmean = np.mean(p, axis=-2)
    p = p - pmean
    n = mrcal.sorted_eig(nps.matmult(nps.transpose(p),p))[1][:,0]
    if abs(n[2]) > np.cos(30.*np.pi/180.):
        return None




    return mask_plane_keep

def find_chessboard_in_view(rt_lidar_board__estimate,
                            lidar_points_vnl,
                            *,
                            p_board_local = None,
                            # identifying string
                            what,
                            viz                          = False,
                            viz_show_only_accepted       = False,
                            viz_show_point_cloud_context = False):

    if rt_lidar_board__estimate is not None:
        if p_board_local is None:
            raise Exception("rt_lidar_board__estimate is given, so p_board_local MUST be given also")

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
    th           = np.arctan2( points[:,1], points[:,0] )

    # I sort the points by ring and angle so that I can easily look at the
    # "next" and "previous" points in a scan
    i = np.argsort( th + ring*100 )
    points = points[i]
    ring   = ring  [i]
    th     = th    [i]

    # Ignore all points <1m or >5m away
    range_sq = nps.norm2(points)
    mask_midrange = (1.*1. < range_sq) * (range_sq < 5.*5.)
    idx_midrange  = np.nonzero(mask_midrange)[0]

    cloud_midrange = pcl.PointCloud(points[mask_midrange].astype(np.float32))

    p_accepted = None
    p_accepted_multiple = False

    i_cluster             = -1
    i_cluster_accepted    = None
    i_subcluster_accepted = None
    for idx_cluster in cluster_points(cloud_midrange,
                                      cluster_tolerance = 0.5):
        # idx_cluster indexes points[idx_midrange]
        # Convert it to index points[]
        idx_cluster = idx_midrange[idx_cluster]
        mask_cluster = np.zeros( (len(points),), dtype=bool)
        mask_cluster[idx_cluster] = 1

        i_cluster += 1
        i_subcluster = -1

        points_cluster = points[mask_cluster]
        ring_cluster   = ring  [mask_cluster]

        mask_plane = None
        while True:

            i_subcluster += 1

            # Remove the previous plane found in this cluster, and go again.
            # This is required if a cluster contains multiple planes
            if mask_plane is not None:

                mask_cluster *= ~mask_plane
                idx_cluster = np.nonzero(mask_cluster)[0]
                points_cluster = points[mask_cluster]
                ring_cluster   = ring  [mask_cluster]

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

            # idx_plane indexes points[idx_cluster]
            # Convert it to index points[]
            idx_plane = idx_cluster[idx_plane]

            mask_plane = np.zeros( (len(points),), dtype=bool)
            mask_plane[idx_plane] = True

            points_plane = points[idx_plane]
            rings_plane  = ring  [idx_plane]

            mask_plane_keep = \
                find_chessboard_in_plane_fit(points, ring, th,
                                             idx_plane,
                                             p_center__estimate,
                                             n__estimate,
                                             # for diagnostics
                                             i_cluster    = i_cluster,
                                             i_subcluster = i_subcluster)
            if mask_plane_keep is None:
                mask_plane_keep = np.zeros( (len(points),), dtype=bool)
                have_acceptable_plane = False
            else:
                have_acceptable_plane = np.any(mask_plane_keep)

                idx_plane_keep = np.nonzero(mask_plane_keep)[0]
                # idx_plane_keep indexes points[idx_plane]
                # Convert it to index points[]
                idx_plane_keep = idx_plane[idx_plane_keep]
                mask_plane_keep = np.zeros( (len(points),), dtype=bool)
                mask_plane_keep[idx_plane_keep] = 1

            if viz and \
               (not viz_show_only_accepted or have_acceptable_plane):

                if have_acceptable_plane: any_accepted = "-SOMEACCEPTED"
                else:                     any_accepted = ""
                hardcopy = f'/tmp/lidar-{what}-{i_cluster}-{i_subcluster}{any_accepted}.gp'

                plot_tuples = \
                    [
                      ( points[mask_cluster * ~mask_plane],
                        dict(_with  = 'points pt 1 ps 1',
                             legend = 'In cluster, not in plane') ),
                      ( points[mask_plane * ~mask_plane_keep],
                        dict(_with  = 'points pt 2 ps 1',
                             legend = 'In cluster, in plane, rejected by find_chessboard_in_plane_fit()') ),
                      ( points[mask_plane_keep],
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
                         title = f"{what}: {i_cluster=} {i_subcluster=}",
                         _3d       = True,
                         square    = True,
                         wait      = True)


                if viz_show_point_cloud_context:
                    plot_tuples = \
                        [
                            ( points[ ~mask_cluster * mask_midrange ],
                              dict(_with  = 'dots',
                                   legend = 'Not in cluster') ),
                            *plot_tuples
                        ]

                if hardcopy is not None:
                    plot_options['hardcopy'] = hardcopy
                gp.plot( *plot_tuples, **plot_options)
                if hardcopy is not None:
                    print(f"Wrote '{hardcopy}'")

            if not have_acceptable_plane:
                continue

            # Found an acceptable set of points on the chessboard in this cluster!

            if p_accepted is not None:
                p_accepted_multiple = True
                # This is an error. I don't fail immediately because I want to
                # generate all the plots
            else:
                p_accepted = points[mask_plane_keep]
                i_cluster_accepted    = i_cluster
                i_subcluster_accepted = i_subcluster







    if p_accepted_multiple:
        print("More than one cluster found that observes a board")
        return None
    if p_accepted is None:
        print("No chessboard found in view")
        return None

    print(f"Accepted cluster={i_cluster_accepted} subcluster={i_subcluster_accepted}")
    return p_accepted

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
                          p_board_local                = None,
                          what,
                          viz                          = False,
                          viz_show_only_accepted       = False,
                          viz_show_point_cloud_context = False):
    lidar_metadata = read_first_message_in_bag(bag, lidar_topic)
    if len(lidar_metadata) == 0:
        raise Exception(f"Couldn't find lidar scan")
    if len(lidar_metadata) != 1:
        raise Exception(f"Found multiple lidar scans. I asked for exactly 1")
    lidar_metadata = lidar_metadata[0]

    lidar_points_filename = lidar_metadata['points']

    return \
        find_chessboard_in_view(None,
                                lidar_points_filename,
                                p_board_local = p_board_local,
                                what          = what,
                                viz                          = viz,
                                viz_show_only_accepted       = viz_show_only_accepted,
                                viz_show_point_cloud_context = viz_show_point_cloud_context)
