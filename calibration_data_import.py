#!/usr/bin/python3

r'''Utilities to extract chessboard data from ROS bags

Used by the higher-level calibration routines
'''

import sys
import os
import numpy as np
import numpysane as nps
import gnuplotlib as gp

import mrcal

import vnlog
import pcl
import io
import cv2
import inspect
import re

from bag_interface import bag_messages_generator

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
               ksearch,
               search_radius,
               distance_threshold,
               normal_distance_weight,
               max_iterations = 100):


    # The logic around ksearch and search_radius is apparently not in libpcl at
    # all, but in the Python wrapper:
    #
    #   https://sources.debian.org/src/python-pcl/0.3.0~rc1%2Bdfsg-14/pcl/minipcl.cpp/#L17
    #
    # The relevant logic:
    #
    #   pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    #   if (ksearch >= 0)
    #       ne.setKSearch (ksearch);
    #   if (searchRadius >= 0.0)
    #       ne.setRadiusSearch (searchRadius);
    #
    # The meaning of these is in the PCL docs; for instance:
    #   https://pcl.readthedocs.io/projects/tutorials/en/latest/normal_estimation.html
    seg = pcl.PointCloud(points.astype(np.float32)). \
        make_segmenter_normals(ksearch      = ksearch,
                               searchRadius = search_radius)

    seg.set_optimize_coefficients(True)

    # The PCL docs describing SACMODEL_NORMAL_PLANE, and the relevant "weight" parameter:
    #   https://pointclouds.org/documentation/classpcl_1_1_sample_consensus_model_normal_plane.html#details
    #   https://pointclouds.org/documentation/classpcl_1_1_s_a_c_segmentation_from_normals.html#adffe38382fbb0b511764faf9490140ca
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

def line_number():
    # back one so that I'm not in the lineno() function
    frame = inspect.currentframe().f_back
    return frame.f_lineno

def find_chessboard_in_plane_fit(points, ring, th,
                                 idx_plane,
                                 p_center__estimate,
                                 n__estimate,
                                 *,
                                 # "width", but assumes the board is square.
                                 # Will mostly work for non-square boards, but
                                 # the logic could be improved in those cases.
                                 # Here I accept different values for min and
                                 # max checks. USUALLY these would be the same,
                                 # but some datasets require micro-management
                                 board_size_for_min,
                                 board_size_for_max,
                                 # for diagnostics
                                 i_cluster    = None,
                                 i_subcluster = None,

                                 # Any gap of > 1*dth means there was a gap in
                                 # the plane scan. I look for the biggest
                                 # interval with no BIG gaps. I allow small gaps
                                 # (hence 4.5 and not 1.5)
                                 max_acceptable_scan_gap = 4.5,

                                 # I examined the data to confirm that the
                                 # points come regularly at an even interval of:
                                 dth = 0.2 * np.pi/180.,
                                 # Validated by making this plot, observing that
                                 # with this dth I get integers with this plot:
                                 # gp.plot(np.diff(th_plane/dth))


                                 # I look at a few LIDAR returns past the edges.
                                 # The board should be in front of everything,
                                 # not behind. So these adjacent LIDAR cannot
                                 # have a shorter range. I look at NscansAtEdge
                                 # LIDAR returns off to either side. This is set
                                 # to a high number: I cannot be near anything
                                 # that might be occluding. Such a high number
                                 # is required to catch the roll cage bars that
                                 # might split a view of a wall
                                 NscansAtEdge = 20,
                                 max_range_ahead_allowed = 0.2,

                                 distance_threshold = 1.0,
                                 offplane_threshold = 0.5,
                                 min_points_in_ring = 20,
                                 min_ratio_of_contiguous_points_in_ring = 0.7,
                                 max_cos_lidar_axis_to_plane_normal = np.cos(30.*np.pi/180.)):

    points_plane = points[idx_plane]
    rings_plane  = ring  [idx_plane]
    th_plane     = th    [idx_plane]

    rings_plane_min = rings_plane[ 0]
    rings_plane_max = rings_plane[-1]

    Nrings = rings_plane_max+1 - rings_plane_min

    mask_ring_accepted = np.zeros((Nrings,), dtype=bool)

    mask_plane_keep_per_ring = [None] * Nrings


    ring_msgs = dict()
    plane_msg = None
    def reject_ring_if(condition, msg, ring, line):
        nonlocal ring_msgs
        if not condition: return False

        print(f"Rejecting ring {ring} on line {line}: {msg}")
        ring_msgs[ring] = f"{ring} rejected on line {line}: {msg}"
        return True
    def reject_plane_if(condition, msg, line):
        nonlocal plane_msg
        if not condition: return False

        print(f"Rejecting plane on line {line_number()}: {msg}")
        plane_msg = f"Plane rejected: {msg}"
        return True


    # For each ring I find the longest contiguous section on my plane
    for iring in range(Nrings):
        # shape (Npoints_plane,); indexes_plane
        idx_ring = np.nonzero(rings_plane ==
                              iring + rings_plane_min)[0]
        if reject_ring_if(len(idx_ring) < min_points_in_ring,
                          f"len(idx_ring) < min_points_in_ring ~~~ {len(idx_ring)} < {min_points_in_ring}",
                          iring+rings_plane_min,
                          line_number()):
            continue

        # Throw out all points that are too far from where we expect the
        # chessboard to be
        if p_center__estimate is not None and \
           n__estimate is not None:

            # shape (Npoints_ring,)
            points_ring_off_center = points_plane[idx_ring] - p_center__estimate

            # shape (Npoints_ring,)
            mask_near_estimate = \
                (np.abs(nps.inner(points_ring_off_center,
                                  n__estimate)) < offplane_threshold ) * \
                (nps.norm2(points_ring_off_center) < distance_threshold*distance_threshold)

            idx_ring = idx_ring[mask_near_estimate]
            if reject_ring_if(len(idx_ring) == 0,
                              "len(idx_ring) == 0",
                              iring+rings_plane_min,
                              line_number()):
                continue

        th_ring = th_plane[idx_ring]

        # the data is pre-sorted, so th_ring is sorted as well

        # Below is logic to look for long continuous scans. Any missing points
        # indicate that we're in a noisy area that maybe isn't in the plane.
        large_diff_ring_plane_gap = np.diff(th_ring/dth) > max_acceptable_scan_gap

        # I look for the largest run of False in large_diff_ring_plane_gap
        # These are inclusive indices into diff(th)
        if reject_ring_if(len(large_diff_ring_plane_gap) == 0,
                          "len(large_diff_ring_plane_gap) == 0",
                          iring+rings_plane_min,
                          line_number()):
            continue
        if reject_ring_if(np.all(large_diff_ring_plane_gap),
                          "np.all(large_diff_ring_plane_gap)",
                          iring+rings_plane_min,
                          line_number()):
            continue

        # i0,i1 are python-style ranges indexing diff(th_ring)
        i0,i1 = longest_run_of_0(large_diff_ring_plane_gap)
        # I convert them to index idx_ring. i0 means "there was a large jump
        # between i0-1,i0". So the segment starts at i0. i1 means "there was
        # a large jump between i1,i1+1". So the segment ends at i1, and to
        # get a python-style range I use i1+1
        i1 += 1

        if reject_ring_if((i1 - i0) / len(large_diff_ring_plane_gap) < min_ratio_of_contiguous_points_in_ring,
                          f"(i1 - i0) / len(large_diff_ring_plane_gap) < min_ratio_of_contiguous_points_in_ring ~~~ ({i1} - {i0}) / {len(large_diff_ring_plane_gap)} < {min_ratio_of_contiguous_points_in_ring}",
                          iring+rings_plane_min,
                          line_number()):
            continue

        idx_ring = idx_ring[i0:i1]

        # If the selected segment is too short, I throw it out as noise
        len_segment = \
            nps.mag(points_plane[idx_ring[-1]] - \
                    points_plane[idx_ring[ 0]])
        if reject_ring_if(len_segment < 0.7*board_size_for_min,
                          f"len_segment < 0.7*board_size_for_min ~~~ {len_segment} < {0.7*board_size_for_min}",
                          iring+rings_plane_min,
                          line_number()):
            continue
        if reject_ring_if(len_segment > np.sqrt(2)*board_size_for_max,
                          f"len_segment > np.sqrt(2)*board_size_for_max ~~~ {len_segment} > {np.sqrt(2)*board_size_for_max}",
                          iring+rings_plane_min,
                          line_number()):
            continue

        i0 = idx_plane[idx_ring[ 0]] # first point index in this segment
        i1 = idx_plane[idx_ring[-1]] # last  point index in this segment

        if False:
            # useful plot
            gp.plot(th            [ring == iring + rings_plane_min],
                    nps.mag(points[ring == iring + rings_plane_min]),
                    xlabel = "Azimuth (rad)",
                    ylabel = "Range (m)",
                    _set=(f'arrow from {th[i0]},   graph 0 to {th[i0]},   graph 1 nohead',
                          f'arrow from {th[i1-1]}, graph 0 to {th[i1-1]}, graph 1 nohead'))

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

            # out-of-bounds i are set to 0 to make ring[i] and such always
            # return valid values. But maks_keep is already False for those, so
            # they will be ignored anyway. I'm just doing this to avoid Python
            # barfing at me
            i[i <  0          ] = 0
            i[i >= len(points)] = 0

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
            if reject_ring_if(np.any(range0 - _range > max_range_ahead_allowed),
                              f"np.any(range0 - _range > max_range_ahead_allowed) ~~~ np.any({range0} - {_range} > {max_range_ahead_allowed})",
                              iring+rings_plane_min,
                              line_number()):
                continue

        i = scan_indices_off_edge(i1, NscansAtEdge)
        if i.size:
            range0  = nps.mag(points[i1])
            _range  = nps.mag(points[i ])
            if reject_ring_if(np.any(range0 - _range > max_range_ahead_allowed),
                              f"np.any(range0 - _range > max_range_ahead_allowed) ~~~ np.any({range0} - {_range} > {max_range_ahead_allowed})",
                              iring+rings_plane_min,
                              line_number()):
                continue



        mask_ring_accepted[iring] = 1

        mask_plane_keep_per_ring[iring] = np.zeros( (len(points_plane),), dtype=bool)
        mask_plane_keep_per_ring[iring][idx_ring] = True

        ring_msgs[iring+rings_plane_min] = f"{iring+rings_plane_min}: accepted"

    # I want at least some number of contiguous rings to have data on my plane
    Nrings_min_threshold = 3
    iring_hasdata_start,iring_hasdata_end = longest_run_of_0(~mask_ring_accepted)
    if reject_plane_if(iring_hasdata_start is None or iring_hasdata_end is None,
                       f"iring_hasdata_start is None or iring_hasdata_end is None ~~~ {iring_hasdata_start} is None or {iring_hasdata_end} is None",
                       line_number()):
        return None,ring_msgs,plane_msg
    if reject_plane_if(iring_hasdata_end-iring_hasdata_start < Nrings_min_threshold,
                       f"iring_hasdata_end-iring_hasdata_start < Nrings_min_threshold ~~~ {iring_hasdata_end}-{iring_hasdata_start} < {Nrings_min_threshold}",
                       line_number()):
        return None,ring_msgs,plane_msg

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
    d = distance_between_furthest_pair_of_points(p)
    if reject_plane_if(d > (np.sqrt(2) + 0.1)*board_size_for_max,
                       f"d > (np.sqrt(2) + 0.1)*board_size_for_max ~~~ {d} > {(np.sqrt(2) + 0.1)}*{board_size_for_max}",
                       line_number()):
        return None,ring_msgs,plane_msg

    # The angle of the plane off the lidar plane should be > some threshold. cos(th) = inner(normal,z)
    pmean = np.mean(p, axis=-2)
    p = p - pmean
    n = mrcal.sorted_eig(nps.matmult(nps.transpose(p),p))[1][:,0]
    if reject_plane_if(abs(n[2]) > max_cos_lidar_axis_to_plane_normal,
                       f"abs(n[2]) > max_cos_lidar_axis_to_plane_normal ~~~ abs({n[2]}) > {max_cos_lidar_axis_to_plane_normal}",
                       line_number()):
        return None,ring_msgs,plane_msg




    return mask_plane_keep,ring_msgs,plane_msg

def find_chessboard_in_view(rt_lidar_board__estimate,
                            points, ring,
                            *,
                            p_board_local = None,
                            # identifying string
                            what,
                            # "width", but assumes the board is square. Will
                            # mostly work for non-square boards, but the logic
                            # could be improved in those cases. Here I accept
                            # different values for min and max checks. USUALLY
                            # these would be the same, but some datasets require
                            # micro-management
                            board_size_for_min,
                            board_size_for_max,
                            viz                          = False,
                            viz_show_only_accepted       = False,
                            viz_show_point_cloud_context = False,

                            far_distance_threshold_m  = 12,
                            near_distance_threshold_m = 1):

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

    th           = np.arctan2( points[:,1], points[:,0] )

    # I sort the points by ring and angle so that I can easily look at the
    # "next" and "previous" points in a scan
    i = np.argsort( th + ring*100 )
    points = points[i]
    ring   = ring  [i]
    th     = th    [i]

    range_sq = nps.norm2(points)
    mask_near     = (near_distance_threshold_m*near_distance_threshold_m > range_sq)
    mask_far      = (range_sq > far_distance_threshold_m*far_distance_threshold_m)
    mask_midrange = (~mask_near) * (~mask_far)
    idx_midrange  = np.nonzero(mask_midrange)[0]

    result = cluster_and_find_planes(points, idx_midrange,
                                     what                         = what,
                                     viz                          = viz,
                                     viz_show_only_accepted       = viz_show_only_accepted,
                                     viz_show_point_cloud_context = viz_show_point_cloud_context,
                                     mask_far                     = mask_far,
                                     p__estimate                  = p__estimate,
                                     p_center__estimate           = p_center__estimate,
                                     n__estimate                  = n__estimate,
                                     ring                         = ring,
                                     th                           = th,
                                     board_size_for_max = board_size_for_max,
                                     board_size_for_min = board_size_for_min)

    if result['p_accepted_multiple']:
        print("More than one cluster found that observes a board")
        return None
    if result['p_accepted'] is None:
        print("No chessboard found in view")
        return None

    print(f"Accepted cluster={result['i_cluster_accepted']} subcluster={result['i_subcluster_accepted']}")
    return result['p_accepted']

def cluster_and_find_planes(points, idx,
                            *,
                            what,
                            viz,
                            viz_show_only_accepted,
                            viz_show_point_cloud_context,
                            mask_far,
                            p__estimate,
                            p_center__estimate,
                            n__estimate,
                            # "width", but assumes the board is square. Will
                            # mostly work for non-square boards, but the logic
                            # could be improved in those cases. Here I accept
                            # different values for min and max checks. USUALLY
                            # these would be the same, but some datasets require
                            # micro-management
                            board_size_for_min,
                            board_size_for_max,
                            ring,
                            th):

    p_accepted          = None
    p_accepted_multiple = False

    i_cluster             = -1
    i_cluster_accepted    = None
    i_subcluster_accepted = None

    for idx_cluster in cluster_points(pcl.PointCloud(points[idx].astype(np.float32)),
                                      cluster_tolerance = 0.5):
        # idx_cluster indexes points[idx]
        # Convert it to index points[]
        idx_cluster = idx[idx_cluster]
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


            print(f"looking for plane within {len(points_cluster)} points ({i_cluster=}, {i_subcluster=})")
            idx_plane = find_plane(points_cluster,
                                   distance_threshold     = 0.05,
                                   ksearch                = -1,
                                   search_radius          = 0.4,
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

            mask_plane_keep,ring_msgs,plane_msg = \
                find_chessboard_in_plane_fit(points, ring, th,
                                             idx_plane,
                                             p_center__estimate,
                                             n__estimate,
                                             board_size_for_max = board_size_for_max,
                                             board_size_for_min = board_size_for_min,
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

                rings_here = np.unique(ring_cluster)
                p_center_of_rings = np.zeros((len(rings_here),3), dtype=float)
                for i,r in enumerate(rings_here):
                    p_center_of_rings[i] = \
                        np.mean(points_cluster[ring_cluster == r],
                                axis=-2)
                ring_labels = np.array([ring_msgs.get(r,str(r)) for r in rings_here])

                plot_tuples = \
                    [
                      ( points[mask_cluster * ~mask_plane],
                        dict(_with  = 'points pt 1 ps 1',
                             legend = 'In cluster, not in plane') ),
                      ( points[mask_plane * ~mask_plane_keep],
                        dict(_with  = 'points pt 2 ps 1',
                             legend = 'In cluster, in plane, rejected by find_chessboard_in_plane_fit()') ),
                      ( points[mask_plane_keep],
                        dict(_with  = 'points pt 4 ps 1 lc "red"',
                             legend = 'ACCEPTED') ),
                      ( *p_center_of_rings.T, ring_labels,
                        dict(_with  = 'labels',
                             legend = "ring annotations",
                             tuplesize = 4) ),
                    ]

                if p__estimate is not None:
                    plot_tuples.append( (p__estimate,
                                         dict(_with  = 'points pt 3 ps 1',
                                              legend = 'Assuming old calibration')),
                                       )

                title = f"{what}: {i_cluster=} {i_subcluster=}"
                if plane_msg is not None: title += f". {plane_msg}"
                plot_options = \
                    dict(cbmin     = 0,
                         cbmax     = 5,
                         tuplesize = -3,
                         xlabel = 'x',
                         ylabel = 'y',
                         zlabel = 'z',
                         title = title,
                         _3d       = True,
                         square    = True,
                         wait      = True)


                if viz_show_point_cloud_context:
                    plot_tuples = \
                        [
                            ( points[ ~mask_cluster * (~mask_far) ],
                              dict(_with  = 'dots',
                                   legend = 'Not in cluster; cutting off far points') ),
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

    return \
        dict(p_accepted            = p_accepted,
             p_accepted_multiple   = p_accepted_multiple,
             i_cluster_accepted    = i_cluster_accepted,
             i_subcluster_accepted = i_subcluster_accepted)

def chessboard_corners(bag, camera_topic,
                       *,
                       bagname,
                       cache = None):

    try:
        import mrgingham
    except:
        print("processing chessboard images requires the 'mrgingham' library",
              file=sys.stderr)
        sys.exit(1)
    if not hasattr(mrgingham, "find_board"):
        print("mrginham too old. Need at least 1.24",
              file=sys.stderr)
        sys.exit(1)



    if cache is not None and camera_topic in cache:
        return cache[camera_topic]

    raise Exception("THIS IS CURRENTLY UNIMPLEMENTED. debag -> rosbags conversion broke it. Bring it back")
    metadata = \
        next(bag_messages_generator(bag, (camera_topic,)),
             None)

    if len(metadata) == 0:
        print(f"NO images in '{bag}', so no chessboard observations")
        return None

    image_filename = metadata[0]['image']

    image_filename_target = \
        os.path.split(image_filename)[0] + '/' + bagname + os.path.splitext(image_filename)[1]
    # make symlink, overwriting if needed
    try:    os.unlink(image_filename_target)
    except: pass
    os.symlink(os.path.split(image_filename)[1],
               image_filename_target)

    print(f"=== Looking for board in image '{image_filename_target}'....")

    image = mrcal.load_image(image_filename_target,
                             bits_per_pixel = 8,
                             channels       = 1)

    if not hasattr(chessboard_corners, 'clahe'):
        chessboard_corners.clahe = cv2.createCLAHE()
        chessboard_corners.clahe.setClipLimit(8)
    image = chessboard_corners.clahe.apply(image)
    cv2.blur(image, (3,3), dst=image)

    q_observed = mrgingham.find_board(image, gridn=14)
    if q_observed is None:
        print(f"NO chessboard in '{image_filename_target}'")
    else:
        print(f"FOUND chessboard in '{image_filename_target}'")

    if cache is not None: cache[camera_topic] = q_observed

    return q_observed

def get_lidar_observation(bag, lidar_topic,
                          *,
                          p_board_local                = None,
                          what,
                          cache                        = None,
                          viz                          = False,
                          viz_show_only_accepted       = False,
                          viz_show_point_cloud_context = False,
                          # "width", but assumes the board is square. Will
                          # mostly work for non-square boards, but the logic
                          # could be improved in those cases. Here I accept
                          # different values for min and max checks. USUALLY
                          # these would be the same, but some datasets require
                          # micro-management
                          board_size_for_min,
                          board_size_for_max):

    if cache is not None and lidar_topic in cache:
        return cache[lidar_topic]


    try:
        msg = next(bag_messages_generator(bag, (lidar_topic,)))
    except:
        raise Exception(f"Bag '{bag}' doesn't have at least one message of {lidar_topic=}")

    p_lidar = \
        find_chessboard_in_view(None,
                                msg['array']['xyz'].astype(np.float64),
                                msg['array']['ring'],
                                p_board_local = p_board_local,
                                what          = what,
                                viz                          = viz,
                                viz_show_only_accepted       = viz_show_only_accepted,
                                viz_show_point_cloud_context = viz_show_point_cloud_context,
                                board_size_for_max = board_size_for_max,
                                board_size_for_min = board_size_for_min)

    if cache is not None: cache[lidar_topic] = p_lidar
    return p_lidar

def canonical_lidar_topic_name(topic):
    # remove leading and trailing /
    topic = re.match('/*(.*?)/*$', topic).group(1)

    return topic.replace('/', '-')

