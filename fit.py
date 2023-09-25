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

    parser.add_argument('--reference-lidar',
                        type=str,
                        required = True,
                        help = '''Which lidar we're looking at, as specified in the reference-geometry file''')

    parser.add_argument('--reference-camera',
                        type=str,
                        required = True,
                        help = '''Which camera we're looking at, as specified in the reference-geometry file''')

    parser.add_argument('--viz',
                        action='store_true',
                        help = '''Visualize the points''')

    parser.add_argument('model',
                        type = str,
                        help='''Camera model from the optical calibration''')

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

sys.path[:0] = '/home/dima/projects/mrcal',
import mrcal







def find_stationary_frame(t, rt_rf):

    rt_rf0 = rt_rf[:-1]
    rt_rf1 = rt_rf[1:]
    rt_f0f1 = \
        mrcal.compose_rt( mrcal.invert_rt(rt_rf0),
                          rt_rf1 )

    tmid = (t[:-1] + t[1:]) / 2.
    dr = nps.mag(rt_f0f1[..., :3])
    dt = nps.mag(rt_f0f1[..., 3:])


    # I look at chunks of time where dr,dt are consistently low. I can visualize
    # like this:

    # gp.plot(tmid,
    #         nps.cat(dr,dt),
    #         xlabel = "Time (s)",
    #         ylabel = 'Frame-frame shift in position (m) and orientation (rad)',
    #         wait = True)

    # Need to look at t, not just tmid: might be in a big jump

    twant = 272
    return np.argmin(np.abs(t - twant))

def estimate__rt_lidar_camera(reference_lidar,
                              reference_camera):

    with open(getattr(args,'reference-geometry')) as f:
        extrinsics_estimate_dict = json.load(f)


    def extrinsics_from_reference(what):
        d = extrinsics_estimate_dict['extrinsics'][what]
        return \
            np.array(( d['angle_axis_x'], d['angle_axis_y'], d['angle_axis_z'],
                       d['x'],            d['y'],            d['z'] ))

    rt_lidar_ref  = extrinsics_from_reference(reference_lidar)
    rt_camera_ref = extrinsics_from_reference(reference_camera)

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

    seg = pcl.PointCloud(points).make_segmenter_normals(ksearch=50)

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

def find_chessboard_in_view(rt_lidar_frame__estimate,
                            lidar_points_vnl,
                            ref_chessboard):
    # shape (N,3)
    p__estimate = \
        nps.clump( \
            mrcal.transform_point_rt(rt_lidar_frame__estimate,
                                     ref_chessboard),
            n = 2 )

    # The center point and normal vector where we expect the chessboard to be
    # observed. In the lidar frame. Assuming our geometry is correct
    p_center__estimate = np.mean(p__estimate, axis=-2)
    n__estimate = mrcal.rotate_point_r(rt_lidar_frame__estimate[:3], np.array((0,0,1.),))



    points, ring = load_lidar_points(lidar_points_vnl)

    if False:
        # azimuth, elevation we can use for visualization and studies
        th  = np.arctan2( points[:,1], points[:,0] )
        phi = np.arctan2( nps.mag(points[:,:2]), points[:,2] )


    cloud = pcl.PointCloud(points.astype(np.float32))

    p_accepted = None

    for idx_cluster in cluster_points(cloud):

        points_cluster = cloud.to_array()[idx_cluster]

        idx_plane    = find_plane(points_cluster)
        if len(idx_plane) == 0:
            continue

        points_plane = points_cluster[idx_plane]


        mask_cluster = np.zeros( (len(points),), dtype=bool)
        mask_cluster[idx_cluster] = True

        mask_plane = np.zeros( (len(points_cluster),), dtype=bool)
        mask_plane[idx_plane] = True

        mask_plane_keep = np.zeros( (len(points_plane),), dtype=bool)




        # For each ring I find the longest contiguous section on my plane
        th_plane = np.arctan2(points_plane[:,1],
                              points_plane[:,0])

        rings_plane = ring[idx_cluster][idx_plane]
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


        if args.viz:
            gp.plot(
                ( points[ ~mask_cluster ],
                  dict(_with  = 'dots',
                       legend = 'Not in cluster') ),
                ( points_cluster[~mask_plane],
                  dict(_with  = 'points pt 1 ps 1',
                       legend = 'In cluster, not in plane') ),
                ( points_cluster[ mask_plane][~mask_plane_keep],
                  dict(_with  = 'points pt 2 ps 1',
                       legend = 'In cluster, in plane, discard') ),
                ( points_cluster[ mask_plane][mask_plane_keep],
                  dict(_with  = 'points pt 7 ps 1',
                       legend = 'In cluster, in plane, keep') ),
                (p__estimate,
                 dict(_with  = 'points pt 7 ps 1',
                      legend = 'Assuming old calibration')),
                cbmin     = 0,
                cbmax     = 5,
                tuplesize = -3,
                xlabel = 'x',
                ylabel = 'y',
                zlabel = 'z',
                title = f"Cluster ",
                _3d       = True,
                square    = True,
                wait      = True)

        if not np.any(mask_plane_keep):
            continue

        # Found an acceptable set of points on the chessboard in this cluster!

        if p_accepted is not None:
            raise Exception("More than one cluster found that observes a board")

        p_accepted = points_cluster[ mask_plane][mask_plane_keep]

    if p_accepted is None:
        raise Exception("No chessboard found in view")

    return p_accepted




model = mrcal.cameramodel(args.model)

optimization_inputs = model.optimization_inputs()

imagepaths      = optimization_inputs['imagepaths']
rt_camera_frame = optimization_inputs['frames_rt_toref']

if len(imagepaths) != len(rt_camera_frame) or \
   optimization_inputs['extrinsics_rt_fromref'].size > 0:
    raise Exception("I'm assuming a monocular camera calibration, with the camera at the origin")

# shape (NchessboardObservations,6)
rt_lidar_frame__estimate = \
    mrcal.compose_rt(estimate__rt_lidar_camera(args.reference_lidar,
                                               args.reference_camera),
                     rt_camera_frame)


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
    # rt_camera_frame = rt_camera_frame[idx]

    raise Exception("Images in the optical calibration set are not in order")


iobservation_calibration = find_stationary_frame(t, rt_camera_frame)

p = find_chessboard_in_view(rt_lidar_frame__estimate[iobservation_calibration],
                            args.lidar,
                            mrcal.ref_calibration_object(optimization_inputs =
                                                         optimization_inputs))



##### initially fit a single camera-lidar combination. There are many joint








# ##############



# if True:
#     model = mrcal.cameramodel('camera-0.cameramodel')
#     optimization_inputs = model.optimization_inputs()
#     rt_cam0_frames = optimization_inputs['frames_rt_toref']
#     frame_ids = [int(re.sub("frame0*([0-9]+?)\.png",r"\1",s)) \
#                  for s in optimization_inputs['imagepaths']]

#     ## I now have corresponding frame_ids,rt_cam0_frames

# if True:
#     # each row of p is (frame_id,x,y,z)
#     pl = np.loadtxt("points.vnl")

#     dl = nps.mag(pl[:,1:])
#     vl = pl[:,1:] / nps.dummy(dl,-1)

# Nframes = len(frame_ids)

# data_per_frame = [None] * Nframes
# pl_center      = np.zeros((Nframes,3), dtype=float)
# pcam_center    = np.zeros((Nframes,3), dtype=float)

# for i in range(Nframes):

#     frame_id      = frame_ids[i]
#     rt_cam0_frame = rt_cam0_frames[i]

#     # I have a chessboard pose I represent its plane as all x where nt x = d in
#     # camera0 coords. n is the normal to the plane from the camera0 origin. d is
#     # the distance to the plane along this normal.
#     Rt_cam0_frame = mrcal.Rt_from_rt(rt_cam0_frame)
#     R_cam0_frame = Rt_cam0_frame[:3,:]
#     t_cam0_frame = Rt_cam0_frame[ 3,:]

#     # The normal is [0,0,1] in board coords. Here I convert it to camera
#     # coordinates
#     n = R_cam0_frame[:,2]

#     # The point [0,0,0] in board coords is on the plane. I convert to camera
#     # coordinates
#     d = nps.inner(t_cam0_frame, n)
#     if d < 0:
#         d *= -1
#         n *= -1

#     i_points = (pl[:,0] == frame_id)
#     pl_here = pl[i_points,1:]
#     vl_here = vl[i_points,1:]
#     dl_here = dl[i_points,1:]
#     data_per_frame[i] = d,n,pl_here,vl_here,dl_here

#     pl_center  [i] = np.mean(pl_here, axis=-2)
#     pcam_center[i] = xxxx
#     # and procrustes to fit







# def fit(Nmeas, data_per_frame):
#     r'''Align the LIDAR and camera geometry

# A plane in camera coordinates: all x where

#   nt xc = d

# In lidar coords:

#   xc = Rcl xl + tcl
#   nt Rcl xl + nt tcl = d
#   (nt Rcl) xl = d - nt tcl

# Lidar measurements are distances along a vector vl. xl = dl vl. I have a
# measurement dl. Along the lidar vector vl, the distance should satisfy

#   (nt Rcl) vl dl = d - nt tcl

# So dl = (d - nt tcl) / ( nt Rcl vl )

# And the error is

#   dl - dl_observed
#     '''

#     def cost(rt_cam0_lidar, *,
#              simple):

#         Rt_cl = mrcal.Rt_from_rt(rt_cam0_lidar)
#         R_cl = Rt_cl[:3,:]
#         t_cl = Rt_cl[ 3,:]

#         x     = np.zeros((Nmeas,), dtype=float)
#         imeas = 0

#         for d,n,pl,vl,dl in data_per_frame:

#             Nmeas_here = len(dl)

#             if simple:
#                 x[imeas:imeas+Nmeas_here] = \
#                     nps.inner(n,
#                               mrcal.transform_point_Rt(Rt_cl,pl)) \
#                     - d
#             else:
#                 dp = ( d - nps.inner(n,t_cl)) / nps.inner(vl, nps.inner(n,R_cl.T))
#                 x[imeas:imeas+Nmeas_here] = dp - dl

#             imeas += Nmeas_here

#         return x


#     res = scipy.optimize.least_squares(cost,

#                                        # I don't bother to seed
#                                        mrcal.identity_rt(),
#                                        method  = 'dogbox',
#                                        verbose = 2,
#                                        kwargs = dict(simple = True))

#     rt_cam0_lidar__seed = res.x
#     res = scipy.optimize.least_squares(cost,

#                                        rt_cam0_lidar__seed,
#                                        method  = 'dogbox',
#                                        verbose = 2,
#                                        kwargs = dict(simple = False))

#     rt_cam0_lidar = res.x

#     return rt_cam0_lidar




# rt_cam0_lidar = fit(Nmeas, data_per_frame)



# # Done. I want to plot the whole thing
# data_tuples, plot_options = \
#     mrcal.show_geometry((model,
#                          mrcal.invert_rt(rt_cam0_lidar)),
#                         cameranames = ('camera', 'lidar'),
#                         axis_scale       = 1.0,
#                         return_plot_args = True)


# points = [ mrcal.transform_point_rt(rt_cam0_lidar, pl) \
#            for d,n,pl,vl,dl in data_per_frame ]

# gp.plot(*data_tuples,
#         *[ (points[i], dict(_with     = 'points',
#                             legend    = f"Points from frame {frame_ids[i]}",
#                             tuplesize = -3)) \
#            for i in range(len(points)) ],
#         **plot_options,
#         hardcopy = '/tmp/tst.gp')
