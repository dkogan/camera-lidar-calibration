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
                        help = '''The topic that contains the images. This is a
                        comma-separated list of topics. Any number of cameras
                        (including none) is supported. The number of camera
                        topics must match the number of given models EXACTLY''')

    parser.add_argument('--bag',
                        type=str,
                        required = True,
                        help = '''Glob for the rosbag that contains the lidar
                        and camera data. This can match multiple files''')

    parser.add_argument('--exclude-bag',
                        type=str,
                        action = 'append',
                        help = '''Bags to exclude from the processing. These are
                        treated as a regex match against the bag paths''')

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
                        nargs='*',
                        help='''Camera model for the optical calibration. Only
                        the intrinsics are used. The number of models given must
                        match the number of --camera-topic EXACTLY''')

    args = parser.parse_args()

    import glob
    import re
    bags = glob.glob(args.bag)

    if args.exclude_bag is not None:
        for ex in args.exclude_bag:
            bags = [b for b in bags if not re.search(ex, b)]

    if len(bags) < 3:
        print(f"--bag '{args.bag}' must match at least 3 files. Instead this matched {len(bags)} files",
              file=sys.stderr)
        sys.exit(1)
    args.bag = bags


    # import pprint
    # pprint.pprint(args.exclude_bag)

    # print("BAGS:")
    # pprint.pprint(args.bag)
    # sys.exit()

    args.lidar_topic  = args.lidar_topic.split(',')
    if args.camera_topic is not None: args.camera_topic = args.camera_topic.split(',')
    else:                             args.camera_topic = []

    if len(args.models) != len(args.camera_topic):
        print(f"The number of models given must match the number of --camera-topic EXACTLY",
              file=sys.stderr)
        sys.exit(1)

    return args


args = parse_args()



import numpy as np
import numpysane as nps
import gnuplotlib as gp
import scipy.optimize
import pickle
import io

import mrcal
import mrcal.calibration

import calibration_data_import


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


def observations_camera(joint_observations):
    for iboard in range(len(joint_observations)):
        q_observed_all = joint_observations[iboard][0]
        for icamera in range(len(q_observed_all)):
            q_observed = q_observed_all[icamera]
            if q_observed is None:
                continue

            yield (q_observed,iboard,icamera)

def observations_lidar(joint_observations):
    for iboard in range(len(joint_observations)):
        plidar_all = joint_observations[iboard][1]
        for ilidar in range(len(plidar_all)):
            plidar = plidar_all[ilidar]
            if plidar is None:
                continue

            yield (plidar,iboard,ilidar)

def normal(p):
    p_mean = np.mean(p, axis=-2)
    p = p - p_mean
    return mrcal.sorted_eig(nps.matmult(nps.transpose(p),p))[1][:,0]



# Copy from mrcal/mrcal/calibration.py. Please consolidate
def _traverse_sensor_connections( Nsensors,
                                  callback__neighbors,
                                  callback__cost_edge,
                                  callback__found_best_path_to_sensor ):
    '''Traverses a connectivity graph of sensors

    Starts from the root sensor, and visits each one in order of total distance
    from the root. Useful to evaluate the whole set of sensors using pairwise
    metrics, building the network up from the best-connected, to the
    worst-connected. Any sensor not connected to the root at all will NOT be
    visited. The caller should check for any unvisited sensors.

    We have Nsensors sensors. Each one is identified by an integer in
    [0,Nsensors). The root is defined to be sensor 0.

    Three callbacks must be passed in:

    - callback__neighbors (i)
      An iterable returning each neigbor of sensor i.

    - callback__cost_edge(i, i_parent)
      The cost between two adjacent nodes

    - callback__found_best_path_to_sensor(i, i_parent)
      Called when the best path to node i is found. This path runs through
      i_parent as the previous sensor

    '''

    import heapq

    class Node:
        def __init__(self, idx):
            self.idx        = idx
            self.idx_parent = -1
            self.cost       = None
            self.done       = False

        def __lt__(self, other):
            return self.cost < other.cost

        def visit(self):
            callback__found_best_path_to_sensor(self.idx,
                                                self.idx_parent)
            self.done = True

            for neighbor_idx in callback__neighbors(self.idx):
                neighbor = nodes[neighbor_idx]

                if neighbor.done:
                    continue

                cost_to_neighbor_via_node = \
                    self.cost + \
                    callback__cost_edge(neighbor_idx,self.idx)

                if neighbor.cost is None:
                    # Haven't seen this node yet
                    neighbor.cost = cost_to_neighbor_via_node
                    neighbor.idx_parent     = self.idx
                    heapq.heappush(heap, neighbor)
                else:
                    # This node is already in the heap, ready to be processed.
                    # If this new path to this node is better, use it
                    if cost_to_neighbor_via_node < neighbor.cost:
                        neighbor.cost = cost_to_neighbor_via_node
                        neighbor.idx_parent     = self.idx
                        heapq.heapify(heap) # is this the most efficient "update" call?


    nodes = [Node(i) for i in range(Nsensors)]
    nodes[0].cost = 0
    heap = []

    nodes[0].visit()
    while heap:
        node_top = heapq.heappop(heap)
        node_top.visit()


def fit_estimate( joint_observations,
                  Nboards, Ncameras, Nlidars,
                  Nmeas_camera_observation,
                  Nmeas_camera_observation_all,
                  Nmeas_lidar_observation_all,
                  p_board_local):

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

    def pcenter_normal_camera(q_observed, iboard, icamera, what,
                              out = None):

        Rt_camera_board = \
            get__Rt_camera_board(q_observed,
                                 iboard,
                                 icamera,
                                 what = what)

        if out is None:
            out = np.zeros((2,3),dtype=float)
        mrcal.transform_point_Rt(Rt_camera_board,
                                 p_center_board,
                                 out = out[0])
        out[1] = Rt_camera_board[:3,2]

        # I make sure that the normal points towards the sensor; for consistency
        if nps.inner(out[0],out[1]) > 0:
            out[1] *= -1
        return out

    def pcenter_normal_lidar(plidar, out = None):
        if out is None:
            out = np.zeros((2,3),dtype=float)
        np.mean(plidar, axis=-2,
                out = out[0])
        out[1] = normal(plidar)
        # I make sure that the normal points towards the sensor; for consistency
        if nps.inner(out[0],out[1]) > 0:
            out[1] *= -1
        return out




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

    def full_symmetric_matrix_from_upper_triangle(U, N = Nsensors):
        # I have an (N,N) symmetric matrix with a 0 diagonal. I store only the
        # upper triangle: (N*(N-1)/2) values. This function returns the full
        # symmetric array. For debugging
        M = np.zeros((N,N), dtype=U.dtype)
        iu = 0
        for i in range(N):
            M[i,i+1:] = U[iu:iu+N-i-1]
            M[i+1:,i] = U[iu:iu+N-i-1]
            iu += N-i-1
        return M

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
        # shared_observation_pcenter_normal[...,0,:,:] is from isensor0 and
        # shared_observation_pcenter_normal[...,1,:,:] is from isensor1
        # where isensor0 < isensor1
        # shape (Npairs,Nbuffer,Nsensors=2,pcenter_normal=2,3)
        shared_observation_pcenter_normal = [ np.zeros((16,2,2,3), dtype=float) \
                                              for i in range(pairwise_N()) ]

        def get_pcloud_normal_next(idx):
            i = shared_observation_counts[idx]
            Nbuffer = shared_observation_pcenter_normal[idx].shape[0]
            if i >= Nbuffer:
                # need to grow buffer
                x = np.zeros((Nbuffer*2,*shared_observation_pcenter_normal[idx].shape[1:]),
                             dtype=float)
                x[:Nbuffer] = shared_observation_pcenter_normal[idx]
                shared_observation_pcenter_normal[idx] = x
            return shared_observation_pcenter_normal[idx][i]


        for cameras,lidars,iboard in observation_sets():

            for ic0 in range(len(cameras)-1):
                icamera0,q_observed0 = cameras[ic0]

                pcenter_normal_camera0 = \
                    pcenter_normal_camera(q_observed0,
                                          iboard,
                                          icamera0,
                                          what = f"{iboard=},icamera={icamera0}")
                for ic1 in range(ic0+1,len(cameras)):
                    icamera1,q_observed1 = cameras[ic1]

                    idx = pairwise_index(Nlidars+icamera0,
                                         Nlidars+icamera1)

                    # shape (Nsensors=2,pcenter_normal=2,3)
                    pcloud_normal_next = get_pcloud_normal_next(idx)
                    pcloud_normal_next[0] = pcenter_normal_camera0
                    pcenter_normal_camera(q_observed1,
                                          iboard,
                                          icamera1,
                                          what = f"{iboard=},icamera={icamera1}",
                                          out = pcloud_normal_next[1])

                    shared_observation_counts[idx] += 1



            for il0 in range(len(lidars)-1):
                ilidar0,plidar0 = lidars[il0]

                pcenter_normal_lidar0 = \
                    pcenter_normal_lidar(plidar0)

                for il1 in range(il0+1,len(lidars)):
                    ilidar1,plidar1 = lidars[il1]

                    idx = pairwise_index(ilidar1,
                                         ilidar0)

                    # shape (Nsensors=2,pcenter_normal=2,3)
                    pcloud_normal_next = get_pcloud_normal_next(idx)
                    pcloud_normal_next[0] = pcenter_normal_lidar0
                    pcenter_normal_lidar(plidar1,
                                         out = pcloud_normal_next[1])

                    shared_observation_counts[idx] += 1


            for ic in range(len(cameras)):
                icamera,q_observed = cameras[ic]

                pcenter_normal_camera0 = \
                    pcenter_normal_camera(q_observed,
                                          iboard,
                                          icamera,
                                          what = f"{iboard=},icamera={icamera}")

                for il in range(len(lidars)):
                    ilidar,plidar = lidars[il]

                    idx = pairwise_index(ilidar,
                                         Nlidars+icamera)

                    # shape (Nsensors=2,pcenter_normal=2,3)
                    pcloud_normal_next = get_pcloud_normal_next(idx)
                    # isensor(camera) > isensor(lidar) always, so I store the
                    # camera into pcloud_normal_next[1] and the lidar into
                    # pcloud_normal_next[0]
                    pcloud_normal_next[1] = pcenter_normal_camera0
                    pcenter_normal_lidar(plidar,
                                         out = pcloud_normal_next[0])

                    shared_observation_counts[idx] += 1


        return shared_observation_counts, shared_observation_pcenter_normal

    def align_point_clouds(isensor0,isensor1):

        idx = pairwise_index(isensor1,isensor0)

        # shape (Nbuffer,Nsensors=2,pcenter_normal=2,3)
        pcloud_normals = shared_observation_pcenter_normal[idx]

        # Nbuffer > N; I cut it down to the real data
        N = shared_observation_counts[idx]
        pcloud_normals = pcloud_normals[:N]

        if isensor1 > isensor0:
            pcloud_normals0 = pcloud_normals[...,0,:,:]
            pcloud_normals1 = pcloud_normals[...,1,:,:]
        else:
            pcloud_normals0 = pcloud_normals[...,1,:,:]
            pcloud_normals1 = pcloud_normals[...,0,:,:]

        pcloud0 = pcloud_normals0[:,0,:]
        pcloud1 = pcloud_normals1[:,0,:]

        normals0 = pcloud_normals0[:,1,:]
        normals1 = pcloud_normals1[:,1,:]

        # If I had lots of points, I'd do a procrustes fit, and I'd be done. But
        # I have few points, so I do this in two steps:
        # - I align the normals to get a high-confidence rotation
        # - I lock down this rotation, and find the best translation
        if not np.all(nps.norm2(pcloud_normals0)) or \
           not np.all(nps.norm2(pcloud_normals1)):
            raise Exception("Aligning uninitialized data")

        Rt01 = np.zeros((4,3), dtype=float)

        Rt01[:3,:] = \
            mrcal.align_procrustes_vectors_R01(normals0, normals1)
        # Errors are reported this way (Rt01=0) in the bleeding-edge mrcal only.
        # So I also check for N
        if len(pcloud_normals) < 2 or not np.any(Rt01[:3,:]):
            raise Exception(f"Insufficient overlap between sensors {isensor0} and {isensor1}")

        # Now the translation. R01 x1 + t01 ~ x0
        Rt01[3,:] = np.mean(pcloud0 - mrcal.rotate_point_R(Rt01[:3,:], pcloud1),
                            axis = -2)

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

                print(f"Estimating pose of camera {icamera1} from camera {icamera0}")
                if not np.any(Rt_lidar0_camera[icamera0]):
                    raise Exception(f"Computing pose of camera {icamera1} from camera {icamera0}, but the pose of camera {icamera0} is not initialized")
                Rt_lidar0_camera[icamera1] = mrcal.compose_Rt(Rt_lidar0_camera[icamera0],
                                                              Rt01)

            else:
                ilidar0 = isensor0

                print(f"Estimating pose of camera {icamera1} from lidar {ilidar0}")
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

                print(f"Estimating pose of lidar {ilidar1} from camera {icamera0}")
                if not np.any(Rt_lidar0_camera[icamera0]):
                    raise Exception(f"Computing pose of lidar {ilidar1} from camera {icamera0}, but the pose of camera {icamera0} is not initialized")
                Rt_lidar0_lidar[ilidar1-1] = mrcal.compose_Rt(Rt_lidar0_camera[icamera0],
                                                              Rt01)

            else:
                ilidar0 = isensor0

                print(f"Estimating pose of lidar {ilidar1} from lidar {ilidar0}")
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

    shared_observation_counts, shared_observation_pcenter_normal = connectivity_matrices()

    print(f"Sensor shared-observations matrix for {Nlidars=} followed by {Ncameras=}:")
    print(full_symmetric_matrix_from_upper_triangle(shared_observation_counts))

    _traverse_sensor_connections \
        ( Nsensors,
          neighbors,
          cost_edge,
          found_best_path_to_node )

    for i in range(len(Rt_lidar0_camera)):
        if not np.any(Rt_lidar0_camera[i]):
            raise Exception(f"ERROR: Don't have complete observations overlap: camera {i} ({args.camera_topic[i]}) not connected")

    for i in range(len(Rt_lidar0_lidar)):
        if not np.any(Rt_lidar0_lidar[i]):
            raise Exception(f"ERROR: Don't have complete observations overlap: lidar {i+1} ({args.lidar_topic[i+1]}) not connected")


    for iboard in range(len(joint_observations)):
        q_observed_all = joint_observations[iboard][0]

        icamera_first = \
            next((i for i in range(len(q_observed_all)) if q_observed_all[i] is not None),
                 None)
        if icamera_first is not None:
            # We have some camera observation. I arbitrarily use the first one

            if not np.any(Rt_camera_board_cache[iboard,icamera_first]):
                raise Exception(f"Rt_camera_board_cache[{iboard=},{icamera_first=}] uninitialized")
            Rt_lidar0_board[iboard] = \
                mrcal.compose_Rt(Rt_lidar0_camera[icamera_first],
                                 Rt_camera_board_cache[iboard,icamera_first])

        else:
            # This board is observed only by LIDARs
            plidar_all = joint_observations[iboard][1]
            ilidar_first = \
                next((i for i in range(len(plidar_all)) if plidar_all[i] is not None),
                     None)
            if ilidar_first is None:
                raise Exception(f"Getting here is a bug: no camera or lidar observations for {iboard=}")

            # I'm looking at the first LIDAR in the list. This is arbitrary. Any
            # LIDAR will do
            plidar = plidar_all[ilidar_first]
            n = normal(plidar)
            plidar_mean = np.mean(plidar, axis=-2)
            # I have the normal to the board, in lidar coordinates. Compute an
            # arbitrary rotation that matches this normal. This is unique only
            # up to yaw
            Rt_board_lidar = np.zeros((4,3), dtype=float)
            Rt_board_lidar[:3,:] = mrcal.R_aligned_to_vector(n)
            # I want p_center_board to map to plidar_mean: R_board_lidar
            # plidar_mean + t_board_lidar = p_center_board
            Rt_board_lidar[3,:] = p_center_board - mrcal.rotate_point_R(Rt_board_lidar[:3,:],plidar_mean)

            if ilidar_first == 0:
                Rt_lidar0_board[iboard] = mrcal.invert_Rt(Rt_board_lidar)
            else:
                Rt_lidar0_board[iboard] = \
                    mrcal.compose_Rt(Rt_lidar0_lidar[ilidar_first - 1],
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
         Nmeas_lidar_observation_all,
         p_board_local,
         seed_kwargs):

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
        if np.any(rt_lidar_ref[0]):
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

    def measurement_indices():

        # Same measurement loop as in cost()

        imeas = 0
        for iboard in range(len(joint_observations)):
            q_observed_all = joint_observations[iboard][0]
            for icamera in range(Ncameras):
                if q_observed_all[icamera] is None:
                    continue

                yield (imeas, f"{iboard=}\\n{icamera=}")

                imeas += Nmeas_camera_observation


        for iboard in range(len(joint_observations)):
            plidar_all = joint_observations[iboard][1]
            for ilidar in range(Nlidars):

                if plidar_all[ilidar] is None:
                    continue

                yield (imeas, f"{iboard=}\\n{ilidar=}")

                imeas += len(plidar_all[ilidar])

        yield (imeas, f"regularization")


    def cost(b, *,

             # simplified computation for seeding
             use_distance_to_plane = False,
             report_imeas          = False):

        x     = np.zeros((Nmeasurements,), dtype=float)
        imeas = 0

        state = unpack_state(b)

        for iboard in range(len(joint_observations)):
            q_observed_all = joint_observations[iboard][0]
            for icamera in range(len(q_observed_all)):
                q_observed = q_observed_all[icamera]

                if q_observed is None:
                    continue

                if report_imeas:
                    print(f"{iboard=} {icamera=} {imeas=}")

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

                if report_imeas:
                    print(f"{iboard=} {ilidar=} {imeas=}")

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


    def plot_residuals(filename_base,
                       *,
                       x_camera,
                       x_lidar,
                       x_regularization):

        imeas_all = list(measurement_indices())

        measurement_boundaries = \
            [ x \
              for imeas,what in imeas_all \
              for x in \
              (f'arrow from {imeas}, graph 0 to {imeas}, graph 1 nohead',
               f'label "{what}" at {imeas},graph 0 left front offset 0,character 2 boxed') ]

        filename = f'{filename_base}.gp'
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
                _set  = measurement_boundaries,
                ylabel  = 'Camera fit residual (pixels)',
                y2label = 'LIDAR fit residual (m)',
                hardcopy = filename)
        print(f"Wrote '{filename}'")

        filename = f'{filename_base}-histogram.gp'
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

    seed = pack_state(**seed_kwargs)

    x = cost(seed, use_distance_to_plane = False, report_imeas = True)
    x_camera = \
        x[imeas_camera_0:
          imeas_camera_0+Nmeas_camera_observation_all]
    x_lidar  = \
        x[imeas_lidar_0:
          imeas_lidar_0+Nmeas_lidar_observation_all]
    x_regularization = \
        x[imeas_regularization_0:
          imeas_regularization_0+Nmeas_regularization]
    plot_residuals("/tmp/residuals-seed",
                   x_camera         = x_camera,
                   x_lidar          = x_lidar,
                   x_regularization = x_regularization)


    # Docs say:
    # * 0 (default) : work silently.
    # * 1 : display a termination report.
    # * 2 : display progress during iterations (not supported by 'lm')
    verbose = 2

    # Crude pre-solve
    res = scipy.optimize.least_squares(cost,
                                       seed,
                                       max_nfev = 20,
                                       method   = 'dogbox',
                                       verbose  = verbose,
                                       kwargs   = dict(use_distance_to_plane = True))

    # Fine refinement. Note: verbosity doesn't work (see note above about 'lm')
    res = scipy.optimize.least_squares(cost,
                                       res.x,
                                       method  = 'lm',
                                       ftol    = 1e-8,
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

        plot_residuals("/tmp/residuals",
                       x_camera         = x_camera,
                       x_lidar          = x_lidar,
                       x_regularization = x_regularization)

    return state

def get_joint_observation(bag,
                          *,
                          # Both input and output
                          cache = None):
    r'''Compute ONE lidar observation and/or ONE camera observation

    from a bag of ostensibly-stationary data'''

    bagname = os.path.split(os.path.splitext(os.path.basename(bag))[0])[1]

    if cache is not None:
        if bagname not in cache:
            cache[bagname] = dict()
        cache = cache[bagname]

    print(f"===== Looking for joint observations in '{bagname}'")

    Ncameras = len(args.camera_topic)
    q_observed = \
        [ calibration_data_import.chessboard_corners( \
                             bag,
                             args.camera_topic[icamera],
                             bagname = bagname,
                             cache   = cache ) \
          for icamera in range(Ncameras) ]

    Nlidars = len(args.lidar_topic)
    p_lidar = \
        [ calibration_data_import.get_lidar_observation( \
                                bag,
                                args.lidar_topic[ilidar],
                                p_board_local = p_board_local,
                                what = f"{bagname}-{calibration_data_import.canonical_lidar_topic_name(args.lidar_topic[ilidar])}",
                                cache = cache,
                                viz                          = args.viz,
                                viz_show_only_accepted       = args.viz_show_only_accepted,
                                viz_show_point_cloud_context = args.viz_show_point_cloud_context) \
          for ilidar in range(Nlidars)]

    if all(x is None for x in q_observed) and \
       all(x is None for x in p_lidar):
        return None

    return q_observed,p_lidar

def plot_geometry(filename,
                  *,
                  rt_ref_board,
                  rt_camera_ref,
                  rt_lidar_ref,
                  only_axes = False):
    data_tuples, plot_options = \
        mrcal.show_geometry(nps.glue(rt_camera_ref,
                                     rt_lidar_ref,
                                     axis = -2),
                            cameranames = (*args.camera_topic,
                                           *args.lidar_topic),
                            show_calobjects  = None,
                            axis_scale       = 1.0,
                            return_plot_args = True)

    if not only_axes:
        points_camera_observations = \
            [ mrcal.transform_point_rt(rt_ref_board[iboard],
                                       nps.clump(p_board_local,n=2) ) \
              for (q_observed,iboard,icamera) in observations_camera(joint_observations) ]
        legend_camera_observations = \
            [ f"{iboard=} {icamera=}" \
              for (q_observed,iboard,icamera) in observations_camera(joint_observations) ]
        points_lidar_observations = \
            [ mrcal.transform_point_rt(mrcal.invert_rt(rt_lidar_ref[ilidar]),
                                       plidar) \
                    for (plidar,iboard,ilidar) in observations_lidar(joint_observations) ]
        legend_lidar_observations = \
            [ f"{iboard=} {ilidar=}" \
              for (plidar,iboard,ilidar) in observations_lidar(joint_observations) ]

        data_tuples = (*data_tuples,
                       *[ (points_camera_observations[i],
                           dict(_with     = 'lines',
                                legend    = legend_camera_observations[i],
                                tuplesize = -3)) \
                          for i in range(len(points_camera_observations)) ],
                       *[ (points_lidar_observations[i],
                           dict(_with     = 'points',
                                legend    = legend_lidar_observations[i],
                                tuplesize = -3)) \
                          for i in range(len(points_lidar_observations)) ] )

    gp.plot(*data_tuples,
            **plot_options,
            hardcopy = filename)
    print(f"Wrote '{filename}'")

def find_multisense_units_lra(topics):
    r'''Converts unordered camera topics into multisense sets

Each multisense unit consists of 3 cameras (listed in their physical order from
left to right):

- "left": monochrome camera
- "aux": color camera (immediately to the right of the "left" camera). Wider FOV
  than the others
- "right": monochrome camera; same sensor, lens as the "left". Placed far to the
  right of the other two

We offload the stereo processing to the multisense hardware, so we must send out
the calibration data to that hardware. It expects results in multisense sets,
not as individual camera units, so we must find the multisense sets in the
topics we read

Each of our camera topics has the form

  PREFIX/multisenseSUFFIX/CAMERA/image_DEPTH

The PREFIX doesn't matter. The SUFFIX identifies the multisense unit. The CAMERA
is one of ("left","aux","right"). The DEPTH is either "mono" or "color". It
doesn't matter

This function returns a dict such as

  dict(multisense_front = np.array((0,5,-1)),
       multisense_left  = np.array((1,2,4)) )

This indicates that we have two multisense units. The "multisense_left" unit has
the left camera in topics[1], the right camera in topics[2] and the aux camera
in topics[4]. The "multisense_front" unit doesn't have an aux camera specified

    '''

    matches = [ re.search("/(multisense[a-zA-Z0-9_]*)/([a-zA-Z0-9_]+)/image", topic) \
                for topic in topics ]

    unit_list = [ m.group(1) if m is not None else None for m in matches ]

    units = set([u for u in unit_list if u is not None])

    units_lra = dict()
    for u in units:
        # initialize all the cameras to -1: none exist until we see them
        units_lra[u] = -np.ones( (3,), dtype=int)

    for i,m in enumerate(matches):
        if m is None: continue
        u = m.group(1)
        c = m.group(2)
        if   c == "left":  units_lra[u][0] = i
        elif c == "right": units_lra[u][1] = i
        elif c == "aux":   units_lra[u][2] = i
        else:
            raise Exception(f"Topic {topics[i]} has an unknown multisense camera: '{c}'. I only know about 'left','right','aux'")

    return units_lra

def write_multisense_calibration(multisense_units_lra):

    def write_intrinsics(D, unit, lra, models):
        if not all(models[i].intrinsics()[0] == 'LENSMODEL_OPENCV8' for i in lra):
            print(f"Multisense unit '{unit}' doesn't have ALL the cameras follow the LENSMODEL_OPENCV8 model. The multisense requires this (I think?). So I'm not writing its calibration file",
                  file = sys.stderr)
            return

        # I mimic the intrinsics files I see in the factory multisense unit:
        #
        # %YAML:1.0
        # M1: !!opencv-matrix
        #    rows: 3
        #    cols: 3
        #    dt: d
        #    data: [ 1287.92260742187500000,    0.00000000000000000,  927.83203125000000000,
        #               0.00000000000000000, 1287.85131835937500000,  611.74780273437500000,
        #               0.00000000000000000,    0.00000000000000000,    1.00000000000000000 ]
        # D1: !!opencv-matrix
        #    rows: 1
        #    cols: 8
        #    dt: d
        #    data: [   -0.12928095459938049,   -0.36398330330848694,    0.00021414959337562,    0.00007285172614502,   -0.02243571542203426,    0.30014526844024658,   -0.52040416002273560,   -0.12218914926052094 ]
        # M2: !!opencv-matrix
        #    rows: 3
        #    cols: 3
        #    dt: d
        #    data: [ 1290.11486816406250000,    0.00000000000000000,  925.88769531250000000,
        #               0.00000000000000000, 1290.20703125000000000,  610.00817871093750000,
        #               0.00000000000000000,    0.00000000000000000,    1.00000000000000000 ]
        # D2: !!opencv-matrix
        #    rows: 1
        #    cols: 8
        #    dt: d
        #    data: [   -0.44484668970108032,   -0.25598752498626709,    0.00005418056753115,    0.00011601681035245,    0.00710464408621192,   -0.01624384894967079,   -0.54431200027465820,   -0.02376704663038254 ]
        # M3: !!opencv-matrix
        #    rows: 3
        #    cols: 3
        #    dt: d
        #    data: [  837.97216796875000000,    0.00000000000000000,  992.33569335937500000,
        #               0.00000000000000000,  837.97766113281250000,  593.07342529296875000,
        #               0.00000000000000000,    0.00000000000000000,    1.00000000000000000 ]
        # D3: !!opencv-matrix
        #    rows: 1
        #    cols: 8
        #    dt: d
        #    data: [   -0.18338683247566223,   -0.37549209594726562,    0.00007732228550594,   -0.00011700890900102,   -0.01655971631407738,    0.18460999429225922,   -0.53588074445724487,   -0.09533628821372986 ]
        def intrinsics_definition(i):
            intrinsics = models[lra[i]].intrinsics()[1]
            fx,fy,cx,cy = intrinsics[:4]

            f = io.StringIO()
            np.savetxt(f,
                       nps.atleast_dims(intrinsics[4:], -2),
                       delimiter=',',
                       newline  = '',
                       fmt='%.12f')

            return \
    f"""M{i+1}: !!opencv-matrix
   rows: 3
   cols: 3
   dt: d
   data: [ {fx},   0.0,    {cx},
           0.0,   {fy},    {cy},
           0.0,    0.0,    1.0 ]
D{i+1}: !!opencv-matrix
   rows: 1
   cols: 8
   dt: d
   data: [ {f.getvalue()} ]
"""

        filename = f"{D}/multisense-intrinsics.yaml"
        with open(filename, "w") as f:
            f.write("%YAML:1.0\n" + ''.join( (intrinsics_definition(i) for i in range(3)) ))
        print(f"Wrote '{filename}'")
        return filename


    def write_extrinsics(D, unit, lra, models):
        # Currently I see this on the multisense:
        #
        # %YAML:1.0
        # R1: !!opencv-matrix
        #    rows: 3
        #    cols: 3
        #    dt: d
        #    data: [    0.99998009204864502,   -0.00628566369414330,   -0.00054030440514907,
        #               0.00628599477931857,    0.99998003244400024,    0.00061379116959870,
        #               0.00053643551655114,   -0.00061717530479655,    0.99999964237213135 ]
        # P1: !!opencv-matrix
        #    rows: 3
        #    cols: 4
        #    dt: d
        #    data: [ 1200.00000000000000000,    0.00000000000000000,  960.00000000000000000,    0.00000000000000000,
        #               0.00000000000000000, 1200.00000000000000000,  600.00000000000000000,    0.00000000000000000,
        #               0.00000000000000000,    0.00000000000000000,    1.00000000000000000,    0.00000000000000000 ]
        # R2: !!opencv-matrix
        #    rows: 3
        #    cols: 3
        #    dt: d
        #    data: [    0.99999642372131348,    0.00264605483971536,    0.00032660007127561,
        #              -0.00264585344120860,    0.99999630451202393,   -0.00061591889243573,
        #              -0.00032822860521264,    0.00061505258781835,    0.99999976158142090 ]
        # P2: !!opencv-matrix
        #    rows: 3
        #    cols: 4
        #    dt: d
        #    data: [ 1200.00000000000000000,    0.00000000000000000,  960.00000000000000000, -323.95281982421875000,
        #               0.00000000000000000, 1200.00000000000000000,  600.00000000000000000,    0.00000000000000000,
        #               0.00000000000000000,    0.00000000000000000,    1.00000000000000000,    0.00000000000000000 ]
        # R3: !!opencv-matrix
        #    rows: 3
        #    cols: 3
        #    dt: d
        #    data: [    0.99995851516723633,    0.00857812166213989,   -0.00305829849094152,
        #              -0.00857601314783096,    0.99996298551559448,    0.00070187030360103,
        #               0.00306420610286295,   -0.00067561317700893,    0.99999505281448364 ]
        # P3: !!opencv-matrix
        #    rows: 3
        #    cols: 4
        #    dt: d
        #    data: [ 1200.00000000000000000,    0.00000000000000000,  960.00000000000000000,  -40.03798675537109375,
        #               0.00000000000000000, 1200.00000000000000000,  600.00000000000000000,    0.00302204862236977,
        #               0.00000000000000000,    0.00000000000000000,    1.00000000000000000,   -0.00060220796149224 ]
        #
        # This is weird. I THINK this is describing a rectified system, NOT the
        # geometry of the actual cameras. The order is probably like in the
        # intrinsics: (left,right,aux). The rectified cameras should be in the
        # same location as the input cameras, and looking at the tranlations we
        # have the left camera at the origin. This makes sense. Reverse
        # engineering tells me:
        #
        # - P[:,3] are scaled translations: t*fx as described here:
        #
        #     https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#stereorectify
        #
        #   Both the yaml data above and the calibrated result tell me that the
        #   left-right separation is 27cm. This is consistent, so that's the
        #   meaning of P[:,3]
        #
        # - These translations in P[:,3]/fx are t_rightrect_leftrect and
        #   t_rightrect_auxrect. Because the physical order in the unit is
        #   left-aux-looooonggap-right. And the fact that the P2[:,3] is [x,0,0]
        #   tells me that P1-P2 are a rectified system: the rightrect camera is
        #   located at a perfect along-the-x-axis translation from the left
        #   camera. So t_rightrect_leftrect and not t_right_left.
        #
        # - The rotations in R are R_camrect_cam. Verified by writing test code
        #   for rectification, and comparing with the rectified images report by
        #   the camera:
        #
        #     imagersize_rectified = np.array((1920,1200), dtype=int)
        #     filenames_input = \
        #         ('/tmp/multisense/left_camera_frame/image00000.png',
        #          '/tmp/multisense/right_camera_frame/image00000.png',
        #          '/tmp/multisense/aux_camera_frame/image00000.png')
        #     qrect = \
        #         np.ascontiguousarray(nps.mv(nps.cat(*np.meshgrid(np.arange(imagersize_rectified[0], dtype=float),
        #                                                          np.arange(imagersize_rectified[1], dtype=float))),
        #                                     0,-1))
        #     for i in range(3):
        #         intrinsics = np.zeros((12,), dtype=float)
        #         intrinsics[0]  = M[i,0,0]
        #         intrinsics[1]  = M[i,1,1]
        #         intrinsics[2]  = M[i,0,2]
        #         intrinsics[3]  = M[i,1,2]
        #         intrinsics[4:] = D[i]
        #         # shape (3,3)
        #         Phere = P[i,:,:3]
        #         R_cam_camrect = nps.transpose(R[i])
        #         t_cam_camrect = P[i,:,3]*0
        #         # assuming Phere[:2,:2] is 1200*I
        #         p_camrect = (qrect - Phere[:2,2])/1200
        #         p_camrect = nps.glue(p_camrect, np.ones(p_camrect.shape[:-1] + (1,)),
        #                              axis = -1)
        #         pcam = mrcal.rotate_point_R(R_cam_camrect, p_camrect) + t_cam_camrect
        #         q = mrcal.project(pcam, 'LENSMODEL_OPENCV8', intrinsics)
        #         image = mrcal.load_image(filenames_input[i])
        #         image_rect = mrcal.transform_image(image,q.astype(np.float32))
        #         filename_rect = f"/tmp/rect-{i}.jpg"
        #         mrcal.save_image(filename_rect, image_rect)
        #         print(f"Wrote '{filename_rect}")
        #
        # - So the P matrices are exactly what's described in the docs:
        #
        #     https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#stereorectify
        #
        #   Note that this is consistent for a left/right stereo pair. It is NOT
        #   consistent for any other stereo pairs: the aux camera does NOT have
        #   P[:,3] in the form [xxx,0,0]. Perhaps this is pinholed in the same
        #   way, but not epipolar-line aligned? This sorta makes sense. The
        #   topics published by the multisense driver:
        #
        #     dima@fatty:~$ ROS_MASTER_URI=http://128.149.135.71:11311 rostopic list | egrep '(image_rect|disparity)$'
        #
        #     /multisense/left/image_rect
        #     /multisense/right/image_rect
        #     /multisense/aux/image_rect
        #     /multisense/left/disparity
        #     /multisense/right/disparity
        #
        #   So we have "rectified" all 3 cameras but we only have disparities
        #   for the left/right pair. What is a "right" disparity? I looked at
        #   both of these. The left disparity makes sense and looks reasonable.
        #   The right disparity is a mess. I'm guessing this isn't working right
        #   and that nobody is using it. So yes. They are "rectifying"
        #   (pinhoing, really) all 3 cameras, but only use the left/right
        #   cameras as a stereo pair
        #
        # This is all enough to write out comparable data for my calibration.

        if np.any(models[lra[0]].imagersize() - np.array((1920,1200))):
            raise Exception("The rectification routine assumes we're looking at full-res left camera data, but we're not")
        if np.any(models[lra[1]].imagersize() - np.array((1920,1200))):
            raise Exception("The rectification routine assumes we're looking at full-res right camera data, but we're not")
        if np.any(models[lra[2]].imagersize() - np.array((1920,1188))):
            raise Exception("The rectification routine assumes we're looking at full-res aux camera data, but we're not")

        # Matching up what I currently see on their hardware
        fx_rect = 1200
        fy_rect = 1200
        cx_rect = 960
        cy_rect = 600

        # I compute the rectified system geometry. Logic stolen from
        # mrcal.stereo._rectified_system_python()

        ileft  = lra[0]
        iright = lra[1]
        iaux   = lra[2]

        # Compute the rectified system. I do this to get the geometry ONLY
        models_rectified = \
            mrcal.rectified_system( (models[ileft], models[iright]),

                                    # FOV numbers are made-up. These are
                                    # used to compute the rectified
                                    # intrinsics, but I use what multisense
                                    # has seen previously instead
                                    az_fov_deg        = 30,
                                    el_fov_deg        = 30,
                                    pixels_per_deg_az = 1000,
                                    pixels_per_deg_el = 1000 )

        Rt_leftrect_left = \
            mrcal.compose_Rt( models_rectified[0].extrinsics_Rt_fromref(),
                              models[ileft].extrinsics_Rt_toref())
        Rt_rightrect_right = \
            mrcal.compose_Rt( models_rectified[1].extrinsics_Rt_fromref(),
                              models[iright].extrinsics_Rt_toref())
        Rt_rightrect_leftrect = \
            mrcal.compose_Rt( models_rectified[1].extrinsics_Rt_fromref(),
                              models_rectified[0].extrinsics_Rt_toref())

        if np.any(np.abs(Rt_rightrect_leftrect[:3,:] - np.eye(3)) > 1e-8):
            raise Exception("Logic error: Rt_rightrect_leftrect should have an identity rotation")
        if np.any(np.abs(Rt_rightrect_leftrect[3,1:]) > 1e-8):
            raise Exception("Logic error: Rt_rightrect_leftrect should have a purely-x translation")

        # results
        R = np.zeros((3,3,3), dtype=float)
        P = np.zeros((3,3,4), dtype=float)


        # The R matrices are R_camrect_cam
        R[0] = Rt_leftrect_left  [:3,:]
        R[1] = Rt_rightrect_right[:3,:]

        Pbase = np.array(((fx_rect, 0,       cx_rect),
                          (      0, fy_rect, cy_rect),
                          (      0, 0,       1,),),)

        P[0] = nps.glue(Pbase,
                        np.zeros((3,1),),
                        axis = -1)
        P[1] = nps.glue(Pbase,
                        nps.dummy(Rt_rightrect_leftrect[3,:],
                                  -1) * fx_rect,
                        axis = -1)

        # Done with the stereo pair I have: (left,right). As described above,
        # "aux" isn't a part of any stereo pair (the left camera would need a
        # different R for that stereo pair). So what exactly is this? I make an
        # educated guess. I need an "auxrect" coordinate system. I translate the
        # aux camera to sit colinearly with l,r.
        Rt_ref_left  = models[ileft ].extrinsics_Rt_toref()
        Rt_ref_right = models[iright].extrinsics_Rt_toref()
        Rt_ref_aux   = models[iaux  ].extrinsics_Rt_toref()

        vbaseline = Rt_ref_right[3,:] - Rt_ref_left[3,:]
        vbaseline /= nps.mag(vbaseline)

        caux           = Rt_ref_aux[3,:] - Rt_ref_left[3,:]
        caux_projected = nps.inner(caux,vbaseline) * vbaseline
        Rt_ref_aux[3,:] = Rt_ref_left[3,:] + caux_projected
        print(f"Translated aux camera by {nps.mag(caux - caux_projected):.3f}m to sit on the left-right baseline.")
        print("  This is completely made-up, but we have to do this to fit into multisense's internal representation.")

        # We now have a compatible aux pose. I rectify it: I reuse the
        # left-rectified rotation (because I'm only allowed to have one, and
        # cannot recompute a better one for the (left,aux) pair)
        baseline_aux = nps.mag(caux_projected)
        Rt_leftrect_auxrect = mrcal.identity_Rt()
        Rt_leftrect_auxrect[3,0] = baseline_aux

        Rt_auxrect_aux = \
            mrcal.compose_Rt( mrcal.invert_Rt(Rt_leftrect_auxrect),
                              Rt_leftrect_left,
                              mrcal.invert_Rt(Rt_ref_left),
                              Rt_ref_aux )
        Rt_auxrect_leftrect = mrcal.invert_Rt(Rt_leftrect_auxrect)

        R[2] = Rt_auxrect_aux[:3,:]
        P[2] = nps.glue(Pbase,
                        nps.dummy(Rt_auxrect_leftrect[3,:],
                                  -1) * fx_rect,
                        axis = -1)




        def extrinsics_definition(i):
            f = io.StringIO()
            np.savetxt(f,
                       nps.atleast_dims(R[i].ravel(), -2),
                       delimiter=',',
                       newline  = '',
                       fmt='%.12f')
            Rstring = f.getvalue()

            f = io.StringIO()
            np.savetxt(f,
                       nps.atleast_dims(P[i].ravel(), -2),
                       delimiter=',',
                       newline  = '',
                       fmt='%.12f')
            Pstring = f.getvalue()

            return \
    f"""R{i+1}: !!opencv-matrix
   rows: 3
   cols: 3
   dt: d
   data: [ {Rstring} ]
P{i+1}: !!opencv-matrix
   rows: 3
   cols: 4
   dt: d
   data: [ {Pstring} ]
"""

        filename = f"{D}/multisense-extrinsics.yaml"
        with open(filename, "w") as f:
            f.write("%YAML:1.0\n" + ''.join( (extrinsics_definition(i) for i in range(3)) ))
        print(f"Wrote '{filename}'")
        return filename


    for unit in multisense_units_lra.keys():
        lra = multisense_units_lra[unit]
        if np.any(lra < 0):
            print(f"Multisense unit '{unit}' doesn't have ALL the cameras specified: lra = {lra}. I'm not writing its calibration file",
                  file = sys.stderr)
            continue

        # The multisense files are ordered by
        #
        # - left
        # - right
        # - aux
        #
        # And the have the left camera at the identity transform (I don't know if
        # this is a requirement or convention). I do this as well. In particular, I
        # write the results to the same directory as the "left" camera
        root,extension = os.path.splitext(args.models[lra[0]])
        D              = os.path.split(root)[0]
        if len(D) == 0: D = '.'

        filename_intrinsics = write_intrinsics(D, unit, lra, models)
        filename_extrinsics = write_extrinsics(D, unit, lra, models)

        print("Send like this:")
        print(f"  rosrun multisense_lib ImageCalUtility -e {filename_extrinsics} -e {filename_intrinsics} -a IP")

def rpy_from_r(r):
    # Wikipedia has some rotation matrix representations of euler angle rotations:
    #
    #     [ ca cb   ca sb sy - sa cy   ca sb cy + sa sy ]
    # R = [ sa cb   sa sb sy + ca cy   sa sb cy - ca sy ]
    #     [ -sb     cb sy              cb cy            ]
    #
    #
    #     [ cb cy   sa sb cy - ca sy   ca sb cy + sa sy ]
    # R = [ cb sy   sa sb sy + ca cy   ca sb sy - sa cy ]
    #     [ -sb     sa cb              ca cb            ]
    #
    # where yaw,pitch,roll = a,b,y

    R = mrcal.R_from_r(r)

    def asin(s): return np.arcsin(np.clip(s,-1,1))

    if True:
        # the first representation
        b = -asin(R[2,0])
        y = np.arctan2(R[2,1], R[2,2])
        a = np.arctan2(R[1,0], R[0,0])

    else:
        # the second representation
        b = -asin(R[2,0])
        a = np.arctan2(R[2,1], R[2,2])
        y = np.arctan2(R[1,0], R[0,0])

    yaw,pitch,roll = a,b,y

    return np.array((roll,pitch,yaw),)




if args.models:

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

else:
    p_board_local = None





if args.read_cache:
    with open(args.cache, "rb") as f:
        cache = pickle.load(f)
else:
    cache = dict()

# read AND write the cache dict
joint_observations = [get_joint_observation(bag, cache=cache) for bag in args.bag ]

# Any boards observed by a single sensor aren't useful, and I get rid of
# them
def num_sensors_observed(o):
    return \
        sum(0 if x is None else 1 for qp in o for x in qp)

mask_observations = np.ones( (len(joint_observations),), dtype=bool)
for i,o in enumerate(joint_observations):
    if o is None or num_sensors_observed(o) <= 1:
        mask_observations[i] = 0

joint_observations = [o for i,o in enumerate(joint_observations) \
                      if mask_observations[i]]

idx_observations = np.nonzero(mask_observations)[0]
for iboard in np.arange(len(joint_observations)):
    print(f"{iboard=} (pre-filter iboard={idx_observations[iboard]}) corresponds to {args.bag[idx_observations[iboard]]}")

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

if p_board_local is not None:
    Nmeas_camera_observation = p_board_local.shape[-3]*p_board_local.shape[-2]*2
    Nmeas_camera_observation_all = Nobservations_camera * Nmeas_camera_observation
else:
    Nmeas_camera_observation     = 0
    Nmeas_camera_observation_all = 0

Nobservations_lidar  = \
    sum(0 if x is None else 1 \
        for o in joint_observations \
        for x in o[1])
Nmeas_lidar_observation_all = \
    sum(0 if x is None else len(x) \
        for o in joint_observations \
        for x in o[1])

if not args.read_cache:
    with open(args.cache, "wb") as f:
        pickle.dump(cache, f)


NcameraObservations = [sum(0 if o[0][icamera] is None else 1 for o in joint_observations) \
                       for icamera in range(Ncameras)]
NlidarObservations = [sum(0 if o[1][ilidar] is None else 1 for o in joint_observations) \
                      for ilidar in range(Nlidars)]
print(f"Got {NcameraObservations=}")
print(f"Got {NlidarObservations=}")

for icamera in range(Ncameras):
    NcameraObservations_this = NcameraObservations[icamera]
    if NcameraObservations_this == 0:
        print(f"I need at least 1 observation of each camera. Got only {NcameraObservations_this} for camera {icamera} from {args.camera_topic[icamera]}",
              file=sys.stderr)
        sys.exit(1)

for ilidar in range(Nlidars):
    NlidarObservations_this = NlidarObservations[ilidar]
    if NlidarObservations_this < 3:
        print(f"I need at least 3 observations of each lidar to unambiguously set the translation (the set of all plane normals must span R^3). Got only {NlidarObservations_this} for lidar {ilidar} from {args.lidar_topic[ilidar]}",
              file=sys.stderr)
        sys.exit(1)


seed_state = \
    fit_estimate( joint_observations,
                  Nboards, Ncameras, Nlidars,
                  Nmeas_camera_observation,
                  Nmeas_camera_observation_all,
                  Nmeas_lidar_observation_all,
                  p_board_local )
plot_geometry("/tmp/geometry-seed.gp",
              **seed_state)
plot_geometry("/tmp/geometry-seed-onlyaxes.gp",
              only_axes = True,
              **seed_state)


solved_state = \
    fit( joint_observations,
         Nboards, Ncameras, Nlidars,
         Nmeas_camera_observation,
         Nmeas_camera_observation_all,
         Nmeas_lidar_observation_all,
         p_board_local,
         seed_state )
plot_geometry("/tmp/geometry.gp",
              **solved_state)
plot_geometry("/tmp/geometry-onlyaxes.gp",
              only_axes = True,
              **solved_state)

for imodel in range(len(args.models)):
    models[imodel].extrinsics_rt_fromref(solved_state['rt_camera_ref'][imodel])
    root,extension = os.path.splitext(args.models[imodel])
    filename = f"{root}-mounted{extension}"
    models[imodel].write(filename)
    print(f"Wrote '{filename}'")

for iobservation in range(len(joint_observations)):
    (q_observed, p_lidar) = joint_observations[iobservation]
    for ilidar in range(Nlidars):
        if p_lidar[ilidar] is None: continue
        for icamera in range(Ncameras):
            if q_observed[icamera] is None: continue

            rt_camera_lidar = mrcal.compose_rt(solved_state['rt_camera_ref'][icamera],
                                               mrcal.invert_rt(solved_state['rt_lidar_ref'][ilidar]))
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
                     yinv    = True,
                     hardcopy = filename)
            print(f"Wrote '{filename}'")

# Write the intra-multisense calibration
multisense_units_lra = find_multisense_units_lra(args.camera_topic)
write_multisense_calibration(multisense_units_lra)


# Write the inter-multisense extrinsics
for unit in multisense_units_lra.keys():
    lra = multisense_units_lra[unit]
    l = lra[0]
    if l < 0:
        continue

    topic = args.camera_topic[l]

    rt_multisenseleft_lidar0 = models[l].extrinsics_rt_fromref()
    rpy = rpy_from_r(rt_multisenseleft_lidar0[:3])
    xyz = rt_multisenseleft_lidar0[3:]
    print(f"rt_multisenseleft_lidar0 pose for {topic}: rt_multisenseleft_lidar0={rt_multisenseleft_lidar0} {rpy=} {xyz=}")

    rt_lidar0_multisenseleft = mrcal.invert_rt(rt_multisenseleft_lidar0)
    rpy = rpy_from_r(rt_lidar0_multisenseleft[:3])
    xyz = rt_lidar0_multisenseleft[3:]
    print(f"rt_lidar0_multisenseleft pose for {topic}: rt_lidar0_multisenseleft={rt_lidar0_multisenseleft} {rpy=} {xyz=}")


# Write the inter-multisense lidar
for ilidar in range(Nlidars):
    topic = args.lidar_topic[ilidar]

    rt_lidar_lidar0 = solved_state['rt_lidar_ref'][ilidar]
    rpy = rpy_from_r(rt_lidar_lidar0[:3])
    xyz = rt_lidar_lidar0[3:]
    print(f"rt_lidar_lidar0 pose for {topic}: rt_lidar_lidar0={rt_lidar_lidar0} {rpy=} {xyz=}")

    rt_lidar0_lidar = mrcal.invert_rt(rt_lidar_lidar0)
    rpy = rpy_from_r(rt_lidar0_lidar[:3])
    xyz = rt_lidar0_lidar[3:]
    print(f"rt_lidar0_lidar pose for {topic}: rt_lidar0_lidar={rt_lidar0_lidar} {rpy=} {xyz=}")
