#!/usr/bin/python3

r'''Calibrate a set of cameras and LIDARs into a common coordinate system

SYNOPSIS

  $ lidars=(/lidar/vl_points_0)
  $ cameras=(/front/multisense/{{left,right}/image_mono_throttle,aux/image_color_throttle})

  $ ./fit.py \
      --lidar-topic  ${(j:,:)lidars}  \
      --camera-topic ${(j:,:)cameras} \
      --bag 'camera-lidar-*.bag'      \
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
                        a regex match against the bag paths''')
    parser.add_argument('--dump',
                        type=str,
                        help = '''Write solver diagnostics into the given
                        .pickle file''')
    parser.add_argument('models',
                        type = str,
                        nargs='*',
                        help='''Camera model for the optical calibration. Only
                        the intrinsics are used. The number of models given must
                        match the number of --camera-topic EXACTLY''')

    args = parser.parse_args()

    import glob
    import re
    args.bag = args.bag.rstrip('/') # to not confuse os.path.splitext()
    bags = sorted(glob.glob(args.bag))

    if args.exclude_bag is not None:
        for ex in args.exclude_bag:
            bags = [b for b in bags if not re.search(ex, b)]

    if len(bags) < 3:
        print(f"--bag '{args.bag}' must match at least 3 files. Instead this matched {len(bags)} files",
              file=sys.stderr)
        sys.exit(1)
    args.bag = bags

    args.lidar_topic  = args.lidar_topic.split(',')
    if args.camera_topic is not None:
        args.camera_topic = args.camera_topic.split(',')
    else:
        args.camera_topic = []

    if len(args.models) != len(args.camera_topic):
        print(f"The number of models given must match the number of --camera-topic EXACTLY",
              file=sys.stderr)
        sys.exit(1)

    return args


args = parse_args()



import numpy as np
import numpysane as nps
import gnuplotlib as gp
import io
import pickle
import mrcal
import clc


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





if len(args.models) > 0:
    m = mrcal.cameramodel(args.models[0])
    o = m.optimization_inputs()
    H,W = o['observations_board'].shape[-3:-1]
    calibration_object_kwargs = \
        dict(object_spacing  = o['calibration_object_spacing'],
             object_width_n  = W,
             object_height_n = H)
else:
    calibration_object_kwargs = dict()

kwargs_calibrate = dict(bags            = args.bag,
                        lidar_topic     = args.lidar_topic,
                        camera_topic    = args.camera_topic,
                        models          = args.models,
                        check_gradient  = False,
                        Npoints_per_segment                = 15,
                        threshold_min_Nsegments_in_cluster = 4,
                        **calibration_object_kwargs)
result = clc.calibrate(dump_optimization_inputs = args.dump is not None,
                       **kwargs_calibrate)


for imodel in range(len(args.models)):
    models[imodel].extrinsics_rt_toref(result['rt_ref_camera'][imodel])
    root,extension = os.path.splitext(args.models[imodel])
    filename = f"{root}-mounted{extension}"
    models[imodel].write(filename)
    print(f"Wrote '{filename}'")

for ilidar,rt_ref_lidar in enumerate(result['rt_ref_lidar']):
    # dummy lidar "cameramodel". The intrinsics are made-up, but the extrinsics
    # are true, and can be visualized with the usual tools
    filename = f"/tmp/lidar{ilidar}-mounted.cameramodel"
    model = mrcal.cameramodel( intrinsics = ('LENSMODEL_PINHOLE',
                                             np.array((1.,1.,0.,0.))),
                               imagersize = (1,1),
                               extrinsics_rt_toref = rt_ref_lidar )
    model.write(filename,
                note = "Intrinsics are made-up and nonsensical")
    print(f"Wrote '{filename}'")


context = \
    dict(result           = result,
         lidar_topic      = args.lidar_topic,
         camera_topic     = args.camera_topic,
         kwargs_calibrate = kwargs_calibrate)
if args.dump is not None:
    with open(args.dump, 'wb') as f:
        pickle.dump( context,
                     f )
    print(f"Wrote '{args.dump}'")




statistics = clc.post_solve_statistics(bag               = args.bag[0],
                                       lidar_topic       = args.lidar_topic,
                                       Nsectors          = 36,
                                       rt_vehicle_lidar0 = mrcal.identity_rt(),
                                       models            = args.models,
                                       **result)




data_tuples_sensor_forward_vectors = \
    clc.get_data_tuples_sensor_forward_vectors(context['result']['rt_ref_lidar' ],
                                               context['result']['rt_ref_camera'],
                                               context['lidar_topic'],
                                               context['camera_topic'])



# shape (Nsensors, Nsectors)
isvisible_per_sensor_per_sector = statistics['isvisible_per_sensor_per_sector']

Nsensors,Nsectors = isvisible_per_sensor_per_sector.shape
dth = np.pi*2./Nsectors
th = np.arange(Nsectors)*dth + dth/2.
plotradius = nps.transpose(np.arange(Nsensors) + 10)
ones = np.ones( (Nsectors,) )

filename = '/tmp/observability.pdf'
gp.plot( (th,                 # angle
          plotradius*ones,    # radius
          ones*dth*0.9,       # angular width of slice
          ones*0.9,           # depth of slice
          isvisible_per_sensor_per_sector,
          dict(_with = 'sectors palette fill solid',
               tuplesize = 5)),

         *data_tuples_sensor_forward_vectors,

         _xrange = (-10-Nsensors,10+Nsensors),
         _yrange = (-10-Nsensors,10+Nsensors),
         square = True,
         unset = 'colorbox',
         title = 'Observability map of each sensor',
         hardcopy = filename,
        )
print(f"Wrote '{filename}'")

stdev_worst = statistics['stdev_worst']
i = stdev_worst != 0
filename = '/tmp/uncertainty.pdf'
gp.plot( (th[i],                 # angle
          10.*ones[i],           # radius
          ones[i]*dth*0.9,       # angular width of slice
          ones[i]*0.9,           # depth of slice
          stdev_worst[i],
          dict(tuplesize = 5,
               _with = 'sectors palette fill solid')),
         *data_tuples_sensor_forward_vectors,
         _xrange = (-11,11),
         _yrange = (-11,11),
         square = True,
         title = 'Worst-case uncertainty. Put the board in high-uncertainty regions',
         hardcopy = filename,
        )
print(f"Wrote '{filename}'")

import IPython
IPython.embed()
sys.exit()


sys.exit()















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


print("The poses follow. The reference is defined to sit at lidar0\n")
print("To visualize an aligned bag, run:\n\n./show-aligned-lidar-pointclouds.py \\")

if Ncameras > 0:
    # Write the inter-multisense extrinsics
    multisense_topics          = []
    str_multisense_poses       = ''
    str_multisense_poses_other = ''
    for unit in multisense_units_lra.keys():
        lra = multisense_units_lra[unit]
        l = lra[0]
        if l < 0:
            continue

        topic = args.camera_topic[l]
        multisense_topics.append(topic)

        rt_multisenseleft_lidar0 = models[l].extrinsics_rt_fromref()
        str_multisense_poses += \
            f"  --rt-multisenseleft-ref \" {','.join(list(str(x) for x in rt_multisenseleft_lidar0))}\" \\\n"

        rpy = rpy_from_r(rt_multisenseleft_lidar0[:3])
        xyz = rt_multisenseleft_lidar0[3:]
        str_multisense_poses_other += \
            f"  {rpy=} {xyz=}\n"
    print(f"  --multisense-topic {','.join(multisense_topics)} \\")
    print(str_multisense_poses)
    print("other poses:")
    print(str_multisense_poses_other)
    print('\n')


# Write the inter-multisense lidar
lidar_topic          = []
str_lidar_poses       = ''
str_lidar_poses_other = ''
for ilidar in range(Nlidars):
    topic = args.lidar_topic[ilidar]
    lidar_topic.append(topic)

    rt_lidar_lidar0 = solved_state['rt_lidar_ref'][ilidar]
    str_lidar_poses += \
        f"  --rt-lidar-ref \" {','.join(list(str(x) for x in rt_lidar_lidar0))}\" \\\n"

    rpy = rpy_from_r(rt_lidar_lidar0[:3])
    xyz = rt_lidar_lidar0[3:]
    str_lidar_poses_other += \
        f"  {rpy=} {xyz=}\n"
print(f"  --lidar-topic {','.join(lidar_topic)} \\")
print(str_lidar_poses, end='')
print('  ' + args.bag[0])
print('\nOr pass in any of the other bags\n')

print("other poses:")
print(str_lidar_poses_other)
