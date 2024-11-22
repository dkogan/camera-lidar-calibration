#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <limits.h>
#include <assert.h>
#include <math.h>
#include <mrcal/mrcal.h>
#include <dogleg.h>
#include "clc.h"
#include "util.h"
#include "minimath/minimath_generated.h"

typedef struct
{
    // The points_and_plane contains indices into the points[] array here

    // pointers to the pools declared above
    clc_point3f_t*                points;
    const clc_points_and_plane_t* points_and_plane;
} points_and_plane_full_t;

typedef struct
{
    #warning "This should be board detections, NOT raw images"
    union
    {
        mrcal_image_uint8_t uint8;
        mrcal_image_bgr_t   bgr;
    } images[clc_Ncameras_max];

    points_and_plane_full_t lidar_scans[clc_Nlidars_max];
} sensor_snapshot_segmented_t;





static int
pairwise_index(const int a, const int b, const int N)
{
    // I have an (N,N) symmetric matrix with a 0 diagonal. I store only the
    // upper triangle: a 1D array of (N*(N-1)/2) values. This function returns
    // the linear index into this upper-triangle-only-row-first array
    //
    // If a > b: a + b*N - sum(1..b+1) = a + b*N - (b+1)*(b+2)/2
    if(a>b) return a + b*N - (b+1)*(b+2)/2;
    else    return b + a*N - (a+1)*(a+2)/2;
}

static int
pairwise_N(const int N)
{
    // I have an (N,N) symmetric matrix with a 0 diagonal. I store only the
    // upper triangle: (N*(N-1)/2) values. This function returns the size of the
    // linear array
    return N*(N-1)/2;
}

static void
print_full_symmetric_matrix_from_upper_triangle(const uint16_t* A,
                                                const int N)
{
    // I have an (N,N) symmetric matrix with a 0 diagonal. I store only the
    // upper triangle: (N*(N-1)/2) values. This function prints the full
    // symmetric array. For debugging
    for(int i=0; i<N; i++)
    {
        for(int j=0; j<N; j++)
            fprintf(stderr, "%d ",
                    i==j ? 0 : A[pairwise_index(i,j,N)]);
        fprintf(stderr, "\n");
    }
}

static void
connectivity_matrix(// out
                    uint16_t* shared_observation_counts,

                    // in
                    const sensor_snapshot_segmented_t* sensor_snapshots_filtered,
                    const int                          Nsensor_snapshots_filtered,

                    // These apply to ALL the sensor_snapshots[]
                    const int Ncameras,
                    const int Nlidars)
{
    /*
      Fills in shared_observation_counts: a matrix of sensor observations

      This is a symmetric (Nsensor,Nsensor) matrix of integers, where each entry
      contains the number of frames containing overlapping observations for that
      pair of sensors. The upper-triangle is stored only, row-first.

      The sensors are the lidars,cameras; in order
    */

    const int Nsensors = Ncameras + Nlidars;
    memset(shared_observation_counts, 0, pairwise_N(Nsensors)*sizeof(shared_observation_counts[0]));

    for(int isnapshot=0; isnapshot < Nsensor_snapshots_filtered; isnapshot++)
    {
        const sensor_snapshot_segmented_t* sensor_snapshot = &sensor_snapshots_filtered[isnapshot];

#if 0
        // camera stuff; not done yet

        // "icamera" and "ilidar" used to mean two things here. Look at fit.py
        for(int icamera0=0; icamera0<Ncameras-1; icamera0++)
        {
            icamera0,q_observed0 = cameras[icamera0];

            pcenter_normal_camera0 = \
                pcenter_normal_camera(q_observed0,
                                      isnapshot,
                                      icamera0,
                                      what = f"{isnapshot=},icamera={icamera0}");
            for(int icamera1=icamera0+1; icamera1<Ncameras; icamera1++)
            {
                icamera1,q_observed1 = cameras[icamera1]

                    idx = pairwise_index(Nlidars+icamera0,
                                         Nlidars+icamera1,
                                         Nsensors);

                // shape (Nsensors=2,pcenter_normal=2,3)
                pcloud_normal_next = get_pcloud_normal_next(idx);
                pcloud_normal_next[0] = pcenter_normal_camera0;
                pcenter_normal_camera(q_observed1,
                                      isnapshot,
                                      icamera1,
                                      what = f"{isnapshot=},icamera={icamera1}",
                                      out = pcloud_normal_next[1]);

                shared_observation_counts[idx] += 1;
            }
        }
#endif

        for(int ilidar0=0; ilidar0<Nlidars-1; ilidar0++)
        {
            if(sensor_snapshot->lidar_scans[ilidar0].points == NULL)
                continue;

            for(int ilidar1=ilidar0+1; ilidar1<Nlidars; ilidar1++)
            {
                if(sensor_snapshot->lidar_scans[ilidar1].points == NULL)
                    continue;

                const int idx = pairwise_index(ilidar1, ilidar0,
                                               Nsensors);
                shared_observation_counts[idx]++;
            }
        }

#if 0
        // camera stuff; not done yet

        // "icamera" and "ilidar" used to mean two things here. Look at fit.py

        for(icamera in range(Ncameras))
        {
            icamera,q_observed = cameras[icamera]

                pcenter_normal_camera0 = \
                pcenter_normal_camera(q_observed,
                                      isnapshot,
                                      icamera,
                                      what = f"{isnapshot=},icamera={icamera}");

            for(ilidar in range(Nlidars))
            {
                ilidar,plidar = lidars[ilidar]

                    idx = pairwise_index(ilidar,
                                         Nlidars+icamera,
                                         Nsensors);

                // shape (Nsensors=2,pcenter_normal=2,3)
                pcloud_normal_next = get_pcloud_normal_next(idx);
                // isensor(camera) > isensor(lidar) always, so I store the
                // camera into pcloud_normal_next[1] and the lidar into
                // pcloud_normal_next[0]
                pcloud_normal_next[1] = pcenter_normal_camera0;
                pcenter_normal_lidar(plidar,
                                     out = pcloud_normal_next[0]);

                shared_observation_counts[idx] += 1;
            }
        }
#endif
    }
}

static void
mrcal_point3_from_clc_point3f( mrcal_point3_t* pout,
                               const clc_point3f_t*  pin)
{
    for(int i=0; i<3; i++)
        pout->xyz[i] = (double)pin->xyz[i];
}

static bool
compute_board_poses(// out
                    double*                            Rt_lidar0_board,
                    // in
                    const sensor_snapshot_segmented_t* sensor_snapshots_filtered,
                    const int                          Nsensor_snapshots_filtered,
                    const int                          Ncameras,
                    const int                          Nlidars,
                    const double*                      Rt_lidar0_lidar,
                    const mrcal_point3_t*              p_center_board)
{
    for(int isnapshot=0; isnapshot < Nsensor_snapshots_filtered; isnapshot++)
    {

        // cameras
#if 0
        q_observed_all = joint_observations[iboard][0];
        icamera_first =
            next((i for i in range(len(q_observed_all)) if q_observed_all[i] is not None),
                 None);
        if(icamera_first is not None)
        {
            // We have some camera observation. I arbitrarily use the first one

            if not np.any(Rt_camera_board_cache[iboard,icamera_first]):
            raise Exception(f"Rt_camera_board_cache[{iboard=},{icamera_first=}] uninitialized");
            Rt_lidar0_board[iboard] =
                mrcal.compose_Rt(Rt_lidar0_camera[icamera_first],
                                 Rt_camera_board_cache[iboard,icamera_first]);
        }
        else
#endif
        {
            // This board is observed only by LIDARs


            const sensor_snapshot_segmented_t* sensor_snapshot = &sensor_snapshots_filtered[isnapshot];

            int ilidar_first;
            for(ilidar_first=0; ilidar_first<Nlidars; ilidar_first++)
                if(sensor_snapshot->lidar_scans[ilidar_first].points != NULL)
                    break;

            if(ilidar_first == Nlidars)
            {
                MSG("Getting here is a bug: no camera or lidar observations for isnapshot=%d",
                    isnapshot);
                return false;
            }

            // I'm looking at the first LIDAR in the list. This is arbitrary. Any
            // LIDAR will do
            mrcal_point3_t n,plidar_mean;
            mrcal_point3_from_clc_point3f(&n,
                                          &sensor_snapshot->lidar_scans[ilidar_first].points_and_plane->plane.n);
            mrcal_point3_from_clc_point3f(&plidar_mean,
                                          &sensor_snapshot->lidar_scans[ilidar_first].points_and_plane->plane.p_mean);

            // I have the normal to the board, in lidar coordinates. Compute an
            // arbitrary rotation that matches this normal. This is unique only
            // up to yaw
            double Rt_board_lidar[4*3];
            mrcal_R_aligned_to_vector(Rt_board_lidar,
                                      n.xyz);
            // I want p_center_board to map to p: R_board_lidar
            // p + t_board_lidar = p_center_board
            for(int i=0; i<3; i++)
            {
                Rt_board_lidar[9+i] = p_center_board->xyz[i];
                for(int j=0; j<3; j++)
                    Rt_board_lidar[9+i] -=
                        Rt_board_lidar[i*3 + j] * plidar_mean.xyz[j];
            }

            double* Rt_lidar0_board__here = &Rt_lidar0_board[isnapshot*4*3];
            if(ilidar_first == 0)
                mrcal_invert_Rt(Rt_lidar0_board__here,
                                Rt_board_lidar);
            else
            {
                double Rt_lidar_board[4*3];
                mrcal_invert_Rt(Rt_lidar_board,
                                Rt_board_lidar);

                mrcal_compose_Rt(Rt_lidar0_board__here,
                                 &Rt_lidar0_lidar[(ilidar_first - 1)*4*3],
                                 Rt_lidar_board);
            }
        }
    }

    return true;
}

static
bool align_point_clouds(// out
                        double* Rt01,
                        // in
                        const uint16_t isensor1,
                        const uint16_t isensor0,

                        const sensor_snapshot_segmented_t* sensor_snapshots_filtered,
                        const int                          Nsensor_snapshots_filtered,
                        const int                          Ncameras,
                        const int                          Nlidars)
{
    // cameras
    if(isensor0 >= Nlidars) assert(0);
    if(isensor1 >= Nlidars) assert(0);

    const int ilidar0 = isensor0;
    const int ilidar1 = isensor1;


    // Nsensor_snapshots_filtered is the max number I will need
    // These are double instead of float, since that's what the alignment code
    // uses
    mrcal_point3_t normals0[Nsensor_snapshots_filtered];
    mrcal_point3_t normals1[Nsensor_snapshots_filtered];
    mrcal_point3_t points0 [Nsensor_snapshots_filtered];
    mrcal_point3_t points1 [Nsensor_snapshots_filtered];
    int Nbuffer = 0;

    // to pacify the compiler
    normals0[0].x = 0;
    normals1[0].x = 0;

    // Loop through all the observations
#warning "I ignore all the cameras here"
    for(int isnapshot=0; isnapshot < Nsensor_snapshots_filtered; isnapshot++)
    {
        const sensor_snapshot_segmented_t* sensor_snapshot = &sensor_snapshots_filtered[isnapshot];

        if(sensor_snapshot->lidar_scans[ilidar0].points == NULL)
            continue;
        if(sensor_snapshot->lidar_scans[ilidar1].points == NULL)
            continue;

        mrcal_point3_from_clc_point3f(&normals0[Nbuffer],
                                      &sensor_snapshot->lidar_scans[ilidar0].points_and_plane->plane.n);
        mrcal_point3_from_clc_point3f(&normals1[Nbuffer],
                                      &sensor_snapshot->lidar_scans[ilidar1].points_and_plane->plane.n);
        mrcal_point3_from_clc_point3f(&points0[Nbuffer],
                                      &sensor_snapshot->lidar_scans[ilidar0].points_and_plane->plane.p_mean);
        mrcal_point3_from_clc_point3f(&points1[Nbuffer],
                                      &sensor_snapshot->lidar_scans[ilidar1].points_and_plane->plane.p_mean);
        Nbuffer++;
    }

    // If I had lots of points, I'd do a procrustes fit, and I'd be done. But
    // I have few points, so I do this in two steps:
    // - I align the normals to get a high-confidence rotation
    // - I lock down this rotation, and find the best translation
    if(!mrcal_align_procrustes_vectors_R01(// out
                                           Rt01,
                                           // in
                                           Nbuffer,
                                           &normals0[0].x,
                                           &normals1[0].x,
                                           NULL))
        return false;

    // We computed a rotation. It should fit the data decently well. If it
    // doesn't, something is broken, and we should complain
    for(int i=0; i<Nbuffer; i++)
    {
        mrcal_point3_t normals0_validation;
        mrcal_rotate_point_R(normals0_validation.xyz, NULL, NULL,
                             Rt01,
                             normals1[i].xyz);

        const double cos_err = mrcal_point3_inner(normals0_validation, normals0[i]);

#warning unhardcode
        const double cos_threshold = cos(5.*M_PI/180.);
        if(cos_err < cos_threshold)
        {
            MSG("Inconsistent seed rotation: th=%.1f deg. Giving up",
                acos(cos_err) * 180./M_PI);
            return false;
        }
    }

    // Now the translation. R01 x1 + t01 ~ x0
    // -> t01 ~ x0 - R01 x1
    //
    // I now have an estimate for t01 for each observation. Ideally they should
    // be self-consistent, so I compute the mean:
    // -> t01 = mean(x0 - R01 x1)
    *(mrcal_point3_t*)(&Rt01[9]) = (mrcal_point3_t){};
    for(int i=0; i<Nbuffer; i++)
    {
        mrcal_point3_t R01_x1;
        mrcal_rotate_point_R(R01_x1.xyz, NULL, NULL,
                             Rt01,
                             points1[i].xyz);
        mrcal_point3_t t01 = mrcal_point3_sub(points0[i],R01_x1);
        for(int j=0; j<3; j++)
            Rt01[9 + j] += t01.xyz[j];
    }
    for(int j=0; j<3; j++)
        Rt01[9 + j] /= (double)Nbuffer;

    // We computed a seed translation as a mean of candidate translations. They
    // should have been self-consistent, and each one should be close to the
    // mean. If not, something is broken, and we should complain
    for(int i=0; i<Nbuffer; i++)
    {
        mrcal_point3_t R01_x1;
        mrcal_rotate_point_R(R01_x1.xyz, NULL, NULL,
                             Rt01,
                             points1[i].xyz);
        mrcal_point3_t t01 = mrcal_point3_sub(points0[i],R01_x1);

        mrcal_point3_t t01_err = mrcal_point3_sub(t01,
                                                  *(mrcal_point3_t*)(&Rt01[9]));
        double norm2_t01_err = mrcal_point3_norm2(t01_err);

#warning unhardcode
        if(norm2_t01_err > 0.5*0.5)
        {
            MSG("Inconsistent seed translation. Giving up");
            return false;
        }
    }

    return true;
}

typedef struct
{
    const sensor_snapshot_segmented_t* sensor_snapshots_filtered;
    const int                          Nsensor_snapshots_filtered;
    const int                          Ncameras;
    const int                          Nlidars;
    double*                            Rt_lidar0_lidar;
} cb_sensor_link_cookie_t;
static
bool cb_sensor_link(const uint16_t isensor1,
                    const uint16_t isensor0,
                    void* _cookie)
{
    if(isensor1 == 0)
        // This is the reference sensor. Nothing to do
        return true;

    cb_sensor_link_cookie_t* cookie = (cb_sensor_link_cookie_t*)_cookie;

    const sensor_snapshot_segmented_t* sensor_snapshots_filtered =
        cookie->sensor_snapshots_filtered;
    const int                          Nsensor_snapshots_filtered =
        cookie->Nsensor_snapshots_filtered;
    const int                          Ncameras =
        cookie->Ncameras;
    const int                          Nlidars =
        cookie->Nlidars;
    double*                            Rt_lidar0_lidar =
        cookie->Rt_lidar0_lidar;



    double Rt01[4*3];
    if(!align_point_clouds(// out
                           Rt01,
                           // in
                           isensor1,
                           isensor0,
                           sensor_snapshots_filtered,
                           Nsensor_snapshots_filtered,
                           Ncameras,
                           Nlidars))
        return false;

    if(isensor1 >= Nlidars)
    {
        assert(0);
#if 0
        icamera1 = idx_to - Nlidars;

        if(idx_from >= Nlidars)
        {
            icamera0 = idx_from - Nlidars;

            print(f"Estimating pose of camera {icamera1} from camera {icamera0}");
            if(not np.any(Rt_lidar0_camera[icamera0]))
                raise Exception(f"Computing pose of camera {icamera1} from camera {icamera0}, but the pose of camera {icamera0} is not initialized");
            Rt_lidar0_camera[icamera1] = mrcal.compose_Rt(Rt_lidar0_camera[icamera0],
                                                          Rt01);
        }
        else
        {
            ilidar0 = idx_from;

            print(f"Estimating pose of camera {icamera1} from lidar {ilidar0}");
            if(ilidar0 == 0)
                // from the reference
                Rt_lidar0_camera[icamera1] = Rt01;
            else
            {
                if(not np.any(Rt_lidar0_lidar[ilidar0-1])):
                    raise Exception(f"Computing pose of camera {icamera1} from lidar {ilidar0}, but the pose of lidar {ilidar0} is not initialized");
                Rt_lidar0_camera[icamera1] = mrcal.compose_Rt(Rt_lidar0_lidar[ilidar0-1],
                                                              Rt01);
            }
        }
#endif
    }
    else
    {
        int ilidar1 = isensor1;

        // ASSUMING SENSOR0 IS A LIDAR; THE CHECK IS IMMEDIATELY BELOW
        int ilidar0 = isensor0;

        // ilidar1 == 0 will not happen; checked above
        if(isensor0 >= Nlidars)
        {
            assert(0);
#if 0
            icamera0 = isensor0 - Nlidars;

            print(f"Estimating pose of lidar {ilidar1} from camera {icamera0}");
            if(not np.any(Rt_lidar0_camera[icamera0]))
                raise Exception(f"Computing pose of lidar {ilidar1} from camera {icamera0}, but the pose of camera {icamera0} is not initialized");
            Rt_lidar0_lidar[ilidar1-1] = mrcal.compose_Rt(Rt_lidar0_camera[icamera0],
                                                          Rt01);
#endif
        }

        MSG("Estimating pose of lidar %d from lidar %d",
            ilidar1,
            ilidar0);
        if(ilidar0 == 0)
            // from the reference
            memcpy(&Rt_lidar0_lidar[ (ilidar1-1)*4*3 ],
                   Rt01,
                   4*3*sizeof(double));
        else
            mrcal_compose_Rt(// out
                             &Rt_lidar0_lidar[ (ilidar1-1)*4*3 ],
                             // in
                             &Rt_lidar0_lidar[ (ilidar0-1)*4*3 ],
                             Rt01);
    }
    return true;
}

static bool Rt_is_all_zero(const double* Rt)
{
    for(int i=0; i<4*3; i++)
        if(Rt[i] != 0.0)
            return false;
    return true;
}

static bool
fit_seed(// out
         double* Rt_lidar0_board,  // Nsensor_snapshots_filtered poses ( (4,3) Rt arrays ) of these to fill
         double* Rt_lidar0_lidar,  // Nlidars-1 poses ( (4,3) Rt arrays ) of these to fill (lidar0 not included)
         double* Rt_lidar0_camera, // Ncameras  poses ( (4,3) Rt arrays ) of these to fill

         // in
         const sensor_snapshot_segmented_t* sensor_snapshots_filtered,
         const unsigned int                 Nsensor_snapshots_filtered,

         // These apply to ALL the sensor_snapshots[]
         const unsigned int Ncameras,
         const unsigned int Nlidars,

         // bits indicating whether a camera in
         // sensor_snapshots.images[] is color or not
         const clc_is_bgr_mask_t is_bgr_mask)
{
    mrcal_point3_t p_center_board = {};

    if(Ncameras > 0)
    {
        // The estimate of the center of the board, in board coords. This doesn't
        // need to be precise. If the board has an even number of corners, I just
        // take the nearest one

        assert(0);

        // Nh,Nw = p_board_local.shape[:2];
        // p_center_board = p_board_local[Nh/2,Nw/2,:];

    }
    else
    {
        // LIDAR-only solve. I don't have the board geometry and I don't know how
        // big the board is. But I eventually only look at distances to an
        // infinite plane (the LIDAR error metric), so it doesn't matter. I set
        // the board origin to 0

        // p_center_board is already 0, so there's nothing to do
    }



    const int Nsensors = Ncameras + Nlidars;
    uint16_t shared_observation_counts[pairwise_N(Nsensors)];
    connectivity_matrix(// out
                        shared_observation_counts,
                        // in
                        sensor_snapshots_filtered,
                        Nsensor_snapshots_filtered,

                        // These apply to ALL the sensor_snapshots[]
                        Ncameras,
                        Nlidars);

    MSG("Sensor shared-observations matrix for Nlidars=%d followed by Ncameras=%d:",
        Nlidars, Ncameras);
    print_full_symmetric_matrix_from_upper_triangle(shared_observation_counts,
                                                    Nsensors);

    cb_sensor_link_cookie_t cookie =
        {

            .sensor_snapshots_filtered  = sensor_snapshots_filtered,
            .Nsensor_snapshots_filtered = Nsensor_snapshots_filtered,
            .Ncameras                   = Ncameras,
            .Nlidars                    = Nlidars,
            .Rt_lidar0_lidar            = Rt_lidar0_lidar
        };


    if(!mrcal_traverse_sensor_links( Nsensors,
                                     shared_observation_counts,
                                     cb_sensor_link,
                                     &cookie))
    {
        MSG("mrcal_traverse_sensor_links() failed");
        return false;
    }

    // Traversed all the sensor links. It's possible that some sensors don't
    // even have transitive overlap to the root sensor (lidar0). I can detect
    // this by seeing an uninitialized Rt_lidar0_camera or Rt_lidar0_lidar.

    for(unsigned int i=0; i<Ncameras; i++)
        if( Rt_is_all_zero(&Rt_lidar0_camera[i*4*3]))
        {
            MSG("ERROR: Don't have complete observations overlap: camera %d not connected",
                i);
            return false;
        }
    for(unsigned int i=0; i<Nlidars-1; i++)
        if( Rt_is_all_zero(&Rt_lidar0_lidar[i*4*3]))
        {
            MSG("ERROR: Don't have complete observations overlap: lidar %d not connected",
                i+1);
            return false;
        }


    // All the sensor-sensor transforms computed. I compute the pose of the
    // boards
    if(!compute_board_poses(// out
                            Rt_lidar0_board,
                            // in
                            sensor_snapshots_filtered,
                            Nsensor_snapshots_filtered,
                            Ncameras,
                            Nlidars,
                            Rt_lidar0_lidar,
                            &p_center_board))
    {
        MSG("compute_board_poses() failed");
        return false;
    }

    // And now I confirm to make sure the seed is reasonable. At each instant in
    // time I make sure that all sensors are observing the board at roughly the
    // same location, orientation. It will be rough, since here we're checking a
    // seed solve
    bool validation_failed = false;

    for(unsigned int isnapshot=0; isnapshot < Nsensor_snapshots_filtered; isnapshot++)
    {
        const sensor_snapshot_segmented_t* sensor_snapshot = &sensor_snapshots_filtered[isnapshot];

        // camera
#if 0
        q_observed_all = joint_observations[isnapshot][0];
        if(any(q is not None for q in q_observed_all))
            raise Exception("Camera observations not supported here at this time; LIDAR only");
        // Looking at LIDAR only
#endif

        const mrcal_point3_t n_lidar0_should = {.x = Rt_lidar0_board[isnapshot*4*3 + 0*3 + 2],
                                                .y = Rt_lidar0_board[isnapshot*4*3 + 1*3 + 2],
                                                .z = Rt_lidar0_board[isnapshot*4*3 + 2*3 + 2]};

        const mrcal_point3_t* p0_lidar0_should = (mrcal_point3_t*)&Rt_lidar0_board[isnapshot*4*3 + 3*3 + 0];


        for(unsigned int ilidar=0; ilidar<Nlidars; ilidar++)
        {
            if(sensor_snapshot->lidar_scans[ilidar].points == NULL)
                continue;

            mrcal_point3_t n_lidar0_observed;
            mrcal_point3_t p0_lidar0_observed;

            mrcal_point3_from_clc_point3f(&n_lidar0_observed,
                                          &sensor_snapshot->lidar_scans[ilidar].points_and_plane->plane.n);
            mrcal_point3_from_clc_point3f(&p0_lidar0_observed,
                                          &sensor_snapshot->lidar_scans[ilidar].points_and_plane->plane.p_mean);
            if(ilidar != 0)
            {
                mrcal_rotate_point_R(n_lidar0_observed.xyz, NULL, NULL,
                                     &Rt_lidar0_lidar[(ilidar-1)*4*3],
                                     n_lidar0_observed.xyz);
                mrcal_transform_point_Rt(p0_lidar0_observed.xyz, NULL, NULL,
                                         &Rt_lidar0_lidar[(ilidar-1)*4*3],
                                         p0_lidar0_observed.xyz);
            }

            double cos_err = mrcal_point3_inner(n_lidar0_observed,n_lidar0_should);
            if(cos_err < -1.0) cos_err = -1.0;
            if(cos_err >  1.0) cos_err =  1.0;

            const double th_err_deg = acos(cos_err) * 180. / M_PI;
            const mrcal_point3_t p0_err = mrcal_point3_sub(*p0_lidar0_should, p0_lidar0_observed);
            const double p0_err_mag = mrcal_point3_mag(p0_err);

#warning "unhardcode"
            bool validation_failed_here = p0_err_mag > 0.5 || th_err_deg > 10.;

            if(validation_failed_here) validation_failed = true;

            MSG("%sisnapshot=%d ilidar=%d p0_err_mag=%.2f th_err_deg=%.2f",
                validation_failed_here ? "FAILED: " : "",
                isnapshot, ilidar, p0_err_mag, th_err_deg);
        }
    }

    return !validation_failed;
}


typedef struct
{
    unsigned int Ncameras;
    unsigned int Nlidars;

    const sensor_snapshot_segmented_t* snapshots;
    unsigned int Nsnapshots;

    // simplified computation for seeding
    bool use_distance_to_plane : 1;

    // extra diagnostics
    bool report_imeas          : 1;
} callback_context_t;


static int state_index_lidar(const int ilidar,
                             const callback_context_t* ctx)
{
    if(ilidar <= 0) return -1; // lidar0 is at the reference, and has no explicit pose
    return
        0 +
        (ilidar-1) * 6;
}
static int num_states_lidars(const callback_context_t* ctx)
{
    return (ctx->Nlidars-1)*6;
}
static int state_index_camera(const int icamera,
                              const callback_context_t* ctx)
{
    return
        num_states_lidars(ctx) +
        icamera * 6;
}
static int num_states_cameras(const callback_context_t* ctx)
{
    return ctx->Ncameras*6;
}
static int state_index_board(const int isnapshot,
                             const callback_context_t* ctx)
{
    return
        num_states_lidars(ctx) +
        num_states_cameras(ctx) +
        isnapshot * 6;
}
static int num_states_boards(const callback_context_t* ctx)
{
    return ctx->Nsnapshots*6;
}

static int num_states(const callback_context_t* ctx)
{
    return
        num_states_lidars(ctx) +
        num_states_cameras(ctx) +
        num_states_boards(ctx);
}

static int measurement_index_camera(const unsigned int isnapshot,
                                    const unsigned int icamera,
                                    const callback_context_t* ctx)
{
    return -1;
}

static int measurement_index_lidar(const unsigned int isnapshot,
                                   const unsigned int ilidar,
                                   const callback_context_t* ctx)
{
    int imeas = 0;

    for(unsigned int _isnapshot=0; _isnapshot < ctx->Nsnapshots; _isnapshot++)
    {
        const sensor_snapshot_segmented_t* sensor_snapshot = &ctx->snapshots[_isnapshot];
        for(unsigned int _ilidar=0; _ilidar<ctx->Nlidars; _ilidar++)
        {
            if(isnapshot == _isnapshot && ilidar == _ilidar)
                return imeas;

            const points_and_plane_full_t* points_and_plane_full = &sensor_snapshot->lidar_scans[_ilidar];

            if(points_and_plane_full->points == NULL)
                continue;

            const clc_ipoint_set_t* set =
                &points_and_plane_full->points_and_plane->ipoint_set;
            imeas += set->n;
        }
    }
    return -1;
}

static int num_measurements_lidars(const callback_context_t* ctx)
{
    int Nmeasurements = 0;

    for(unsigned int isnapshot=0; isnapshot < ctx->Nsnapshots; isnapshot++)
    {
        const sensor_snapshot_segmented_t* sensor_snapshot = &ctx->snapshots[isnapshot];
        for(unsigned int ilidar=0; ilidar<ctx->Nlidars; ilidar++)
        {
            const points_and_plane_full_t* points_and_plane_full = &sensor_snapshot->lidar_scans[ilidar];

            if(points_and_plane_full->points == NULL)
                continue;

            const clc_ipoint_set_t* set =
                &points_and_plane_full->points_and_plane->ipoint_set;
            Nmeasurements += set->n;
        }
    }
    return Nmeasurements;
}
static int num_measurements_cameras(const callback_context_t* ctx)
{
    return 0;
}
static int measurement_index_regularization(const callback_context_t* ctx)
{
    return
        num_measurements_lidars(ctx) +
        num_measurements_cameras(ctx);
}

static int num_measurements_regularization(const callback_context_t* ctx)
{
    return 6*ctx->Nsnapshots;
}
static int num_measurements(const callback_context_t* ctx)
{
    return
        num_measurements_lidars(ctx) +
        num_measurements_cameras(ctx) +
        num_measurements_regularization(ctx);
}
static int num_j_nonzero(const callback_context_t* ctx)
{
    int nnz = 0;

    for(unsigned int isnapshot=0; isnapshot < ctx->Nsnapshots; isnapshot++)
    {
        const sensor_snapshot_segmented_t* sensor_snapshot = &ctx->snapshots[isnapshot];

        for(unsigned int ilidar=0; ilidar<ctx->Nlidars; ilidar++)
        {
            const points_and_plane_full_t* points_and_plane_full = &sensor_snapshot->lidar_scans[ilidar];

            if(points_and_plane_full->points == NULL)
                continue;

            const clc_ipoint_set_t* set =
                &points_and_plane_full->points_and_plane->ipoint_set;

            if(ilidar == 0) nnz +=   6*set->n;
            else            nnz += 2*6*set->n;
        }
    }

    // regularization
    nnz += ctx->Nsnapshots * 6;

    return nnz;
}

// From mrcal/scales.h. These work well for mrcal, and should work well here as
// well
#define SCALE_ROTATION_CAMERA         (0.1 * M_PI/180.0)
#define SCALE_TRANSLATION_CAMERA      1.0
#define SCALE_ROTATION_FRAME          (15.0 * M_PI/180.0)
#define SCALE_TRANSLATION_FRAME       1.0

#define SCALE_ROTATION_LIDAR    SCALE_ROTATION_CAMERA
#define SCALE_TRANSLATION_LIDAR SCALE_TRANSLATION_CAMERA

#define SCALE_MEASUREMENT_PX               0.15   /* expected noise levels */
#define SCALE_MEASUREMENT_M                0.015  /* expected noise levels */
#define SCALE_MEASUREMENT_REGULARIZATION_r 100.   /* rad */
#define SCALE_MEASUREMENT_REGULARIZATION_t 10000. /* meters */

// The reference coordinate system is defined by the coord system of the
// first lidar
static void
pack_solver_state(// out
                  double* b,
                  // in
                  // shape (Nlidars-1, 6)
                  const double* rt_lidar0_lidar,
                  // shape (Ncameras, 6)
                  const double* rt_lidar0_camera,
                  // shape (Nsnapshots, 6)
                  const double* rt_lidar0_board,

                  const callback_context_t* ctx)
{
    int istate = 0;
    for(unsigned int i=0; i<ctx->Nlidars-1; i++)
    {
        for(int j=0; j<3; j++) b[istate++] = rt_lidar0_lidar[i*6 + j] / SCALE_ROTATION_LIDAR;
        for(int j=3; j<6; j++) b[istate++] = rt_lidar0_lidar[i*6 + j] / SCALE_TRANSLATION_LIDAR;
    }
#if 0
    for(unsigned int i=0; i<ctx->Ncameras; i++)
    {
        for(int j=0; j<3; j++) b[istate++] = rt_lidar0_camera[i*6 + j] / SCALE_ROTATION_CAMERA;
        for(int j=3; j<6; j++) b[istate++] = rt_lidar0_camera[i*6 + j] / SCALE_TRANSLATION_CAMERA;
    }
#endif
    for(unsigned int i=0; i<ctx->Nsnapshots; i++)
    {
        for(int j=0; j<3; j++) b[istate++] = rt_lidar0_board[i*6 + j] / SCALE_ROTATION_FRAME;
        for(int j=3; j<6; j++) b[istate++] = rt_lidar0_board[i*6 + j] / SCALE_TRANSLATION_FRAME;
    }
}

static void
unpack_solver_state(// out
                    // shape (Nlidars-1, 6)
                    double* rt_lidar0_lidar,
                    // shape (Ncameras, 6)
                    double* rt_lidar0_camera,
                    // shape (Nsnapshots, 6)
                    double* rt_lidar0_board,
                    // in
                    const double* b,

                    const callback_context_t* ctx)
{
    int istate = 0;
    for(unsigned int i=0; i<ctx->Nlidars-1; i++)
    {
        for(int j=0; j<3; j++) rt_lidar0_lidar[i*6 + j] = b[istate++] * SCALE_ROTATION_LIDAR;
        for(int j=3; j<6; j++) rt_lidar0_lidar[i*6 + j] = b[istate++] * SCALE_TRANSLATION_LIDAR;
    }
#if 0
    for(unsigned int i=0; i<ctx->Ncameras; i++)
    {
        for(int j=0; j<3; j++) rt_lidar0_camera[i*6 + j] = b[istate++] * SCALE_ROTATION_CAMERA;
        for(int j=3; j<6; j++) rt_lidar0_camera[i*6 + j] = b[istate++] * SCALE_TRANSLATION_CAMERA;
    }
#endif
    for(unsigned int i=0; i<ctx->Nsnapshots; i++)
    {
        for(int j=0; j<3; j++) rt_lidar0_board[i*6 + j] = b[istate++] * SCALE_ROTATION_FRAME;
        for(int j=3; j<6; j++) rt_lidar0_board[i*6 + j] = b[istate++] * SCALE_TRANSLATION_FRAME;
    }
}


static double x_dot_nlidar0(// out
                            mrcal_point3_t* dy_dr,
                            // in
                            const mrcal_point3_t* x,
                            const mrcal_point3_t* nlidar0,
                            const double*         dnlidar0__dr_lidar0_board)
{
    // y = inner(nlidar0,x)
    // dy/dr = transpose(x) dnlidar0/dr
    *dy_dr =
        (mrcal_point3_t)
        { .x = x->xyz[0] * dnlidar0__dr_lidar0_board[0] +
               x->xyz[1] * dnlidar0__dr_lidar0_board[3] +
               x->xyz[2] * dnlidar0__dr_lidar0_board[6],
          .y = x->xyz[0] * dnlidar0__dr_lidar0_board[1] +
               x->xyz[1] * dnlidar0__dr_lidar0_board[4] +
               x->xyz[2] * dnlidar0__dr_lidar0_board[7],
          .z = x->xyz[0] * dnlidar0__dr_lidar0_board[2] +
               x->xyz[1] * dnlidar0__dr_lidar0_board[5] +
               x->xyz[2] * dnlidar0__dr_lidar0_board[8] };
    return mrcal_point3_inner(*nlidar0, *x);
}

static void cost(const double*   b,
                 double*         x,
                 cholmod_sparse* Jt,
                 const callback_context_t* ctx)
{
    int*    Jrowptr = Jt ? (int*)   Jt->p : NULL;
    int*    Jcolidx = Jt ? (int*)   Jt->i : NULL;
    double* Jval    = Jt ? (double*)Jt->x : NULL;

    int iJacobian    = 0;
    int iMeasurement = 0;


#define STORE_JACOBIAN(col, g)                  \
    do                                          \
    {                                           \
        if(Jt) {                                \
            Jcolidx[ iJacobian ] = col;         \
            Jval   [ iJacobian ] = g;           \
        }                                       \
        iJacobian++;                            \
    } while(0)



    double rt_lidar0_lidar_all [(ctx->Nlidars-1) *6];
    double rt_lidar0_camera_all[ctx->Ncameras    *6];
    double rt_lidar0_board_all [ctx->Nsnapshots  *6];

    unpack_solver_state(rt_lidar0_lidar_all,
                        rt_lidar0_camera_all,
                        rt_lidar0_board_all,
                        b,
                        ctx);

    for(unsigned int isnapshot=0; isnapshot < ctx->Nsnapshots; isnapshot++)
    {
        const sensor_snapshot_segmented_t* sensor_snapshot = &ctx->snapshots[isnapshot];

        double* rt_lidar0_board = &rt_lidar0_board_all[isnapshot*6];

        mrcal_point3_t z = {.z = 1.};
        mrcal_point3_t nlidar0;
        double dnlidar0__dr_lidar0_board[3*3];
        mrcal_rotate_point_r(nlidar0.xyz, dnlidar0__dr_lidar0_board, NULL,
                             &rt_lidar0_board[0], z.xyz);

        mrcal_point3_t dd1__dr_lidar0_board;
        mrcal_point3_t* t = (mrcal_point3_t*)(&rt_lidar0_board[3]);
        double d1 = x_dot_nlidar0(// out
                                  &dd1__dr_lidar0_board,
                                  // in
                                  t, &nlidar0, dnlidar0__dr_lidar0_board);
        mrcal_point3_t* dd1__dt_lidar0_board = &nlidar0;

        if(ctx->use_distance_to_plane)
        {
            // Simplified error: look at perpendicular distance off the
            // plane
            //
            // The pose of the board is Rt_lidar0_board. The board is z=0 in the
            // board coords so the normal to the plane is nlidar0 =
            // R_lidar0_board[:,2]. I define the board as
            //
            //   all x where d = inner(nlidar0,xlidar0) = inner(nlidar0, Rt_lidar0_board xy0)
            //
            // So
            //
            //   d1 = inner(nlidar0, R_lidar0_board xy0 + t_lidar0_board) =
            //      = inner(nlidar0, R_lidar0_board[:,0] x + R_lidar0_board[:,1] y + t_lidar0_board)
            //      = inner(nlidar0, t_lidar0_board)
            //
            // For any lidar-observed point p I can compute its distance to the board:
            //
            //   d2 = inner(nlidar0, Rt_lidar0_lidar p)
            //      = inner(nlidar0, R_lidar0_lidar p + t_lidar0_lidar)
            //      = inner(nlidar0, v)
            //
            // where v = R_lidar0_lidar p + t_lidar0_lidar
            // So
            //
            //   err = d1 - d2 =
            //       = inner(nlidar0, t_lidar0_board - v)
            for(unsigned int ilidar=0; ilidar<ctx->Nlidars; ilidar++)
            {
                const points_and_plane_full_t* points_and_plane_full = &sensor_snapshot->lidar_scans[ilidar];

                if(points_and_plane_full->points == NULL)
                    continue;

                if(ctx->report_imeas)
                    MSG("isnapshot=%d ilidar=%d iMeasurement=%d",
                        isnapshot, ilidar, iMeasurement);

                const clc_ipoint_set_t* set =
                    &points_and_plane_full->points_and_plane->ipoint_set;
                if(ilidar == 0)
                {
                    // this is lidar0; it sits at the reference, and doesn't
                    // have an explicit pose or a rt_lidar0_lidar gradient
                    for(unsigned int iipoint=0; iipoint<set->n; iipoint++)
                    {
                        mrcal_point3_t p;
                        int ipoint = set->ipoint[iipoint];
                        mrcal_point3_from_clc_point3f(&p,
                                                      &points_and_plane_full->points[ipoint]);


                        #warning "mrcal_transform_point_rt() for the same rt in a loop will be faster if I compute Rt first"

                        mrcal_point3_t* v = &p;

                        mrcal_point3_t dd2__dr_lidar0_board;
                        double d2 = x_dot_nlidar0(// out
                                                  &dd2__dr_lidar0_board,
                                                  // in
                                                  v, &nlidar0, dnlidar0__dr_lidar0_board);

                        if(Jt) Jrowptr[iMeasurement] = iJacobian;
                        x[iMeasurement] = (d1 - d2) / SCALE_MEASUREMENT_M;

                        int ivar;

                        ivar = state_index_board(isnapshot, ctx);
                        for(int i=0; i<3; i++)
                            STORE_JACOBIAN(ivar + i,
                                           (dd1__dr_lidar0_board .xyz[i] - dd2__dr_lidar0_board .xyz[i])
                                           * SCALE_ROTATION_FRAME/SCALE_MEASUREMENT_M);
                        for(int i=0; i<3; i++)
                            STORE_JACOBIAN(ivar + i + 3,
                                           (dd1__dt_lidar0_board->xyz[i])
                                           * SCALE_TRANSLATION_FRAME/SCALE_MEASUREMENT_M);
                        iMeasurement++;
                    }
                }
                else
                {
                    double* rt_lidar0_lidar = &rt_lidar0_lidar_all[6*(ilidar-1)];

                    for(unsigned int iipoint=0; iipoint<set->n; iipoint++)
                    {
                        mrcal_point3_t p;
                        int ipoint = set->ipoint[iipoint];
                        mrcal_point3_from_clc_point3f(&p,
                                                      &points_and_plane_full->points[ipoint]);


                        #warning "mrcal_transform_point_rt() for the same rt in a loop will be faster if I compute Rt first"
                        mrcal_point3_t v;
                        double dv__dr_lidar0_lidar[3*3];
                        mrcal_rotate_point_r(v.xyz, dv__dr_lidar0_lidar, NULL,
                                             rt_lidar0_lidar, p.xyz);
                        for(int i=0; i<3; i++)
                            v.xyz[i] += rt_lidar0_lidar[3+i];

                        mrcal_point3_t dd2__dr_lidar0_board;
                        double d2 = x_dot_nlidar0(// out
                                                  &dd2__dr_lidar0_board,
                                                  // in
                                                  &v, &nlidar0, dnlidar0__dr_lidar0_board);

                        mrcal_point3_t dd2__dr_lidar0_lidar =
                            { .x = nlidar0.xyz[0] * dv__dr_lidar0_lidar[0] +
                                   nlidar0.xyz[1] * dv__dr_lidar0_lidar[3] +
                                   nlidar0.xyz[2] * dv__dr_lidar0_lidar[6],
                              .y = nlidar0.xyz[0] * dv__dr_lidar0_lidar[1] +
                                   nlidar0.xyz[1] * dv__dr_lidar0_lidar[4] +
                                   nlidar0.xyz[2] * dv__dr_lidar0_lidar[7],
                              .z = nlidar0.xyz[0] * dv__dr_lidar0_lidar[2] +
                                   nlidar0.xyz[1] * dv__dr_lidar0_lidar[5] +
                                   nlidar0.xyz[2] * dv__dr_lidar0_lidar[8] };

                        mrcal_point3_t* dd2__dt_lidar0_lidar = &nlidar0;

                        if(Jt) Jrowptr[iMeasurement] = iJacobian;
                        x[iMeasurement] = (d1 - d2) / SCALE_MEASUREMENT_M;

                        int ivar;

                        ivar = state_index_board(isnapshot, ctx);
                        for(int i=0; i<3; i++)
                            STORE_JACOBIAN(ivar + i,
                                           (dd1__dr_lidar0_board .xyz[i] - dd2__dr_lidar0_board .xyz[i])
                                           * SCALE_ROTATION_FRAME/SCALE_MEASUREMENT_M);
                        for(int i=0; i<3; i++)
                            STORE_JACOBIAN(ivar + i + 3,
                                           (dd1__dt_lidar0_board->xyz[i])
                                           * SCALE_TRANSLATION_FRAME/SCALE_MEASUREMENT_M);

                        ivar = state_index_lidar(ilidar, ctx);
                        for(int i=0; i<3; i++)
                            STORE_JACOBIAN(ivar + i,
                                           (-dd2__dr_lidar0_lidar .xyz[i])
                                           * SCALE_ROTATION_LIDAR/SCALE_MEASUREMENT_M);
                        for(int i=0; i<3; i++)
                            STORE_JACOBIAN(ivar + i + 3,
                                           (-dd2__dt_lidar0_lidar->xyz[i])
                                           * SCALE_TRANSLATION_LIDAR/SCALE_MEASUREMENT_M);
                        iMeasurement++;
                    }
                }
            }
        }
        else
        {
            // More complex, but more realistic error
            //
            // A plane is zboard = 0
            // A lidar point plidar = vlidar dlidar
            //
            // pboard = Rbl plidar + tbl
            //        = T_b_l0 T_l0_l plidar
            // 0 = zboard = pboard[2] = inner(Rbl[2,:],plidar) + tbl[2]
            // -> inner(Rbl[2,:],vlidar)*dlidar = -tbl[2]
            // -> dlidar = -tbl[2] / inner(Rbl[2,:],vlidar)
            //           = -tbl[2] / (inner(Rbl[2,:],plidar) / mag(plidar))
            //           = -tbl[2] mag(plidar) / inner(Rbl[2,:],plidar)
            //
            // And the error is
            //
            //   err = dlidar_observed - dlidar
            //       = mag(plidar) - dlidar
            //       = mag(plidar) + tbl[2] mag(plidar) / inner(Rbl[2,:],plidar)
            //       = mag(plidar) * (1 + tbl[2] / inner(Rbl[2,:],plidar) )
            //
            // Rbl[2,:] = Rlb[:,2] = R_lidar_board z = R_lidar_lidar0 nlidar0
            //
            // tbl[2]   = (R_board_lidar0 t_lidar0_lidar + t_board_lidar0)[2]
            //          = R_board_lidar0[2,:] t_lidar0_lidar + t_board_lidar0[2]
            //          = R_lidar0_board[:,2] t_lidar0_lidar + t_board_lidar0[2]
            //          = inner(nlidar0,t_lidar0_lidar) + t_board_lidar0[2]
            //
            // R_lidar0_board pb + t_lidar0_board = pl0
            // -> pb = R_board_lidar0 pl0 - R_board_lidar0 t_lidar0_board
            // -> t_board_lidar0 = - R_board_lidar0 t_lidar0_board
            // -> t_board_lidar0[2] = - R_board_lidar0[2,:] t_lidar0_board
            //                      = - R_lidar0_board[:,2] t_lidar0_board
            //                      = - inner(nlidar0, t_lidar0_board)
            //                      = -d1 (the same d1 as in the crude solve above)
            for(unsigned int ilidar=0; ilidar<ctx->Nlidars; ilidar++)
            {
                const points_and_plane_full_t* points_and_plane_full = &sensor_snapshot->lidar_scans[ilidar];

                if(points_and_plane_full->points == NULL)
                    continue;

                if(ctx->report_imeas)
                    MSG("isnapshot=%d ilidar=%d iMeasurement=%d",
                        isnapshot, ilidar, iMeasurement);

                const clc_ipoint_set_t* set =
                    &points_and_plane_full->points_and_plane->ipoint_set;
                if(ilidar == 0)
                {
                    // this is lidar0; it sits at the reference, and doesn't
                    // have an explicit pose or a rt_lidar0_lidar gradient
                    for(unsigned int iipoint=0; iipoint<set->n; iipoint++)
                    {
                        mrcal_point3_t p;
                        int ipoint = set->ipoint[iipoint];
                        mrcal_point3_from_clc_point3f(&p,
                                                      &points_and_plane_full->points[ipoint]);

                        double magp = mrcal_point3_mag(p);

                        mrcal_point3_t* Rbl2 = &nlidar0;

                        double neg_tbl2 = d1;
                        mrcal_point3_t* extra_neg_dtbl2__dt_lidar0_board =  dd1__dt_lidar0_board;
                        mrcal_point3_t* extra_neg_dtbl2__dr_lidar0_board = &dd1__dr_lidar0_board;

                        double inv_Rbl2_p = 1. / mrcal_point3_inner(*Rbl2, p);
                        double dinv_Rbl2_p__dRbl2[3] = { -p.x * inv_Rbl2_p*inv_Rbl2_p,
                                                         -p.y * inv_Rbl2_p*inv_Rbl2_p,
                                                         -p.z * inv_Rbl2_p*inv_Rbl2_p };

                        if(Jt) Jrowptr[iMeasurement] = iJacobian;
                        x[iMeasurement] =
                            magp * (1. - neg_tbl2 * inv_Rbl2_p )
                            / SCALE_MEASUREMENT_M;

                        // Gradients:
                        //   Rbl2 has these gradients:
                        //     dRbl2__dnlidar0 = identity
                        //   nlidar0 has these gradients:
                        //     dnlidar0__dr_lidar0_board
                        //   tbl2 has these gradients:
                        //     extra_neg_dtbl2__dt_lidar0_board
                        //     extra_neg_dtbl2__dr_lidar0_board


                        int ivar;

                        ivar = state_index_board(isnapshot, ctx);
                        //////////// dr_lidar0_board
                        //   d(tbl2 * inv_Rbl2_p)
                        // = dtbl2 * inv_Rbl2_p + tbl2 * dinv_Rbl2_p
                        // = (- extra_neg_dtbl2__dr_lidar0_board) inv_Rbl2_p +
                        //    tbl2 dinv_Rbl2_p__dRbl2 dnlidar0__dr_lidar0_board
                        double *M = dnlidar0__dr_lidar0_board;
                        for(int i=0; i<3; i++)
                        {
                            STORE_JACOBIAN(ivar + i,
                                           ( (- extra_neg_dtbl2__dr_lidar0_board->xyz[i]) * inv_Rbl2_p
                                             - neg_tbl2 *
                                             ( dinv_Rbl2_p__dRbl2[0] * M[0*3 + i] +
                                               dinv_Rbl2_p__dRbl2[1] * M[1*3 + i] +
                                               dinv_Rbl2_p__dRbl2[2] * M[2*3 + i] )
                                           )
                                           * SCALE_ROTATION_FRAME/SCALE_MEASUREMENT_M * magp);
                        }
                        //////////// dt_lidar0_board
                        //   d(tbl2 * inv_Rbl2_p)
                        // = dtbl2 * inv_Rbl2_p + tbl2 * dinv_Rbl2_p
                        // = - extra_neg_dtbl2__dt_lidar0_board inv_Rbl2_p
                        for(int i=0; i<3; i++)
                            STORE_JACOBIAN(ivar + i + 3,
                                           - extra_neg_dtbl2__dt_lidar0_board->xyz[i] * inv_Rbl2_p
                                           * SCALE_TRANSLATION_FRAME/SCALE_MEASUREMENT_M * magp);

                        iMeasurement++;
                    }
                }
                else
                {
                    double* rt_lidar0_lidar = &rt_lidar0_lidar_all[6*(ilidar-1)];

                    for(unsigned int iipoint=0; iipoint<set->n; iipoint++)
                    {
                        mrcal_point3_t p;
                        int ipoint = set->ipoint[iipoint];
                        mrcal_point3_from_clc_point3f(&p,
                                                      &points_and_plane_full->points[ipoint]);

                        double magp = mrcal_point3_mag(p);

                        mrcal_point3_t Rbl2;
                        double dRbl2__dr_lidar0_lidar[3*3];
                        double dRbl2__dnlidar0[3*3];
                        mrcal_rotate_point_r_inverted(Rbl2.xyz, dRbl2__dr_lidar0_lidar, dRbl2__dnlidar0,
                                                      rt_lidar0_lidar, nlidar0.xyz);

                        mrcal_point3_t dtbl2__dr_lidar0_board;
                        double tbl2 = x_dot_nlidar0(// out
                                                    &dtbl2__dr_lidar0_board,
                                                    // in
                                                    (mrcal_point3_t*)&rt_lidar0_lidar[3], &nlidar0, dnlidar0__dr_lidar0_board);
                        mrcal_point3_t* dtbl2__dt_lidar0_lidar = &nlidar0;

                        tbl2 -= d1;
                        mrcal_point3_t* extra_neg_dtbl2__dt_lidar0_board =  dd1__dt_lidar0_board;
                        mrcal_point3_t* extra_neg_dtbl2__dr_lidar0_board = &dd1__dr_lidar0_board;

                        double inv_Rbl2_p = 1. / mrcal_point3_inner(Rbl2, p);
                        double dinv_Rbl2_p__dRbl2[3] = { -p.x * inv_Rbl2_p*inv_Rbl2_p,
                                                         -p.y * inv_Rbl2_p*inv_Rbl2_p,
                                                         -p.z * inv_Rbl2_p*inv_Rbl2_p };


                        if(Jt) Jrowptr[iMeasurement] = iJacobian;
                        x[iMeasurement] =
                            magp * (1. + tbl2 * inv_Rbl2_p )
                            / SCALE_MEASUREMENT_M;

                        // Gradients:
                        //   Rbl2 has these gradients:
                        //     dRbl2__dr_lidar0_lidar
                        //     dRbl2__dnlidar0
                        //   nlidar0 has these gradients:
                        //     dnlidar0__dr_lidar0_board
                        //   tbl2 has these gradients:
                        //     dtbl2__dr_lidar0_board
                        //     dtbl2__dt_lidar0_lidar
                        //     extra_neg_dtbl2__dt_lidar0_board
                        //     extra_neg_dtbl2__dr_lidar0_board


                        int ivar;

                        ivar = state_index_board(isnapshot, ctx);
                        //////////// dr_lidar0_board
                        //   d(tbl2 * inv_Rbl2_p)
                        // = dtbl2 * inv_Rbl2_p + tbl2 * dinv_Rbl2_p
                        // = (dtbl2__dr_lidar0_board - extra_neg_dtbl2__dr_lidar0_board) inv_Rbl2_p +
                        //    tbl2 dinv_Rbl2_p__dRbl2 dRbl2__dnlidar0 dnlidar0__dr_lidar0_board
                        // = (dtbl2__dr_lidar0_board - extra_neg_dtbl2__dr_lidar0_board) inv_Rbl2_p +
                        //    tbl2 dinv_Rbl2_p__dRbl2 M
                        double M[3*3];
                        mul_genN3_gen33_vout(3, dRbl2__dnlidar0, dnlidar0__dr_lidar0_board, M);
                        for(int i=0; i<3; i++)
                        {
                            STORE_JACOBIAN(ivar + i,
                                           ( (dtbl2__dr_lidar0_board.xyz[i] - extra_neg_dtbl2__dr_lidar0_board->xyz[i]) * inv_Rbl2_p
                                             + tbl2 *
                                             ( dinv_Rbl2_p__dRbl2[0] * M[0*3 + i] +
                                               dinv_Rbl2_p__dRbl2[1] * M[1*3 + i] +
                                               dinv_Rbl2_p__dRbl2[2] * M[2*3 + i] )
                                           )
                                           * SCALE_ROTATION_FRAME/SCALE_MEASUREMENT_M * magp);
                        }
                        //////////// dt_lidar0_board
                        //   d(tbl2 * inv_Rbl2_p)
                        // = dtbl2 * inv_Rbl2_p + tbl2 * dinv_Rbl2_p
                        // = - extra_neg_dtbl2__dt_lidar0_board inv_Rbl2_p
                        for(int i=0; i<3; i++)
                            STORE_JACOBIAN(ivar + i + 3,
                                           - extra_neg_dtbl2__dt_lidar0_board->xyz[i] * inv_Rbl2_p
                                           * SCALE_TRANSLATION_FRAME/SCALE_MEASUREMENT_M * magp);

                        ivar = state_index_lidar(ilidar, ctx);
                        //////////// dr_lidar0_lidar
                        //   d(tbl2 * inv_Rbl2_p)
                        // = dtbl2 * inv_Rbl2_p + tbl2 * dinv_Rbl2_p
                        // = tbl2 dinv_Rbl2_p__dRbl2 dRbl2__dr_lidar0_lidar
                        for(int i=0; i<3; i++)
                            STORE_JACOBIAN(ivar + i,
                                           tbl2 *
                                           ( dinv_Rbl2_p__dRbl2[0] * dRbl2__dr_lidar0_lidar[0*3 + i] +
                                             dinv_Rbl2_p__dRbl2[1] * dRbl2__dr_lidar0_lidar[1*3 + i] +
                                             dinv_Rbl2_p__dRbl2[2] * dRbl2__dr_lidar0_lidar[2*3 + i] )
                                           * SCALE_ROTATION_LIDAR/SCALE_MEASUREMENT_M * magp);
                        //////////// dt_lidar0_lidar
                        //   d(tbl2 * inv_Rbl2_p)
                        // = dtbl2 * inv_Rbl2_p + tbl2 * dinv_Rbl2_p
                        // = dtbl2__dt_lidar0_lidar inv_Rbl2_p
                        for(int i=0; i<3; i++)
                            STORE_JACOBIAN(ivar + i + 3,
                                           dtbl2__dt_lidar0_lidar->xyz[i] * inv_Rbl2_p
                                           * SCALE_TRANSLATION_LIDAR/SCALE_MEASUREMENT_M * magp);
                        iMeasurement++;
                    }
                }
            }
        }
    }

    // camera stuff
#if 0
    for(iboard in range(len(joint_observations)))
    {
        q_observed_all = joint_observations[iboard][0];
        for(icamera in range(len(q_observed_all)))
        {
            q_observed = q_observed_all[icamera];

            if(q_observed is None)
                     continue;

            if(report_imeas)
                     print(f"{iboard=} {icamera=} {iMeasurement=}");

            rt_ref_board  = state['rt_ref_board'] [iboard];
            rt_camera_ref = state['rt_camera_ref'][icamera];

            Rt_ref_board  = mrcal.Rt_from_rt(rt_ref_board);
            Rt_camera_ref = mrcal.Rt_from_rt(rt_camera_ref);

            Rt_camera_board = mrcal.compose_Rt( Rt_camera_ref,
                                                Rt_ref_board );
            q = mrcal.project( mrcal.transform_point_Rt(Rt_camera_board,
                                                        p_board_local),
                               *models[icamera].intrinsics() );
            q = nps.clump(q,n=2);
            x[iMeasurement:iMeasurement+Nmeas_camera_observation] =
                (q - q_observed).ravel() / SCALE_MEASUREMENT_PX;

            iMeasurement += Nmeas_camera_observation;
        }
    }
#endif


    /////// Regularization
    // The regularization lightly pulls every element of rt_ref_board towards
    // zero. This is necessary because LIDAR-only observations of the board have
    // only 3 DOF: the board is free to translate and yaw in its plane. I can
    // accomplish the same thing with a different T_ref_board representation for
    // LIDAR-only observations: (n,d). That disparate representation would take
    // more typing, so I don't do that just yet
    int ivar = state_index_board(0, ctx);
    for(unsigned int isnapshot=0; isnapshot < ctx->Nsnapshots; isnapshot++)
    {
        double* rt_lidar0_board = &rt_lidar0_board_all[isnapshot*6];

        for(int i=0; i<3; i++)
        {
            if(Jt) Jrowptr[iMeasurement] = iJacobian;
            x[iMeasurement] = rt_lidar0_board[i] / SCALE_MEASUREMENT_REGULARIZATION_r;
            STORE_JACOBIAN(ivar,
                           SCALE_ROTATION_FRAME/SCALE_MEASUREMENT_REGULARIZATION_r);
            ivar++;
            iMeasurement++;
        }
        for(int i=0; i<3; i++)
        {
            if(Jt) Jrowptr[iMeasurement] = iJacobian;
            x[iMeasurement] = rt_lidar0_board[i+3] / SCALE_MEASUREMENT_REGULARIZATION_t;
            STORE_JACOBIAN(ivar,
                           SCALE_TRANSLATION_FRAME/SCALE_MEASUREMENT_REGULARIZATION_t);
            ivar++;
            iMeasurement++;
        }
    }

    const int Nmeasurements = num_measurements(ctx);
    if(iMeasurement != Nmeasurements)
    {
        MSG("cost() wrote an unexpected number of measurements: iMeasurement=%d, Nmeasurements=%d; this is bug that must be fixed before any of this will work. Exiting",
            iMeasurement, Nmeasurements);
        exit(1);
    }

    const int Njnnz = num_j_nonzero(ctx);
    if(iJacobian != Njnnz)
    {
        MSG("cost() wrote an unexpected number of jacobian entries: iJacobian=%d, Njnnz=%d; this is bug that must be fixed before any of this will work. Exiting",
            iJacobian, Njnnz);
        exit(1);
    }

    if(Jrowptr != NULL)
        Jrowptr[Nmeasurements] = iJacobian;
}

static bool
plot_residuals(const char* filename_base,
               const double* x,
               const callback_context_t* ctx)
{
    char filename[1024];
    char cmd[2048];
    FILE* fp;
    int result;

    const int imeas_lidar_0                = measurement_index_lidar(0,0, ctx);
    const int Nmeas_lidar_observation_all  = num_measurements_lidars(ctx);
    const int imeas_camera_0               = measurement_index_camera(0,0, ctx);
    const int Nmeas_camera_observation_all = num_measurements_cameras(ctx);
    const int imeas_regularization_0       = measurement_index_regularization(ctx);
    const int Nmeas_regularization         = num_measurements_regularization(ctx);

    if( (int)sizeof(filename) <= snprintf(filename, sizeof(filename),
                                          "%s.gp", filename_base) )
    {
        MSG("sizeof(filename) exceeded. Giving up making the plot");
        return false;
    }

#warning "plot the measurement boundaries"
    // imeas_all = list(measurement_indices());

    // measurement_boundaries =
    //     [ x
    //       for imeas,what in imeas_all
    //       for x in
    //       (f'arrow from {imeas}, graph 0 to {imeas}, graph 1 nohead',
    //        f'label "{what}" at {imeas},graph 0 left front offset 0,character 2 boxed') ];
    //_set  = measurement_boundaries +

    if( (int)sizeof(cmd) <= snprintf(cmd, sizeof(cmd),
                                     "feedgnuplot "
                                     "--domain --dataid "
                                     "--legend lidar 'LIDAR residuals' "
                                     "--legend camera 'Camera residuals' "
                                     "--legend regularization 'Regularization residuals; plotted in pixels on the left y axis' "
                                     "--y2 lidar "
                                     "--with points "
                                     "--set 'link y2 via y*%f inverse y*%f' "
                                     "--ylabel   'Camera fit residual (pixels)' "
                                     "--y2label  'LIDAR fit residual (m)' "
                                     "--hardcopy '%s' ",
                                     SCALE_MEASUREMENT_M/SCALE_MEASUREMENT_PX,
                                     SCALE_MEASUREMENT_PX/SCALE_MEASUREMENT_M,
                                     filename) )
    {
        MSG("sizeof(cmd) exceeded. Giving up making the plot");
        return false;
    }

    fp = popen(cmd, "w");
    if(fp == NULL)
    {
        MSG("popen(feedgnuplot ...) failed");
        return false;
    }

    for(int i=0; i<Nmeas_lidar_observation_all; i++)
        fprintf(fp, "%d lidar %f\n",
                imeas_lidar_0 + i,
                x[imeas_lidar_0 + i] * SCALE_MEASUREMENT_M);
    for(int i=0; i<Nmeas_camera_observation_all; i++)
        fprintf(fp, "%d camera %f\n",
                imeas_camera_0 + i,
                x[imeas_camera_0 + i] * SCALE_MEASUREMENT_PX);
    for(int i=0; i<Nmeas_regularization; i++)
        fprintf(fp, "%d regularization %f\n",
                imeas_regularization_0 + i,
                x[imeas_regularization_0 + i] * SCALE_MEASUREMENT_PX);

    result = pclose(fp);
    if(result < 0)
    {
        MSG("pclose() failed. Giving up making the plot");
        return false;
    }
    if(result > 0)
    {
        MSG("feedgnuplot failed. Giving up making the plot");
        return false;
    }
    MSG("Wrote '%s'", filename);



    if( (int)sizeof(filename) <= snprintf(filename, sizeof(filename),
                                          "%s-histogram-lidar.gp", filename_base) )
    {
        MSG("sizeof(filename) exceeded. Giving up making the plot");
        return false;
    }

    if( (int)sizeof(cmd) <= snprintf(cmd, sizeof(cmd),
                                     "feedgnuplot "
                                     "--histogram 0 "
                                     "--binwidth %f "
                                     "--xmin %f --xmax %f "
                                     "--xlabel 'LIDAR residual (m)' "
                                     "--ylabel 'frequency' "
                                     "--hardcopy '%s' ",
                                     SCALE_MEASUREMENT_M/10.,
                                     -3.*SCALE_MEASUREMENT_M,
                                     3.*SCALE_MEASUREMENT_M,
                                     filename) )

    {
        MSG("sizeof(cmd) exceeded. Giving up making the plot");
        return false;
    }

    fp = popen(cmd, "w");
    if(fp == NULL)
    {
        MSG("popen(feedgnuplot ...) failed");
        return false;
    }
    for(int i=0; i<Nmeas_lidar_observation_all; i++)
        fprintf(fp, "%f\n",
                x[imeas_lidar_0 + i] * SCALE_MEASUREMENT_M);
    result = pclose(fp);
    if(result < 0)
    {
        MSG("pclose() failed. Giving up making the plot");
        return false;
    }
    if(result > 0)
    {
        MSG("feedgnuplot failed. Giving up making the plot");
        return false;
    }
    MSG("Wrote '%s'", filename);


    if(Nmeas_camera_observation_all)
    {
        if( (int)sizeof(filename) <= snprintf(filename, sizeof(filename),
                                              "%s-histogram-camera.gp", filename_base) )
        {
            MSG("sizeof(filename) exceeded. Giving up making the plot");
            return false;
        }

        if( (int)sizeof(cmd) <= snprintf(cmd, sizeof(cmd),
                                         "feedgnuplot "
                                         "--histogram 0 "
                                         "--binwidth %f "
                                         "--xmin %f --xmax %f "
                                         "--xlabel 'CAMERA residual (px)' "
                                         "--ylabel 'frequency' "
                                         "--hardcopy '%s' ",
                                         SCALE_MEASUREMENT_PX/10.,
                                         -3.*SCALE_MEASUREMENT_PX,
                                         3.*SCALE_MEASUREMENT_PX,
                                         filename) )

        {
            MSG("sizeof(cmd) exceeded. Giving up making the plot");
            return false;
        }

        fp = popen(cmd, "w");
            if(fp == NULL)
            {
                MSG("popen(feedgnuplot ...) failed");
                return false;
            }
        for(int i=0; i<Nmeas_camera_observation_all; i++)
            fprintf(fp, "%f\n",
                    x[imeas_camera_0 + i] * SCALE_MEASUREMENT_M);

        result = pclose(fp);
        if(result < 0)
        {
            MSG("pclose() failed. Giving up making the plot");
            return false;
        }
        if(result > 0)
        {
            MSG("feedgnuplot failed. Giving up making the plot");
            return false;
        }
        MSG("Wrote '%s'", filename);
    }

    return true;
}

#warning "disabled plot_geometry() for now"
#if 1
#define plot_geometry(...)
#else
def plot_geometry(filename,
                  *,
                  rt_ref_board,
                  rt_camera_ref,
                  rt_lidar_ref,
                  only_axes = False):


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
#endif

// Align the LIDAR and camera geometry
static bool
fit(// in,out
    // seed state on input
    double* Rt_lidar0_board,  // Nsensor_snapshots_filtered poses ( (4,3) Rt arrays ) of these to fill
    double* Rt_lidar0_lidar,  // Nlidars-1 poses ( (4,3) Rt arrays ) of these to fill (lidar0 not included)
    double* Rt_lidar0_camera, // Ncameras  poses ( (4,3) Rt arrays ) of these to fill

    // in
    const sensor_snapshot_segmented_t* sensor_snapshots_filtered,
    const unsigned int                 Nsensor_snapshots_filtered,

    // These apply to ALL the sensor_snapshots[]
    const unsigned int Ncameras,
    const unsigned int Nlidars,

    // bits indicating whether a camera in
    // sensor_snapshots.images[] is color or not
    const clc_is_bgr_mask_t is_bgr_mask,

    bool check_gradient__use_distance_to_plane,
    bool check_gradient )
{
    double rt_lidar0_board [6*Nsensor_snapshots_filtered];
    double rt_lidar0_lidar [6*(Nlidars-1)];
    double rt_lidar0_camera[6*Ncameras];
    for(unsigned int i=0; i<Nsensor_snapshots_filtered; i++)
        mrcal_rt_from_Rt(&rt_lidar0_board[i*6], NULL,
                         &Rt_lidar0_board[i*4*3]);
    for(unsigned int i=0; i<(Nlidars-1); i++)
        mrcal_rt_from_Rt(&rt_lidar0_lidar[i*6], NULL,
                         &Rt_lidar0_lidar[i*4*3]);
    for(unsigned int i=0; i<Ncameras; i++)
        mrcal_rt_from_Rt(&rt_lidar0_camera[i*6], NULL,
                         &Rt_lidar0_camera[i*4*3]);



    callback_context_t ctx = {.Ncameras              = Ncameras,
                              .Nlidars               = Nlidars,
                              .Nsnapshots            = Nsensor_snapshots_filtered,
                              .snapshots             = sensor_snapshots_filtered,
                              .use_distance_to_plane = false,
                              .report_imeas          = false};
    dogleg_parameters2_t dogleg_parameters;
    dogleg_getDefaultParameters(&dogleg_parameters);
    if(!(check_gradient__use_distance_to_plane || check_gradient))
        dogleg_parameters.dogleg_debug = DOGLEG_DEBUG_VNLOG;

    dogleg_solverContext_t* solver_context = NULL;

    const int Nstate        = num_states(&ctx);
    const int Nmeasurements = num_measurements(&ctx);
    const int Njnnz         = num_j_nonzero(&ctx);

    printf("## Nstate=%d Nmeasurements=%d Nlidars=%d Ncameras=%d Nsnapshots=%d\n",
           Nstate,Nmeasurements,Nlidars,Ncameras,Nsensor_snapshots_filtered);
    if(Ncameras > 0)
        printf("## States rt_lidar0_lidar in [%d,%d], rt_lidar0_camera in [%d,%d], rt_lidar0_board in [%d,%d]\n",
               state_index_lidar (1, &ctx), state_index_lidar (1, &ctx)+num_states_lidars (&ctx)-1,
               state_index_camera(0, &ctx), state_index_camera(0, &ctx)+num_states_cameras(&ctx)-1,
               state_index_board (0, &ctx), state_index_board (0, &ctx)+num_states_boards (&ctx)-1);
    else
        printf("## States rt_lidar0_lidar in [%d,%d], rt_lidar0_board in [%d,%d]\n",
               state_index_lidar (1, &ctx), state_index_lidar (1, &ctx)+num_states_lidars (&ctx)-1,
               state_index_board (0, &ctx), state_index_board (0, &ctx)+num_states_boards (&ctx)-1);

    double b[Nstate];
    double x[Nmeasurements];
    double norm2x;

    pack_solver_state(// out
                      b,
                      // in
                      rt_lidar0_lidar,
                      rt_lidar0_camera,
                      rt_lidar0_board,
                      &ctx);

    cost(b, x, NULL, &ctx);
    plot_residuals("/tmp/residuals-seed", x, &ctx);

    for(int i=0; i<Nmeasurements; i++)
        if( fabs(x[i])*SCALE_MEASUREMENT_PX > 1000 ||
            fabs(x[i])*SCALE_MEASUREMENT_M  > 100 )
        {
            MSG("Error: seed has unbelievably-high errors. Giving up");
            return false;
        }

    MSG("Starting pre-solve");
    ctx.use_distance_to_plane = true;
    if(!check_gradient__use_distance_to_plane)
    {
        norm2x = dogleg_optimize2(b,
                                  Nstate, Nmeasurements, Njnnz,
                                  (dogleg_callback_t*)&cost, &ctx,
                                  &dogleg_parameters,
                                  &solver_context);
    }
    else
    {
        for(int ivar=0; ivar<Nstate; ivar++)
            dogleg_testGradient(ivar, b,
                                Nstate, Nmeasurements, Njnnz,
                                (dogleg_callback_t*)&cost, &ctx);
        return true;
    }

    MSG("Finished pre-solve; started full solve;");

    ctx.use_distance_to_plane = false;
    if(!check_gradient)
    {
        norm2x = dogleg_optimize2(b,
                                  Nstate, Nmeasurements, Njnnz,
                                  (dogleg_callback_t*)&cost, &ctx,
                                  &dogleg_parameters,
                                  &solver_context);
    }
    else
    {
        for(int ivar=0; ivar<Nstate; ivar++)
            dogleg_testGradient(ivar, b,
                                Nstate, Nmeasurements, Njnnz,
                                (dogleg_callback_t*)&cost, &ctx);
        return true;
    }

    MSG("Finished full solve");

// uncertainty; not done yet
#if 0
    // I propagate uncertainty. This is similar to what I do in mrcal, but much
    // simpler. In mrcal, every camera is described by its intrinsics and
    // extrinsics, and much effort goes into disentangling the two. Here we have
    // only the poses.
    //
    // The blueprint of what to do is in https://mrcal.secretsauce.net/uncertainty.html
    //
    // I assume SCALE_MEASUREMENT_PX and SCALE_MEASUREMENT_M are the stdev of the
    // measurement. So
    // Var(qref) = I. The scale is applied in the cost function, so W = I also. So
    //
    // Var(b) = inv(JtJ) JobsT_Jobs inv(JtJ)
    J = res.jac;
    JtJ        = nps.matmult(J.T,J);

    Jobs       = J[:-Nmeas_regularization,:];
    JobsT_Jobs = nps.matmult(Jobs.T,Jobs);

    L,islower          = scipy.linalg.cho_factor(JtJ);
    inv_JtJ_JobsT_Jobs = scipy.linalg.cho_solve((L,islower), JobsT_Jobs);
    Var_b              = scipy.linalg.cho_solve((L,islower), inv_JtJ_JobsT_Jobs.T);

    inv_JtJ = np.linalg.inv(JtJ);
    Var_b_direct = nps.matmult(inv_JtJ, JobsT_Jobs, inv_JtJ);

    pinv = np.linalg.pinv(J)[:-Nmeas_regularization];
    Var_b_pinv = nps.matmult(pinv, pinv.T);

    // I don't care about the poses of the chessboards, so I pull out the
    // variance of the poses of the sensors only.
    istate_sensors_0 = min(istate_camera_pose_0,istate_lidar_pose_0);
    Nstate_sensors   = Nstate_camera_pose + Nstate_lidar_pose;
    istatesensors_camera_pose_0 = istate_camera_pose_0 - istate_sensors_0;
    istatesensors_lidar_pose_0  = istate_lidar_pose_0  - istate_sensors_0;
    if(!(istatesensors_camera_pose_0 + Nstate_camera_pose == istatesensors_lidar_pose_0 or
         istatesensors_lidar_pose_0  + Nstate_lidar_pose  == istatesensors_camera_pose_0 ))
        raise Exception("Uncertainty code assumes the camera and lidar pose states live next to each other");
    Var_b = Var_b[istate_sensors_0:istate_sensors_0+Nstate_sensors,
                  istate_sensors_0:istate_sensors_0+Nstate_sensors];

    Nsensors_in_state     = Nstate_sensors/6;
    Var_b = Var_b.reshape(Nsensors_in_state,6,Nsensors_in_state,6);

    // J uses packed state. I need to unpack it; J has state in the denominator,
    // so I unpack it by PACKING (/SCALE). Var(b) ~ inv(JtJ), so this is
    // inverted, and I UNPACK (*SCALE) in both dimensions.
    slice_state_cameras =
        slice(istatesensors_camera_pose_0/6,
              (istatesensors_camera_pose_0+Nstate_camera_pose)/6);
    slice_state_lidars =
        slice(istatesensors_lidar_pose_0/6,
              (istatesensors_lidar_pose_0+Nstate_lidar_pose)/6);

    Var_b[slice_state_cameras,:,:,:] *= nps.mv(SCALE_RT_CAMERA_REF,-1,-3);
    Var_b[:,:,slice_state_cameras,:] *= SCALE_RT_CAMERA_REF;
    Var_b[slice_state_lidars,:,:,:] *= nps.mv(SCALE_RT_LIDAR_REF,-1,-3);
    Var_b[:,:,slice_state_lidars,:] *= SCALE_RT_LIDAR_REF;
#endif

    unpack_solver_state(// out
                        rt_lidar0_lidar,
                        rt_lidar0_camera,
                        rt_lidar0_board,
                        // in
                        b,
                        &ctx);
    for(unsigned int i=0; i<Nsensor_snapshots_filtered; i++)
        mrcal_Rt_from_rt(&Rt_lidar0_board[i*4*3], NULL,
                         &rt_lidar0_board[i*6]);
    for(unsigned int i=0; i<(Nlidars-1); i++)
        mrcal_Rt_from_rt(&Rt_lidar0_lidar[i*4*3], NULL,
                         &rt_lidar0_lidar[i*6]);
    for(unsigned int i=0; i<Ncameras; i++)
        mrcal_Rt_from_rt(&Rt_lidar0_camera[i*4*3], NULL,
                         &rt_lidar0_camera[i*6]);

    cost(b, x, NULL, &ctx);

    MSG("RMS fit error: %.2f normalized units",
        sqrt(norm2x / (double)Nmeasurements));

    // if(Ncameras > 0)
    //     MSG("RMS fit error (camera): %.3f pixels",
    //         sqrt(np.mean(x_camera*x_camera))*SCALE_MEASUREMENT_PX);
    // MSG("RMS fit error (lidar): %.3f m",
    //     sqrt(np.mean(x_lidar *x_lidar ))*SCALE_MEASUREMENT_M);
    // MSG("norm2(error_regularization)/norm2(error): %.3f m",
    //     norm2(x_regularization)/norm2(x));

    plot_residuals("/tmp/residuals", x, &ctx);

    if(solver_context != NULL)
        dogleg_freeContext(&solver_context);

    return true;
}

static
bool _clc_internal(// out
         mrcal_pose_t* rt_ref_lidar,  // Nlidars  of these to fill
         mrcal_pose_t* rt_ref_camera, // Ncameras of these to fill

         // in

         // Exactly one of these should be non-NULL
         const clc_sensor_snapshot_unsorted_t* sensor_snapshots_unsorted,
         const clc_sensor_snapshot_sorted_t*   sensor_snapshots_sorted,
         const clc_sensor_snapshot_segmented_t*sensor_snapshots_segmented,

         const unsigned int                    Nsensor_snapshots,
         // The stride, in bytes, between each successive points or rings value
         // in clc_lidar_scan_unsorted_t
         const unsigned int           lidar_packet_stride,

         // These apply to ALL the sensor_snapshots[]
         const unsigned int Ncameras,
         const unsigned int Nlidars,

         // bits indicating whether a camera in
         // sensor_snapshots.images[] is color or not
         const clc_is_bgr_mask_t is_bgr_mask,

         bool check_gradient__use_distance_to_plane,
         bool check_gradient )
{
    if(1 !=
       (sensor_snapshots_unsorted != NULL) +
       (sensor_snapshots_sorted   != NULL) +
       (sensor_snapshots_segmented!= NULL))
    {
        MSG("Exactly one of (sensor_snapshots_sorted,sensor_snapshots_unsorted,sensor_snapshots_segmented) should be non-NULL");
        return false;
    }

    bool result = false;

    // contains the points_pool and the points_and_plane_pool
    uint8_t* pool = NULL;

    // I start by finding the chessboard in the raw sensor data, and throwing
    // out the sensor snapshots with too few shared-sensor observations
    sensor_snapshot_segmented_t sensor_snapshots_filtered[Nsensor_snapshots];

    int NcameraObservations[Ncameras];
    int NlidarObservations [Nlidars ];


    if(Ncameras != 0)
    {
        MSG("Only LIDAR-LIDAR calibrations implemented for now");
        goto done;
    }




#warning hardcoded
    const unsigned int Nrings = 32;

    // If we need to sort or segment the lidar data, I need to allocate memory.
    // Here I get the buffer sizes
    int Npoints_buffer      = 0;
    int Nlidar_scans_buffer = 0;
    for(unsigned int isnapshot=0; isnapshot < Nsensor_snapshots; isnapshot++)
        for(unsigned int ilidar=0; ilidar<Nlidars; ilidar++)
        {
            if(sensor_snapshots_unsorted != NULL)
                Npoints_buffer += sensor_snapshots_unsorted[isnapshot].lidar_scans[ilidar].Npoints;
            Nlidar_scans_buffer++;
        }

#warning "This is ugly. I should only store one plane's worth of info, and clc_lidar_segmentation() should tell me if it would have reported more"
    const int Nplanes_max = 2;
    // I allocate Nplanes_max extra planes so that clc_lidar_segmentation()
    // can use them, but I only use one plane's worth

    pool = malloc(Npoints_buffer * sizeof(clc_point3f_t) +
                  (Nlidar_scans_buffer + Nplanes_max) * sizeof(clc_points_and_plane_t));
    if(pool == NULL)
    {
        MSG("malloc() failed. Giving up");
        goto done;
    }

    clc_point3f_t*          points_pool           = (clc_point3f_t*)pool;
    clc_points_and_plane_t* points_and_plane_pool = (clc_points_and_plane_t*)&pool[Npoints_buffer * sizeof(clc_point3f_t)];

    int points_pool_index           = 0;
    int points_and_plane_pool_index = 0;


    int Nsensor_snapshots_filtered = 0;
    for(unsigned int isnapshot=0; isnapshot < Nsensor_snapshots; isnapshot++)
    {
        int Nsensors_observing = 0;

        const clc_sensor_snapshot_unsorted_t* sensor_snapshot_unsorted =
            (sensor_snapshots_unsorted != NULL ) ?
            &sensor_snapshots_unsorted[isnapshot] :
            NULL;
        const clc_sensor_snapshot_sorted_t* sensor_snapshot_sorted =
            (sensor_snapshots_sorted != NULL ) ?
            &sensor_snapshots_sorted[isnapshot] :
            NULL;
        const clc_sensor_snapshot_segmented_t* sensor_snapshot_segmented =
            (sensor_snapshots_segmented != NULL ) ?
            &sensor_snapshots_segmented[isnapshot] :
            NULL;

        for(unsigned int icamera=0; icamera<Ncameras; icamera++)
            sensor_snapshots_filtered[Nsensor_snapshots_filtered].images[icamera].uint8 =
                (mrcal_image_uint8_t){};
        if(Ncameras) assert(0); // a lot more to do above

        for(unsigned int ilidar=0; ilidar<Nlidars; ilidar++)
        {
            sensor_snapshots_filtered[Nsensor_snapshots_filtered].lidar_scans[ilidar] =
                (points_and_plane_full_t){};

            const clc_lidar_scan_unsorted_t* scan_unsorted =
                (sensor_snapshot_unsorted != NULL) ?
                &sensor_snapshot_unsorted->lidar_scans[ilidar] :
                NULL;
            const clc_lidar_scan_sorted_t* scan_sorted =
                (sensor_snapshot_sorted != NULL) ?
                &sensor_snapshot_sorted->lidar_scans[ilidar] :
                NULL;
            const clc_lidar_scan_segmented_t* scan_segmented =
                (sensor_snapshot_segmented != NULL) ?
                &sensor_snapshot_segmented->lidar_scans[ilidar] :
                NULL;


            // We have data from this lidar. Try to find the chessboard

            clc_point3f_t* points_here = NULL;
            unsigned int   _Npoints[Nrings];
            unsigned int*  Npoints = _Npoints;

            if(scan_unsorted != NULL ||
               scan_sorted   != NULL)
            {

                if(scan_unsorted != NULL)
                {
                    if(scan_unsorted->Npoints == 0)
                        continue;

                    points_here = &points_pool[points_pool_index];
                    points_pool_index += scan_unsorted->Npoints;

                    uint32_t ipoint_unsorted_in_sorted_order[scan_unsorted->Npoints];
                    clc_lidar_sort(// out
                                   //
                                   // These buffers must be pre-allocated
                                   // length sum(Npoints). Sorted by ring and then by azimuth
                                   points_here,
                                   ipoint_unsorted_in_sorted_order,
                                   // length Nrings
                                   Npoints,

                                   // in
                                   Nrings,
                                   // The stride, in bytes, between each successive points or
                                   // rings value in clc_lidar_scan_unsorted_t
                                   lidar_packet_stride,
                                   scan_unsorted);
                }
                else
                {
                    points_here = scan_sorted->points;
                    Npoints     = scan_sorted->Npoints;
                }

                clc_points_and_plane_t* points_and_plane_here =
                    &points_and_plane_pool[points_and_plane_pool_index];
                points_and_plane_pool_index++;


                clc_lidar_segmentation_context_t ctx;
                clc_lidar_segmentation_default_context(&ctx);

                int8_t Nplanes_found =
                    clc_lidar_segmentation_sorted(// out
                                                  points_and_plane_here,
                                                  // in
                                                  Nplanes_max,
                                                  &(clc_lidar_scan_sorted_t){.points  = points_here,
                                                                             .Npoints = Npoints},
                                                  &ctx);

                // If we didn't see a clear plane, I keep the previous
                // ..._pool_bytes_used value, reusing this memory on the next round.
                // If we see a clear plane, but filter this data out at a later
                // point, I will not be reusing the memory; instead I'll carry it
                // around until the whole thing is freed at the end of clc_unsorted()
                if(Nplanes_found == 0)
                {
                    // MSG("No planes found for isnapshot=%d ilidar=%d.",
                    //     isnapshot, ilidar);
                    continue;
                }
                if(Nplanes_found > 1)
                {
                    // MSG("Too many planes found for isnapshot=%d ilidar=%d.",
                    //     isnapshot, ilidar);
                    continue;
                }

                // Keep this scan
                sensor_snapshots_filtered[Nsensor_snapshots_filtered].lidar_scans[ilidar] =
                    (points_and_plane_full_t){ .points           = points_here,
                                               .points_and_plane = points_and_plane_here };

                Nsensors_observing++;
            }
            else
            {
                // scan_segmented
                if(scan_segmented->points_and_plane.ipoint_set.n == 0)
                    continue;

                // Keep this scan
                sensor_snapshots_filtered[Nsensor_snapshots_filtered].lidar_scans[ilidar] =
                    (points_and_plane_full_t){ .points           = scan_segmented->points,
                                               .points_and_plane = &scan_segmented->points_and_plane };
                Nsensors_observing++;
            }
        }


        if(Nsensors_observing < 2)
        {
            // MSG("Sensor snapshot %d observed by %d sensors",
            //     isnapshot, Nsensors_observing);
            // MSG("Need at least 2. Throwing out this snapshot");
            continue;
        }

        MSG("Sensor snapshot %d observed by %d sensors",
            isnapshot, Nsensors_observing);
        MSG("isnapshot_original=%d corresponds to isnapshot_filtered=%d",
            isnapshot, Nsensor_snapshots_filtered);
        Nsensor_snapshots_filtered++;
    }

    MSG("Have %d joint observations", Nsensor_snapshots_filtered);


    // I pass through the observations again, to make sure that I count the
    // FILTERED snapshots, not the original ones
    for(unsigned int i=0; i<Ncameras; i++) NcameraObservations[i] = 0;
    for(unsigned int i=0; i<Nlidars;  i++) NlidarObservations [i] = 0;

    for(int isnapshot=0; isnapshot < Nsensor_snapshots_filtered; isnapshot++)
    {
        const sensor_snapshot_segmented_t* sensor_snapshot = &sensor_snapshots_filtered[isnapshot];

        for(unsigned int icamera=0; icamera<Ncameras; icamera++)
            if(sensor_snapshot->images[icamera].uint8.width != 0)
                NcameraObservations[icamera]++;

        for(unsigned int ilidar=0; ilidar<Nlidars; ilidar++)
            if(sensor_snapshot->lidar_scans[ilidar].points != NULL)
                NlidarObservations[ilidar]++;
    }
    for(unsigned int i=0; i<Ncameras; i++)
    {
        const int NcameraObservations_this = NcameraObservations[i];
        if (NcameraObservations_this == 0)
            MSG("I need at least 1 observation of each camera. Got only %d for camera %d",
                NcameraObservations_this, i);
        goto done;
    }

    for(unsigned int i=0; i<Nlidars; i++)
    {
        const int NlidarObservations_this = NlidarObservations[i];
        if (NlidarObservations_this < 3)
        {
            MSG("I need at least 3 observations of each lidar to unambiguously set the translation (the set of all plane normals must span R^3). Got only %d for lidar %d",
                NlidarObservations_this, i);
            goto done;
        }
    }


    {
        double Rt_lidar0_board [Nsensor_snapshots_filtered * 4*3];
        double Rt_lidar0_lidar [(Nlidars-1)                * 4*3];
        double Rt_lidar0_camera[Ncameras                   * 4*3];
        if(!fit_seed(// out
                     Rt_lidar0_board,
                     Rt_lidar0_lidar,
                     Rt_lidar0_camera,

                     // in
                     sensor_snapshots_filtered,
                     Nsensor_snapshots_filtered,

                     // These apply to ALL the sensor_snapshots[]
                     Ncameras,
                     Nlidars,

                     // bits indicating whether a camera in
                     // sensor_snapshots.images[] is color or not
                     is_bgr_mask))
        {
            MSG("fit_seed() failed");
            goto done;
        }

        plot_geometry("/tmp/geometry-seed.gp",
                      **seed_state);
        plot_geometry("/tmp/geometry-seed-onlyaxes.gp",
                      only_axes = True,
                      **seed_state);

        if(!fit(// in,out
                // seed state on input
                Rt_lidar0_board,
                Rt_lidar0_lidar,
                Rt_lidar0_camera,

                // in
                sensor_snapshots_filtered,
                Nsensor_snapshots_filtered,

                // These apply to ALL the sensor_snapshots[]
                Ncameras,
                Nlidars,

                // bits indicating whether a camera in
                // sensor_snapshots.images[] is color or not
                is_bgr_mask,

                check_gradient__use_distance_to_plane,
                check_gradient))
        {
            MSG("fit_seed() failed");
            goto done;
        }

        if(check_gradient__use_distance_to_plane || check_gradient)
        {
            result = true;
            goto done;
        }

        plot_geometry("/tmp/geometry.gp",
                      **solved_state);
        plot_geometry("/tmp/geometry-onlyaxes.gp",
                      only_axes = True,
                      **solved_state);

/*
        for(imodel in range(len(args.models)))
        {
            models[imodel].extrinsics_rt_fromref(solved_state['rt_camera_ref'][imodel]);
            root,extension = os.path.splitext(args.models[imodel]);
            filename = f"{root}-mounted{extension}";
            models[imodel].write(filename);
            print(f"Wrote '{filename}'");
        }

        for(iobservation in range(len(joint_observations)))
        {
            (q_observed, p_lidar) = joint_observations[iobservation]
            for ilidar in range(Nlidars):
                if p_lidar[ilidar] is None: continue
                for icamera in range(Ncameras):
                    if q_observed[icamera] is None: continue

                    rt_camera_lidar = mrcal.compose_rt(solved_state['rt_camera_ref'][icamera],
                                                       mrcal.invert_rt(solved_state['rt_lidar_ref'][ilidar]));
                    p = mrcal.transform_point_rt(rt_camera_lidar, p_lidar[ilidar]);
                    q_from_lidar = mrcal.project(p, *models[icamera].intrinsics());

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
                             hardcopy = filename);
                    print(f"Wrote '{filename}'");
        }

        // Write the intra-multisense calibration
        multisense_units_lra = find_multisense_units_lra(args.camera_topic);
        write_multisense_calibration(multisense_units_lra);
*/
/*


        print("The poses follow. The reference is defined to sit at lidar0\n");
        print("To visualize an aligned bag, run:\n\n./show-aligned-lidar-pointclouds.py \\");

        if(Ncameras > 0)
        {
            // Write the inter-multisense extrinsics
            multisense_topics          = []
            str_multisense_poses       = ''
            str_multisense_poses_other = ''
            for unit in multisense_units_lra.keys():
                lra = multisense_units_lra[unit]
                l = lra[0]
                if l < 0:
                    continue

                topic = args.camera_topic[l]
                multisense_topics.append(topic);

                rt_multisenseleft_lidar0 = models[l].extrinsics_rt_fromref();
                str_multisense_poses +=
                    f"  --rt-multisenseleft-ref \" {','.join(list(str(x) for x in rt_multisenseleft_lidar0))}\" \\\n"

                rpy = rpy_from_r(rt_multisenseleft_lidar0[:3]);
                xyz = rt_multisenseleft_lidar0[3:]
                str_multisense_poses_other +=
                    f"  {rpy=} {xyz=}\n"
            print(f"  --multisense-topic {','.join(multisense_topics)} \\");
            print(str_multisense_poses);
            print("other poses:");
            print(str_multisense_poses_other);
            print('\n');
        }

        // Write the inter-multisense lidar
        lidar_topics          = [];
        str_lidar_poses       = '';
        str_lidar_poses_other = '';
        for(ilidar in range(Nlidars))
        {
            topic = args.lidar_topic[ilidar];
            lidar_topics.append(topic);

            rt_lidar_lidar0 = solved_state['rt_lidar_ref'][ilidar];
            str_lidar_poses +=
                f"  --rt-lidar-ref \" {','.join(list(str(x) for x in rt_lidar_lidar0))}\" \\\n";

            rpy = rpy_from_r(rt_lidar_lidar0[:3]);
            xyz = rt_lidar_lidar0[3:];
            str_lidar_poses_other += f"  {rpy=} {xyz=}\n";
        }

        print(f"  --lidar-topic {','.join(lidar_topics)} \\");
        print(str_lidar_poses, end='');
        print('  ' + args.bag[0]);
        print('\nOr pass in any of the other bags\n');

        print("other poses:");
        print(str_lidar_poses_other);
*/

        for(unsigned int i=0; i<Ncameras; i++)
        {
            mrcal_rt_from_Rt(rt_ref_camera[i].r.xyz, NULL,
                             &Rt_lidar0_camera[i*4*3]);
            rt_ref_camera[i].t = *(mrcal_point3_t*)&Rt_lidar0_camera[i*4*3 + 9];
        }
        memset(rt_ref_lidar, 0, sizeof(*rt_ref_lidar)); // lidar0 has the reference transform
        for(unsigned int i=0; i<Nlidars-1; i++)
        {
            mrcal_rt_from_Rt(rt_ref_lidar[i+1].r.xyz, NULL,
                             &Rt_lidar0_lidar[i*4*3]);
            rt_ref_lidar[i+1].t = *(mrcal_point3_t*)&Rt_lidar0_lidar[i*4*3 + 9];
        }

        // for(unsigned int i=0; i<Nsensor_snapshots; i++)
        // {
        //     mrcal_rt_from_Rt(rt_ref_camera[i].r.xyz, NULL,
        //                      &Rt_lidar0_camera[i*4*3]);
        //     rt_ref_camera[i].t = *(mrcal_point3_t*)&Rt_lidar0_camera[i*4*3 + 9];
        // }

    }

    result = true;

 done:
    free(pool);
    return result;
}

bool clc_unsorted(// out
         mrcal_pose_t* rt_ref_lidar,  // Nlidars  of these to fill
         mrcal_pose_t* rt_ref_camera, // Ncameras of these to fill

         // in
         const clc_sensor_snapshot_unsorted_t* sensor_snapshots,
         const unsigned int                    Nsensor_snapshots,
         // The stride, in bytes, between each successive points or rings value
         // in clc_lidar_scan_unsorted_t
         const unsigned int           lidar_packet_stride,

         // These apply to ALL the sensor_snapshots[]
         const unsigned int Ncameras,
         const unsigned int Nlidars,

         // bits indicating whether a camera in
         // sensor_snapshots.images[] is color or not
         const clc_is_bgr_mask_t is_bgr_mask,

         bool check_gradient__use_distance_to_plane,
         bool check_gradient)
{
    return _clc_internal(// out
                         rt_ref_lidar,
                         rt_ref_camera,

                         // in
                         sensor_snapshots, NULL, NULL,
                         Nsensor_snapshots,
                         lidar_packet_stride,
                         Ncameras,
                         Nlidars,
                         is_bgr_mask,

                         check_gradient__use_distance_to_plane,
                         check_gradient);
}

bool clc_sorted(// out
         mrcal_pose_t* rt_ref_lidar,  // Nlidars  of these to fill
         mrcal_pose_t* rt_ref_camera, // Ncameras of these to fill

         // in
         const clc_sensor_snapshot_sorted_t* sensor_snapshots,
         const unsigned int                  Nsensor_snapshots,

         // These apply to ALL the sensor_snapshots[]
         const unsigned int Ncameras,
         const unsigned int Nlidars,

         // bits indicating whether a camera in
         // sensor_snapshots.images[] is color or not
         const clc_is_bgr_mask_t is_bgr_mask,

         bool check_gradient__use_distance_to_plane,
         bool check_gradient)
{
    return _clc_internal(// out
                         rt_ref_lidar,
                         rt_ref_camera,

                         // in
                         NULL, sensor_snapshots, NULL,
                         Nsensor_snapshots,
                         0,
                         Ncameras,
                         Nlidars,
                         is_bgr_mask,

                         check_gradient__use_distance_to_plane,
                         check_gradient);
}

bool clc_lidar_segmented(// out
         mrcal_pose_t* rt_ref_lidar,  // Nlidars  of these to fill
         mrcal_pose_t* rt_ref_camera, // Ncameras of these to fill

         // in
         const clc_sensor_snapshot_segmented_t* sensor_snapshots,
         const unsigned int                     Nsensor_snapshots,

         // These apply to ALL the sensor_snapshots[]
         const unsigned int Ncameras,
         const unsigned int Nlidars,

         // bits indicating whether a camera in
         // sensor_snapshots.images[] is color or not
         const clc_is_bgr_mask_t is_bgr_mask,

         bool check_gradient__use_distance_to_plane,
         bool check_gradient)
{
    return _clc_internal(// out
                         rt_ref_lidar,
                         rt_ref_camera,

                         // in
                         NULL, NULL, sensor_snapshots,
                         Nsensor_snapshots,
                         0,
                         Ncameras,
                         Nlidars,
                         is_bgr_mask,

                         check_gradient__use_distance_to_plane,
                         check_gradient);
}

