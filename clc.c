#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <limits.h>
#include <assert.h>
#include <math.h>
#include <mrcal/mrcal.h>

#include "clc.h"
#include "util.h"


typedef struct
{
    // The points_and_plane contains indices into the points[] array here

    // pointers to the pools declared above
    clc_point3f_t*          points;
    clc_points_and_plane_t* points_and_plane;
} points_and_plane_full_t;

typedef struct
{
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
                               clc_point3f_t*  pin)
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

    // Now the translation. R01 x1 + t01 ~ x0
    // -> t01 ~ x0 - R01 x1
    //
    // I now have an estimate for t01 for each observation. Ideally they should
    // be self-consistent, so I compute the mean:
    // -> t01 = mean(x0 - R01 x1)
    *(mrcal_point3_t*)(&Rt01[9]) = (mrcal_point3_t){};
    for(int i=0; i<Nbuffer; i++)
    {
        double R01_x1[3];
        mrcal_rotate_point_R(R01_x1, NULL, NULL,
                             Rt01,
                             points1[i].xyz);
        for(int j=0; j<3; j++)
            Rt01[9 + j] += points0[i].xyz[j] - R01_x1[j];
    }
    for(int j=0; j<3; j++)
        Rt01[9 + j] /= (double)Nbuffer;

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

            bool validation_failed_here = p0_err_mag > 4. || th_err_deg > 20.;

            if(validation_failed_here) validation_failed = true;

            MSG("%sisnapshot=%d ilidar=%d p0_err_mag=%.2f th_err_deg=%.2f",
                validation_failed_here ? "FAILED: " : "",
                isnapshot, ilidar, p0_err_mag, th_err_deg);
        }
    }

    return !validation_failed;
}

static
bool _clc_internal(// out
         mrcal_pose_t* rt_ref_lidar,  // Nlidars  of these to fill
         mrcal_pose_t* rt_ref_camera, // Ncameras of these to fill

         // in

         // Exactly one of these should be non-NULL
         const clc_sensor_snapshot_unsorted_t* sensor_snapshots_unsorted,
         const clc_sensor_snapshot_sorted_t*   sensor_snapshots_sorted,
         const unsigned int                    Nsensor_snapshots,
         // The stride, in bytes, between each successive points or rings value
         // in clc_lidar_scan_unsorted_t
         const unsigned int           lidar_packet_stride,

         // These apply to ALL the sensor_snapshots[]
         const unsigned int Ncameras,
         const unsigned int Nlidars,

         // bits indicating whether a camera in
         // sensor_snapshots.images[] is color or not
         const clc_is_bgr_mask_t is_bgr_mask)
{
    if(1 !=
       (sensor_snapshots_unsorted != NULL) +
       (sensor_snapshots_sorted != NULL))
    {
        MSG("Exactly one of (sensor_snapshots_sorted,sensor_snapshots_unsorted) should be non-NULL");
        return false;
    }

    bool result = false;

    clc_point3f_t* points_pool = NULL;
    int points_pool_bytes_used = 0;

    clc_points_and_plane_t* points_and_plane_pool            = NULL;
    int                     points_and_plane_pool_bytes_used = 0;

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


            // We have data from this lidar. Try to find the chessboard

            int            points_pool_bytes_used_here = 0;
            clc_point3f_t* points_here                 = NULL;
            unsigned int   _Npoints[Nrings];
            unsigned int*  Npoints = _Npoints;

            if(scan_unsorted != NULL)
            {
                if(scan_unsorted->Npoints == 0)
                    continue;

                // We need another chunk of memory. realloc()
                points_pool_bytes_used_here =
                    scan_unsorted->Npoints * sizeof(points_pool[0]);
                points_pool =
                    (clc_point3f_t*)realloc(points_pool,
                                            points_pool_bytes_used +
                                            points_pool_bytes_used_here);
                if(points_pool == NULL)
                {
                    MSG("realloc() failed. Giving up");
                    goto done;
                }
                points_here =
                    (clc_point3f_t*)
                    &(((uint8_t*)points_pool          )[points_pool_bytes_used]);

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


#warning "This is ugly. I should only store one plane's worth of info, and clc_lidar_segmentation() should tell me if it would have reported more"
            const int Nplanes_max = 2;
            // I allocate Nplanes_max planes so that clc_lidar_segmentation()
            // can use them, but I only use one plane's worth. So
            // points_and_plane_pool_bytes_used_here has just one plane
            const int points_and_plane_pool_bytes_used_here =
                sizeof(points_and_plane_pool[0]);
            points_and_plane_pool =
                (clc_points_and_plane_t*)realloc(points_and_plane_pool,
                                                 points_and_plane_pool_bytes_used +
                                                 Nplanes_max * points_and_plane_pool_bytes_used_here);
            if(points_and_plane_pool == NULL)
            {
                MSG("realloc() failed. Giving up");
                goto done;
            }

            clc_points_and_plane_t* points_and_plane_here =
                (clc_points_and_plane_t*)
                &(((uint8_t*)points_and_plane_pool)[points_and_plane_pool_bytes_used]);


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
            points_pool_bytes_used           += points_pool_bytes_used_here;
            points_and_plane_pool_bytes_used += points_and_plane_pool_bytes_used_here;

            Nsensors_observing++;
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

    }

    result = true;

 done:
    free(points_pool);
    free(points_and_plane_pool);
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
         const clc_is_bgr_mask_t is_bgr_mask)
{
    return _clc_internal(// out
                         rt_ref_lidar,
                         rt_ref_camera,

                         // in
                         sensor_snapshots, NULL,
                         Nsensor_snapshots,
                         lidar_packet_stride,
                         Ncameras,
                         Nlidars,
                         is_bgr_mask);
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
         const clc_is_bgr_mask_t is_bgr_mask)
{
    return _clc_internal(// out
                         rt_ref_lidar,
                         rt_ref_camera,

                         // in
                         NULL, sensor_snapshots,
                         Nsensor_snapshots,
                         0,
                         Ncameras,
                         Nlidars,
                         is_bgr_mask);
}
