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
#include "opencv-c-bridge.hh"
#include "bitarray.h"


#warning hardcoded
static
const unsigned int _Nrings = 32;

typedef struct
{
    clc_point3f_t* points;

    unsigned int   n;
    // If non-NULL, indices into points[]. Otherwise the points[] are used
    // densely
    const uint32_t*ipoint;

    clc_plane_t    plane;
} points_and_plane_full_t;

typedef struct
{
    // If no chessboard corners were found, each pointer is NULL
    //
    // If the corners were found, this is a object_width_n*object_height_n buffer of
    // points. object_width_n and object_height_n are available in the caller
    mrcal_point2_t* chessboard_corners[clc_Ncameras_max];

    points_and_plane_full_t lidar_scans[clc_Nlidars_max];
} sensor_snapshot_segmented_t;



// Basic multiplication function, with arbitrary strides. The strides are given
// in terms of ELEMENTS, not bytes
static void multiply_matrix_matrix( // out
                                    double* P, // (N,L) matrix
                                    const int P_stride0, const int P_stride1,
                                    // in
                                    const double* A, // (N,M) matrix
                                    const int A_stride0, const int A_stride1,
                                    const double* B, // (M,L) matrix
                                    const int B_stride0, const int B_stride1,
                                    const int N, const int M, const int L,
                                    const bool accumulate)
{
    for(int i=0; i<N; i++)
        for(int j=0; j<L; j++)
        {
            if(!accumulate)
                P[i*P_stride0 + j*P_stride1] = 0.;
            // we're writing P[i,j]
            for(int k=0; k<M; k++)
            {
                P[i*P_stride0 + j*P_stride1] +=
                    A[i*A_stride0 + k*A_stride1] *
                    B[k*B_stride0 + j*B_stride1];
            }
        }
}


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



#define LOOP_SNAPSHOT(CTXREF)                   \
    for(int isnapshot=0;                        \
        isnapshot < CTXREF Nsnapshots;          \
        isnapshot++)

#define LOOP_SNAPSHOT_HEADER(CTXREF,CONST)                              \
        if(!bitarray64_check(CTXREF bitarray_snapshots_selected, isnapshot)) \
            continue;                                                   \
        CONST sensor_snapshot_segmented_t* snapshot = &CTXREF snapshots[isnapshot]




static void
connectivity_matrix(// out
                    uint16_t* shared_observation_counts,

                    // in
                    const sensor_snapshot_segmented_t* snapshots,
                    const int                          Nsnapshots,
                    const uint64_t*                    bitarray_snapshots_selected,

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

    LOOP_SNAPSHOT()
    {
        LOOP_SNAPSHOT_HEADER(,const);

        for(int icamera0=0; icamera0<Ncameras; icamera0++)
        {
            // camera-camera links

            if(snapshot->chessboard_corners[icamera0] == NULL)
                continue;

            for(int icamera1=icamera0+1; icamera1<Ncameras; icamera1++)
            {
                if(snapshot->chessboard_corners[icamera1] == NULL)
                    continue;

                const int idx = pairwise_index(Nlidars+icamera0,
                                               Nlidars+icamera1,
                                               Nsensors);
                shared_observation_counts[idx]++;
            }

            // camera-lidar links
            for(int ilidar1=0; ilidar1<Nlidars; ilidar1++)
            {
                if(snapshot->lidar_scans[ilidar1].points == NULL)
                    continue;

                const int idx = pairwise_index(ilidar1,
                                               Nlidars+icamera0,
                                               Nsensors);
                shared_observation_counts[idx]++;
            }
        }

        // lidar-lidar links
        for(int ilidar0=0; ilidar0<Nlidars-1; ilidar0++)
        {
            if(snapshot->lidar_scans[ilidar0].points == NULL)
                continue;

            for(int ilidar1=ilidar0+1; ilidar1<Nlidars; ilidar1++)
            {
                if(snapshot->lidar_scans[ilidar1].points == NULL)
                    continue;

                const int idx = pairwise_index(ilidar1, ilidar0,
                                               Nsensors);
                shared_observation_counts[idx]++;
            }
        }
    }
}

static void
mrcal_point3_from_clc_point3f( mrcal_point3_t* pout,
                               const clc_point3f_t*  pin)
{
    for(int i=0; i<3; i++)
        pout->xyz[i] = (double)pin->xyz[i];
}

static bool Rt_uninitialized(const double* Rt)
{
    for(int i=0; i<3*3; i++)
        if(Rt[i] != 0.)
            return false;
    return true;
}

// Exact C port of
// mrcal.calibration._estimate_camera_pose_from_fixed_point_observations(). Not
// pushed to mrcal itself because it calls OpenCV, but mrcal does not link to it
// in its C library
bool
clc_estimate_camera_pose_from_fixed_point_observations(// out
                                                    double* Rt_cam_points,
                                                    // in
                                                    const mrcal_lensmodel_t* lensmodel,
                                                    const double*            intrinsics,
                                                    const mrcal_point2_t*    observations,
                                                    const mrcal_point3_t*    points_ref,
                                                    const int                N)
{
    // if z<0, try again with bigger f
    // if too few points: try again with smaller f
    double scale = 1.0;
    while(true)
    {
        const double fx = intrinsics[0];
        const double fy = intrinsics[1];
        const double cx = intrinsics[2];
        const double cy = intrinsics[3];

        const double camera_matrix[] =
            { fx*scale, 0,       cx,
              0,       fy*scale, cy,
              0,       0,        1 };

        mrcal_point2_t observations_local[N];
        mrcal_point3_t ref_object        [N];
        int Nobservations = 0;
        for(int i=0; i<N; i++)
        {
            mrcal_point2_t q =
                { .x = (observations[i].x - cx) / scale + cx,
                  .y = (observations[i].y - cy) / scale + cy };
            mrcal_point3_t v;
            if(!mrcal_unproject( &v,
                                 &q, 1, lensmodel, intrinsics))
                continue;
            if( !isfinite(v.x) ||
                !isfinite(v.y) ||
                !isfinite(v.z) )
            {
                continue;
            }

            mrcal_project( &observations_local[Nobservations], NULL, NULL,
                           &v, 1, &(mrcal_lensmodel_t){.type = MRCAL_LENSMODEL_PINHOLE}, intrinsics );

            // observation_qxqy_pinhole = (observation_qxqy_pinhole - cxy)*scale + cxy
            observations_local[i].x *= scale;
            observations_local[i].y *= scale;
            observations_local[i].x += cx*(1. - scale);
            observations_local[i].y += cy*(1. - scale);

            // accept point
            ref_object[Nobservations] = points_ref[i];
            Nobservations++;
        }

        if(Nobservations < 4)
        {
            if(scale == 1.0)
            {
                scale = 0.7;
                continue;
            }
            MSG("Insufficient observations; need at least 4; got %d instead. Cannot estimate initial extrinsics",
                Nobservations);
            return false;
        }

        double rtvec[6];
        mrcal_point3_t* rvec = (mrcal_point3_t*)&rtvec[0];
        mrcal_point3_t* tvec = (mrcal_point3_t*)&rtvec[3];
        if(!cv_solvePnP(// in/out
                        rvec, tvec,
                        // in
                        ref_object,
                        observations_local,
                        Nobservations,
                        camera_matrix,
                        false))
        {
            MSG("solvePnP() failed! Cannot estimate initial extrinsics");
            return false;
        }
        if(tvec->z <= 0)
        {
            // The object ended up behind the camera. I flip it, and try to solve
            // again
            for(int i=0; i<3; i++) tvec->xyz[i] *= -1;

            if(!cv_solvePnP(// in/out
                            rvec, tvec,
                            // in
                            ref_object,
                            observations_local,
                            Nobservations,
                            camera_matrix,
                            true))
            {
                MSG("Retried solvePnP() failed! Cannot estimate initial extrinsics");
                return false;
            }
            if(tvec->z <= 0)
            {
                if(scale == 1.0)
                {
                    scale = 1.5;
                    continue;
                }
                MSG("Retried solvePnP() insists that tvec->z <= 0 (i.e. the chessboard is behind us). Cannot estimate initial extrinsics");
                return false;
            }
        }

        mrcal_Rt_from_rt(Rt_cam_points, NULL,
                         rtvec);
        return true;
    }

    // unreachable
    return false;
}

void
clc_ref_calibration_object(// out
                       mrcal_point3_t*            points_ref,
                       // in
                       const int                  object_height_n,
                       const int                  object_width_n,
                       const double               object_spacing)

{
    // The board geometry is usually computed by mrcal.ref_calibration_object():
    // the coordinates move as
    //
    //   x = linspace(0,object_width_n-1,object_width_n)*object_spacing
    //-> x = i*object_spacing
    // In the center i = (object_width_n-1)/2
    //-> x_center = (object_width_n-1)/2*object_spacing
    //
    // I'm also assuming no board warp, so z=0

    for(int i=0; i<object_height_n; i++)
        for(int j=0; j<object_width_n; j++)
        {
            points_ref[i*object_width_n + j].x = (double)j*object_spacing;
            points_ref[i*object_width_n + j].y = (double)i*object_spacing;
            points_ref[i*object_width_n + j].z = 0.;
        }
}

bool clc_fit_Rt_camera_board(// out
                 double*                    Rt_camera_board,
                 // in
                 const mrcal_cameramodel_t* model,
                 const mrcal_point2_t*      observations,
                 const int                  object_height_n,
                 const int                  object_width_n,
                 const double               object_spacing)
{
    const int N = object_height_n*object_width_n;
    mrcal_point3_t points_ref[N];
    clc_ref_calibration_object(points_ref, object_height_n, object_width_n, object_spacing);

    if(!clc_estimate_camera_pose_from_fixed_point_observations( Rt_camera_board,
                                                             &model->lensmodel,
                                                             model->intrinsics,
                                                             observations,
                                                             points_ref,
                                                             N))
        return false;
    if(Rt_camera_board[3*3 + 2] <= 0)
    {
        MSG("Chessboard is behind the camera");
        memset(Rt_camera_board, 0, 4*3*sizeof(Rt_camera_board[0]));
        return false;
    }
    return true;
}

static bool
fit_Rt_camera_board_withcache(// out
                              double*                    Rt_camera_board,
                              // in
                              const mrcal_cameramodel_t* model,
                              const mrcal_point2_t*      observations,
                              const int                  object_height_n,
                              const int                  object_width_n,
                              const double               object_spacing)
{
    if(Rt_uninitialized(Rt_camera_board))
        if(!clc_fit_Rt_camera_board(// out
                                    Rt_camera_board,
                                    // in
                                    model,
                                    observations,
                                    object_height_n,
                                    object_width_n,
                                    object_spacing))
            return false;
    return true;
}

static
void get_pboardcenter_board(// out
                            double* p,
                            // in
                            const int    object_height_n,
                            const int    object_width_n,
                            const double object_spacing,
                            const int    Ncameras)
{
    if(Ncameras > 0)
    {
        // We have cameras; we know where the center of the board is
        p[0] = (object_width_n -1)/2*object_spacing;
        p[1] = (object_height_n-1)/2*object_spacing;
        p[2] = 0.;
        return;
    }

    // No cameras; we don't know where the center of the board is. We leave it
    // at 0
    p[0] = 0.;
    p[1] = 0.;
    p[2] = 0.;
}

static bool
compute_board_poses(// out
                    double*                            Rt_lidar0_board,
                    // in,out
                    double*                            Rt_camera_board_cache,
                    // in
                    const sensor_snapshot_segmented_t* snapshots,
                    const int                          Nsnapshots,
                    const uint64_t*                    bitarray_snapshots_selected,
                    const int                          Ncameras,
                    const int                          Nlidars,
                    const double*                      Rt_lidar0_lidar,
                    const double*                      Rt_lidar0_camera,
                    const mrcal_cameramodel_t*const*   models, // Ncameras of these
                    const int                          object_height_n,
                    const int                          object_width_n,
                    const double                       object_spacing,
                    bool verbose)
{
    LOOP_SNAPSHOT()
    {
        LOOP_SNAPSHOT_HEADER(,const);

        // cameras

        // I arbitrarily use the first camera observation, if there is one. I
        // prefer cameras. If camera observations are available, I use them
        bool done = false;
        for(int icamera=0; icamera<Ncameras; icamera++)
        {
            if(snapshot->chessboard_corners[icamera] == NULL)
                continue;

            double* Rt_camera_board = &Rt_camera_board_cache[ (isnapshot*Ncameras + icamera) *4*3];
            if(!fit_Rt_camera_board_withcache(// out
                                              Rt_camera_board,
                                              // in
                                              models[icamera],
                                              snapshot->chessboard_corners[icamera],
                                              object_height_n,
                                              object_width_n,
                                              object_spacing))
                return false;

            mrcal_compose_Rt(// out
                             &Rt_lidar0_board[isnapshot*4*3],
                             // in
                             &Rt_lidar0_camera[icamera*4*3],
                             Rt_camera_board);
            done = true;
            break;
        }
        if(done)
            continue;




        // This board is observed only by LIDARs

        int ilidar_first=0;
        for(; ilidar_first<Nlidars; ilidar_first++)
            if(snapshot->lidar_scans[ilidar_first].points != NULL)
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
                                      &snapshot->lidar_scans[ilidar_first].plane.n);
        mrcal_point3_from_clc_point3f(&plidar_mean,
                                      &snapshot->lidar_scans[ilidar_first].plane.p_mean);

        // I have the normal to the board, in lidar coordinates. Compute an
        // arbitrary rotation that matches this normal. This is unique only
        // up to yaw
        double Rt_board_lidar[4*3] = {};
        mrcal_R_aligned_to_vector(Rt_board_lidar,
                                  n.xyz);

        // I want pboardcenter_board to map to p: R_board_lidar
        // plidar_mean + t_board_lidar = pboardcenter_board
        // -> t_board_lidar = pboardcenter_board - R_board_lidar plidar_mean
        double* t_board_lidar = &Rt_board_lidar[3*3];
        for(int i=0; i<3; i++)
            get_pboardcenter_board(// out
                                   t_board_lidar,
                                   // in
                                   object_height_n,
                                   object_width_n,
                                   object_spacing,
                                   Ncameras);

        for(int i=0; i<3; i++)
            for(int j=0; j<3; j++)
                t_board_lidar[i] -=
                    Rt_board_lidar[i*3 + j] * plidar_mean.xyz[j];

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

    return true;
}







// These two macros MUST appear consecutively. One opens a block, the other
// closes it. This is required because of the local _filename variable
#define PLOT_MAKE_FILENAME(fmt, ...)                                    \
    {                                                                   \
        char _filename[1024];                                           \
        if( (int)sizeof(_filename) <= snprintf(_filename, sizeof(_filename), \
                                               fmt, ##__VA_ARGS__) )    \
        {                                                               \
            MSG("sizeof(filename) exceeded. Giving up making the plot"); \
            return false;                                               \
        }

#define PLOT(block, fmt, ...) {                                         \
        char cmd[65536];                                                \
        FILE* fp;                                                       \
                                                                        \
        if( (int)sizeof(cmd) <= snprintf(cmd, sizeof(cmd),              \
                                         "feedgnuplot --hardcopy '%s' " fmt, \
                                         _filename,                     \
                                         ##__VA_ARGS__) )               \
        {                                                               \
            MSG("sizeof(cmd) exceeded. Giving up making the plot");     \
            return false;                                               \
        }                                                               \
                                                                        \
        fp = popen(cmd, "w");                                           \
        if(fp == NULL)                                                  \
        {                                                               \
            MSG("popen(feedgnuplot ...) failed");                       \
            return false;                                               \
        }                                                               \
                                                                        \
        block;                                                          \
                                                                        \
        int _result = pclose(fp);                                       \
        if(_result < 0)                                                 \
        {                                                               \
            MSG("pclose() failed. Giving up making the plot");          \
            return false;                                               \
        }                                                               \
        if(_result > 0)                                                 \
        {                                                               \
            MSG("feedgnuplot failed. Giving up making the plot");       \
            return false;                                               \
        }                                                               \
        MSG_IF_VERBOSE("Wrote '%s'", _filename);                        \
    }                                                                   \
}







static
bool boardcenter_normal__camera(// out
                           mrcal_point3_t*            pboardcenter_camera,
                           mrcal_point3_t*            boardnormal_camera,
                           // in
                           const double*              Rt_camera_board,
                           const mrcal_cameramodel_t* model,
                           const mrcal_point2_t*      observations,
                           const int                  object_height_n,
                           const int                  object_width_n,
                           const double               object_spacing,
                           bool verbose)
{
    // diagnostics
    if(false)
    {
        static int plot_seq = 0;

        PLOT_MAKE_FILENAME("/tmp/%s-%03d.gp",
                           __func__, plot_seq);

        PLOT( ({ double rms_error = 0.;
                 for(int i=0; i<object_height_n; i++)
                     for(int j=0; j<object_width_n; j++)
                     {
                         mrcal_point3_t p = {.x = (double)j*object_spacing,
                                             .y = (double)i*object_spacing,
                                             .z = 0.};

                         mrcal_transform_point_Rt(p.xyz, NULL,NULL,
                                                  Rt_camera_board, p.xyz);

                         mrcal_point2_t q;
                         mrcal_project(&q,NULL,NULL,
                                       &p, 1,
                                       &model->lensmodel,
                                       model->intrinsics);

                         const mrcal_point2_t* q_observation =
                             &observations[i*object_width_n + j];
                         fprintf(fp,
                                 "%f observation %f\n"
                                 "%f fitted %f\n",
                                 q_observation->x, q_observation->y,
                                 q.x, q.y);

                         rms_error += mrcal_point2_norm2(mrcal_point2_sub(q,
                                                                          *q_observation));
                     }

                 rms_error = sqrt(rms_error / (double)(object_width_n*object_height_n));
                 MSG_IF_VERBOSE("RMS error: %.2f pixels", rms_error);
                }),

            "--domain --dataid "
            "--with 'linespoints pt 2 ps 2 lw 2' "
            "--autolegend "
            "--title 'Validation of the %d-th %s() call' ",
            plot_seq, __func__);

        plot_seq++;
    }

    // The estimate of the center of the board, in board coords
    const mrcal_point3_t pboardcenter_board =
        { .x = (object_width_n -1)/2*object_spacing,
          .y = (object_height_n-1)/2*object_spacing,
          .z = 0. };
    mrcal_transform_point_Rt(pboardcenter_camera->xyz, NULL, NULL,
                             Rt_camera_board, pboardcenter_board.xyz);
    boardnormal_camera->x = Rt_camera_board[3*0 + 2];
    boardnormal_camera->y = Rt_camera_board[3*1 + 2];
    boardnormal_camera->z = Rt_camera_board[3*2 + 2];

    // I make sure that the normal points away from the sensor; for consistency
    // with the lidar code
    if(mrcal_point3_inner(*boardnormal_camera,
                          *pboardcenter_camera) < 0)
    {
        *boardnormal_camera =
            mrcal_point3_scale(*boardnormal_camera, -1.);
    }
    return true;
}


static
bool boardcenter_normal__sensor(// out
                           mrcal_point3_t* pboardcenter_sensor,
                           mrcal_point3_t* boardnormal_sensor,
                           double*         Rt_camera_board_cache,
                           // in
                           const uint16_t                     isensor,
                           const int                          isnapshot,
                           const sensor_snapshot_segmented_t* snapshot,
                           const int                          Nlidars,
                           const int                          Ncameras,
                           const mrcal_cameramodel_t*const*   models,
                           const int                          object_height_n,
                           const int                          object_width_n,
                           const double                       object_spacing,
                           bool verbose)
{
    if(isensor < Nlidars)
    {
        // this is a lidar
        const int ilidar = isensor;

        const points_and_plane_full_t* lidar_scan = &snapshot->lidar_scans[ilidar];
        if(lidar_scan->points == NULL)
            return false;

        mrcal_point3_from_clc_point3f(pboardcenter_sensor, &lidar_scan->plane.p_mean);
        mrcal_point3_from_clc_point3f(boardnormal_sensor, &lidar_scan->plane.n);
    }
    else
    {
        // this is a camera
        const int icamera = isensor - Nlidars;

        const mrcal_point2_t* chessboard_corners = snapshot->chessboard_corners[icamera];
        if(chessboard_corners == NULL)
            return false;

        double* Rt_camera_board = &Rt_camera_board_cache[ (isnapshot*Ncameras + icamera) *4*3];
        if(!fit_Rt_camera_board_withcache(// out
                                          Rt_camera_board,
                                          // in
                                          models[icamera],
                                          chessboard_corners,
                                          object_height_n,
                                          object_width_n,
                                          object_spacing))
            return false;

        if(!boardcenter_normal__camera(// out
                                  pboardcenter_sensor,
                                  boardnormal_sensor,
                                  // in
                                  Rt_camera_board,
                                  models[icamera],
                                  chessboard_corners,
                                  object_height_n,
                                  object_width_n,
                                  object_spacing,
                                  verbose))
            return false;
    }

    return true;
}

static
bool print_sensor_points(// out
                         FILE* fp,
                         // in
                         const int                          isnapshot,
                         const char*                        id_pair,
                         const uint16_t                     isensor,
                         const sensor_snapshot_segmented_t* snapshot,
                         const double*                      Rt_camera_board_cache,
                         const int                          Nlidars,
                         const int                          Ncameras,
                         const mrcal_point3_t*              chessboard_points_ref,
                         const int                          object_height_n,
                         const int                          object_width_n)
{
    char id[32];
    snprintf(id, sizeof(id), "isnapshot=%03d-%s", isnapshot, id_pair);

    if(isensor < Nlidars)
    {
        // this is a lidar
        const int ilidar = isensor;

        const points_and_plane_full_t* lidar_scan = &snapshot->lidar_scans[ilidar];
        if(lidar_scan->points == NULL)
            return false;

        for(int i=0; i<(int)lidar_scan->n; i++)
        {
            const clc_point3f_t* p =
                (lidar_scan->ipoint != NULL) ?
                &lidar_scan->points[ lidar_scan->ipoint[i] ] :
                &lidar_scan->points[i];
            fprintf(fp,
                    "%f %f %s %f\n",
                    p->x, p->y, id, p->z);
        }
    }
    else
    {
        // this is a camera
        const int icamera = isensor - Nlidars;

        const double* Rt_camera_board =
            &Rt_camera_board_cache[ (isnapshot*Ncameras + icamera) *4*3];

        int N = object_height_n*object_width_n;

        for(int i=0; i<object_height_n*object_width_n; i++)
        {
            mrcal_point3_t p;
            mrcal_transform_point_Rt(p.xyz, NULL,NULL,
                                     Rt_camera_board, chessboard_points_ref[i].xyz);
            fprintf(fp,
                    "%f %f %s %f\n",
                    p.x, p.y, id, p.z);
        }
    }

    return true;
}

// Estimate the RELATIVE transform between two sensors
static
bool align_point_clouds(// out
                        double* Rt01,
                        double* Rt_camera_board_cache,
                        // in
                        const uint16_t isensor1,
                        const uint16_t isensor0,

                        const sensor_snapshot_segmented_t* snapshots,
                        const int                          Nsnapshots,
                        const uint64_t*                    bitarray_snapshots_selected,
                        const int                          Nlidars,
                        const int                          Ncameras,
                        const mrcal_cameramodel_t*const*   models,
                        const int                          object_height_n,
                        const int                          object_width_n,
                        const double                       object_spacing,
                        const double                       fit_seed_position_err_threshold,
                        const double                       fit_seed_cos_angle_err_threshold,
                        bool verbose)
{
    // Nsnapshots is the max number I will need
    // These are double instead of float, since that's what the alignment code
    // uses
    mrcal_point3_t points0 [Nsnapshots];
    mrcal_point3_t points1 [Nsnapshots];
    mrcal_point3_t normals0[Nsnapshots];
    mrcal_point3_t normals1[Nsnapshots];
    int isnapshot_fit[Nsnapshots]; // for diagnostics
    int Nfit_snapshot = 0;

    // to pacify the compiler
    normals0[0].x = 0;
    normals1[0].x = 0;

    // Loop through all the observations
    LOOP_SNAPSHOT()
    {
        LOOP_SNAPSHOT_HEADER(,const);

        if(!boardcenter_normal__sensor(// out
                                  &points0[Nfit_snapshot],
                                  &normals0[Nfit_snapshot],
                                  Rt_camera_board_cache,
                                  // in
                                  isensor0,
                                  isnapshot,
                                  &snapshots[isnapshot],
                                  Nlidars,
                                  Ncameras,
                                  models,
                                  object_height_n,
                                  object_width_n,
                                  object_spacing,
                                  verbose) ||
           !boardcenter_normal__sensor(// out
                                  &points1[Nfit_snapshot],
                                  &normals1[Nfit_snapshot],
                                  Rt_camera_board_cache,
                                  // in
                                  isensor1,
                                  isnapshot,
                                  &snapshots[isnapshot],
                                  Nlidars,
                                  Ncameras,
                                  models,
                                  object_height_n,
                                  object_width_n,
                                  object_spacing,
                                  verbose))
        {
            continue;
        }

        isnapshot_fit[Nfit_snapshot] = isnapshot;
        Nfit_snapshot++;
    }

    // If I had lots of points, I'd do a procrustes fit, and I'd be done. But
    // I have few points, so I do this in two steps:
    // - I align the normals to get a high-confidence rotation
    // - I lock down this rotation, and find the best translation
    if(!mrcal_align_procrustes_vectors_R01(// out
                                           Rt01,
                                           // in
                                           Nfit_snapshot,
                                           normals0[0].xyz,
                                           normals1[0].xyz,
                                           NULL))
    {
        MSG("mrcal_align_procrustes_vectors_R01() failed: couldn't align sensors %d and %d: available data is likely insufficient",
            isensor0, isensor1);
        return false;
    }

    // diagnostics
    if(false)
    {
        char sensor0_name[32];
        char sensor1_name[32];
        if(isensor0 < Nlidars) snprintf(sensor0_name,sizeof(sensor0_name), "lidar-%03d",isensor0);
        else                   snprintf(sensor0_name,sizeof(sensor0_name), "camera-%03d",isensor0-Nlidars);
        if(isensor1 < Nlidars) snprintf(sensor1_name,sizeof(sensor1_name), "lidar-%03d",isensor1);
        else                   snprintf(sensor1_name,sizeof(sensor1_name), "camera-%03d",isensor1-Nlidars);

        PLOT_MAKE_FILENAME("/tmp/%s-rotated.gp", __func__);
        PLOT( ({ for(int i=0; i<Nfit_snapshot; i++)
                    {
                        const int isnapshot = isnapshot_fit[i];

                        char id[32];
                        sprintf(id, "isnapshot=%03d", isnapshot);

                        mrcal_point3_t n1_sensor0;
                        mrcal_rotate_point_R(n1_sensor0.xyz, NULL,NULL,
                                             Rt01, normals1[i].xyz);

                        fprintf(fp,
                                "0 0 %s 0 %f %f %f\n"
                                "0 0 %s 0 %f %f %f\n",
                                id, normals0  [i].x,     normals0  [i].y,     normals0  [i].z,
                                id, n1_sensor0   .x*1.5, n1_sensor0   .y*1.5, n1_sensor0   .z*1.5);
                    }
                }),

            "--3d --domain --dataid "
            "--square "
            "--autolegend "
            "--with vectors --tuplesizeall 6 "
            "--title 'Aligned board normals in the sensor0 coordinate system; sensors: (%s,%s)' "
            "--xlabel x --ylabel y --zlabel z ",
            sensor0_name,sensor1_name);

        PLOT_MAKE_FILENAME("/tmp/%s-unrotated.gp", __func__);
        PLOT( ({

                 int N = 0;
                 if(Ncameras > 0)
                     N = object_height_n*object_width_n;
                 mrcal_point3_t chessboard_points_ref[N];
                 if(Ncameras > 0)
                     clc_ref_calibration_object(chessboard_points_ref, object_height_n, object_width_n, object_spacing);

                 for(int i=0; i<Nfit_snapshot; i++)
                 {
                     const int isnapshot = isnapshot_fit[i];

                     // plot the normals
                     fprintf(fp,
                             "0 0 sensor0 0 %f %f %f\n"
                             "%f %f label %f %d\n"
                             "0 0 sensor1 0 %f %f %f\n"
                             "%f %f label %f %d\n",
                             normals0[i].x,         normals0[i].y,         normals0[i].z,
                             normals0[i].x*1.1,     normals0[i].y*1.1,     normals0[i].z*1.1,     isnapshot,
                             normals1[i].x*1.5,     normals1[i].y*1.5,     normals1[i].z*1.5,
                             normals1[i].x*1.5*1.1, normals1[i].y*1.5*1.1, normals1[i].z*1.5*1.1, isnapshot);

                     print_sensor_points(fp,
                                         isnapshot,
                                         "sensor0",
                                         isensor0,
                                         &snapshots[isnapshot],
                                         Rt_camera_board_cache,
                                         Nlidars,
                                         Ncameras,
                                         chessboard_points_ref,
                                         object_height_n,
                                         object_width_n);
                     print_sensor_points(fp,
                                         isnapshot,
                                         "sensor1",
                                         isensor1,
                                         &snapshots[isnapshot],
                                         Rt_camera_board_cache,
                                         Nlidars,
                                         Ncameras,
                                         chessboard_points_ref,
                                         object_height_n,
                                         object_width_n);
                 }
                }),

            "--3d --domain --dataid "
            "--square "
            "--autolegend "
            "--with points --tuplesizeall 3 "
            "--style sensor0 'with vectors' --tuplesize sensor0 6 "
            "--style sensor1 'with vectors' --tuplesize sensor1 6 "
            "--style label   'with labels'  --tuplesize label   4 "
            "--title 'Aligned board normals in their local coordinate system: sensors: (%s,%s)' "
            "--xlabel x --ylabel y --zlabel z ",
            sensor0_name,sensor1_name);
    }


    // We computed a rotation. It should fit the data decently well. If it
    // doesn't, something is broken, and we should complain
    for(int i=0; i<Nfit_snapshot; i++)
    {
        const int isnapshot = isnapshot_fit[i];

#if 0
        // Print the first segmentation point. Used to make sure that the
        // segmentation for this specific snapshot was done properly. We have
        // this isnapshot (it is filtered); the preceding console output tells
        // us which input (unfiltered) snapshot this corresponsds to. This is a
        // rosbag. We can then segment this bag ourselves, and get a
        // visualization of the segmentation, and we can look at the first
        // point, which should match what we get here. For instance:
        //
        //     $ ./lidar-segmentation-test.py \
        //         /vl_points_0 \
        //         lidar-34.bag
        //
        //     In [1]: segmentation['points'][0][0]
        //     Out[1]: array([ 3.4646409 , -0.1949154 , -0.20211124], dtype=float32)
        if(isnapshot == 12)
        {
            int ilidar;
            if(isensor0 < Nlidars) ilidar = isensor0;
            else if(isensor1 < Nlidars) ilidar = isensor1;
            else assert(0);

            if(snapshots[isnapshot].lidar_scans[ilidar].ipoint != NULL)
            {
                int i0 = snapshots[isnapshot].lidar_scans[ilidar].ipoint[0];
                const clc_point3f_t* p = &snapshots[isnapshot].lidar_scans[ilidar].points[i0];
                MSG("p0 = %f %f %f", p->x, p->y, p->z);
            }
            else
            {
                const clc_point3f_t* p = &snapshots[isnapshot].lidar_scans[ilidar].points[0];
                MSG("p0 = %f %f %f", p->x, p->y, p->z);
            }
        }
#endif

        mrcal_point3_t normals0_validation;
        mrcal_rotate_point_R(normals0_validation.xyz, NULL, NULL,
                             Rt01,
                             normals1[i].xyz);

        const double cos_err = mrcal_point3_inner(normals0_validation, normals0[i]);

        // This isn't the same metric as the checks in fit_seed(), and these are
        // warnings, while the checks in fit_seed() are errors. So I complain
        // HERE only if the errors are really large. I want the threshold to be
        // 2x the error. th -> 2th. costh ~ 1 - th^2/2 -> cos(2th) ~ costh*costh
        // - sinth*sinth ~ costh*costh
        if(cos_err < fit_seed_cos_angle_err_threshold*fit_seed_cos_angle_err_threshold)
            MSG("WARNING: inconsistent seed rotation aligning sensors (%d,%d), for isnapshot=%d: th=%.1f deg. Most likely this is wrong, and we're about to fail the fit_seed() validation",
                isensor0, isensor1, isnapshot,
                acos(cos_err) * 180./M_PI);
        else
            MSG_IF_VERBOSE("Seed rotation aligning sensors (%d,%d), for isnapshot=%d: th=%.1f deg",
                           isensor0, isensor1, isnapshot,
                           acos(cos_err) * 180./M_PI);
    }

    // Now the translation. R01 x1 + t01 ~ x0
    // -> t01 ~ x0 - R01 x1
    //
    // I now have an estimate for t01 for each observation. Ideally they should
    // be self-consistent, so I compute the mean:
    // -> t01 = mean(x0 - R01 x1)
    *(mrcal_point3_t*)(&Rt01[9]) = (mrcal_point3_t){};
    for(int i=0; i<Nfit_snapshot; i++)
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
        Rt01[9 + j] /= (double)Nfit_snapshot;

    // We computed a seed translation as a mean of candidate translations. They
    // should have been self-consistent, and each one should be close to the
    // mean. If not, something is broken, and we should complain
    for(int i=0; i<Nfit_snapshot; i++)
    {
        const int isnapshot = isnapshot_fit[i];

        mrcal_point3_t R01_x1;
        mrcal_rotate_point_R(R01_x1.xyz, NULL, NULL,
                             Rt01,
                             points1[i].xyz);
        mrcal_point3_t t01 = mrcal_point3_sub(points0[i],R01_x1);

        mrcal_point3_t t01_err = mrcal_point3_sub(t01,
                                                  *(mrcal_point3_t*)(&Rt01[9]));
        double norm2_t01_err = mrcal_point3_norm2(t01_err);

        // This isn't the same metric as the checks in fit_seed(), and these are
        // warnings, while the checks in fit_seed() are errors. So I complain
        // HERE only if the errors are really large. I want the threshold to be
        // 2x the error.
        if(norm2_t01_err > 4.*fit_seed_position_err_threshold*fit_seed_position_err_threshold)
            MSG("WARNING: inconsistent seed translation for aligning sensors (%d,%d), isnapshot=%d: mag(t01_err)=%.1f. Most likely this is wrong, and we're about to fail the fit_seed() validation",
                isensor0, isensor1, isnapshot,
                sqrt(norm2_t01_err));
        else
            MSG_IF_VERBOSE("Seed translation for aligning sensors (%d,%d), isnapshot=%d: mag(t01_err)=%.1f",
                           isensor0, isensor1, isnapshot,
                sqrt(norm2_t01_err));
    }

    return true;
}

typedef struct
{
    const sensor_snapshot_segmented_t* snapshots;
    const int                          Nsnapshots;
    const uint64_t*                    bitarray_snapshots_selected;
    const int                          Nlidars;
    const int                          Ncameras;
    const mrcal_cameramodel_t*const*   models; // Ncameras of these
    // The dimensions of the chessboard grid being detected in the images
    const int                          object_height_n;
    const int                          object_width_n;
    const double                       object_spacing;
    const double                       fit_seed_position_err_threshold;
    const double                       fit_seed_cos_angle_err_threshold;
    double*                            Rt_lidar0_lidar;
    double*                            Rt_lidar0_camera;
    double*                            Rt_camera_board_cache;
    const bool                         verbose;
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

    const sensor_snapshot_segmented_t* snapshots                        = cookie->snapshots;
    const int                          Nsnapshots                       = cookie->Nsnapshots;
    const uint64_t*                    bitarray_snapshots_selected      = cookie->bitarray_snapshots_selected;
    const int                          Nlidars                          = cookie->Nlidars;
    const int                          Ncameras                         = cookie->Ncameras;
    const mrcal_cameramodel_t*const*   models                           = cookie->models;
    const int                          object_height_n                  = cookie->object_height_n;
    const int                          object_width_n                   = cookie->object_width_n;
    const double                       object_spacing                   = cookie->object_spacing;
    const double                       fit_seed_position_err_threshold  = cookie->fit_seed_position_err_threshold;
    const double                       fit_seed_cos_angle_err_threshold = cookie->fit_seed_cos_angle_err_threshold;
    const bool                         verbose                          = cookie->verbose;

    double*                            Rt_lidar0_lidar            = cookie->Rt_lidar0_lidar;
    double*                            Rt_lidar0_camera           = cookie->Rt_lidar0_camera;
    double*                            Rt_camera_board_cache      = cookie->Rt_camera_board_cache;
    double Rt01[4*3];
    if(!align_point_clouds(// out
                           Rt01,
                           Rt_camera_board_cache,
                           // in
                           isensor1,
                           isensor0,
                           snapshots,
                           Nsnapshots,
                           bitarray_snapshots_selected,
                           Nlidars,
                           Ncameras,
                           models,
                           object_height_n,
                           object_width_n,
                           object_spacing,
                           fit_seed_position_err_threshold,
                           fit_seed_cos_angle_err_threshold,
                           verbose))
        return false;

    if(isensor1 >= Nlidars)
    {
        const int icamera1 = isensor1 - Nlidars;

        if(isensor0 >= Nlidars)
        {
            const int icamera0 = isensor0 - Nlidars;

            if(Rt_uninitialized(&Rt_lidar0_camera[icamera0*4*3]))
            {
                MSG("Computing pose of camera %d from camera %d, but it is not initialized",
                    icamera1, icamera0);
                return false;
            }
            mrcal_compose_Rt(// out
                             &Rt_lidar0_camera[icamera1*4*3],
                             // in
                             &Rt_lidar0_camera[icamera0*4*3],
                             Rt01);
        }
        else
        {
            const int ilidar0 = isensor0;

            if(ilidar0 == 0)
                // from the reference
                memcpy(&Rt_lidar0_camera[icamera1*4*3],
                       Rt01,
                       4*3*sizeof(double));
            else
            {
                if(Rt_uninitialized(&Rt_lidar0_lidar[(ilidar0-1)*4*3]))
                {
                    MSG("Computing pose of camera %d from lidar %d, but it is not initialized",
                        icamera1, ilidar0);
                    return false;
                }
                mrcal_compose_Rt(// out
                                 &Rt_lidar0_camera[icamera1*4*3],
                                 // in
                                 &Rt_lidar0_lidar[ (ilidar0-1)*4*3 ],
                                 Rt01);
            }
        }
    }
    else
    {
        const int ilidar1 = isensor1;

        if(isensor0 >= Nlidars)
        {
            const int icamera0 = isensor0 - Nlidars;

            if(Rt_uninitialized(&Rt_lidar0_camera[icamera0*4*3]))
            {
                MSG("Computing pose of lidar %d from camera %d, but it is not initialized",
                    ilidar1, icamera0);
                return false;
            }
            mrcal_compose_Rt(// out
                             &Rt_lidar0_lidar[ (ilidar1-1)*4*3 ],
                             // in
                             &Rt_lidar0_camera[icamera0*4*3],
                             Rt01);
        }
        else
        {
            const int ilidar0 = isensor0;
            if(ilidar0 == 0)
                // from the reference
                memcpy(&Rt_lidar0_lidar[ (ilidar1-1)*4*3 ],
                       Rt01,
                       4*3*sizeof(double));
            else
            {
                if(Rt_uninitialized(&Rt_lidar0_lidar[(ilidar0-1)*4*3]))
                {
                    MSG("Computing pose of lidar %d from lidar %d, but it is not initialized",
                        ilidar1, ilidar0);
                    return false;
                }
                mrcal_compose_Rt(// out
                                 &Rt_lidar0_lidar[ (ilidar1-1)*4*3 ],
                                 // in
                                 &Rt_lidar0_lidar[ (ilidar0-1)*4*3 ],
                                 Rt01);
            }
        }
    }
    return true;
}

static bool
fit_seed(// in/out
         // if(use_given_seed_geometry) then we already have Rt_lidar0_lidar and
         // Rt_lidar0_camera. We still need to compute Rt_lidar0_board
         double* Rt_lidar0_board,  // Nsnapshots poses ( (4,3) Rt arrays ) of these to fill
         double* Rt_lidar0_lidar,  // Nlidars-1 poses ( (4,3) Rt arrays ) of these to fill (lidar0 not included)
         double* Rt_lidar0_camera, // Ncameras  poses ( (4,3) Rt arrays ) of these to fill
         bool use_given_seed_geometry,

         // in
         const sensor_snapshot_segmented_t* snapshots,
         const int                          Nsnapshots,
         uint64_t*                          bitarray_snapshots_selected,
         // Nsnapshots*Ncameras arrays of shape (4,3)
         double*                            Rt_camera_board_cache,

         const unsigned int Nlidars,
         const unsigned int Ncameras,
         const mrcal_cameramodel_t*const* models, // Ncameras of these
         // The dimensions of the chessboard grid being detected in the images
         const int object_height_n,
         const int object_width_n,
         const double object_spacing,
         const double fit_seed_position_err_threshold,
         const double fit_seed_cos_angle_err_threshold,
         bool verbose)
{
    if(!use_given_seed_geometry)
    {
        // compute Rt_lidar0_lidar and Rt_lidar0_camera

        const int Nsensors = Ncameras + Nlidars;
        const int Nshared_observation_counts = pairwise_N(Nsensors);
        uint16_t shared_observation_counts[Nshared_observation_counts];
        connectivity_matrix(// out
                            shared_observation_counts,
                            // in
                            snapshots,
                            Nsnapshots,
                            bitarray_snapshots_selected,
                            Ncameras,
                            Nlidars);

        if(verbose)
        {
            MSG("Sensor shared-observations matrix for Nlidars=%d followed by Ncameras=%d:",
                Nlidars, Ncameras);
            print_full_symmetric_matrix_from_upper_triangle(shared_observation_counts,
                                                            Nsensors);
        }

        // For the purposes of seeding I ignore any links with too-few
        // connections. At least 3 is a hard requirement: the procrustes solve
        // will otherwise not converge, and mrcal_traverse_sensor_links() will
        // fail. And I ask for a few more for redundancy. I ignore links by
        // setting them to 0; mrcal_traverse_sensor_links() will not try to use
        // those connections at all then, and opt for multi-hopping
        const int Nlinks_min = 4;
        for(int i=0; i<Nshared_observation_counts; i++)
            if(shared_observation_counts[i] < Nlinks_min)
                shared_observation_counts[i] = 0;
        if(verbose)
        {
            MSG("Sensor shared-observations matrix for Nlidars=%d followed by Ncameras=%d, after removing connections worse than Nlinks_min=%d:",
                Nlidars, Ncameras, Nlinks_min);
            print_full_symmetric_matrix_from_upper_triangle(shared_observation_counts,
                                                            Nsensors);
        }

        cb_sensor_link_cookie_t cookie =
            {
                .snapshots                        = snapshots,
                .Nsnapshots                       = Nsnapshots,
                .bitarray_snapshots_selected      = bitarray_snapshots_selected,
                .Nlidars                          = Nlidars,
                .Ncameras                         = Ncameras,
                .models                           = models,
                .object_height_n                  = object_height_n,
                .object_width_n                   = object_width_n,
                .object_spacing                   = object_spacing,
                .fit_seed_position_err_threshold  = fit_seed_position_err_threshold,
                .fit_seed_cos_angle_err_threshold = fit_seed_cos_angle_err_threshold,
                .Rt_lidar0_lidar                  = Rt_lidar0_lidar,
                .Rt_lidar0_camera                 = Rt_lidar0_camera,
                .Rt_camera_board_cache            = Rt_camera_board_cache,
                .verbose                          = verbose
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
        bool any_connections_missing = false;
        for(unsigned int i=0; i<Ncameras; i++)
            if( Rt_uninitialized(&Rt_lidar0_camera[i*4*3]))
            {
                MSG("ERROR: Don't have complete observations overlap: camera %d not connected",
                    i);
                any_connections_missing = true;
            }
        for(unsigned int i=0; i<Nlidars-1; i++)
            if( Rt_uninitialized(&Rt_lidar0_lidar[i*4*3]))
            {
                MSG("ERROR: Don't have complete observations overlap: lidar %d not connected",
                    i+1);
                any_connections_missing = true;
            }
        if(any_connections_missing)
            return false;
    }

    // All the sensor-sensor transforms computed. I compute the pose of the
    // boards
    if(!compute_board_poses(// out
                            Rt_lidar0_board,
                            Rt_camera_board_cache,
                            // in
                            snapshots,
                            Nsnapshots,
                            bitarray_snapshots_selected,
                            Ncameras,
                            Nlidars,
                            Rt_lidar0_lidar,
                            Rt_lidar0_camera,
                            models,
                            object_height_n,
                            object_width_n,
                            object_spacing,
                            verbose))
    {
        MSG("compute_board_poses() failed");
        return false;
    }

    // And now I confirm to make sure the seed is reasonable. At each instant in
    // time I make sure that all sensors are observing the board at roughly the
    // same location, orientation. It will be rough, since here we're checking a
    // seed solve. Nevertheless, anything that fails is flagged an an outlier,
    // and the whole snapshot is thrown away
    LOOP_SNAPSHOT()
    {
        LOOP_SNAPSHOT_HEADER(,const);

        const mrcal_point3_t n_lidar0_should = {.x = Rt_lidar0_board[isnapshot*4*3 + 0*3 + 2],
                                                .y = Rt_lidar0_board[isnapshot*4*3 + 1*3 + 2],
                                                .z = Rt_lidar0_board[isnapshot*4*3 + 2*3 + 2]};

        mrcal_point3_t p0_lidar0_should;
        if(Ncameras > 0)
        {
            // We have cameras; we know where the center of the board is
            const mrcal_point3_t pboardcenter_board =
                { .x = (object_width_n -1)/2*object_spacing,
                  .y = (object_height_n-1)/2*object_spacing,
                  .z = 0. };

            mrcal_transform_point_Rt(p0_lidar0_should.xyz, NULL, NULL,
                                     &Rt_lidar0_board[isnapshot*4*3], pboardcenter_board.xyz);
        }
        else
        {
            // No cameras; we don't know where the center of the board
            // is. We leave it at 0
            p0_lidar0_should = *(mrcal_point3_t*)(&Rt_lidar0_board[isnapshot*4*3 + 3*3]);
        }


        for(unsigned int ilidar=0; ilidar<Nlidars; ilidar++)
        {
            if(snapshot->lidar_scans[ilidar].points == NULL)
                continue;

            mrcal_point3_t n_lidar0_observed;
            mrcal_point3_t p0_lidar0_observed;

            mrcal_point3_from_clc_point3f(&n_lidar0_observed,
                                          &snapshot->lidar_scans[ilidar].plane.n);
            mrcal_point3_from_clc_point3f(&p0_lidar0_observed,
                                          &snapshot->lidar_scans[ilidar].plane.p_mean);
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

            const mrcal_point3_t p0_err = mrcal_point3_sub(p0_lidar0_should, p0_lidar0_observed);
            const double p0_err_mag = mrcal_point3_mag(p0_err);

            bool validation_failed_here =
                p0_err_mag > fit_seed_position_err_threshold ||
                cos_err    < fit_seed_cos_angle_err_threshold;

            MSG_IF_VERBOSE("%sisnapshot=%d ilidar=%d th_err_deg=%.2f p0_err_mag=%.2f",
                           validation_failed_here ? "FAILED: " : "",
                           isnapshot, ilidar, acos(cos_err) * 180. / M_PI, p0_err_mag);
            if(validation_failed_here)
            {
                MSG_IF_VERBOSE("Throwing out this snapshot as an outlier");
                bitarray64_clear(bitarray_snapshots_selected, isnapshot);

                // I don't NEED to continue here, and a "break" would be fine.
                // But the extra diagnostics I get without the "break" are
                // valuable, so I keep going
            }
        }

        for(unsigned int icamera=0; icamera<Ncameras; icamera++)
        {
            if(!bitarray64_check(bitarray_snapshots_selected, isnapshot))
                // just threw this out as an outlier
                break;

            if(snapshot->chessboard_corners[icamera] == NULL)
                continue;

            for(int i=0; i<object_height_n; i++)
                for(int j=0; j<object_width_n; j++)
                {
                    mrcal_point3_t p = {.x = (double)j*object_spacing,
                                        .y = (double)i*object_spacing,
                                        .z = 0.};

                    mrcal_transform_point_Rt(p.xyz, NULL, NULL,
                                             &Rt_lidar0_board[isnapshot*4*3],
                                             p.xyz);
                    mrcal_transform_point_Rt_inverted(p.xyz, NULL, NULL,
                                                      &Rt_lidar0_camera[icamera*4*3],
                                                      p.xyz);
                    mrcal_point2_t q;
                    mrcal_project(&q, NULL, NULL,
                                  &p, 1,
                                  &models[icamera]->lensmodel,
                                  models[icamera]->intrinsics);

                    double errsq =
                        mrcal_point2_norm2( mrcal_point2_sub(q,
                                                             snapshot->chessboard_corners[icamera][i*object_width_n + j]));
#warning "unhardcode"
                    bool validation_failed_here = errsq > 10.*10.;

                    if(validation_failed_here)
                    {
                        MSG("FAILED: isnapshot=%d icamera=%d i=%d j=%d reprojection error too high: %.1f pixels",
                            isnapshot, icamera, i,j, sqrt(errsq));

                        // move on to the next camera
                        i = INT_MAX-1;
                        j = INT_MAX-1;

                        MSG_IF_VERBOSE("Throwing out this snapshot as an outlier");
                        bitarray64_clear(bitarray_snapshots_selected, isnapshot);
                    }
                }
        }
    }

    return true;
}


typedef struct
{
    unsigned int Ncameras;
    unsigned int Nlidars;

    const sensor_snapshot_segmented_t* snapshots;
    const int Nsnapshots;
    const uint64_t* bitarray_snapshots_selected;

    const mrcal_cameramodel_t*const* models; // Ncameras of these
    // The dimensions of the chessboard grid being detected in the images
    const int object_height_n;
    const int object_width_n;
    const double object_spacing;

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
static int state_index_board(const int _isnapshot,
                             const callback_context_t* ctx)
{
    if(bitarray64_check_all_clear(ctx->bitarray_snapshots_selected, ctx->Nsnapshots))
    {
        MSG("ERROR: bitarray_snapshots_selected is all clear. Did we forget to initialize it?");
        return -1;
    }
#warning "this is slow. Cache this function, or make the callers use it cumulatively"
    int isnapshot_state = 0;
    LOOP_SNAPSHOT(ctx->)
    {
        LOOP_SNAPSHOT_HEADER(ctx->,const);

        if(isnapshot >= _isnapshot) break;
        isnapshot_state++;
    }

    return
        num_states_lidars(ctx) +
        num_states_cameras(ctx) +
        isnapshot_state * 6;
}
static int num_states_boards(const callback_context_t* ctx)
{
    if(bitarray64_check_all_clear(ctx->bitarray_snapshots_selected, ctx->Nsnapshots))
    {
        MSG("ERROR: bitarray_snapshots_selected is all clear. Did we forget to initialize it?");
        return -1;
    }
#warning "this is slow. Cache this function, or make the callers use it cumulatively"
    int isnapshot_state = 0;
    LOOP_SNAPSHOT(ctx->)
    {
        LOOP_SNAPSHOT_HEADER(ctx->,const);

        isnapshot_state++;
    }

    return isnapshot_state*6;
}

static int num_states(const callback_context_t* ctx)
{
    return
        num_states_lidars(ctx) +
        num_states_cameras(ctx) +
        num_states_boards(ctx);
}


static int measurement_index_lidar(const unsigned int _isnapshot,
                                   const unsigned int _ilidar,
                                   const callback_context_t* ctx)
{
    int imeasurement = 0;

    LOOP_SNAPSHOT(ctx->)
    {
        LOOP_SNAPSHOT_HEADER(ctx->,const);
        for(unsigned int ilidar=0; ilidar<ctx->Nlidars; ilidar++)
        {
            if(_isnapshot <= isnapshot && _ilidar == ilidar)
                return imeasurement;

            const points_and_plane_full_t* lidar_scan = &snapshot->lidar_scans[ilidar];

            if(lidar_scan->points == NULL)
                continue;

            imeasurement += lidar_scan->n;
        }
    }
    return -1;
}

static int num_measurements_lidars(const callback_context_t* ctx)
{
    int Nmeasurements = 0;

    LOOP_SNAPSHOT(ctx->)
    {
        LOOP_SNAPSHOT_HEADER(ctx->,const);
        for(unsigned int ilidar=0; ilidar<ctx->Nlidars; ilidar++)
        {
            const points_and_plane_full_t* lidar_scan = &snapshot->lidar_scans[ilidar];

            if(lidar_scan->points == NULL)
                continue;

            Nmeasurements += lidar_scan->n;
        }
    }
    return Nmeasurements;
}
static int measurement_index_camera(const unsigned int _isnapshot,
                                    const unsigned int icamera,
                                    const callback_context_t* ctx)
{
    int imeasurement = 0;

    LOOP_SNAPSHOT(ctx->)
    {
        LOOP_SNAPSHOT_HEADER(ctx->,const);
        for(int _icamera=0; _icamera<(int)ctx->Ncameras; _icamera++)
        {
            const mrcal_point2_t* chessboard_corners =
                snapshot->chessboard_corners[_icamera];
            if(chessboard_corners == NULL) continue;

            if(_isnapshot <= isnapshot && (int)icamera == _icamera)
                return
                    num_measurements_lidars(ctx) +
                    imeasurement;

            imeasurement +=
                ctx->object_width_n *
                ctx->object_height_n *
                2;
        }
    }

    return -1;
}
static int num_measurements_cameras(const callback_context_t* ctx)
{
    int Nobservations_camera = 0;

    LOOP_SNAPSHOT(ctx->)
    {
        LOOP_SNAPSHOT_HEADER(ctx->,const);
        for(int icamera=0; icamera<(int)ctx->Ncameras; icamera++)
        {
            const mrcal_point2_t* chessboard_corners =
                snapshot->chessboard_corners[icamera];
            if(chessboard_corners != NULL) Nobservations_camera++;
        }
    }

    return
        Nobservations_camera *
        ctx->object_width_n *
        ctx->object_height_n *
        2;
}
static int measurement_index_regularization(const callback_context_t* ctx)
{
    return
        num_measurements_lidars(ctx) +
        num_measurements_cameras(ctx);
}

static int num_measurements_regularization(const callback_context_t* ctx)
{
    int isnapshot_state = 0;
    LOOP_SNAPSHOT(ctx->)
    {
        LOOP_SNAPSHOT_HEADER(ctx->,const);

        isnapshot_state++;
    }

    return 6*isnapshot_state;
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

    // lidar
    LOOP_SNAPSHOT(ctx->)
    {
        LOOP_SNAPSHOT_HEADER(ctx->,const);

        for(unsigned int ilidar=0; ilidar<ctx->Nlidars; ilidar++)
        {
            const points_and_plane_full_t* lidar_scan = &snapshot->lidar_scans[ilidar];

            if(lidar_scan->points == NULL)
                continue;

            if(ilidar == 0) nnz +=   6*lidar_scan->n;
            else            nnz += 2*6*lidar_scan->n;
        }
    }

    nnz += num_measurements_cameras(ctx) * 6 * 2;
    nnz += num_measurements_regularization(ctx);

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
#define SCALE_MEASUREMENT_M                0.03   /* expected noise levels */
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
    for(unsigned int i=0; i<ctx->Ncameras; i++)
    {
        for(int j=0; j<3; j++) b[istate++] = rt_lidar0_camera[i*6 + j] / SCALE_ROTATION_CAMERA;
        for(int j=3; j<6; j++) b[istate++] = rt_lidar0_camera[i*6 + j] / SCALE_TRANSLATION_CAMERA;
    }
    LOOP_SNAPSHOT(ctx->)
    {
        LOOP_SNAPSHOT_HEADER(ctx->,const);
        for(int j=0; j<3; j++) b[istate++] = rt_lidar0_board[isnapshot*6 + j] / SCALE_ROTATION_FRAME;
        for(int j=3; j<6; j++) b[istate++] = rt_lidar0_board[isnapshot*6 + j] / SCALE_TRANSLATION_FRAME;
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
    for(unsigned int i=0; i<ctx->Ncameras; i++)
    {
        for(int j=0; j<3; j++) rt_lidar0_camera[i*6 + j] = b[istate++] * SCALE_ROTATION_CAMERA;
        for(int j=3; j<6; j++) rt_lidar0_camera[i*6 + j] = b[istate++] * SCALE_TRANSLATION_CAMERA;
    }

    // The skipped poses are reset to 0
    memset(rt_lidar0_board, 0, ctx->Nsnapshots*6*sizeof(rt_lidar0_board[0]));
    LOOP_SNAPSHOT(ctx->)
    {
        LOOP_SNAPSHOT_HEADER(ctx->,const);
        for(int j=0; j<3; j++) rt_lidar0_board[isnapshot*6 + j] = b[istate++] * SCALE_ROTATION_FRAME;
        for(int j=3; j<6; j++) rt_lidar0_board[isnapshot*6 + j] = b[istate++] * SCALE_TRANSLATION_FRAME;
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

// https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
static double randn(void)
{
    static bool have_precomputed_value;
    static double precomputed_value;
    if(have_precomputed_value)
    {
        have_precomputed_value = false;
        return precomputed_value;
    }

    double u1, u2;
    do {
        u1 = drand48();
    } while (u1 == 0.);
    u2 = drand48();

    const double mag = sqrt(-2.0 * log(u1));
    have_precomputed_value = true;
    precomputed_value = mag * cos(2.0 * M_PI * u2);
    return mag * sin(2.0 * M_PI * u2);
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
#define STORE_JACOBIAN3(col, g, scale)          \
    do                                          \
    {                                           \
        if(Jt) {                                \
            for(int _i=0; _i<3; _i++) {         \
                Jcolidx[ iJacobian ] = (col) + _i;      \
                Jval   [ iJacobian ] = (g)[_i] * (scale);       \
                iJacobian++;                    \
            }                                   \
        }                                       \
        else                                    \
            iJacobian+=3;                       \
    } while(0)



    double rt_lidar0_lidar_all [(ctx->Nlidars-1) *6];
    double rt_lidar0_camera_all[ctx->Ncameras    *6];
    double rt_lidar0_board_all [ctx->Nsnapshots  *6];

    unpack_solver_state(rt_lidar0_lidar_all,
                        rt_lidar0_camera_all,
                        rt_lidar0_board_all,
                        b,
                        ctx);

    LOOP_SNAPSHOT(ctx->)
    {
        LOOP_SNAPSHOT_HEADER(ctx->,const);

        bool any_lidar_data_here = false;
        for(unsigned int ilidar=0; ilidar<ctx->Nlidars; ilidar++)
        {
            const points_and_plane_full_t* lidar_scan = &snapshot->lidar_scans[ilidar];
            if(lidar_scan->points != NULL)
            {
                any_lidar_data_here = true;
                break;
            }
        }
        if(!any_lidar_data_here)
            continue;


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
                const points_and_plane_full_t* lidar_scan = &snapshot->lidar_scans[ilidar];

                if(lidar_scan->points == NULL)
                    continue;

                if(ctx->report_imeas)
                    MSG("isnapshot=%d ilidar=%d iMeasurement=%d",
                        isnapshot, ilidar, iMeasurement);

                if(ilidar == 0)
                {
                    // this is lidar0; it sits at the reference, and doesn't
                    // have an explicit pose or a rt_lidar0_lidar gradient
                    for(unsigned int iipoint=0; iipoint<lidar_scan->n; iipoint++)
                    {
                        mrcal_point3_t p;
                        int ipoint = (lidar_scan->ipoint != NULL) ?
                            lidar_scan->ipoint[iipoint] :
                            iipoint;
                        mrcal_point3_from_clc_point3f(&p,
                                                      &lidar_scan->points[ipoint]);


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

                    for(unsigned int iipoint=0; iipoint<lidar_scan->n; iipoint++)
                    {
                        mrcal_point3_t p;
                        int ipoint = (lidar_scan->ipoint != NULL) ?
                            lidar_scan->ipoint[iipoint] :
                            iipoint;
                        mrcal_point3_from_clc_point3f(&p,
                                                      &lidar_scan->points[ipoint]);


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
                const points_and_plane_full_t* lidar_scan = &snapshot->lidar_scans[ilidar];

                if(lidar_scan->points == NULL)
                    continue;

                if(ctx->report_imeas)
                    MSG("isnapshot=%d ilidar=%d iMeasurement=%d",
                        isnapshot, ilidar, iMeasurement);

                if(ilidar == 0)
                {
                    // this is lidar0; it sits at the reference, and doesn't
                    // have an explicit pose or a rt_lidar0_lidar gradient
                    for(unsigned int iipoint=0; iipoint<lidar_scan->n; iipoint++)
                    {
                        mrcal_point3_t p;
                        int ipoint = (lidar_scan->ipoint != NULL) ?
                            lidar_scan->ipoint[iipoint] :
                            iipoint;
                        mrcal_point3_from_clc_point3f(&p,
                                                      &lidar_scan->points[ipoint]);

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

                    for(unsigned int iipoint=0; iipoint<lidar_scan->n; iipoint++)
                    {
                        mrcal_point3_t p;
                        int ipoint = (lidar_scan->ipoint != NULL) ?
                            lidar_scan->ipoint[iipoint] :
                            iipoint;
                        mrcal_point3_from_clc_point3f(&p,
                                                      &lidar_scan->points[ipoint]);

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
    int istate_board = state_index_board(0, ctx);

    LOOP_SNAPSHOT(ctx->)
    {
        LOOP_SNAPSHOT_HEADER(ctx->,const);

        int istate_camera = state_index_camera(0, ctx);

        for(int icamera=0;
            icamera<(int)ctx->Ncameras;
            icamera++, istate_camera += 6)
        {
            const mrcal_point2_t* chessboard_corners =
                snapshot->chessboard_corners[icamera];
            if(chessboard_corners == NULL)
                continue;

            if(ctx->report_imeas)
                MSG("isnapshot=%d icamera=%d iMeasurement=%d",
                    isnapshot, icamera, iMeasurement);

            double rt_camera_board[6];
            double drcb_drl0c[3*3];
            double drcb_drl0b[3*3];
            double dtcb_drl0c[3*3];
            double dtcb_drl0b[3*3];
            double dtcb_dtl0c[3*3];
            double dtcb_dtl0b[3*3];
            mrcal_compose_rt_inverted0( rt_camera_board,
                                        drcb_drl0c,drcb_drl0b,dtcb_drl0c,dtcb_drl0b,dtcb_dtl0c,dtcb_dtl0b,
                                        &rt_lidar0_camera_all[icamera*6],
                                        &rt_lidar0_board_all[isnapshot*6] );

            for(int i=0; i<ctx->object_height_n; i++)
                for(int j=0; j<ctx->object_width_n; j++)
                {
                    mrcal_point2_t q;
                    mrcal_point3_t pref = {.x = (double)j*ctx->object_spacing,
                                           .y = (double)i*ctx->object_spacing,
                                           .z = 0.};
                    mrcal_point3_t pcam;
                    double dpcam_drtcb[3*6];
                    mrcal_point3_t dq_dpcam[2];
                    mrcal_transform_point_rt(pcam.xyz, dpcam_drtcb, NULL,
                                             rt_camera_board, pref.xyz);
                    mrcal_project(&q, dq_dpcam, NULL,
                                  &pcam,1,
                                  &ctx->models[icamera]->lensmodel,
                                  ctx->models[icamera]->intrinsics);

                    for(int k=0; k<2; k++)
                    {
                        const mrcal_point3_t* dqi_dpcam = &dq_dpcam[k];
                        double dqi_drtcb[6];
                        for(int l=0; l<6; l++)
                        {
                            dqi_drtcb[l] = 0.;
                            for(int m=0;m<3;m++)
                                dqi_drtcb[l] +=
                                    dqi_dpcam->xyz[m] *
                                    dpcam_drtcb[6*m + l];
                        }

                        if(Jt) Jrowptr[iMeasurement] = iJacobian;
                        x[iMeasurement] = (q.xy[k] - chessboard_corners[i*ctx->object_width_n+j].xy[k]) / SCALE_MEASUREMENT_PX;

                        double dqi[3];

                        mul_vec3_gen33_vout(// in
                                            &dqi_drtcb[0],
                                            drcb_drl0c,
                                            // out
                                            dqi);
                        mul_vec3_gen33_vaccum(// in
                                            &dqi_drtcb[3],
                                            dtcb_drl0c,
                                            // out
                                            dqi);
                        STORE_JACOBIAN3(istate_camera,
                                        dqi,
                                        SCALE_ROTATION_CAMERA/SCALE_MEASUREMENT_PX);

                        mul_vec3_gen33_vout(// in
                                            &dqi_drtcb[3],
                                            dtcb_dtl0c,
                                            // out
                                            dqi);
                        STORE_JACOBIAN3(istate_camera+3,
                                        dqi,
                                        SCALE_TRANSLATION_CAMERA/SCALE_MEASUREMENT_PX);

                        mul_vec3_gen33_vout(// in
                                            &dqi_drtcb[0],
                                            drcb_drl0b,
                                            // out
                                            dqi);
                        mul_vec3_gen33_vaccum(// in
                                            &dqi_drtcb[3],
                                            dtcb_drl0b,
                                            // out
                                            dqi);
                        STORE_JACOBIAN3(istate_board,
                                        dqi,
                                        SCALE_ROTATION_FRAME/SCALE_MEASUREMENT_PX);

                        mul_vec3_gen33_vout(// in
                                            &dqi_drtcb[3],
                                            dtcb_dtl0b,
                                            // out
                                            dqi);
                        STORE_JACOBIAN3(istate_board+3,
                                        dqi,
                                        SCALE_TRANSLATION_FRAME/SCALE_MEASUREMENT_PX);

                        iMeasurement++;
                    }
                }
        }

        istate_board += 6;
    }


    /////// Regularization
    // The regularization lightly pulls every element of rt_lidar0_board towards
    // zero. This is necessary because LIDAR-only observations of the board have
    // only 3 DOF: the board is free to translate and yaw in its plane. I can
    // accomplish the same thing with a different T_ref_board representation for
    // LIDAR-only observations: (n,d). That disparate representation would take
    // more typing, so I don't do that just yet
    int ivar = state_index_board(0, ctx);
    LOOP_SNAPSHOT(ctx->)
    {
        LOOP_SNAPSHOT_HEADER(ctx->,const);
        double* rt_lidar0_board = &rt_lidar0_board_all[isnapshot*6];

        for(int i=0; i<3; i++)
        {
            if(Jt) Jrowptr[iMeasurement] = iJacobian;
            x[iMeasurement] = rt_lidar0_board[i] / SCALE_MEASUREMENT_REGULARIZATION_r;
            STORE_JACOBIAN(ivar,
                           1.0 *
                           SCALE_ROTATION_FRAME/SCALE_MEASUREMENT_REGULARIZATION_r);
            ivar++;
            iMeasurement++;
        }
        for(int i=0; i<3; i++)
        {
            if(Jt) Jrowptr[iMeasurement] = iJacobian;
            x[iMeasurement] = rt_lidar0_board[i+3] / SCALE_MEASUREMENT_REGULARIZATION_t;
            STORE_JACOBIAN(ivar,
                           1.0 *
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
               const callback_context_t* ctx,
               bool verbose)
{
    const int imeas_lidar_0                = measurement_index_lidar(0,0, ctx);
    const int Nmeas_lidar_observation_all  = num_measurements_lidars(ctx);
    const int imeas_camera_0               = measurement_index_camera(0,0, ctx);
    const int Nmeas_camera_observation_all = num_measurements_cameras(ctx);
    const int imeas_regularization_0       = measurement_index_regularization(ctx);
    const int Nmeas_regularization         = num_measurements_regularization(ctx);

    char*  str_measurement_indices  = NULL;
    size_t size_measurement_indices = 0;
    FILE* fp_measurement_indices = open_memstream(&str_measurement_indices,
                                                  &size_measurement_indices);

    if(fp_measurement_indices == NULL)
    {
        MSG("Couldn't open_memstream() for the measurement indices");
        return false;
    }
    const char* fmt_measurement_indices_lidar =
        "--set 'arrow from %1$d, graph 0 to %1$d, graph 1 nohead' "
        "--set 'label \"isnapshot=%2$d\\nilidar=%3$d\" at %1$d,graph 0 left front offset 0,character 2 boxed' ";
    const char* fmt_measurement_indices_camera =
        "--set 'arrow from %1$d, graph 0 to %1$d, graph 1 nohead' "
        "--set 'label \"isnapshot=%2$d\\nicamera=%3$d\" at %1$d,graph 0 left front offset 0,character 2 boxed' ";
    const char* fmt_measurement_indices_regularization =
        "--set 'arrow from %1$d, graph 0 to %1$d, graph 1 nohead' "
        "--set 'label \"regularization\" at %1$d,graph 0 left front offset 0,character 2 boxed' ";
    int iMeasurement = 0;
    LOOP_SNAPSHOT(ctx->)
    {
        LOOP_SNAPSHOT_HEADER(ctx->,const);
        for(unsigned int ilidar=0; ilidar<ctx->Nlidars; ilidar++)
        {
            const points_and_plane_full_t* lidar_scan = &snapshot->lidar_scans[ilidar];
            if(lidar_scan->points == NULL)
                continue;
            fprintf(fp_measurement_indices,
                    fmt_measurement_indices_lidar,
                    iMeasurement, isnapshot, ilidar);
            iMeasurement += lidar_scan->n;
        }
    }
    LOOP_SNAPSHOT(ctx->)
    {
        LOOP_SNAPSHOT_HEADER(ctx->,const);
        for(int icamera=0;
            icamera<(int)ctx->Ncameras;
            icamera++)
        {
            const mrcal_point2_t* chessboard_corners =
                snapshot->chessboard_corners[icamera];
            if(chessboard_corners == NULL)
                continue;
            fprintf(fp_measurement_indices,
                    fmt_measurement_indices_camera,
                    iMeasurement, isnapshot, icamera);
            iMeasurement +=
                ctx->object_height_n *
                ctx->object_width_n *
                2;
        }
    }
    fprintf(fp_measurement_indices,
            fmt_measurement_indices_regularization,
            iMeasurement);
    fclose(fp_measurement_indices);

    PLOT_MAKE_FILENAME("%s.gp", filename_base);
    PLOT( ({ // All measurements plotted using the nominal SCALE_MEASUREMENT_M. The
             // camera residuals in pixels are shown on the y2 axis for convenience
            for(int i=0; i<Nmeas_lidar_observation_all; i++)
                fprintf(fp, "%d lidar %f\n",
                        imeas_lidar_0 + i,
                        x[imeas_lidar_0 + i] * SCALE_MEASUREMENT_M);
            for(int i=0; i<Nmeas_camera_observation_all; i++)
                fprintf(fp, "%d camera %f\n",
                        imeas_camera_0 + i,
                        x[imeas_camera_0 + i] * SCALE_MEASUREMENT_M);
            for(int i=0; i<Nmeas_regularization; i++)
                fprintf(fp, "%d regularization %f\n",
                        imeas_regularization_0 + i,
                        x[imeas_regularization_0 + i] * SCALE_MEASUREMENT_M);
            }),

        "--domain --dataid "
        "--legend lidar 'LIDAR residuals' "
        "--legend camera 'Camera residuals' "
        "--legend regularization 'Regularization residuals; plotted in pixels on the left y axis' "
        "--y2 camera "
        "--with points "
        "--set 'link y2 via y*%f inverse y*%f' "
        "--ylabel  'LIDAR fit residual (m)' "
        "--y2label 'Camera fit residual (pixels)' "
        "%s",
        SCALE_MEASUREMENT_PX/SCALE_MEASUREMENT_M,
        SCALE_MEASUREMENT_M/SCALE_MEASUREMENT_PX,
        str_measurement_indices);
    free(str_measurement_indices);

    PLOT_MAKE_FILENAME("%s-histogram-lidar.gp", filename_base);
    PLOT( ({ for(int i=0; i<Nmeas_lidar_observation_all; i++)
                 fprintf(fp, "%f\n",
                         x[imeas_lidar_0 + i] * SCALE_MEASUREMENT_M);
          }),

        "--histogram 0 "
        "--binwidth %f "
        "--xmin %f --xmax %f "
        "--xlabel 'LIDAR residual (m)' "
        "--ylabel 'frequency' ",
        SCALE_MEASUREMENT_M/10.,
        -3.*SCALE_MEASUREMENT_M,
        3.*SCALE_MEASUREMENT_M);


    if(Nmeas_camera_observation_all)
    {
        PLOT_MAKE_FILENAME("%s-histogram-camera.gp", filename_base);
        PLOT( ({ for(int i=0; i<Nmeas_camera_observation_all; i++)
                     fprintf(fp, "%f\n",
                             x[imeas_camera_0 + i] * SCALE_MEASUREMENT_M);
                }),

            "--histogram 0 "
            "--binwidth %f "
            "--xmin %f --xmax %f "
            "--xlabel 'CAMERA residual (px)' "
            "--ylabel 'frequency' ",
            SCALE_MEASUREMENT_PX/10.,
            -3.*SCALE_MEASUREMENT_PX,
            3.*SCALE_MEASUREMENT_PX);
    }

    return true;
}

static void
write_axes(FILE* fp,
           const char* curveid,
           const double* Rt_ref_sensor,
           const double scale)
{
    if(Rt_ref_sensor != NULL)
    {
        fprintf(fp, "%f %f %s %f %f %f %f\n",
                Rt_ref_sensor[3*3 + 0],
                Rt_ref_sensor[3*3 + 1],
                "axes",
                Rt_ref_sensor[3*3 + 2],
                Rt_ref_sensor[0*3 + 0] * scale,
                Rt_ref_sensor[1*3 + 0] * scale,
                Rt_ref_sensor[2*3 + 0] * scale);
        fprintf(fp, "%f %f %s %f %f %f %f\n",
                Rt_ref_sensor[3*3 + 0],
                Rt_ref_sensor[3*3 + 1],
                "axes",
                Rt_ref_sensor[3*3 + 2],
                Rt_ref_sensor[0*3 + 1] * scale,
                Rt_ref_sensor[1*3 + 1] * scale,
                Rt_ref_sensor[2*3 + 1] * scale);
        fprintf(fp, "%f %f %s %f %f %f %f\n",
                Rt_ref_sensor[3*3 + 0],
                Rt_ref_sensor[3*3 + 1],
                "axes",
                Rt_ref_sensor[3*3 + 2],
                Rt_ref_sensor[0*3 + 2] * scale * 2.,// "forward": longer axis
                Rt_ref_sensor[1*3 + 2] * scale * 2.,// "forward": longer axis
                Rt_ref_sensor[2*3 + 2] * scale * 2. // "forward": longer axis
                );

        fprintf(fp, "%f %f labels %f x\n",
                Rt_ref_sensor[3*3 + 0] + Rt_ref_sensor[0*3 + 0] * scale      * 1.01,
                Rt_ref_sensor[3*3 + 1] + Rt_ref_sensor[1*3 + 0] * scale      * 1.01,
                Rt_ref_sensor[3*3 + 2] + Rt_ref_sensor[2*3 + 0] * scale      * 1.01);
        fprintf(fp, "%f %f labels %f y\n",
                Rt_ref_sensor[3*3 + 0] + Rt_ref_sensor[0*3 + 1] * scale      * 1.01,
                Rt_ref_sensor[3*3 + 1] + Rt_ref_sensor[1*3 + 1] * scale      * 1.01,
                Rt_ref_sensor[3*3 + 2] + Rt_ref_sensor[2*3 + 1] * scale      * 1.01);
        fprintf(fp, "%f %f labels %f z\n",
                Rt_ref_sensor[3*3 + 0] + Rt_ref_sensor[0*3 + 2] * scale * 2. * 1.01,
                Rt_ref_sensor[3*3 + 1] + Rt_ref_sensor[1*3 + 2] * scale * 2. * 1.01,
                Rt_ref_sensor[3*3 + 2] + Rt_ref_sensor[2*3 + 2] * scale * 2. * 1.01);

        fprintf(fp, "%f %f labels %f %s\n",
                Rt_ref_sensor[3*3 + 0] - (Rt_ref_sensor[0*3 + 0] +
                                          Rt_ref_sensor[0*3 + 1] +
                                          Rt_ref_sensor[0*3 + 2]) * scale * .1,
                Rt_ref_sensor[3*3 + 1] - (Rt_ref_sensor[1*3 + 0] +
                                          Rt_ref_sensor[1*3 + 1] +
                                          Rt_ref_sensor[1*3 + 2]) * scale * .1,
                Rt_ref_sensor[3*3 + 2] - (Rt_ref_sensor[2*3 + 0] +
                                          Rt_ref_sensor[2*3 + 1] +
                                          Rt_ref_sensor[2*3 + 2]) * scale * .1,
                curveid);
    }
    else
    {
        // No explicit Rt_ref_sensor. Use the identity
        fprintf(fp, "%f %f %s %f %f %f %f\n",
                0., 0.,
                "axes",
                0.,
                scale,0.,0.);
        fprintf(fp, "%f %f %s %f %f %f %f\n",
                0., 0.,
                "axes",
                0.,
                0.,scale,0.);
        fprintf(fp, "%f %f %s %f %f %f %f\n",
                0., 0.,
                "axes",
                0.,
                0.,0.,scale*2 // "forward": longer axis
                );

        fprintf(fp, "%f 0 labels 0  x\n",
                scale * 1.01);
        fprintf(fp, "0 %f labels 0  y\n",
                scale * 1.01);
        fprintf(fp, "0  0 labels %f z\n",
                scale * 2. * 1.01);


        fprintf(fp, "%f %f labels %f %s\n",
                - scale * .1,
                - scale * .1,
                - scale * .1,
                curveid);
    }
}

static bool
_plot_geometry(FILE* fp,

               const double* Rt_lidar0_board,  // Nsnapshots poses ( (4,3) Rt arrays ) of these to fill
               const double* Rt_lidar0_lidar,  // Nlidars-1 poses ( (4,3) Rt arrays ) of these to fill (lidar0 not included)
               const double* Rt_lidar0_camera, // Ncameras  poses ( (4,3) Rt arrays ) of these to fill

               const sensor_snapshot_segmented_t* snapshots,
               const int                          Nsnapshots,
               const uint64_t*                    bitarray_snapshots_selected,

               const unsigned int Nlidars,
               const unsigned int Ncameras,
               const int object_height_n,
               const int object_width_n,
               const double object_spacing,
               bool only_axes)
{
    const double axis_scale = 1.0;


    for(unsigned int i=0; i<Nlidars; i++)
    {
        if(i == 0)
            write_axes(fp, "lidar0", NULL, axis_scale);
        else
        {
            const double* Rt_lidar0_sensor = &Rt_lidar0_lidar[4*3*(i-1)];

            char curveid[16];
            if( (int)sizeof(curveid) <= snprintf(curveid, sizeof(curveid),
                                                 "lidar%d",
                                                 i) )
            {
                MSG("sizeof(curveid) exceeded. Giving up making the plot");
                return false;
            }

            write_axes(fp, curveid, Rt_lidar0_sensor, axis_scale);

        }
    }

    for(unsigned int i=0; i<Ncameras; i++)
    {
        const double* Rt_lidar0_sensor = &Rt_lidar0_camera[4*3*i];

        char curveid[16];
        if( (int)sizeof(curveid) <= snprintf(curveid, sizeof(curveid),
                                             "camera%d",
                                             i) )
        {
            MSG("sizeof(curveid) exceeded. Giving up making the plot");
            return false;
        }

        write_axes(fp, curveid, Rt_lidar0_sensor, axis_scale);
    }

    if(!only_axes)
    {
        LOOP_SNAPSHOT()
        {
            LOOP_SNAPSHOT_HEADER(,const);

            char curveid[Nlidars][32];
            for(unsigned int ilidar=0; ilidar<Nlidars; ilidar++)
            {
                if( (int)sizeof(curveid[0]) <= snprintf(curveid[ilidar], sizeof(curveid[0]),
                                                        "isnapshot=%d_ilidar=%d",
                                                        isnapshot,ilidar) )
                {
                    MSG("sizeof(curveidxs]) exceeded. Giving up making the plot");
                    return false;
                }
            }

            for(unsigned int ilidar=0; ilidar<Nlidars; ilidar++)
            {
                const points_and_plane_full_t* lidar_scan = &snapshot->lidar_scans[ilidar];

                if(lidar_scan->points == NULL)
                    continue;

                const double* Rt_lidar0_sensor = NULL;
                if(ilidar > 0)
                    Rt_lidar0_sensor = &Rt_lidar0_lidar[4*3*(ilidar-1)];

                for(unsigned int iipoint=0; iipoint<lidar_scan->n; iipoint++)
                {
                    mrcal_point3_t p;
                    int ipoint = (lidar_scan->ipoint != NULL) ?
                        lidar_scan->ipoint[iipoint] :
                        iipoint;
                    mrcal_point3_from_clc_point3f(&p,
                                                  &lidar_scan->points[ipoint]);

                    if(Rt_lidar0_sensor != NULL)
                        mrcal_transform_point_Rt(p.xyz, NULL, NULL,
                                                 Rt_lidar0_sensor, p.xyz);

                    fprintf(fp, "%f %f %s %f\n",
                            p.x, p.y, curveid[ilidar], p.z);
                }
            }


            // Reference board poses; plot only for those snapshots that have
            // any camera observations
            for(int icamera=0; icamera<(int)Ncameras; icamera++)
            {
                if(snapshot->chessboard_corners[icamera] == NULL)
                    continue;

                for(int i=0; i<object_height_n; i++)
                    for(int j=0; j<object_width_n; j++)
                    {
                        mrcal_point3_t pref = {.x = (double)j*object_spacing,
                                               .y = (double)i*object_spacing,
                                               .z = 0.};
                        mrcal_point3_t p;
                        mrcal_transform_point_Rt(p.xyz, NULL, NULL,
                                                 &Rt_lidar0_board[4*3*isnapshot], pref.xyz);
                        fprintf(fp, "%f %f boards-ref %f\n",
                                p.x, p.y, p.z);
                    }
                // break the line
                fprintf(fp, "nan nan boards-ref nan\n");

                break;
            }
        }
    }

    return true;
}

static bool
plot_geometry(const char* filename,

              const double* Rt_lidar0_board,  // Nsnapshots poses ( (4,3) Rt arrays ) of these to fill
              const double* Rt_lidar0_lidar,  // Nlidars-1 poses ( (4,3) Rt arrays ) of these to fill (lidar0 not included)
              const double* Rt_lidar0_camera, // Ncameras  poses ( (4,3) Rt arrays ) of these to fill

              const sensor_snapshot_segmented_t* snapshots,
              const int                          Nsnapshots,
              const uint64_t*                    bitarray_snapshots_selected,

              const unsigned int Nlidars,
              const unsigned int Ncameras,
              const int object_height_n,
              const int object_width_n,
              const double object_spacing,
              bool only_axes,
              bool verbose)
{
    bool result = false;

    PLOT_MAKE_FILENAME("%s", filename);
    PLOT( ({ result = _plot_geometry(fp,
                                     Rt_lidar0_board,
                                     Rt_lidar0_lidar,
                                     Rt_lidar0_camera,
                                     snapshots,
                                     Nsnapshots,
                                     bitarray_snapshots_selected,
                                     Nlidars,
                                     Ncameras,
                                     object_height_n,
                                     object_width_n,
                                     object_spacing,
                                     only_axes);
          }),

        "--domain --dataid --3d "
        "--square "
        "--xlabel x "
        "--ylabel y "
        "--zlabel z "
        "--title \"Camera geometry\" "
        "--autolegend "
        "--style axes \"with vectors\"  --tuplesize axes   6 "
        "--style labels \"with labels\" --tuplesize labels 4 "
        "--style boards-ref \"with lines\"  --tuplesize boards 3 "
        "--maxcurves 300 "
        "--with points --tuplesizeall 3 ");

    return result;
}

// same inputs as fit()
static bool dump_inputs(
    // on success, these encode the data buffer. The caller must
    // free(*buf_inputs_dump) when done
    char**  buf_inputs_dump,
    size_t* size_inputs_dump,

    // in
    const double* Rt_lidar0_board_seed,   // Nsnapshots poses ( (4,3) Rt arrays ) of these to fill
    const double* Rt_lidar0_lidar_seed,   // Nlidars-1 poses ( (4,3) Rt arrays ) of these to fill (lidar0 not included)
    const double* Rt_lidar0_camera_seed,  // Ncameras  poses ( (4,3) Rt arrays ) of these to fill
    const double* Rt_lidar0_board_solve,  // Nsnapshots poses ( (4,3) Rt arrays ) of these to fill
    const double* Rt_lidar0_lidar_solve,  // Nlidars-1 poses ( (4,3) Rt arrays ) of these to fill (lidar0 not included)
    const double* Rt_lidar0_camera_solve, // Ncameras  poses ( (4,3) Rt arrays ) of these to fill

    const sensor_snapshot_segmented_t* snapshots,
    const int                          Nsnapshots,
    const uint64_t*                    bitarray_snapshots_selected,

    const unsigned int Nlidars,
    const unsigned int Ncameras,
    const mrcal_cameramodel_t*const* models, // Ncameras of these

    // The dimensions of the chessboard grid being detected in the images
    const int object_height_n,
    const int object_width_n,
    const double object_spacing)
{
    bool result = false;

    *buf_inputs_dump  = NULL;
    *size_inputs_dump = 0;
    FILE* fp = open_memstream(buf_inputs_dump, size_inputs_dump);
    if(fp == NULL)
    {
        MSG("open_memstream( optimization-inputs-dump ) failed");
        goto done;
    }

    fprintf(fp, "Nsnapshots      = %d\n", Nsnapshots);
    fprintf(fp, "Nlidars         = %d\n", Nlidars);
    fprintf(fp, "Ncameras        = %d\n", Ncameras);
    fprintf(fp, "object_height_n = %d\n", object_height_n);
    fprintf(fp, "object_width_n  = %d\n", object_width_n);
    fprintf(fp, "object_spacing  = %f\n", object_spacing);

    fwrite(Rt_lidar0_board_seed,        sizeof(double),   Nsnapshots *4*3,               fp);
    fwrite(Rt_lidar0_lidar_seed,        sizeof(double),   (Nlidars-1)*4*3,               fp);
    fwrite(Rt_lidar0_camera_seed,       sizeof(double),   Ncameras   *4*3,               fp);
    fwrite(Rt_lidar0_board_solve,       sizeof(double),   Nsnapshots *4*3,               fp);
    fwrite(Rt_lidar0_lidar_solve,       sizeof(double),   (Nlidars-1)*4*3,               fp);
    fwrite(Rt_lidar0_camera_solve,      sizeof(double),   Ncameras   *4*3,               fp);
    fwrite(bitarray_snapshots_selected, sizeof(uint64_t), bitarray64_nwords(Nsnapshots), fp);

    LOOP_SNAPSHOT()
    {
        LOOP_SNAPSHOT_HEADER(,const);

        for(int icamera=0; icamera<Ncameras; icamera++)
        {
            if(snapshot->chessboard_corners[icamera] == NULL)
            {
                // indicate that this camera does NOT have observations
                fwrite(&(uint8_t){0}, 1,1, fp);
                continue;
            }

            // indicate that this camera DOES have observations
            fwrite(&(uint8_t){1}, 1,1, fp);
            fwrite(snapshot->chessboard_corners[icamera], sizeof(mrcal_point2_t), object_width_n*object_height_n, fp);
        }

        for(int ilidar=0; ilidar<Nlidars; ilidar++)
        {
            const points_and_plane_full_t* points_and_plane_full = &snapshot->lidar_scans[ilidar];

            fwrite(&points_and_plane_full->n, sizeof(points_and_plane_full->n),1, fp);
            if(points_and_plane_full->n == 0)
                continue;

            if(points_and_plane_full->ipoint == NULL)
            {
                // No ipoints[]: points[] are used densely. Indicate this with a
                // 0 here: Npoints = n
                fwrite(&(int){0}, sizeof(int),1, fp);
                fwrite(points_and_plane_full->points, sizeof(points_and_plane_full->points[0]), points_and_plane_full->n, fp);
            }
            else
            {
                // Have ipoints[]: points[] are indexed by ipoints[]. Here I can
                // have Npoints != n: some of the points[] may not be used. I
                // write the minumum I need
                int ipoint_max = 0;
                for(int i=0; i<points_and_plane_full->n; i++)
                    if(ipoint_max < points_and_plane_full->ipoint[i])
                        ipoint_max = points_and_plane_full->ipoint[i];
                const int Npoints = ipoint_max+1;
                fwrite(&Npoints, sizeof(int),1, fp);

                fwrite(points_and_plane_full->points, sizeof(points_and_plane_full->points[0]), Npoints, fp);
                fwrite(points_and_plane_full->ipoint, sizeof(points_and_plane_full->ipoint[0]), points_and_plane_full->n, fp);
            }

            fwrite(&points_and_plane_full->plane, sizeof(points_and_plane_full->plane), 1, fp);
        }
    }

    for(int icamera=0; icamera<Ncameras; icamera++)
    {
        const mrcal_cameramodel_t* model = models[icamera];
        const int Nbytes_here =
            sizeof(mrcal_cameramodel_t) +
            mrcal_lensmodel_num_params(&model->lensmodel) * sizeof(double);

        fwrite(&Nbytes_here, sizeof(int), 1, fp);
        fwrite((uint8_t*)model,
               Nbytes_here,
               1,
               fp);
    }

    result = true;

 done:
    if(fp != NULL)
        fclose(fp);
    if(!result)
    {
        free(*buf_inputs_dump);
        *buf_inputs_dump  = NULL;
        *size_inputs_dump = 0;
    }
    return result;
}

// Align the LIDAR and camera geometry
static bool
fit(// out

    // if non-NULL, the solver context is returned in *solver_context, and it's
    // the caller's responsibility to dogleg_freeContext(solver_context). It is
    // assumed that no existing context is passed in *solver_context, and we
    // always create a new one
    dogleg_solverContext_t** solver_context,

    // Optimal state on output
    double* rt_lidar0_board,
    double* rt_lidar0_lidar,
    double* rt_lidar0_camera,

    // in,out
    // seed state on input
    double* Rt_lidar0_board,  // Nsnapshots poses ( (4,3) Rt arrays ) of these to fill
    double* Rt_lidar0_lidar,  // Nlidars-1 poses ( (4,3) Rt arrays ) of these to fill (lidar0 not included)
    double* Rt_lidar0_camera, // Ncameras  poses ( (4,3) Rt arrays ) of these to fill

    // in
    const sensor_snapshot_segmented_t* snapshots,
    const unsigned int                 Nsnapshots,
    const uint64_t*                    bitarray_snapshots_selected,

    const unsigned int Nlidars,
    const unsigned int Ncameras,
    const mrcal_cameramodel_t*const* models, // Ncameras of these

    // The dimensions of the chessboard grid being detected in the images
    const int object_height_n,
    const int object_width_n,
    const double object_spacing,

    bool check_gradient__use_distance_to_plane,
    bool check_gradient,
    bool skip_presolve,
    bool skip_plots,
    bool verbose)

{
    bool result = false;

    LOOP_SNAPSHOT()
    {
        LOOP_SNAPSHOT_HEADER(,const);
        mrcal_rt_from_Rt(&rt_lidar0_board[isnapshot*6], NULL,
                         &Rt_lidar0_board[isnapshot*4*3]);
    }
    for(unsigned int i=0; i<(Nlidars-1); i++)
        mrcal_rt_from_Rt(&rt_lidar0_lidar[i*6], NULL,
                         &Rt_lidar0_lidar[i*4*3]);
    for(unsigned int i=0; i<Ncameras; i++)
        mrcal_rt_from_Rt(&rt_lidar0_camera[i*6], NULL,
                         &Rt_lidar0_camera[i*4*3]);

    callback_context_t ctx = {.Ncameras              = Ncameras,
                              .Nlidars               = Nlidars,
                              .Nsnapshots            = Nsnapshots,
                              .snapshots             = snapshots,
                              .bitarray_snapshots_selected = bitarray_snapshots_selected,
                              .use_distance_to_plane = false,
                              .report_imeas          = false,
                              .models                = models,
                              .object_height_n       = object_height_n,
                              .object_width_n        = object_width_n,
                              .object_spacing        = object_spacing};
    dogleg_parameters2_t dogleg_parameters;
    dogleg_getDefaultParameters(&dogleg_parameters);
    if(verbose &&
       !(check_gradient__use_distance_to_plane || check_gradient))
        dogleg_parameters.dogleg_debug = DOGLEG_DEBUG_VNLOG;

    dogleg_solverContext_t* _solver_context;
    dogleg_solverContext_t** pp_solver_context;
    if(solver_context != NULL)
        pp_solver_context = solver_context;
    else
        pp_solver_context = &_solver_context;
    *pp_solver_context = NULL;

    const int Nstate        = num_states(&ctx);
    const int Nmeasurements = num_measurements(&ctx);
    const int Njnnz         = num_j_nonzero(&ctx);

    if(check_gradient__use_distance_to_plane || check_gradient)
    {
        // lidar1 is the first optimized state: lidar0 defines the coordinate system
        // reference
        const int istate_lidar1  = state_index_lidar (1, &ctx);
        const int Nstate_lidar   = num_states_lidars (&ctx);
        const int istate_camera0 = state_index_camera(0, &ctx);
        const int Nstate_camera  = num_states_cameras(&ctx);
        const int istate_board0  = state_index_board (0, &ctx);
        const int Nstate_board   = num_states_boards (&ctx);

        printf("## Nstate=%d Nmeasurements=%d Nlidars=%d Ncameras=%d Nsnapshots=%d\n",
               Nstate,Nmeasurements,Nlidars,Ncameras,Nsnapshots);
        if(Ncameras > 0)
            printf("## States rt_lidar0_lidar in [%d,%d], rt_lidar0_camera in [%d,%d], rt_lidar0_board in [%d,%d]\n",
                   istate_lidar1,  istate_lidar1  + Nstate_lidar -1,
                   istate_camera0, istate_camera0 + Nstate_camera-1,
                   istate_board0,  istate_board0  + Nstate_board -1);
        else
            printf("## States rt_lidar0_lidar in [%d,%d], rt_lidar0_board in [%d,%d]\n",
                   istate_lidar1, istate_lidar1 + Nstate_lidar-1,
                   istate_board0, istate_board0 + Nstate_board-1);
    }

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
    if(!skip_plots)
        plot_residuals("/tmp/residuals-seed", x, &ctx, verbose);

    for(int i=0; i<Nmeasurements; i++)
        if( fabs(x[i])*SCALE_MEASUREMENT_PX > 1000 ||
            fabs(x[i])*SCALE_MEASUREMENT_M  > 100 )
        {
            MSG("Error: seed has unbelievably-high errors. Giving up");
            goto done;
        }

    if(!skip_presolve)
    {
        MSG_IF_VERBOSE("Starting pre-solve");
        ctx.use_distance_to_plane = true;
        if(!check_gradient__use_distance_to_plane)
        {
            norm2x = dogleg_optimize2(b,
                                      Nstate, Nmeasurements, Njnnz,
                                      (dogleg_callback_t*)&cost, &ctx,
                                      &dogleg_parameters,
                                      pp_solver_context);
        }
        else
        {
            for(int ivar=0; ivar<Nstate; ivar++)
                dogleg_testGradient(ivar, b,
                                    Nstate, Nmeasurements, Njnnz,
                                    (dogleg_callback_t*)&cost, &ctx);
            result = true;
            goto done;
        }
        MSG_IF_VERBOSE("Finished pre-solve; started full solve;");
    }

    ctx.use_distance_to_plane = false;
    if(check_gradient)
    {
        for(int ivar=0; ivar<Nstate; ivar++)
            dogleg_testGradient(ivar, b,
                                Nstate, Nmeasurements, Njnnz,
                                (dogleg_callback_t*)&cost, &ctx);
        result = true;
        goto done;
    }


    norm2x = dogleg_optimize2(b,
                              Nstate, Nmeasurements, Njnnz,
                              (dogleg_callback_t*)&cost, &ctx,
                              &dogleg_parameters,
                              pp_solver_context);

    MSG_IF_VERBOSE("Finished full solve");



    unpack_solver_state(// out
                        rt_lidar0_lidar,
                        rt_lidar0_camera,
                        rt_lidar0_board,
                        // in
                        b,
                        &ctx);
    LOOP_SNAPSHOT()
    {
        LOOP_SNAPSHOT_HEADER(,const);
        mrcal_Rt_from_rt(&Rt_lidar0_board[isnapshot*4*3], NULL,
                         &rt_lidar0_board[isnapshot*6]);
    }
    for(unsigned int i=0; i<(Nlidars-1); i++)
        mrcal_Rt_from_rt(&Rt_lidar0_lidar[i*4*3], NULL,
                         &rt_lidar0_lidar[i*6]);
    for(unsigned int i=0; i<Ncameras; i++)
        mrcal_Rt_from_rt(&Rt_lidar0_camera[i*4*3], NULL,
                         &rt_lidar0_camera[i*6]);

    cost(b, x, NULL, &ctx);
    MSG_IF_VERBOSE("RMS fit error: %.2f normalized units",
                   sqrt(norm2x / (double)Nmeasurements));

    const int imeas_lidar_0                = measurement_index_lidar(0,0, &ctx);
    const int Nmeas_lidar_observation_all  = num_measurements_lidars(&ctx);
    const int imeas_camera_0               = measurement_index_camera(0,0, &ctx);
    const int Nmeas_camera_observation_all = num_measurements_cameras(&ctx);
    const int imeas_regularization_0       = measurement_index_regularization(&ctx);
    const int Nmeas_regularization         = num_measurements_regularization(&ctx);

    if(verbose)
    {
        if(Ncameras > 0)
        {
            double norm2x_camera = 0;
            for(int i=imeas_camera_0; i<imeas_camera_0+Nmeas_camera_observation_all; i++)
                norm2x_camera += x[i]*x[i];
            MSG("RMS fit error (camera): %.2f pixels",
                sqrt(norm2x_camera / (Nmeas_camera_observation_all/2) )*SCALE_MEASUREMENT_PX);
        }
        double norm2x_lidar = 0;
        for(int i=imeas_lidar_0; i<imeas_lidar_0+Nmeas_lidar_observation_all; i++)
            norm2x_lidar += x[i]*x[i];
        MSG("RMS fit error (lidar): %.3f m",
            sqrt(norm2x_lidar / Nmeas_lidar_observation_all )*SCALE_MEASUREMENT_M);
        double norm2x_regularization = 0;
        for(int i=imeas_regularization_0; i<imeas_regularization_0+Nmeas_regularization; i++)
            norm2x_regularization += x[i]*x[i];
        MSG("norm2(error_regularization)/norm2(error): %.2f",
            norm2x_regularization/norm2x);
    }
    if(!skip_plots)
        plot_residuals("/tmp/residuals", x, &ctx, verbose);

    result = true;

 done:
    if(solver_context == NULL)
    {
        // caller didn't ask for a context; if we have one, get rid of it
        if(_solver_context != NULL)
            dogleg_freeContext(&_solver_context);
    }
    else
    {
        // caller DID ask for a context; if we failed, get rid of it
        if(!result && *solver_context != NULL)
            dogleg_freeContext(solver_context);
    }

    return result;
}

static void free_snapshot(sensor_snapshot_segmented_t* snapshot, const int Nlidars, const int Ncameras)
{
    for(int icamera=0; icamera<Ncameras; icamera++)
        free(snapshot->chessboard_corners[icamera]);

    for(int ilidar=0; ilidar<Nlidars; ilidar++)
    {
        points_and_plane_full_t* points_and_plane_full = &snapshot->lidar_scans[ilidar];
        if(points_and_plane_full != NULL)
        {
            free(points_and_plane_full->points);
            free((uint32_t*)points_and_plane_full->ipoint);
        }
    }
}

static bool is_R_rotation(const double* R)
{
    // each row/col must be orthonormal. Making sure that R Rt = I does that
    for(int i=0; i<3; i++)
        for(int j=i; j<3; j++)
        {
            double dot = 0.;
            for(int k=0; k<3; k++)
                dot += R[i*3 + k]*R[j*3 + k];
            if(i == j)
            {
                if(fabs(dot-1.0) > 1e-9)
                    return false;
            }
            else
            {
                if(fabs(dot) > 1e-9)
                    return false;
            }
        }

    // Should also have det(R) = +1. Otherwise there could be a mirror here
    const double det =
        R[0]*(R[4]*R[8]-R[5]*R[7])-R[1]*(R[3]*R[8]-R[5]*R[6])+R[2]*(R[3]*R[7]-R[4]*R[6]);
    if(fabs(det-1.0) > 1e-9)
        return false;

    return true;
}

static bool
confirm_Rt_are_valid(// in
                     const double* Rt_all,
                     const int N,
                     const char* what)
{
    for(int i=0; i<N; i++)
    {
        const double* Rt = &Rt_all[4*3*i];
        const double* R  = &Rt[0*3];
        const double* t  = &Rt[3*3];

        if(!is_R_rotation(R))
        {
            MSG("ERROR: %s[%d] does NOT contain a valid rotation matrix",
                what, i);
            return false;
        }
    }
    return true;
}

bool clc_fit_from_inputs_dump(// out
                              int* Nlidars,
                              int* Ncameras,
                              // Allocated by the function on success.
                              // It's the caller's responsibility to
                              // free() these
                              mrcal_pose_t** rt_lidar0_lidar,
                              mrcal_pose_t** rt_lidar0_camera,
                              // in
                              const char* buf_inputs_dump,
                              size_t      size_inputs_dump,
                              const int*  isnapshot_exclude, // NULL to not exclude any
                              const int   Nisnapshot_exclude,

                              const double fit_seed_position_err_threshold,
                              const double fit_seed_cos_angle_err_threshold,

                              // if(!do_fit_seed && !do_inject_noise) { fit(previous fit_seed() result)     }
                              // if(!do_fit_seed &&  do_inject_noise) { fit(previous fit() result)          }
                              // if(do_fit_seed)                      { fit( fit_seed() )                   }
                              bool do_fit_seed,
                              // if true, the observations are noised; regardless of do_fit_seed
                              bool do_inject_noise,
                              bool do_skip_plots,
                              bool verbose)
{
    bool result = false;

    *rt_lidar0_lidar  = NULL;
    *rt_lidar0_camera = NULL;

    FILE* fp = fmemopen((char*)buf_inputs_dump, size_inputs_dump, "r");
    if(fp == NULL)
    {
        MSG("Couldn't open buffer for reading");
        return false;
    }

    char*  lineptr = NULL;
    size_t nline   = 0;

    unsigned int Nsnapshots;
    int          object_height_n;
    int          object_width_n;
    double       object_spacing;

    if(0 >= getline(&lineptr, &nline, fp)) goto done;
    if(1 != sscanf(lineptr, "Nsnapshots = %d\n",  &Nsnapshots)) goto done;
    if(0 >= getline(&lineptr, &nline, fp)) goto done;
    if(1 != sscanf(lineptr, "Nlidars                    = %d\n",  Nlidars)) goto done;
    if(0 >= getline(&lineptr, &nline, fp)) goto done;
    if(1 != sscanf(lineptr, "Ncameras                   = %d\n",  Ncameras)) goto done;
    if(0 >= getline(&lineptr, &nline, fp)) goto done;
    if(1 != sscanf(lineptr, "object_height_n            = %d\n",  &object_height_n)) goto done;
    if(0 >= getline(&lineptr, &nline, fp)) goto done;
    if(1 != sscanf(lineptr, "object_width_n             = %d\n",  &object_width_n)) goto done;
    if(0 >= getline(&lineptr, &nline, fp)) goto done;
    if(1 != sscanf(lineptr, "object_spacing             = %lf\n", &object_spacing)) goto done;

    *rt_lidar0_lidar  = malloc(*Nlidars  * sizeof(mrcal_pose_t));
    if(*rt_lidar0_lidar == NULL)
        goto done;
    *rt_lidar0_camera = malloc(*Ncameras * sizeof(mrcal_pose_t));
    if(*rt_lidar0_camera == NULL)
        goto done;


    {
        double Rt_lidar0_board_seed  [Nsnapshots *4*3];
        double Rt_lidar0_lidar_seed  [(*Nlidars-1)               *4*3];
        double Rt_lidar0_camera_seed [*Ncameras                  *4*3];
        double Rt_lidar0_board_solve [Nsnapshots *4*3];
        double Rt_lidar0_lidar_solve [(*Nlidars-1)               *4*3];
        double Rt_lidar0_camera_solve[*Ncameras                  *4*3];

        double rt_lidar0_board__optimization [6*Nsnapshots];
        double rt_lidar0_lidar__optimization [6*(*Nlidars-1)];
        double rt_lidar0_camera__optimization[6*(*Ncameras)];

        sensor_snapshot_segmented_t snapshots[Nsnapshots];
        memset(snapshots, 0, Nsnapshots*sizeof(snapshots[0]));

        const int Nwords = bitarray64_nwords(Nsnapshots);
        uint64_t bitarray_snapshots_selected[Nwords];

        mrcal_cameramodel_t* models[*Ncameras];

        if(Nsnapshots *4*3 != fread(Rt_lidar0_board_seed,   sizeof(double), Nsnapshots *4*3, fp))
            goto done_inner;
        if((*Nlidars-1)               *4*3 != fread(Rt_lidar0_lidar_seed,   sizeof(double), (*Nlidars-1)               *4*3, fp))
            goto done_inner;
        if(*Ncameras                  *4*3 != fread(Rt_lidar0_camera_seed,  sizeof(double), *Ncameras                  *4*3, fp))
            goto done_inner;
        if(Nsnapshots *4*3 != fread(Rt_lidar0_board_solve,  sizeof(double), Nsnapshots *4*3, fp))
            goto done_inner;
        if((*Nlidars-1)               *4*3 != fread(Rt_lidar0_lidar_solve,  sizeof(double), (*Nlidars-1)               *4*3, fp))
            goto done_inner;
        if(*Ncameras                  *4*3 != fread(Rt_lidar0_camera_solve, sizeof(double), *Ncameras                  *4*3, fp))
            goto done_inner;
        if(Nwords                          != fread(bitarray_snapshots_selected, sizeof(uint64_t), Nwords, fp))
            goto done_inner;

        int Ncamera_observations = 0;

        LOOP_SNAPSHOT()
        {
            LOOP_SNAPSHOT_HEADER(,); // NOT const

            for(int icamera=0; icamera<*Ncameras; icamera++)
            {
                uint8_t camera_has_observations;
                if(1 != fread(&camera_has_observations, 1,1, fp))
                    goto done_inner;
                if(!camera_has_observations)
                {
                    snapshot->chessboard_corners[icamera] = NULL;
                    continue;
                }

                snapshot->chessboard_corners[icamera] = malloc(sizeof(mrcal_point2_t)*object_width_n*object_height_n);
                if(NULL == snapshot->chessboard_corners[icamera])
                    goto done_inner;
                if(object_width_n*object_height_n !=
                   fread(snapshot->chessboard_corners[icamera], sizeof(mrcal_point2_t), object_width_n*object_height_n, fp))
                    goto done_inner;
            }

            for(int ilidar=0; ilidar<*Nlidars; ilidar++)
            {
                points_and_plane_full_t* points_and_plane_full = &snapshot->lidar_scans[ilidar];

                if(1 != fread(&points_and_plane_full->n, sizeof(points_and_plane_full->n),1, fp))
                    goto done_inner;
                if(points_and_plane_full->n == 0)
                    continue;

                int Npoints;
                if(1 != fread(&Npoints, sizeof(int),1, fp))
                    goto done_inner;

                if(Npoints == 0)
                {
                    // No ipoints[]: points[] are used densely. Indicated here
                    // with a 0: Npoints = n
                    points_and_plane_full->ipoint = NULL;

                    points_and_plane_full->points = malloc(sizeof(points_and_plane_full->points[0])*points_and_plane_full->n);
                    if(NULL == points_and_plane_full->points)
                        goto done_inner;
                    if(points_and_plane_full->n !=
                       fread(points_and_plane_full->points, sizeof(points_and_plane_full->points[0]), points_and_plane_full->n, fp))
                        goto done_inner;
                }
                else
                {
                    // Have ipoints[]: points[] are indexed by ipoints[]. Here I can
                    // have Npoints != n: some of the points[] may not be used. I
                    // have the Npoints I need
                    points_and_plane_full->points = malloc(sizeof(points_and_plane_full->points[0])*Npoints);
                    if(NULL == points_and_plane_full->points)
                        goto done_inner;
                    points_and_plane_full->ipoint = malloc(sizeof(points_and_plane_full->ipoint[0])*points_and_plane_full->n);
                    if(NULL == points_and_plane_full->ipoint)
                        goto done_inner;
                    if(Npoints != fread(points_and_plane_full->points, sizeof(points_and_plane_full->points[0]), Npoints, fp))
                        goto done_inner;

                    if(do_inject_noise)
                    {
                        for(int i=0; i<Npoints; i++)
                        {
                            clc_point3f_t* p = &points_and_plane_full->points[i];

                            const double distance_have = sqrtf( p->x*p->x +
                                                                p->y*p->y +
                                                                p->z*p->z );
                            const double distance_shift = randn()*SCALE_MEASUREMENT_M;
                            for(int i=0; i<3; i++)
                                p->xyz[i] *= 1. + distance_shift/distance_have;
                        }
                    }

                    if(points_and_plane_full->n != fread((uint32_t*)points_and_plane_full->ipoint, sizeof(points_and_plane_full->ipoint[0]), points_and_plane_full->n, fp))
                        goto done_inner;

                    for(int i=0; i<points_and_plane_full->n; i++)
                    {
                        const clc_point3f_t* p = &points_and_plane_full->points[ points_and_plane_full->ipoint[i] ];
                        if(p->x > 1000 ||p->y > 1000 ||p->z > 1000 || p->x < -1000 ||p->y < -1000 ||p->z < -1000)
                        {
                            MSG("******** READ BOGUS POINT FROM DUMP FILE: isnapshot=%d ilidar=%d iipt = %d ipt = %d p=(%f,%f,%f)",
                                isnapshot, ilidar, i, points_and_plane_full->ipoint[i], p->x, p->y, p->z);
                        }
                    }

                }

                if(1 != fread(&points_and_plane_full->plane, sizeof(points_and_plane_full->plane), 1, fp))
                    goto done_inner;
            }
        }


        if(isnapshot_exclude != NULL)
            for(int i=0; i<Nisnapshot_exclude; i++)
                bitarray64_clear(bitarray_snapshots_selected, isnapshot_exclude[i]);

        for(int icamera=0; icamera<*Ncameras; icamera++)
        {
            int Nbytes_here;
            if(1 != fread(&Nbytes_here, sizeof(Nbytes_here), 1, fp))
                goto done_inner;
            models[icamera] = malloc(Nbytes_here);
            if(models[icamera] == NULL)
                goto done_inner;
            if(Nbytes_here != fread((uint8_t*)models[icamera],
                                    1,
                                    Nbytes_here,
                                    fp))
                goto done_inner;
        }


        if(!do_skip_plots)
        {
            plot_geometry("/tmp/geometry-seed-dump.gp",
                          Rt_lidar0_board_seed,
                          Rt_lidar0_lidar_seed,
                          Rt_lidar0_camera_seed,
                          snapshots,
                          Nsnapshots,
                          bitarray_snapshots_selected,
                          *Nlidars,
                          *Ncameras,
                          object_height_n,
                          object_width_n,
                          object_spacing,
                          false,
                          verbose);
            plot_geometry("/tmp/geometry-seed-dump-onlyaxes.gp",
                          Rt_lidar0_board_seed,
                          Rt_lidar0_lidar_seed,
                          Rt_lidar0_camera_seed,
                          snapshots,
                          Nsnapshots,
                          bitarray_snapshots_selected,
                          *Nlidars,
                          *Ncameras,
                          object_height_n,
                          object_width_n,
                          object_spacing,
                          true,
                          verbose);
            plot_geometry("/tmp/geometry-solve-dump.gp",
                          Rt_lidar0_board_solve,
                          Rt_lidar0_lidar_solve,
                          Rt_lidar0_camera_solve,
                          snapshots,
                          Nsnapshots,
                          bitarray_snapshots_selected,
                          *Nlidars,
                          *Ncameras,
                          object_height_n,
                          object_width_n,
                          object_spacing,
                          false,
                          verbose);
            plot_geometry("/tmp/geometry-solve-dump-onlyaxes.gp",
                          Rt_lidar0_board_solve,
                          Rt_lidar0_lidar_solve,
                          Rt_lidar0_camera_solve,
                          snapshots,
                          Nsnapshots,
                          bitarray_snapshots_selected,
                          *Nlidars,
                          *Ncameras,
                          object_height_n,
                          object_width_n,
                          object_spacing,
                          true,
                          verbose);
        }

        // if(!do_fit_seed && !do_inject_noise) { fit(previous fit_seed() result)     }
        // if(!do_fit_seed &&  do_inject_noise) { fit(previous fit() result)          }
        // if(do_fit_seed)                      { fit( fit_seed() )                   }
        double* Rt_lidar0_board;
        double* Rt_lidar0_lidar;
        double* Rt_lidar0_camera;
        if(!do_fit_seed)
        {
            if(!do_inject_noise)
            {
                // No added noise. I start from the same place we started from
                // the last time, to reproduce the previous solve faithfully
                Rt_lidar0_board  = Rt_lidar0_board_seed;
                Rt_lidar0_lidar  = Rt_lidar0_lidar_seed;
                Rt_lidar0_camera = Rt_lidar0_camera_seed;
            }
            else
            {
                // Added noise. I start from the previous final solution. The
                // noised solves are used in Monte Carlo simulations, and I'm
                // not trying to reproduce the previous runs; rather I'm trying
                // to quantify the sensitivity of the solution to perturbations.
                // So I want this to finish quickly.
                Rt_lidar0_board  = Rt_lidar0_board_solve;
                Rt_lidar0_lidar  = Rt_lidar0_lidar_solve;
                Rt_lidar0_camera = Rt_lidar0_camera_solve;
            }

            // We use the seed from the dump. If it isn't valid, we throw an
            // error
            if(!(confirm_Rt_are_valid(Rt_lidar0_lidar_seed,   (*Nlidars-1), "Rt_lidar0_lidar_seed") &&
                 confirm_Rt_are_valid(Rt_lidar0_camera_seed,  *Ncameras,    "Rt_lidar0_camera_seed")))
            {
                goto done_inner;
            }

            LOOP_SNAPSHOT()
            {
                LOOP_SNAPSHOT_HEADER(,const);

                if(!is_R_rotation(&Rt_lidar0_board_seed[4*3*isnapshot]))
                {
                    MSG("ERROR: %s[isnapshot=%d] does NOT contain a valid rotation matrix",
                        "Rt_lidar0_board_seed", isnapshot);
                    return false;
                }
            }


            // The seed result. Here we just read it from a file, so this was
            // successful
            result = true;
        }
        else
        {
            // Use the _seed buffers. This doesn't matter: either the _seed or
            // _solve buffers would work, as long as this is consistent. I'm
            // recomputing both here anyway
            Rt_lidar0_board  = Rt_lidar0_board_seed;
            Rt_lidar0_lidar  = Rt_lidar0_lidar_seed;
            Rt_lidar0_camera = Rt_lidar0_camera_seed;

            double Rt_camera_board_cache[Nsnapshots*(*Ncameras) * 4*3];
            memset(Rt_camera_board_cache,
                   0,
                   Nsnapshots*(*Ncameras) * 4*3*sizeof(Rt_camera_board_cache[0]));
            // The Rt_camera_board_cache[] entries are all 0, which means "invalid"

            result =
                fit_seed(// out
                         Rt_lidar0_board,  // Nsnapshots poses ( (4,3) Rt arrays ) of these to fill
                         Rt_lidar0_lidar,  // Nlidars-1 poses ( (4,3) Rt arrays ) of these to fill (lidar0 not included)
                         Rt_lidar0_camera, // Ncameras  poses ( (4,3) Rt arrays ) of these to fill
                         false,

                         // in
                         snapshots,
                         Nsnapshots,
                         bitarray_snapshots_selected,
                         Rt_camera_board_cache,

                         *Nlidars,
                         *Ncameras,
                         (const mrcal_cameramodel_t * const*)models, // Ncameras of these

                         // The dimensions of the chessboard grid being detected in the images
                         object_height_n,
                         object_width_n,
                         object_spacing,
                         fit_seed_position_err_threshold,
                         fit_seed_cos_angle_err_threshold,
                         verbose);


            if(result && !do_skip_plots)
            {
                plot_geometry("/tmp/geometry-seed.gp",
                              Rt_lidar0_board,
                              Rt_lidar0_lidar,
                              Rt_lidar0_camera,
                              snapshots,
                              Nsnapshots,
                              bitarray_snapshots_selected,
                              *Nlidars,
                              *Ncameras,
                              object_height_n,
                              object_width_n,
                              object_spacing,
                              false,
                              verbose);
                plot_geometry("/tmp/geometry-seed-onlyaxes.gp",
                              Rt_lidar0_board,
                              Rt_lidar0_lidar,
                              Rt_lidar0_camera,
                              snapshots,
                              Nsnapshots,
                              bitarray_snapshots_selected,
                              *Nlidars,
                              *Ncameras,
                              object_height_n,
                              object_width_n,
                              object_spacing,
                              true,
                              verbose);
            }
        }
        result =
            result &&
            fit(NULL,
                rt_lidar0_board__optimization,
                rt_lidar0_lidar__optimization,
                rt_lidar0_camera__optimization,

                // in,out
                // seed state on input
                Rt_lidar0_board,  // Nsnapshots poses ( (4,3) Rt arrays ) of these to fill
                Rt_lidar0_lidar,  // Nlidars-1 poses ( (4,3) Rt arrays ) of these to fill (lidar0 not included)
                Rt_lidar0_camera, // Ncameras  poses ( (4,3) Rt arrays ) of these to fill

                // in
                snapshots,
                Nsnapshots,
                bitarray_snapshots_selected,

                *Nlidars,
                *Ncameras,
                (const mrcal_cameramodel_t * const*)models, // Ncameras of these

                // The dimensions of the chessboard grid being detected in the images
                object_height_n,
                object_width_n,
                object_spacing,

                false,false,
                true,
                do_skip_plots,
                verbose);

        if(result)
        {
            if(!do_skip_plots)
            {
                plot_geometry("/tmp/geometry.gp",
                              Rt_lidar0_board,
                              Rt_lidar0_lidar,
                              Rt_lidar0_camera,
                              snapshots,
                              Nsnapshots,
                              bitarray_snapshots_selected,
                              *Nlidars,
                              *Ncameras,
                              object_height_n,
                              object_width_n,
                              object_spacing,
                              false,
                              verbose);
                plot_geometry("/tmp/geometry-onlyaxes.gp",
                              Rt_lidar0_board,
                              Rt_lidar0_lidar,
                              Rt_lidar0_camera,
                              snapshots,
                              Nsnapshots,
                              bitarray_snapshots_selected,
                              *Nlidars,
                              *Ncameras,
                              object_height_n,
                              object_width_n,
                              object_spacing,
                              true,
                              verbose);
            }
            (*rt_lidar0_lidar)[0] = (mrcal_pose_t){};
            for(int i=1; i<*Nlidars; i++)
                mrcal_rt_from_Rt( (double*)&(*rt_lidar0_lidar)[i], NULL,
                                  &Rt_lidar0_lidar[(i-1)*4*3] );
            for(int i=0; i<*Ncameras; i++)
                mrcal_rt_from_Rt( (double*)&(*rt_lidar0_camera)[i], NULL,
                                  &Rt_lidar0_camera[i*4*3] );
        }
    done_inner:
        for(int isnapshot=0; isnapshot<Nsnapshots; isnapshot++)
        {
            // don't check bitarray_snapshots_selected to free EVERYTHING, even
            // the snapshots we just excluded
            sensor_snapshot_segmented_t* snapshot = &snapshots[isnapshot];
            free_snapshot(snapshot, *Nlidars, *Ncameras);
        }

        for(int icamera=0; icamera<*Ncameras; icamera++)
            free(models[icamera]);
    }

 done:
    free(lineptr);
    fclose(fp);

    if(!result)
    {
        free(*rt_lidar0_lidar);
        *rt_lidar0_lidar = NULL;
        free(*rt_lidar0_camera);
        *rt_lidar0_camera = NULL;
    }
    return result;
}


static
bool lidar_segmentation(// out
                        points_and_plane_full_t* lidar_scan,

                        clc_point3f_t* points_pool,
                        clc_points_and_plane_t* points_and_plane_pool,
                        int*           points_pool_index,
                        int*           points_and_plane_pool_index,

                        // in
                        const clc_sensor_snapshot_unsorted_t* sensor_snapshot_unsorted,
                        const clc_sensor_snapshot_sorted_t* sensor_snapshot_sorted,
                        const int ilidar,
                        const int Nrings,
                        const unsigned int lidar_packet_stride,
                        const int Nplanes_max,
                        const clc_lidar_segmentation_context_t* ctx)
{
    const clc_lidar_scan_unsorted_t* scan_unsorted =
        (sensor_snapshot_unsorted != NULL) ?
        &sensor_snapshot_unsorted->lidar_scans[ilidar] :
        NULL;
    const clc_lidar_scan_sorted_t* scan_sorted =
        (sensor_snapshot_sorted != NULL) ?
        &sensor_snapshot_sorted->lidar_scans[ilidar] :
        NULL;

    clc_point3f_t* points_here = NULL;
    unsigned int   _Npoints[Nrings];
    unsigned int*  Npoints = _Npoints;

    if(scan_unsorted != NULL)
    {
        if(scan_unsorted->Npoints == 0)
            return false;

        points_here = &points_pool[*points_pool_index];
        *points_pool_index += scan_unsorted->Npoints;

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
        &points_and_plane_pool[*points_and_plane_pool_index];
    (*points_and_plane_pool_index)++;

    int8_t Nplanes_found =
        clc_lidar_segmentation_sorted(// out
                                      points_and_plane_here,
                                      // in
                                      Nplanes_max,
                                      &(clc_lidar_scan_sorted_t){.points  = points_here,
                                                                 .Npoints = Npoints},
                                      ctx);

    // If we didn't see a clear plane, I keep the previous
    // ..._pool_bytes_used value, reusing this memory on the next round.
    // If we see a clear plane, but filter this data out at a later
    // point, I will not be reusing the memory; instead I'll carry it
    // around until the whole thing is freed at the end of clc_unsorted()
    if(Nplanes_found == 0)
    {
        // MSG("No planes found for isnapshot=%d ilidar=%d.",
        //     isnapshot, ilidar);
        return false;
    }
    if(Nplanes_found > 1)
    {
        // MSG("Too many planes found for isnapshot=%d ilidar=%d.",
        //     isnapshot, ilidar);
        return false;
    }

    // Keep this scan
    *lidar_scan =
        (points_and_plane_full_t){ .points = points_here,
                                   .n      = points_and_plane_here->n,
                                   .ipoint = points_and_plane_here->ipoint,
                                   .plane  = points_and_plane_here->plane};

    return true;
}

static bool make_reprojected_plots( const double* Rt_lidar0_lidar,
                                    const double* Rt_lidar0_camera,
                                    const sensor_snapshot_segmented_t* snapshots,
                                    const int                          Nsnapshots,
                                    const uint64_t*                    bitarray_snapshots_selected,
                                    const int                          Nlidars,
                                    const int                          Ncameras,
                                    const mrcal_cameramodel_t*const*   models, // Ncameras of these
                                    const int                          object_height_n,
                                    const int                          object_width_n,
                                    bool  verbose)
{
    LOOP_SNAPSHOT()
    {
        LOOP_SNAPSHOT_HEADER(,const);

        for(int ilidar=0; ilidar<Nlidars; ilidar++)
        {
            const points_and_plane_full_t* lidar_scan = &snapshot->lidar_scans[ilidar];
            if(lidar_scan->points == NULL)
                continue;

            const double* Rt_lidar0_lidar_here =
                ilidar > 0 ?
                &Rt_lidar0_lidar[(ilidar-1)*4*3] :
                NULL;

            for(int icamera=0; icamera<Ncameras; icamera++)
            {
                const mrcal_point2_t* chessboard_corners = snapshot->chessboard_corners[icamera];
                if(chessboard_corners == NULL)
                    continue;

                // these plots ultimately aren't all that useful, so I disable
                // them
#if 0
                PLOT_MAKE_FILENAME("/tmp/reprojected-snapshot%d-camera%d-lidar%d.gp",
                                   isnapshot, icamera, ilidar);
                PLOT( {{
                            const double* Rt_lidar0_camera_here =
                                &Rt_lidar0_camera[icamera*4*3];

                            double Rt_camera_lidar[4*3];
                            if(Rt_lidar0_lidar_here != NULL)
                                mrcal_compose_Rt_inverted0(Rt_camera_lidar,
                                                           Rt_lidar0_camera_here,
                                                           Rt_lidar0_lidar_here);
                            else
                                mrcal_invert_Rt(Rt_camera_lidar,
                                                Rt_lidar0_camera_here);

                            for(unsigned int iipoint=0; iipoint<lidar_scan->n; iipoint++)
                            {
                                mrcal_point3_t p;
                                int ipoint = (lidar_scan->ipoint != NULL) ?
                                    lidar_scan->ipoint[iipoint] :
                                    iipoint;
                                mrcal_point3_from_clc_point3f(&p,
                                                              &lidar_scan->points[ipoint]);

                                mrcal_transform_point_Rt(p.xyz, NULL,NULL,
                                                         Rt_camera_lidar, p.xyz);

                                mrcal_point2_t q;
                                mrcal_project(&q, NULL,NULL,
                                              &p, 1,
                                              &models[icamera]->lensmodel, models[icamera]->intrinsics);

                                fprintf(fp, "%f lidar %f\n",
                                        q.x, q.y);
                            }

                            for(int i=0; i<object_height_n; i++)
                                for(int j=0; j<object_width_n; j++)
                                {
                                    fprintf(fp, "%f camera %f\n",
                                            chessboard_corners[i*object_width_n+j].x,
                                            chessboard_corners[i*object_width_n+j].y);
                                }
                        }},

                    "--domain --dataid "
                    "--legend camera 'Chessboard corners from the image' "
                    "--legend lidar  'Reprojected LIDAR points' "
                    "--style  camera 'with linespoints' "
                    "--style  lidar  'with points' "
                    "--square --set 'yrange [:] reverse' "
                    );
#endif
            }
        }
    }
    return true;
}

static int isector_from_pvehicle(const double* xy,
                                 const int Nsectors)
{
    double yaw = atan2(xy[1], xy[0]); // yaw is in [-pi..pi]
    if(yaw < 0.) yaw += 2.*M_PI;  // yaw is in [0..2pi]

    const double sector_width_rad = 2.*M_PI/(double)Nsectors;
    int isector = yaw / sector_width_rad;
    // just in case; for round-off
    if(     isector < 0        ) isector = 0;
    else if(isector >= Nsectors) isector = Nsectors-1;

    return isector;
}

static bool evaluate_lidar_visibility(// out
                                      // A dense array of shape (Nlidars,Nsectors)
                                      uint8_t* isvisible_per_lidar_per_sector,
                                      // in
                                      const double*                      Rt_vehicle_lidar0, // may be NULL
                                      const int                          Nsectors,
                                      const int                          Nrings,

                                      const double*                      Rt_lidar0_lidar_notfirst,
                                      const double                       threshold_valid_lidar_range,
                                      const double                       threshold_valid_lidar_Npoints,
                                      const clc_lidar_scan_unsorted_t* lidar_scans, // Nlidars of these
                                      // The stride, in bytes, between each successive points or rings value
                                      // in clc_lidar_scan_unsorted_t
                                      // <=0 means "points stored densely"
                                      int                                lidar_packet_stride,
                                      const int                          Nlidars)
{
    if(lidar_packet_stride <= 0)
        // stored densely
        lidar_packet_stride = sizeof(clc_point3f_t);

    for(int ilidar=0; ilidar<Nlidars; ilidar++)
    {
        double Rt_vehicle_lidar[4*3];
        if(Rt_vehicle_lidar0 != NULL)
        {
            if(ilidar == 0)
                memcpy(Rt_vehicle_lidar,
                       Rt_vehicle_lidar0,
                       sizeof(Rt_vehicle_lidar));
            else
                mrcal_compose_Rt(Rt_vehicle_lidar,
                                 Rt_vehicle_lidar0, &Rt_lidar0_lidar_notfirst[4*3*(ilidar-1)]);
        }
        else
        {
            if(ilidar == 0)
                mrcal_identity_Rt(Rt_vehicle_lidar);
            else
                memcpy(Rt_vehicle_lidar,
                       &Rt_lidar0_lidar_notfirst[4*3*(ilidar-1)],
                       sizeof(Rt_vehicle_lidar));
        }

        int Nobservations_per_sector[Nsectors];
        memset(Nobservations_per_sector, 0, Nsectors*sizeof(Nobservations_per_sector[0]));

        int Npoints;
        clc_point3f_t* points;

        const clc_lidar_scan_unsorted_t* lidar_scan = &lidar_scans[ilidar];
        if(lidar_scan->points == NULL)
        {
            memset(&isvisible_per_lidar_per_sector[ilidar * Nsectors],
                   0,
                   Nsectors*sizeof(isvisible_per_lidar_per_sector[0]));
            continue;
        }
        Npoints = lidar_scan->Npoints;
        points  = lidar_scan->points;

        for(int ipoint=0; ipoint<Npoints; ipoint++)
        {
            mrcal_point3_t p;
            mrcal_point3_from_clc_point3f(&p,
                                          (clc_point3f_t*)((uint8_t*)points + lidar_packet_stride*ipoint));
            if(mrcal_point3_norm2(p) >=
               threshold_valid_lidar_range*threshold_valid_lidar_range)
            {
                mrcal_transform_point_Rt(p.xyz,NULL,NULL,
                                         Rt_vehicle_lidar,
                                         p.xyz);

                const int isector = isector_from_pvehicle(p.xyz, Nsectors);
                Nobservations_per_sector[isector]++;
            }
        }

        for(int isector=0; isector<Nsectors; isector++)
            isvisible_per_lidar_per_sector[ilidar * Nsectors + isector] =
                (uint8_t)(Nobservations_per_sector[isector] >= threshold_valid_lidar_Npoints);
    }

    return true;
}

// Computes Var = inv(JtJ). This is potentially inefficient. We should use the
// factorization to compute the thing that we actually use the covariance for
static bool compute_covariance(// out
                               double* Var_rt_lidar0_sensor,
                               // in
                               const sensor_snapshot_segmented_t* snapshots,
                               const int                          Nsnapshots,
                               const uint64_t*                    bitarray_snapshots_selected,

                               const unsigned int Nlidars,
                               const unsigned int Ncameras,
                               const mrcal_cameramodel_t*const* models, // Ncameras of these

                               // The dimensions of the chessboard grid being detected in the images
                               const int object_height_n,
                               const int object_width_n,
                               const double object_spacing,

                               dogleg_solverContext_t* solver_context)
{
    bool result = false;


    // To validate the result
#if 0
    cholmod_dense* Jt_dense =
        cholmod_sparse_to_dense(solver_context->beforeStep->Jt,
                                &solver_context->common);
    const char* filename = "/tmp/J.dat";
    MSG("Dumping Jt of nrow=%d ncol=%d to disk at '%s'",
        Jt_dense->nrow, Jt_dense->ncol, filename);
    FILE* fp = fopen(filename, "wb");
    fwrite(Jt_dense->x, sizeof(double), Jt_dense->nrow*Jt_dense->ncol,
           fp);
    fclose(fp);

    /* This Python program compares the direct Python results and the C
       results. They should match 100%, up to numerical precision

#!/usr/bin/python3

import numpy as np
import numpysane as nps

SCALE_ROTATION_CAMERA    = (0.1 * np.pi/180.0)
SCALE_TRANSLATION_CAMERA = 1.0
SCALE_ROTATION_LIDAR     = SCALE_ROTATION_CAMERA
SCALE_TRANSLATION_LIDAR  = SCALE_TRANSLATION_CAMERA


J = np.fromfile("/tmp/J.dat")
Nstate = 96
Nmeas = len(J) // Nstate

J = J.reshape((Nmeas,Nstate),)
JtJ = np.dot(J.T,J)

inv_JtJ = np.linalg.inv(JtJ)

gp.plotimage(J       != 0, square=1, wait=1)
gp.plotimage(JtJ     != 0, square=1, wait=1)
gp.plotimage(inv_JtJ != 0, square=1, wait=1)

Var = inv_JtJ[:6,:6]
Var[:3,:] *= SCALE_ROTATION_LIDAR
Var[3:,:] *= SCALE_TRANSLATION_LIDAR
Var[:,:3] *= SCALE_ROTATION_LIDAR
Var[:,3:] *= SCALE_TRANSLATION_LIDAR

print(Var)

Var_c = np.array(([ 1.47171e-07, 2.38701e-07, -1.19087e-08, 3.48535e-08, 7.14979e-08, 9.08954e-07,  ],
              [ 2.38701e-07, 8.38225e-07, 1.45342e-08, -1.29072e-07, 2.48447e-07, 2.8084e-06,  ],
              [ -1.19087e-08, 1.45342e-08, 4.47514e-08, -6.84524e-08, -1.43358e-08, -2.84269e-08,  ],
              [ 3.48535e-08, -1.29072e-07, -6.84524e-08, 6.34311e-07, -1.40987e-07, 7.4675e-08,  ],
              [ 7.14979e-08, 2.48447e-07, -1.43358e-08, -1.40987e-07, 3.01596e-07, 6.34809e-07,  ],
              [ 9.08954e-07, 2.8084e-06, -2.84269e-08, 7.4675e-08, 6.34809e-07, 1.05999e-05,  ],),)
print(Var - Var_c)
     */
#endif

    // This is similar to what I do in mrcal, but much simpler. In mrcal, every
    // camera is described by its intrinsics and extrinsics, and much effort
    // goes into disentangling the two. Here we have only the poses.
    //
    // The blueprint of what to do is in https://mrcal.secretsauce.net/uncertainty.html
    //
    // I assume SCALE_MEASUREMENT_PX and SCALE_MEASUREMENT_M are the stdev of
    // the measurements. This is applied in the cost function So Var(x) = I. So
    //
    //   Var(b) = inv(JtJ) Jobst Jobs inv(JtJ)
    //
    // The regularization is meant to make all the state well-defined even if
    // the data in the problem does not do that. The intent is that
    // regularization is very light, and only has an appreciable effect on the
    // solution if the problem if the input data isn't very good. If I ignore
    // the regularization here (by assuming that Var(b) = inv(JtJ)) then I will
    // either
    //
    // - end up with a very similar result, if the data is sufficient
    //
    // - come up with an over-estimate of confidence for any state that wasn't
    //   well-defined. But any state that's defined purely by the regularization
    //   terms will still display a very high uncertainty, so qualitatively,
    //   this is still fine
    //
    // So I make my life easy, and simply compute Var(b) = inv(JtJ)
    //
    // Here I only care about the covariance of the sensor poses: I don't care
    // about the poses of the chessboards. So I retrieve only that block from
    // inv(JtJ)

    // make sure pp_solver_context->factorization exists
    dogleg_computeJtJfactorization(solver_context->beforeStep,
                                   solver_context);

    callback_context_t ctx = {.Ncameras              = Ncameras,
                              .Nlidars               = Nlidars,
                              .Nsnapshots            = Nsnapshots,
                              .snapshots             = snapshots,
                              .bitarray_snapshots_selected = bitarray_snapshots_selected,

                              .models                = models,
                              .object_height_n       = object_height_n,
                              .object_width_n        = object_width_n,
                              .object_spacing        = object_spacing};

    // lidar1 is the first optimized state: lidar0 defines the coordinate system
    // reference
    const int istate_lidar1  = state_index_lidar (1, &ctx);
    const int Nstate_lidar   = num_states_lidars (&ctx);
    const int istate_camera0 = state_index_camera(0, &ctx);
    const int Nstate_camera  = num_states_cameras(&ctx);

    const int Nstate         = num_states(&ctx);


    if(! (istate_lidar1 == 0 &&
          ( Nstate_camera == 0 ||
            istate_camera0 == istate_lidar1 + Nstate_lidar )))
    {
        MSG("Uncertainty computation assumes the state vector begins with lidar poses followed immediately by camera poses");
        return false;
    }


    // Used in the uncertainty computation
    cholmod_dense*  Y    = NULL;
    cholmod_dense*  E    = NULL;
    cholmod_sparse* Xset = NULL;


    // I get inv(JtJ) by solving JtJ x = I. I could do this with a sparse rhs,
    // by specifying Bset, but this is not reliable at this time:
    // https://github.com/DrTimothyAldenDavis/SuiteSparse/issues/892

    // Internally cholmod likes to do things in chunks of 4 vectors, so I let it

    double b_x[Nstate*4];
    cholmod_dense B = {
        .nrow  = Nstate,
        .ncol  = 4,
        .nzmax = Nstate*4,
        .d     = Nstate,
        .x     = b_x,
        .xtype = CHOLMOD_REAL,
        .dtype = CHOLMOD_DOUBLE };
    memset(b_x, 0, sizeof(b_x[0]) * Nstate*4);

    double v_x[Nstate*4];
    cholmod_dense V = {
        .nrow  = Nstate,
        .ncol  = 4,
        .nzmax = Nstate*4,
        .d     = Nstate,
        .x     = v_x,
        .xtype = CHOLMOD_REAL,
        .dtype = CHOLMOD_DOUBLE };
    cholmod_dense* pV = &V;

    const int Nstate_sensor_poses = Nstate_lidar + Nstate_camera;
    for(int i=0; i<Nstate_sensor_poses; i++)
    {
        b_x[ (i%4)*Nstate + i ] = 1.0;

        if( (i%4) == 3 || i == Nstate_sensor_poses-1 )
        {
            // The last column is filled-in. Solve.
            if(i == Nstate_sensor_poses-1 && (i%4) != 3)
            {
                // Last solve of  < 4 columns
                B.ncol = Nstate_sensor_poses % 4;
                V.ncol = Nstate_sensor_poses % 4;
            }

            if(!cholmod_solve2( CHOLMOD_A, solver_context->factorization,
                                &B, NULL,
                                &pV, &Xset, &Y, &E,
                                &solver_context->common))
            {
                MSG("cholmod_solve2() failed when computing the uncertainty. Giving up\n");
                goto done;
            }

            for(int j=0; j<(int)(V.ncol); j++)
                memcpy(&Var_rt_lidar0_sensor[(i-V.ncol+1 + j)*Nstate_sensor_poses],
                       &v_x[j*Nstate],
                       Nstate_sensor_poses*sizeof(Var_rt_lidar0_sensor[0]));


            // A1 = factorization.solve_xt_JtJ_bt( dF_dbpacked, sys='P' )
            // del dF_dbpacked
            // A2 = factorization.solve_xt_JtJ_bt( A1,          sys='L' )
            // del A1
            // A3 = factorization.solve_xt_JtJ_bt( A2,          sys='D' )
            // Var_dF = nps.matmult(A2, nps.transpose(A3))

            // reset b for the next round
            b_x[ ((i-0)%4)*Nstate + (i-0) ] = 0.0;
            b_x[ ((i-1)%4)*Nstate + (i-1) ] = 0.0;
            b_x[ ((i-2)%4)*Nstate + (i-2) ] = 0.0;
            b_x[ ((i-3)%4)*Nstate + (i-3) ] = 0.0;
        }
    }

    // Var_rt_lidar0_sensor[] is now computed, but everything is in respect to
    // unpacked variables. I repack it
    //
    // Scale the rows
    int k;
    k = 0;
    for(int i=0; i<Nstate_sensor_poses; i++)
    {
        for(int j=0; j<(Nstate_lidar)/6; j++)
        {
            Var_rt_lidar0_sensor[k++] *= SCALE_ROTATION_LIDAR;
            Var_rt_lidar0_sensor[k++] *= SCALE_ROTATION_LIDAR;
            Var_rt_lidar0_sensor[k++] *= SCALE_ROTATION_LIDAR;
            Var_rt_lidar0_sensor[k++] *= SCALE_TRANSLATION_LIDAR;
            Var_rt_lidar0_sensor[k++] *= SCALE_TRANSLATION_LIDAR;
            Var_rt_lidar0_sensor[k++] *= SCALE_TRANSLATION_LIDAR;
        }
        for(int j=0; j<(Nstate_camera)/6; j++)
        {
            Var_rt_lidar0_sensor[k++] *= SCALE_ROTATION_CAMERA;
            Var_rt_lidar0_sensor[k++] *= SCALE_ROTATION_CAMERA;
            Var_rt_lidar0_sensor[k++] *= SCALE_ROTATION_CAMERA;
            Var_rt_lidar0_sensor[k++] *= SCALE_TRANSLATION_CAMERA;
            Var_rt_lidar0_sensor[k++] *= SCALE_TRANSLATION_CAMERA;
            Var_rt_lidar0_sensor[k++] *= SCALE_TRANSLATION_CAMERA;
        }
    }
    // And again, but scaling the columns
    k = 0;
    for(int j=0; j<(Nstate_lidar)/6; j++)
    {
        for(int i=0; i<Nstate_sensor_poses*3; i++)
            Var_rt_lidar0_sensor[k++] *= SCALE_ROTATION_LIDAR;
        for(int i=0; i<Nstate_sensor_poses*3; i++)
            Var_rt_lidar0_sensor[k++] *= SCALE_TRANSLATION_LIDAR;
    }
    for(int j=0; j<(Nstate_camera)/6; j++)
    {
        for(int i=0; i<Nstate_sensor_poses*3; i++)
            Var_rt_lidar0_sensor[k++] *= SCALE_ROTATION_CAMERA;
        for(int i=0; i<Nstate_sensor_poses*3; i++)
            Var_rt_lidar0_sensor[k++] *= SCALE_TRANSLATION_CAMERA;
    }


#if 0
    printf("np.array((");
    for(int i=0;i<Nstate_sensor_poses; i++)
    {
        printf("[ ");
        for(int j=0;j<Nstate_sensor_poses; j++)
        {
            printf("%g, ", Var_rt_lidar0_sensor[i*Nstate_sensor_poses + j]);
        }
        printf(" ],\n");
    }
    printf("),)");
#endif

    result = true;

 done:
    if(E    != NULL) cholmod_free_dense (&E,    &solver_context->common);
    if(Y    != NULL) cholmod_free_dense (&Y,    &solver_context->common);
    if(Xset != NULL) cholmod_free_sparse(&Xset, &solver_context->common);

    return result;
}


// Var_p1 += Ma Vab Mbt + Mb Vba Mat
static void
Ma_Var_Mbt_block_accumulate(// out

                            // (3,3) symmetric matrix; upper-triangle only stored; 6 values
                            double* Var_p1,
                            // in

                            // Nstate_sensor_poses x Nstate_sensor_poses matrix
                            const double* Var_rt_lidar0_sensor,
                            const int Nstate_sensor_poses,

                            // each is 3x6 matrix, starting at the given row/col of Var_rt_lidar0_sensor
                            const double* dp1__drt_lidar0_lidar0, const int istate_lidar0,
                            const double* dp1__drt_lidar0_lidar1, const int istate_lidar1)
{
    const double* V0 = &Var_rt_lidar0_sensor[Nstate_sensor_poses*istate_lidar0 + istate_lidar1];

    double V0M1t[6*3];
    multiply_matrix_matrix(// out
                           V0M1t, 3, 1,
                           // in
                           V0, Nstate_sensor_poses, 1,
                           dp1__drt_lidar0_lidar1, 1, 6,
                           6,6,3,
                           false);

    double Ma_Var_Mbt[3*3];

    multiply_matrix_matrix(// out
                           Ma_Var_Mbt, 3, 1,
                           // in
                           dp1__drt_lidar0_lidar0, 6, 1,
                           V0M1t, 3, 1,
                           3,6,3,
                           false);

    // I now have Ma_Var_Mbt. I compute Ma_Var_Mbt + (Ma_Var_Mbt)t
    int k=0;
    for(int i=0; i<3; i++)
        for(int j=i; j<3; j++)
        {
            double t =
                Ma_Var_Mbt[i*3 + j] +
                Ma_Var_Mbt[j*3 + i];
            Var_p1[k] += t;

            k++;
        }
}

// Var_p1 += Ma Vab Mat
static void
M_Var_Mt_block_accumulate(// out

                          // (3,3) symmetric matrix; upper-triangle only stored; 6 values
                          double* Var_p1,
                          // in

                          // Nstate_sensor_poses x Nstate_sensor_poses matrix
                          const double* Var_rt_lidar0_sensor,
                          const int Nstate_sensor_poses,

                          // each is 3x6 matrix, starting at the given row/col of Var_rt_lidar0_sensor
                          const double* dp1__drt, const int istate)
{
    const double* V0 = &Var_rt_lidar0_sensor[Nstate_sensor_poses*istate + istate];

    double V0M1t[6*3];
    multiply_matrix_matrix(// out
                           V0M1t, 3, 1,
                           // in
                           V0, Nstate_sensor_poses, 1,
                           dp1__drt, 1, 6,
                           6,6,3,
                           false);

    double M_Var_Mt[3*3];
    multiply_matrix_matrix(// out
                           M_Var_Mt, 3, 1,
                           // in
                           dp1__drt, 6, 1,
                           V0M1t, 3, 1,
                           3,6,3,
                           false);

    int k=0;
    for(int i=0; i<3; i++)
        for(int j=i; j<3; j++)
        {
            Var_p1[k] += M_Var_Mt[i*3 + j];
            k++;
        }
}


static void accumulate_covariance_pairwise(// out
                                           // (3,3) symmetric matrix; upper-triangle only stored; 6 values
                                           double*               Var_p1,
                                           // in
                                           const mrcal_point3_t* pquery_lidar0_recomputed,
                                           // uninitialized if isensor0==0; same as istate_sensor0<0
                                           const double*         dpref__drt_lidar0_sensor0,
                                           // guaranteed != NULL
                                           const double*         rt_lidar0_sensor1,
                                           const double*         Var_rt_sensor0_sensor,
                                           const int             Nstate_sensor_poses,
                                           // may be <0
                                           const int             istate_sensor0,
                                           // guaranteed >= 0
                                           const int             istate_sensor1)
{
    memset(Var_p1, 0, 6*sizeof(double));

    double dp1__drt_lidar0_sensor1[3*6];
    double dp1__dpref             [3*3];
    mrcal_point3_t p1; // don't need to store this; I just want the gradient
    // rt_lidar0_sensor1!=NULL is guaranteed here
    mrcal_transform_point_rt_inverted(p1.xyz,dp1__drt_lidar0_sensor1,dp1__dpref,
                                      rt_lidar0_sensor1, pquery_lidar0_recomputed->xyz);

    // I have a point from the sensor0 frame expressed in
    // sensor1 coords. The covariance of this quantity is the
    // uncertainty between these two sensors in this
    // location.
    //
    // p1 = f( rt_lidar0_sensor1, pref )
    //    = f( rt_lidar0_sensor1, g( rt_lidar0_sensor0, p0 ) )
    // dp1 ~ dp1/drt_lidar0_sensor1 drt_lidar0_sensor1 + dp1/dpref dpref/drt_lidar0_sensor0 drt_lidar0_sensor0
    //
    // Var(dp1) = M Var(b) Mt
    //
    // where M is dp1/db where b is the solver state. This
    // solver state contains the scaled quantities
    // rt_lidar0_sensor0 and rt_lidar0_sensor1. So we have
    //
    //   (dp1/db)[:,istate_rt_lidar0_sensor0] = dp1/dpref dpref/drt_lidar0_sensor0
    //   (dp1/db)[:,istate_rt_lidar0_sensor1] = dp1/drt_lidar0_sensor1
    //

    // I really should use the factorization here instead of
    // the precomputed Var=inv(JtJ).

    // I need M Var Mt; M has two non-zero blocks: one for
    // drt_lidar0_sensor0 and another for drt_lidar0_sensor1. So I
    // only need to look at those blocks of M:
    //
    // M Var Mt = d0 Var00 d0t + d0 Var01 d1t + d1 Var10 d0t + d1 Var11 d1t
    //
    // Var is symmetric so (d0 Var01 d1t)t = d1 Var10 d0t
    if(istate_sensor0 >= 0)
    {
        double dp1__drt_lidar0_sensor0[3*6];
        multiply_matrix_matrix(// out
                               dp1__drt_lidar0_sensor0, 6, 1,
                               // in
                               dp1__dpref, 3, 1,
                               dpref__drt_lidar0_sensor0, 6, 1,
                               3,3,6,
                               false);

        Ma_Var_Mbt_block_accumulate(// out
                                    Var_p1,
                                    // in
                                    Var_rt_sensor0_sensor,
                                    Nstate_sensor_poses,
                                    dp1__drt_lidar0_sensor0, istate_sensor0,
                                    dp1__drt_lidar0_sensor1, istate_sensor1);

        M_Var_Mt_block_accumulate(// out
                                  Var_p1,
                                  // in
                                  Var_rt_sensor0_sensor,
                                  Nstate_sensor_poses,
                                  dp1__drt_lidar0_sensor0, istate_sensor0);
    }
    M_Var_Mt_block_accumulate(// out
                              Var_p1,
                              // in
                              Var_rt_sensor0_sensor,
                              Nstate_sensor_poses,
                              dp1__drt_lidar0_sensor1, istate_sensor1);
}

// returns true if isector is observed by isensor
static void
lidar_camera_indices_from_sensor(// out
                                 // <0 if we're at the reference
                                 int* istate_sensor,
                                 const double** rt_lidar0_sensor,
                                 // in
                                 const int isensor,
                                 const double* rt_lidar0_lidar__optimization,
                                 const double* rt_lidar0_camera__optimization,
                                 const unsigned int Nlidars,
                                 const callback_context_t* ctx)
{

    if(isensor < Nlidars)
    {
        const int ilidar = isensor;
        if(ilidar == 0)
        {
            *rt_lidar0_sensor = NULL;
            *istate_sensor    = -1;
        }
        else
        {
            *rt_lidar0_sensor = &rt_lidar0_lidar__optimization[(ilidar-1)*6];
            *istate_sensor = state_index_lidar(ilidar, ctx);
        }
    }
    else
    {
        const int icamera = isensor-Nlidars;
        *rt_lidar0_sensor = &rt_lidar0_camera__optimization[icamera*6];
        *istate_sensor = state_index_camera(icamera, ctx);
    }
}

// The worst-case transformation uncertainty in the given sector is returned in
// *stdev_worst. If we return *stdev_worst=0, that means that there isn't any
// pair of sensors that can see the given sector
static bool
transformation_uncertainty_in_sector(// out
                                   double* stdev_worst,
                                   // dense array of shape (2,); corresponds to stdev_worst
                                   uint16_t* isensors_pair_stdev_worst,
                                   // in
                                   const mrcal_point3_t* pquery_lidar0,
                                   const double* rt_lidar0_lidar__optimization,
                                   const double* rt_lidar0_camera__optimization,
                                   const int isector,
                                   const int Nsectors,
                                   // A dense array of shape (Nsensors,Nsectors)
                                   const uint8_t* isvisible_per_sensor_per_sector,

                                   const double* Var_rt_lidar0_sensor,
                                   // in
                                   const unsigned int Nlidars,
                                   const unsigned int Ncameras)
{
    const
        callback_context_t ctx = {.Ncameras = Ncameras,
                                  .Nlidars  = Nlidars};

    const int Nstate_lidar        = num_states_lidars (&ctx);
    const int Nstate_camera       = num_states_cameras(&ctx);
    const int Nstate_sensor_poses = Nstate_lidar + Nstate_camera;

    const int Nsensors = Nlidars + Ncameras;

    // 0 by default, the best-possible uncertainty value. If we end up returning
    // this, that means that there isn't any pair of sensors that can see this
    // sector
    double l_worst = 0.0;

    isensors_pair_stdev_worst[0] = 0;
    isensors_pair_stdev_worst[1] = 0;

    for(int isensor0=0; isensor0<Nsensors-1; isensor0++)
    {
        // <0 if we're at the reference
        int istate_sensor0;
        const double* rt_lidar0_sensor0;

        if(!isvisible_per_sensor_per_sector[isensor0 * Nsectors + isector])
            continue;
        lidar_camera_indices_from_sensor(// out
                                         // <0 if we're at the reference
                                         &istate_sensor0,
                                         &rt_lidar0_sensor0,
                                         // in
                                         isensor0,
                                         rt_lidar0_lidar__optimization,
                                         rt_lidar0_camera__optimization,
                                         Nlidars,
                                         &ctx);

        mrcal_point3_t p0;
        if(rt_lidar0_sensor0 != NULL)
            mrcal_transform_point_rt_inverted(p0.xyz,NULL,NULL,
                                              rt_lidar0_sensor0, pquery_lidar0->xyz);
        else
            p0 = *pquery_lidar0;

        double dpref__drt_lidar0_sensor0[3*6];
        mrcal_point3_t pquery_lidar0_recomputed; // will be pquery_lidar0 again. I probably don't NEED to store it
        if(rt_lidar0_sensor0 != NULL)
            mrcal_transform_point_rt(pquery_lidar0_recomputed.xyz,dpref__drt_lidar0_sensor0,NULL,
                                     rt_lidar0_sensor0, p0.xyz);
        else
            pquery_lidar0_recomputed = p0;

        for(int isensor1=isensor0+1; isensor1<Nsensors; isensor1++)
        {
            // <1 if we're at the reference
            int istate_sensor1;
            const double* rt_lidar0_sensor1;

            if(!isvisible_per_sensor_per_sector[isensor1 * Nsectors + isector])
                continue;
            lidar_camera_indices_from_sensor(// out
                                             // <1 if we're at the reference
                                             &istate_sensor1,
                                             &rt_lidar0_sensor1,
                                             // in
                                             isensor1,
                                             rt_lidar0_lidar__optimization,
                                             rt_lidar0_camera__optimization,
                                             Nlidars,
                                             &ctx);

            // These two sensors can BOTH observe points in this sector. So their transformation uncertainty should be low

            // (3,3) symmetric matrix; upper-triangle only stored; 6 values
            double Var_p1[6];
            accumulate_covariance_pairwise(// out
                                           Var_p1,
                                           // in
                                           &pquery_lidar0_recomputed,
                                           // uninitialized if isensor0==0; same as istate_sensor0<0
                                           dpref__drt_lidar0_sensor0,
                                           // guaranteed != NULL
                                           rt_lidar0_sensor1,
                                           Var_rt_lidar0_sensor,
                                           Nstate_sensor_poses,
                                           // may be <0
                                           istate_sensor0,
                                           // guaranteed >= 0
                                           istate_sensor1);

            double l[3]; // ALL the eigenvalues, in ascending order

            eig_real_symmetric_3x3( // out
                                    NULL,
                                    NULL,
                                    l,          // ALL the eigenvalues, in ascending order
                                    // in
                                    Var_p1,     // shape (6,); packed storage; row-first
                                    false );

            // I find the worst worst-case eigenvalue
            if(l[2] > l_worst)
            {
                l_worst = l[2];
                isensors_pair_stdev_worst[0] = isensor0;
                isensors_pair_stdev_worst[1] = isensor1;
            }
        }
    }

    if(l_worst >= 0.0)
        *stdev_worst = sqrt(l_worst);
    else
    {
        // round-off path
        // We have overlapping data, but it's VERY uncertain
        *stdev_worst = 1e-20;
    }

    return true;
}


static bool is_point_in_view_of_camera(// in
                                       const mrcal_point3_t* p,
                                       const mrcal_cameramodel_t* model)
{
    mrcal_point2_t q;
    if(!mrcal_project(&q, NULL,NULL,
                      p,
                      1,
                      &model->lensmodel, model->intrinsics))
        return false;

    if(q.x < 0 || q.y < 0 ||
       q.x > model->imagersize[0]-1 || q.y > model->imagersize[1]-1)
        return false;

    // project() found a pixel q that maps to the point p. The nonlinear effects
    // of the equations out-of-bounds of the camera could have played a role
    // here. I unproject q back to the world and compare with the requested p.
    // If I get the vector in space, then I know this point WAS in view
    mrcal_point3_t p1;
    if(!mrcal_unproject(&p1,
                        &q,
                        1,
                        &model->lensmodel, model->intrinsics))
        return false;

    // p1 is not normalized. And it doesn't have the same magnitude as p. I look
    // at the angle error between them
    const double cos_error_mag_mag =
        mrcal_point3_inner(*p, p1);
    const double mag_mag_sq = mrcal_point3_norm2(*p) * mrcal_point3_norm2(p1);
    const double cos_err_max = cos(0.01 * M_PI/180.);
    if( cos_error_mag_mag*cos_error_mag_mag < cos_err_max*cos_err_max*mag_mag_sq )
        return false;

    return true;
}

static void evaluate_camera_visibility(// out
                                       // dense array of shape (Ncameras,Nsectors)
                                       uint8_t* isvisible_per_camera_per_sector,
                                       // in
                                       const double* Rt_vehicle_lidar0, // may be NULL
                                       const double uncertainty_quantification_range,
                                       const int Nsectors,
                                       const int Ncameras,
                                       const double* Rt_lidar0_camera,
                                       const mrcal_cameramodel_t*const* models)
{
    const double sector_width_rad = 2.*M_PI/(double)Nsectors;
    const double c = cos(sector_width_rad);
    const double s = sin(sector_width_rad);

    // I will rotate this by sector_width_rad with each step
    mrcal_point3_t pquery_vehicle_left =
        {.x = uncertainty_quantification_range * 1.,
         .y = uncertainty_quantification_range * 0.,
         .z = 0};

    mrcal_point3_t pquery_lidar0_left;
    if(Rt_vehicle_lidar0 != NULL)
        mrcal_transform_point_Rt_inverted(pquery_lidar0_left.xyz,NULL,NULL,
                                          Rt_vehicle_lidar0, pquery_vehicle_left.xyz);
    else
        pquery_lidar0_left = pquery_vehicle_left;

    const int Nwords_cameras = bitarray64_nwords(Ncameras);

    uint64_t bitarray_isvisible_left_per_camera[Nwords_cameras];
    for(int icamera=0; icamera<Ncameras; icamera++)
    {
        mrcal_point3_t pquery_cam_left;
        mrcal_transform_point_Rt_inverted(pquery_cam_left.xyz,NULL,NULL,
                                          &Rt_lidar0_camera[4*3*icamera], pquery_lidar0_left.xyz);
        if(is_point_in_view_of_camera(&pquery_cam_left, models[icamera]))
            bitarray64_set(  bitarray_isvisible_left_per_camera, icamera);
        else
            bitarray64_clear(bitarray_isvisible_left_per_camera, icamera);
    }



    for(int isector=0; isector<Nsectors; isector++)
    {
        const double x = pquery_vehicle_left.x;
        const double y = pquery_vehicle_left.y;
        mrcal_point3_t pquery_vehicle_right =
            {.x = x*c - y*s,
             .y = y*c + x*s,
             .z = 0};

        mrcal_point3_t pquery_lidar0_right;
        if(Rt_vehicle_lidar0 != NULL)
            mrcal_transform_point_Rt_inverted(pquery_lidar0_right.xyz,NULL,NULL,
                                              Rt_vehicle_lidar0, pquery_vehicle_right.xyz);
        else
            pquery_lidar0_right = pquery_vehicle_right;

        uint64_t bitarray_isvisible_right_per_camera[Nwords_cameras];
        for(int icamera=0; icamera<Ncameras; icamera++)
        {
            mrcal_point3_t pquery_cam_right;
            mrcal_transform_point_Rt_inverted(pquery_cam_right.xyz,NULL,NULL,
                                              &Rt_lidar0_camera[4*3*icamera], pquery_lidar0_right.xyz);

            if(is_point_in_view_of_camera(&pquery_cam_right, models[icamera]))
                bitarray64_set(  bitarray_isvisible_right_per_camera, icamera);
            else
            {
                bitarray64_clear(bitarray_isvisible_right_per_camera, icamera);
                isvisible_per_camera_per_sector[icamera * Nsectors + isector] = 0;
                continue;
            }

            // I just checked if we're visible on the right. If it's ALSO
            // visible on the left, report this sector as visible
            isvisible_per_camera_per_sector[icamera * Nsectors + isector] =
                bitarray64_check(bitarray_isvisible_left_per_camera,  icamera);
        }

        pquery_vehicle_left = pquery_vehicle_right;
        pquery_lidar0_left  = pquery_lidar0_right;
        for(int iword=0; iword<Nwords_cameras; iword++)
            bitarray_isvisible_left_per_camera[iword] = bitarray_isvisible_right_per_camera[iword];
    }
}

static bool check_sufficient_observations(const sensor_snapshot_segmented_t* snapshots,
                                          const int                          Nsnapshots,
                                          const uint64_t*                    bitarray_snapshots_selected,
                                          const int                          Ncameras,
                                          const int                          Nlidars)
{
    int NlidarObservations [Nlidars ];
    int NcameraObservations[Ncameras];

    // I pass through the observations again, to make sure that I count the
    // FILTERED snapshots, not the original ones
    for(unsigned int i=0; i<Ncameras; i++) NcameraObservations[i] = 0;
    for(unsigned int i=0; i<Nlidars;  i++) NlidarObservations [i] = 0;

    LOOP_SNAPSHOT()
    {
        LOOP_SNAPSHOT_HEADER(,const);
        for(unsigned int ilidar=0; ilidar<Nlidars; ilidar++)
            if(snapshot->lidar_scans[ilidar].points != NULL)
                NlidarObservations[ilidar]++;

        for(unsigned int icamera=0; icamera<Ncameras; icamera++)
            if(snapshot->chessboard_corners[icamera] != NULL)
                NcameraObservations[icamera]++;

    }
    for(unsigned int i=0; i<Ncameras; i++)
    {
        const int NcameraObservations_this = NcameraObservations[i];
        if (NcameraObservations_this == 0)
        {
            MSG("I need at least 1 observation of each camera. Got only %d for camera %d",
                NcameraObservations_this, i);
            return false;
        }
    }

    for(unsigned int i=0; i<Nlidars; i++)
    {
        const int NlidarObservations_this = NlidarObservations[i];
        if (NlidarObservations_this < 3)
        {
            MSG("I need at least 3 observations of each lidar to unambiguously set the translation (the set of all plane normals must span R^3). Got only %d for lidar %d",
                NlidarObservations_this, i);
            return false;
        }
    }
    return true;
}


static void
get_observations_per_sector(// out
                            // dense array of shape (Nsectors,)
                            uint16_t* observations_per_sector,
                            // in
                            const sensor_snapshot_segmented_t* snapshots,
                            const uint64_t* bitarray_snapshots_selected,
                            const int Nsnapshots,
                            const int Nlidars,
                            const int Ncameras,
                            const int Nsectors,
                            const int object_height_n,
                            const int object_width_n,
                            const double object_spacing,
                            const mrcal_cameramodel_t*const* models, // Ncameras of these
                            // Nsnapshots (4,3) arrays
                            const double* Rt_lidar0_board,
                            // Nlidars-1 (4,3) arrays
                            const double* Rt_lidar0_lidar,
                            // Ncameras (4,3) arrays
                            const double* Rt_lidar0_camera,
                            // may be NULL,
                            const double* Rt_vehicle_lidar0,
                            // may be NULL; used only if we have
                            // Rt_lidar0_lidar/camera; and NOT Rt_lidar0_board
                            double* Rt_camera_board_cache)
{
    // The estimate of the center of the board, in board coords. We
    // compute_board_poses()
    mrcal_point3_t pboardcenter_board;
    get_pboardcenter_board(// out
                           pboardcenter_board.xyz,
                           // in
                           object_height_n,
                           object_width_n,
                           object_spacing,
                           Ncameras);
    memset(observations_per_sector, 0, Nsectors*sizeof(observations_per_sector[0]));

    LOOP_SNAPSHOT()
    {
        LOOP_SNAPSHOT_HEADER(,const);

        // This snapshot was used in the solve, so it has at least 2
        // observing sensors; LOOP_SNAPSHOT_HEADER() skipped the ones that
        // didn't
        mrcal_point3_t pboardcenter_ref;

        if(Rt_lidar0_board != NULL)
            mrcal_transform_point_Rt(pboardcenter_ref.xyz, NULL, NULL,
                                     &Rt_lidar0_board[isnapshot*4*3], pboardcenter_board.xyz);
        else
        {
            // Don't have the board poses, but I have the sensor poses, so I can
            // approximate. I estimate the board center in ref coords (lidar0 or
            // vehicle) for each sensor, and I average them

            mrcal_point3_t pboardcenter;
            int Naccumulated = 0;
            for(int ilidar=0; ilidar<Nlidars; ilidar++)
            {
                if(snapshot->lidar_scans[ilidar].points == NULL)
                    continue;

                mrcal_point3_from_clc_point3f(&pboardcenter,
                                              &snapshot->lidar_scans[ilidar].plane.p_mean);

                if(ilidar>0)
                    mrcal_transform_point_Rt(pboardcenter.xyz,NULL,NULL,
                                             &Rt_lidar0_lidar[(ilidar-1)*4*3], pboardcenter.xyz);
                pboardcenter_ref = mrcal_point3_add(pboardcenter_ref,pboardcenter);
                Naccumulated++;
            }

            for(int icamera=0; icamera<Ncameras; icamera++)
            {
                if(snapshot->chessboard_corners[icamera] == NULL)
                    continue;

                double* Rt_camera_board = &Rt_camera_board_cache[ (isnapshot*Ncameras + icamera) *4*3];
                if(!fit_Rt_camera_board_withcache(// out
                                                  Rt_camera_board,
                                                  // in
                                                  models[icamera],
                                                  snapshot->chessboard_corners[icamera],
                                                  object_height_n,
                                                  object_width_n,
                                                  object_spacing))
                    return false;

                mrcal_transform_point_Rt(pboardcenter.xyz,NULL,NULL,
                                         Rt_camera_board, pboardcenter_board.xyz);
                mrcal_transform_point_Rt(pboardcenter.xyz,NULL,NULL,
                                         &Rt_lidar0_camera[icamera*4*3], pboardcenter.xyz);
                pboardcenter_ref = mrcal_point3_add(pboardcenter_ref,pboardcenter);
                Naccumulated++;
            }

            pboardcenter_ref = mrcal_point3_scale(pboardcenter_ref, 1. / Naccumulated);
        }

        if(Rt_vehicle_lidar0 != NULL)
            mrcal_transform_point_Rt(pboardcenter_ref.xyz,NULL,NULL,
                                     Rt_vehicle_lidar0, pboardcenter_ref.xyz);
        observations_per_sector[ isector_from_pvehicle(pboardcenter_ref.xyz, Nsectors) ]++;
    }
}

static bool
get_isvisible_per_sensor_per_sector(// out
                                    // A dense array of shape (Nsensors,Nsectors); may be NULL
                                    uint8_t* isvisible_per_sensor_per_sector,
                                    // in
                                    const int Nlidars,
                                    const int Ncameras,
                                    const int Nsectors,
                                    // Nlidars-1 (4,3) arrays
                                    const double* Rt_lidar0_lidar,
                                    // Ncameras (4,3) arrays
                                    const double* Rt_lidar0_camera,
                                    // may be NULL,
                                    const double* Rt_vehicle_lidar0,
                                    // used for isvisible_per_sensor_per_sector
                                    const double threshold_valid_lidar_range,
                                    const int    threshold_valid_lidar_Npoints,
                                    const double uncertainty_quantification_range,
                                    const clc_lidar_scan_unsorted_t* lidar_scans_for_isvisible,
                                    const unsigned int lidar_packet_stride,
                                    // Ncameras of these
                                    const mrcal_cameramodel_t*const* models)
{
    // It is more numerically stable to not compute Var=inv(JtJ) and
    // then compute M Var Mt, but instead to use the factorization of
    // JtJ to compute M Var Mt directly. But using the covariance code I
    // already wrote is easier, so I do that for now


    // vehicle coords are
    // x: forward
    // y: left
    // z: up
    if(!evaluate_lidar_visibility(// out
                                  isvisible_per_sensor_per_sector,
                                  // in
                                  Rt_vehicle_lidar0,
                                  Nsectors,
                                  _Nrings,
                                  Rt_lidar0_lidar,
                                  threshold_valid_lidar_range,
                                  threshold_valid_lidar_Npoints,

                                  lidar_scans_for_isvisible,
                                  lidar_packet_stride,
                                  Nlidars))
        return false;

    evaluate_camera_visibility(// out
                               &isvisible_per_sensor_per_sector[Nlidars * Nsectors],
                               // in
                               Rt_vehicle_lidar0,
                               uncertainty_quantification_range,
                               Nsectors,
                               Ncameras,
                               Rt_lidar0_camera,
                               models);

    return true;
}


// Each sensor is uniquely identified by its position in the
// sensor_snapshots[].lidar_scans[] or .images[] arrays. An unobserved sensor in
// some sensor snapshot should be indicated by lidar_scans[] = {} or images[] =
// {}
//
// The rt_lidar0|vehicle_lidar|camera arrays are the input/output. Some (but not
// all) may be NULL
//
// We return true on success
bool clc(// in/out
         // On input:
         //   if(     use_given_seed_geometry_lidar0):  rt_lidar0_...  are used as a seed
         //   else if(use_given_seed_geometry_vehicle): rt_vehicle_... are used as a seed
         //   else: neither is used as the seed, and we COMPUTE the initial geometry
         //
         //   if a seed is given, rt_lidar0_lidar[0] MUST be the identity transform
         //
         // On output:
         //   we store the solution into all of these that are != NULL. If
         //   possible, both rt_lidar0_... and rt_vehicle_... are populated
         //
         // If rt_vehicle_... are given (for input or output), then
         // rt_vehicle_lidar0 MUST be non-NULL
         mrcal_pose_t* rt_lidar0_lidar,   // Nlidars  of these; some may be NULL
         mrcal_pose_t* rt_lidar0_camera,  // Ncameras of these; some may be NULL
         mrcal_pose_t* rt_vehicle_lidar,  // Nlidars  of these; some may be NULL
         mrcal_pose_t* rt_vehicle_camera, // Ncameras of these; some may be NULL
         // at most one of these should be true
         bool          use_given_seed_geometry_lidar0,
         bool          use_given_seed_geometry_vehicle,

         // Covariance of the output. Symmetric matrix of shape
         // (Nstate_sensor_poses,Nstate_sensor_poses) stored densely, written on
         // output. Nstate_sensor_poses = (Nlidars-1 + Ncameras)*6; may be NULL
         double*       Var_rt_lidar0_sensor,
         // dense array of shape (Nsectors,); may be NULL
         // Will try to set this even if clc() failed: from
         // rt_lidar0/vehicle_lidar_camera and use_given_seed_geometry_... If it
         // really couldn't be computed from those either, all entries will be
         // set to 0
         uint16_t* observations_per_sector,

         // A dense array of shape (Nsensors,Nsectors); may be NULL
         // Needs lidar_scans_for_isvisible!=NULL
         // Will try to set this even if clc() failed: from
         // rt_lidar0/vehicle_lidar_camera and use_given_seed_geometry_... If it
         // really couldn't be computed from those either, all entries will be
         // set to 0
         uint8_t* isvisible_per_sensor_per_sector,

         // array of shape (Nsectors,); may be NULL
         // if not NULL, requires that
         //   isvisible_per_sensor_per_sector!=NULL && Var_rt_lidar0_sensor!=NULL
         double* stdev_worst_per_sector,
         // dense array of shape (Nsectors,2); may be NULL
         uint16_t* isensors_pair_stdev_worst,
         const int Nsectors,
         // used for isvisible_per_sensor_per_sector
         const double threshold_valid_lidar_range,
         const int    threshold_valid_lidar_Npoints,
         // used for isvisible_per_sensor_per_sector and stdev_worst_per_sector
         const double uncertainty_quantification_range,

         // Pass non-NULL to get the fit-inputs dump. These encode the data
         // buffer. The caller must free(*buf_inputs_dump) when done. Even when
         // this call fails
         char**  buf_inputs_dump,
         size_t* size_inputs_dump,

         // in

         // Exactly one of these should be non-NULL
         const clc_sensor_snapshot_unsorted_t*        sensor_snapshots_unsorted,
         const clc_sensor_snapshot_sorted_t*          sensor_snapshots_sorted,
         const clc_sensor_snapshot_segmented_t*       sensor_snapshots_segmented,
         const clc_sensor_snapshot_segmented_dense_t* sensor_snapshots_segmented_dense,

         const unsigned int                    Nsnapshots,
         // The stride, in bytes, between each successive points or rings value
         // in sensor_snapshots_unsorted and lidar_scans_for_isvisible; unused
         // if either of those is NULL
         const unsigned int           lidar_packet_stride,

         // Nlidars of these. Required if isvisible_per_sensor_per_sector!=NULL
         const clc_lidar_scan_unsorted_t* lidar_scans_for_isvisible,

         const unsigned int Nlidars,
         const unsigned int Ncameras,
         const mrcal_cameramodel_t*const* models, // Ncameras of these
         // The dimensions of the chessboard grid being detected in the images
         const int object_height_n,
         const int object_width_n,
         const double object_spacing,

         // bits indicating whether a camera in sensor_snapshots.images[] is
         // color or not
         // unused if sensor_snapshots_unsorted==NULL &&
         // sensor_snapshots_sorted==NULL
         const clc_is_bgr_mask_t is_bgr_mask,
         // unused if sensor_snapshots_unsorted==NULL &&
         // sensor_snapshots_sorted==NULL
         const clc_lidar_segmentation_context_t* ctx,

         const mrcal_pose_t* rt_vehicle_lidar0,

         const double fit_seed_position_err_threshold,
         const double fit_seed_cos_angle_err_threshold,
         bool check_gradient__use_distance_to_plane,
         bool check_gradient,
         bool verbose)
{
    // Reset isvisible_per_sensor_per_sector, observations_per_sector. I will
    // try to set these even if clc() failed (from the seed values), and and
    // all-0 results will indicate that this failed also
    if(isvisible_per_sensor_per_sector != NULL)
        memset(isvisible_per_sensor_per_sector, 0,
               (Nlidars+Ncameras)*Nsectors*sizeof(isvisible_per_sensor_per_sector[0]));
    if(observations_per_sector != NULL)
        memset(observations_per_sector, 0,
               Nsectors*sizeof(observations_per_sector[0]));



    if(1 !=
       (sensor_snapshots_unsorted        != NULL) +
       (sensor_snapshots_sorted          != NULL) +
       (sensor_snapshots_segmented       != NULL) +
       (sensor_snapshots_segmented_dense != NULL))
    {
        MSG("Exactly one of (sensor_snapshots_sorted,sensor_snapshots_unsorted,sensor_snapshots_segmented,sensor_snapshots_segmented_dense) should be non-NULL");
        return false;
    }

    if( (!!use_given_seed_geometry_lidar0) +
        (!!use_given_seed_geometry_vehicle) > 1)
    {
        MSG("At most one of (use_given_seed_geometry_lidar0,use_given_seed_geometry_vehicle) may be true");
        return false;
    }

    if(rt_vehicle_lidar0 == NULL &&
       !(rt_vehicle_lidar == NULL && rt_vehicle_camera == NULL))
    {
        MSG("rt_vehicle_lidar0 is NULL, so both of (rt_vehicle_lidar,rt_vehicle_camera) MUST be NULL");
        return false;
    }

    if(use_given_seed_geometry_lidar0 &&
       !(rt_lidar0_lidar != NULL &&
         (Ncameras==0 || rt_lidar0_camera != NULL)))
    {
        MSG("use_given_seed_geometry_lidar0 is true, so both of (rt_lidar0_lidar,rt_lidar0_camera) MUST be non-NULL; rt_lidar0_camera may be NULL if Ncameras==0");
        return false;
    }
    if(use_given_seed_geometry_vehicle &&
       !(rt_vehicle_lidar != NULL && (Ncameras==0 || rt_vehicle_camera != NULL) && rt_vehicle_lidar0 != NULL))
    {
        MSG("use_given_seed_geometry_vehicle is true, so all of (rt_vehicle_lidar,rt_vehicle_camera,rt_vehicle_lidar0) MUST be non-NULL; ; rt_vehicle_camera may be NULL if Ncameras==0");
        return false;
    }

    if(stdev_worst_per_sector != NULL &&
       !(isvisible_per_sensor_per_sector!=NULL && Var_rt_lidar0_sensor!=NULL))
    {
        MSG("stdev_worst_per_sector is not NULL, so we MUST have (isvisible_per_sensor_per_sector!=NULL && Var_rt_lidar0_sensor!=NULL)");
        return false;
    }

    if(isvisible_per_sensor_per_sector != NULL &&
       lidar_scans_for_isvisible == NULL)
    {
        MSG("isvisible_per_sensor_per_sector!=NULL so lidar_scans_for_isvisible MUST be !=NULL as well");
        return false;
    }

    bool result = false;

    // I need to segment the LIDAR data. And I need to select a subset of the
    // sensor snapshots (by looking for overlapping sensor measurements,
    // throwing out inconsistent data as outliers, etc). The segmentation
    // results go here. The snapshot selections are in
    // bitarray_snapshots_selected
    sensor_snapshot_segmented_t snapshots[Nsnapshots];

    // I start with ALL the snapshots DISABLED. As I get overlapping
    // observations, I add them one by one
    const int Nwords_bitarray_snapshots_selected = bitarray64_nwords(Nsnapshots);
    uint64_t bitarray_snapshots_selected[Nwords_bitarray_snapshots_selected];
    bitarray64_clear_all(bitarray_snapshots_selected, Nwords_bitarray_snapshots_selected);


    // If we need to sort or segment the lidar data, I need to allocate memory.
    // Here I get the buffer sizes
    int Npoints_buffer      = 0;
    int Nlidar_scans_buffer = 0;
    for(unsigned int isnapshot=0; isnapshot < Nsnapshots; isnapshot++)
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

    // round up to the next multiple of 8
    const int Nbytes_pool_lidar =
        ( Npoints_buffer * sizeof(clc_point3f_t) +
          (Nlidar_scans_buffer + Nplanes_max) * sizeof(clc_points_and_plane_t)
          + (8-1)) & ~(8-1);
    const int Nbytes_pool_camera =
        ( sensor_snapshots_segmented       != NULL ||
          sensor_snapshots_segmented_dense != NULL ) ?
        0 :
        Nsnapshots*Ncameras*object_width_n*object_height_n*sizeof(mrcal_point2_t);

    // contains the points_pool and the points_and_plane_pool and the chessboard
    // grid detections
    uint8_t* pool = malloc(Nbytes_pool_lidar +
                           Nbytes_pool_camera);
    if(pool == NULL)
    {
        MSG("malloc() failed. Giving up");
        goto done;
    }

    clc_point3f_t*          points_pool           = (clc_point3f_t*)pool;
    clc_points_and_plane_t* points_and_plane_pool = (clc_points_and_plane_t*)&pool[Npoints_buffer * sizeof(clc_point3f_t)];
    int points_pool_index           = 0;
    int points_and_plane_pool_index = 0;

    mrcal_point2_t* chessboard_corners_pool =
        ( sensor_snapshots_segmented       != NULL ||
          sensor_snapshots_segmented_dense != NULL ) ?
        NULL :
        (mrcal_point2_t*)&pool[Nbytes_pool_lidar];


    int Nsnapshots_selected = 0;
    for(unsigned int isnapshot=0; isnapshot < Nsnapshots; isnapshot++)
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
        const clc_sensor_snapshot_segmented_dense_t* sensor_snapshot_segmented_dense =
            (sensor_snapshots_segmented_dense != NULL ) ?
            &sensor_snapshots_segmented_dense[isnapshot] :
            NULL;

        for(unsigned int ilidar=0; ilidar<Nlidars; ilidar++)
        {
            snapshots[isnapshot].lidar_scans[ilidar] =
                (points_and_plane_full_t){};


            if(sensor_snapshot_unsorted != NULL ||
               sensor_snapshot_sorted   != NULL)
            {
                if(!lidar_segmentation(&snapshots[isnapshot].lidar_scans[ilidar],

                                       points_pool,
                                       points_and_plane_pool,
                                       &points_pool_index,
                                       &points_and_plane_pool_index,


                                       sensor_snapshot_unsorted,
                                       sensor_snapshot_sorted,
                                       ilidar,
                                       _Nrings,
                                       lidar_packet_stride,
                                       Nplanes_max,
                                       ctx))
                    continue;
            }
            else if(sensor_snapshot_segmented != NULL)
            {
                const clc_lidar_scan_segmented_t* scan_segmented = &sensor_snapshot_segmented->lidar_scans[ilidar];
                if(scan_segmented->points_and_plane.n == 0)
                    continue;

                snapshots[isnapshot].lidar_scans[ilidar] =
                    (points_and_plane_full_t){ .points = scan_segmented->points,
                                               .n      = scan_segmented->points_and_plane.n,
                                               .ipoint = scan_segmented->points_and_plane.ipoint,
                                               .plane  = scan_segmented->points_and_plane.plane};
            }
            else if(sensor_snapshot_segmented_dense != NULL)
            {
                const clc_lidar_scan_segmented_dense_t* scan_segmented = &sensor_snapshot_segmented_dense->lidar_scans[ilidar];
                if(scan_segmented->points_and_plane.n == 0)
                    continue;

                snapshots[isnapshot].lidar_scans[ilidar] =
                    (points_and_plane_full_t){ .points = scan_segmented->points,
                                               .n      = scan_segmented->points_and_plane.n,
                                               .ipoint = NULL,
                                               .plane  = scan_segmented->points_and_plane.plane};
            }
            else
            {
                MSG("Getting here is a bug");
                return false;
            }


            MSG_IF_VERBOSE("Sensor snapshot %d observed by sensor %d (lidar%d)",
                           isnapshot, ilidar, ilidar);
            Nsensors_observing++;
        }

        for(unsigned int icamera=0; icamera<Ncameras; icamera++)
        {
            if( sensor_snapshot_segmented != NULL)
            {
                snapshots[isnapshot].chessboard_corners[icamera] =
                    sensor_snapshot_segmented->chessboard_corners[icamera];
            }
            else if(sensor_snapshot_segmented_dense != NULL)
            {
                snapshots[isnapshot].chessboard_corners[icamera] =
                    sensor_snapshot_segmented_dense->chessboard_corners[icamera];
            }
            else
            {
                snapshots[isnapshot].chessboard_corners[icamera] =
                    &chessboard_corners_pool[ (isnapshot*Ncameras +
                                               icamera) * object_width_n*object_height_n ];

                // using uint8 type here; the image might be color. This is
                // specified using the is_bgr_mask below
                const mrcal_image_uint8_t* image;
                if(     sensor_snapshot_unsorted  != NULL) image = &sensor_snapshot_unsorted ->images[icamera].uint8;
                else if(sensor_snapshot_sorted    != NULL) image = &sensor_snapshot_sorted   ->images[icamera].uint8;
                else
                    assert(0);
                if(image->data == NULL ||
                   !clc_camera_chessboard_detection(snapshots[isnapshot].chessboard_corners[icamera],

                                                    image,
                                                    is_bgr_mask & (1U << icamera),
                                                    object_height_n,
                                                    object_width_n))
                {
                    snapshots[isnapshot].chessboard_corners[icamera] = NULL;
                    continue;
                }
            }

            MSG_IF_VERBOSE("Sensor snapshot %d observed by sensor %d (camera%d)",
                           isnapshot, Nlidars+icamera, icamera);
            Nsensors_observing++;
        }

        MSG_IF_VERBOSE("Sensor snapshot %d observed by %d sensors",
                       isnapshot, Nsensors_observing);

        if(Nsensors_observing < 2)
        {
            MSG_IF_VERBOSE("Need at least 2. Throwing out isnapshot=%d",
                           isnapshot);
            continue;
        }

        // This snapshot has observations from sufficient overlapping sensors. I
        // use it
        bitarray64_set(bitarray_snapshots_selected, isnapshot);
        MSG_IF_VERBOSE("isnapshot=%d selected", isnapshot);
        Nsnapshots_selected++;
    }

    MSG_IF_VERBOSE("Have %d joint observations", Nsnapshots_selected);

    double Rt_vehicle_lidar0[4*3];
    if(rt_vehicle_lidar0 != NULL)
        mrcal_Rt_from_rt(Rt_vehicle_lidar0, NULL,
                         (const double*)rt_vehicle_lidar0);

    {
        double Rt_lidar0_board_seed  [Nsnapshots                     * 4*3];
        double Rt_lidar0_lidar_seed  [(Nlidars-1)                    * 4*3];
        double Rt_lidar0_camera_seed [Ncameras                       * 4*3];
        double Rt_lidar0_board_solve [Nsnapshots                     * 4*3];
        double Rt_lidar0_lidar_solve [(Nlidars-1)                    * 4*3];
        double Rt_lidar0_camera_solve[Ncameras                       * 4*3];
        memset(Rt_lidar0_board_seed,   0, sizeof(double)*Nsnapshots  * 4*3);
        memset(Rt_lidar0_lidar_seed,   0, sizeof(double)*(Nlidars-1) * 4*3);
        memset(Rt_lidar0_camera_seed,  0, sizeof(double)*Ncameras    * 4*3);
        memset(Rt_lidar0_board_solve,  0, sizeof(double)*Nsnapshots  * 4*3);
        memset(Rt_lidar0_lidar_solve,  0, sizeof(double)*(Nlidars-1) * 4*3);
        memset(Rt_lidar0_camera_solve, 0, sizeof(double)*Ncameras    * 4*3);

        if(use_given_seed_geometry_lidar0)
        {
            for(int i=0; i<3; i++)
                if(fabs(rt_lidar0_lidar[0].r.xyz[i]) > 1e-8)
                {
                    MSG("use_given_seed_geometry_lidar0 is true, so rt_lidar0_lidar[0] MUST be the identity transformation");
                    return false;
                }
            for(int i=0; i<3; i++)
                if(fabs(rt_lidar0_lidar[0].t.xyz[i]) > 1e-8)
                {
                    MSG("use_given_seed_geometry_lidar0 is true, so rt_lidar0_lidar[0] MUST be the identity transformation");
                    return false;
                }
            for(int i=1; i<Nlidars; i++)
                mrcal_Rt_from_rt(&Rt_lidar0_lidar_seed[4*3*(i-1)], NULL,
                                 (const double*)&rt_lidar0_lidar[i]);
            for(int i=0; i<Ncameras; i++)
                mrcal_Rt_from_rt(&Rt_lidar0_camera_seed[4*3*i], NULL,
                                 (const double*)&rt_lidar0_camera[i]);
        }
        else if(use_given_seed_geometry_vehicle)
        {
            // temporary storage, in case rt_lidar0_lidar or rt_lidar0_camera are NULL
            mrcal_pose_t _rt_lidar0_lidar [Nlidars];
            mrcal_pose_t _rt_lidar0_camera[Ncameras];

            for(int i=0; i<Nlidars; i++)
                mrcal_compose_rt_inverted0((double*)&_rt_lidar0_lidar[i], NULL,NULL,NULL,NULL,NULL,NULL,
                                           (double*)rt_vehicle_lidar0,(double*)&rt_vehicle_lidar[i]);
            for(int i=0; i<Ncameras; i++)
                mrcal_compose_rt_inverted0((double*)&_rt_lidar0_camera[i], NULL,NULL,NULL,NULL,NULL,NULL,
                                           (double*)rt_vehicle_lidar0,(double*)&rt_vehicle_camera[i]);

            for(int i=0; i<3; i++)
                if(fabs(_rt_lidar0_lidar[0].r.xyz[i]) > 1e-8)
                {
                    MSG("use_given_seed_geometry_vehicle is true, so rt_lidar0_lidar[0] MUST be the identity transformation");
                    return false;
                }
            for(int i=0; i<3; i++)
                if(fabs(_rt_lidar0_lidar[0].t.xyz[i]) > 1e-8)
                {
                    MSG("use_given_seed_geometry_vehicle is true, so rt_lidar0_lidar[0] MUST be the identity transformation");
                    return false;
                }
            for(int i=1; i<Nlidars; i++)
                mrcal_Rt_from_rt(&Rt_lidar0_lidar_seed[4*3*(i-1)], NULL,
                                 (const double*)&_rt_lidar0_lidar[i]);
            for(int i=0; i<Ncameras; i++)
                mrcal_Rt_from_rt(&Rt_lidar0_camera_seed[4*3*i], NULL,
                                 (const double*)&_rt_lidar0_camera[i]);
        }

        if(!check_sufficient_observations(snapshots,
                                          Nsnapshots,
                                          bitarray_snapshots_selected,
                                          Ncameras,
                                          Nlidars))
            goto done;

        double Rt_camera_board_cache[Nsnapshots*Ncameras * 4*3];
        memset(Rt_camera_board_cache,
               0,
               Nsnapshots*Ncameras * 4*3*sizeof(Rt_camera_board_cache[0]));
        // The Rt_camera_board_cache[] entries are all 0, which means "invalid"

        if(use_given_seed_geometry_lidar0 || use_given_seed_geometry_vehicle)
        {
            if(observations_per_sector != NULL)
                get_observations_per_sector(// out
                                            // dense array of shape (Nsectors,); may be NULL
                                            observations_per_sector,
                                            // in
                                            snapshots,
                                            bitarray_snapshots_selected,
                                            Nsnapshots,
                                            Nlidars,
                                            Ncameras,
                                            Nsectors,
                                            object_height_n,
                                            object_width_n,
                                            object_spacing,
                                            models,
                                            NULL,
                                            // Nlidars-1 (4,3) arrays
                                            Rt_lidar0_lidar_seed,
                                            // Ncameras (4,3) arrays
                                            Rt_lidar0_camera_seed,
                                            // may be NULL,
                                            rt_vehicle_lidar0 == NULL ? NULL : Rt_vehicle_lidar0,
                                            Rt_camera_board_cache);

            if(isvisible_per_sensor_per_sector != NULL)
                if(!get_isvisible_per_sensor_per_sector(// out
                                                        // A dense array of shape (Nsensors,Nsectors); may be NULL
                                                        isvisible_per_sensor_per_sector,
                                                        // in
                                                        Nlidars,
                                                        Ncameras,
                                                        Nsectors,
                                                        // Nlidars-1 (4,3) arrays
                                                        Rt_lidar0_lidar_seed,
                                                        // Ncameras (4,3) arrays
                                                        Rt_lidar0_camera_seed,
                                                        // may be NULL,
                                                        rt_vehicle_lidar0 == NULL ? NULL : Rt_vehicle_lidar0,
                                                        // used for isvisible_per_sensor_per_sector
                                                        threshold_valid_lidar_range,
                                                        threshold_valid_lidar_Npoints,
                                                        uncertainty_quantification_range,
                                                        lidar_scans_for_isvisible,
                                                        lidar_packet_stride,
                                                        // Ncameras of these
                                                        models))
                    return false;
        }

        if(!fit_seed(// in/out
                     Rt_lidar0_board_seed,
                     Rt_lidar0_lidar_seed,
                     Rt_lidar0_camera_seed,
                     use_given_seed_geometry_lidar0 || use_given_seed_geometry_vehicle,

                     // in
                     snapshots,
                     Nsnapshots,
                     bitarray_snapshots_selected,
                     Rt_camera_board_cache,

                     Nlidars,
                     Ncameras,
                     models,
                     object_height_n,
                     object_width_n,
                     object_spacing,
                     fit_seed_position_err_threshold,
                     fit_seed_cos_angle_err_threshold,
                     verbose))
        {
            MSG("fit_seed() failed");

            if(buf_inputs_dump != NULL)
            {
                if(!dump_inputs(buf_inputs_dump,
                                size_inputs_dump,
                                Rt_lidar0_board_seed,
                                Rt_lidar0_lidar_seed,
                                Rt_lidar0_camera_seed,
                                Rt_lidar0_board_solve,
                                Rt_lidar0_lidar_solve,
                                Rt_lidar0_camera_solve,

                                snapshots,
                                Nsnapshots,
                                bitarray_snapshots_selected,

                                Nlidars,
                                Ncameras,
                                models,
                                object_height_n,
                                object_width_n,
                                object_spacing))
                {
                    MSG("dump_inputs() failed");
                }
            }

            goto done;
        }

        // We tossed some outliers. Make sure we have enough observations still
        if(!check_sufficient_observations(snapshots,
                                          Nsnapshots,
                                          bitarray_snapshots_selected,
                                          Ncameras,
                                          Nlidars))
            goto done;


        memcpy(Rt_lidar0_board_solve,  Rt_lidar0_board_seed,  sizeof(double)*Nsnapshots  * 4*3);
        memcpy(Rt_lidar0_lidar_solve,  Rt_lidar0_lidar_seed,  sizeof(double)*(Nlidars-1) * 4*3);
        memcpy(Rt_lidar0_camera_solve, Rt_lidar0_camera_seed, sizeof(double)*Ncameras    * 4*3);

        plot_geometry("/tmp/geometry-seed.gp",
                      Rt_lidar0_board_seed,
                      Rt_lidar0_lidar_seed,
                      Rt_lidar0_camera_seed,
                      snapshots,
                      Nsnapshots,
                      bitarray_snapshots_selected,
                      Nlidars,
                      Ncameras,
                      object_height_n,
                      object_width_n,
                      object_spacing,
                      false,
                      verbose);
        plot_geometry("/tmp/geometry-seed-onlyaxes.gp",
                      Rt_lidar0_board_seed,
                      Rt_lidar0_lidar_seed,
                      Rt_lidar0_camera_seed,
                      snapshots,
                      Nsnapshots,
                      bitarray_snapshots_selected,
                      Nlidars,
                      Ncameras,
                      object_height_n,
                      object_width_n,
                      object_spacing,
                      true,
                      verbose);

        dogleg_solverContext_t* solver_context;
        double rt_lidar0_board__optimization [6*Nsnapshots];
        double rt_lidar0_lidar__optimization [6*(Nlidars-1)];
        double rt_lidar0_camera__optimization[6*Ncameras];
        bool fit_result =
            fit(&solver_context,
                rt_lidar0_board__optimization,
                rt_lidar0_lidar__optimization,
                rt_lidar0_camera__optimization,
                // in,out
                // seed state on input
                Rt_lidar0_board_solve,
                Rt_lidar0_lidar_solve,
                Rt_lidar0_camera_solve,

                // in
                snapshots,
                Nsnapshots,
                bitarray_snapshots_selected,

                Nlidars,
                Ncameras,
                models,
                object_height_n,
                object_width_n,
                object_spacing,

                check_gradient__use_distance_to_plane,
                check_gradient,
                false,    // skip_presolve
                !verbose, // skip_plots
                verbose   // verbose
                );

        if(buf_inputs_dump != NULL)
        {
            if(!dump_inputs(buf_inputs_dump,
                            size_inputs_dump,
                            Rt_lidar0_board_seed,
                            Rt_lidar0_lidar_seed,
                            Rt_lidar0_camera_seed,
                            Rt_lidar0_board_solve,
                            Rt_lidar0_lidar_solve,
                            Rt_lidar0_camera_solve,

                            snapshots,
                            Nsnapshots,
                            bitarray_snapshots_selected,

                            Nlidars,
                            Ncameras,
                            models,
                            object_height_n,
                            object_width_n,
                            object_spacing))
            {
                MSG("dump_inputs() failed");
            }
        }

        if(!fit_result)
        {
            MSG("fit() failed");
            goto done;
        }

        if(Var_rt_lidar0_sensor != NULL)
            if(!compute_covariance(Var_rt_lidar0_sensor,
                                   snapshots,
                                   Nsnapshots,
                                   bitarray_snapshots_selected,
                                   Nlidars,
                                   Ncameras,
                                   models,
                                   object_height_n,
                                   object_width_n,
                                   object_spacing,
                                   solver_context))
            {
                MSG("compute_covariance() failed");
                goto done;
            }

        dogleg_freeContext(&solver_context);

        if(check_gradient__use_distance_to_plane || check_gradient)
        {
            result = true;
            goto done;
        }

        plot_geometry("/tmp/geometry.gp",
                      Rt_lidar0_board_solve,
                      Rt_lidar0_lidar_solve,
                      Rt_lidar0_camera_solve,
                      snapshots,
                      Nsnapshots,
                      bitarray_snapshots_selected,
                      Nlidars,
                      Ncameras,
                      object_height_n,
                      object_width_n,
                      object_spacing,
                      false,
                      verbose);
        plot_geometry("/tmp/geometry-onlyaxes.gp",
                      Rt_lidar0_board_solve,
                      Rt_lidar0_lidar_solve,
                      Rt_lidar0_camera_solve,
                      snapshots,
                      Nsnapshots,
                      bitarray_snapshots_selected,
                      Nlidars,
                      Ncameras,
                      object_height_n,
                      object_width_n,
                      object_spacing,
                      true,
                      verbose);

        make_reprojected_plots( Rt_lidar0_lidar_solve,
                                Rt_lidar0_camera_solve,
                                snapshots,
                                Nsnapshots,
                                bitarray_snapshots_selected,
                                Nlidars,
                                Ncameras,
                                models, // Ncameras of these
                                object_height_n,
                                object_width_n,
                                verbose);

        if(observations_per_sector != NULL)
            get_observations_per_sector(// out
                                        // dense array of shape (Nsectors,); may be NULL
                                        observations_per_sector,
                                        // in
                                        snapshots,
                                        bitarray_snapshots_selected,
                                        Nsnapshots,
                                        Nlidars,
                                        Ncameras,
                                        Nsectors,
                                        object_height_n,
                                        object_width_n,
                                        object_spacing,
                                        models,
                                        // Nsnapshots (4,3) arrays
                                        Rt_lidar0_board_solve,
                                        NULL,NULL,
                                        // may be NULL,
                                        rt_vehicle_lidar0 == NULL ? NULL : Rt_vehicle_lidar0,
                                        NULL);


        // write the output
        if(rt_lidar0_lidar != NULL)
        {
            rt_lidar0_lidar[0] = (mrcal_pose_t){}; // lidar0 has the reference transform
            for(unsigned int i=0; i<Nlidars-1; i++)
                mrcal_rt_from_Rt(rt_lidar0_lidar[i+1].r.xyz, NULL,
                                 &Rt_lidar0_lidar_solve[i*4*3]);
        }
        if(rt_lidar0_camera != NULL)
            for(unsigned int i=0; i<Ncameras; i++)
                mrcal_rt_from_Rt(rt_lidar0_camera[i].r.xyz, NULL,
                                 &Rt_lidar0_camera_solve[i*4*3]);
        if(rt_vehicle_lidar != NULL)
        {
            rt_vehicle_lidar[0] = *rt_vehicle_lidar0;
            for(unsigned int i=0; i<Nlidars-1; i++)
            {
                double Rt_vehicle_lidar[4*3];
                mrcal_compose_Rt(Rt_vehicle_lidar,
                                 Rt_vehicle_lidar0, &Rt_lidar0_lidar_solve[i*4*3]);
                mrcal_rt_from_Rt(rt_vehicle_lidar[i+1].r.xyz, NULL,
                                 Rt_vehicle_lidar);
            }
        }
        if(rt_vehicle_camera != NULL)
        {
            for(unsigned int i=0; i<Ncameras; i++)
            {
                double Rt_vehicle_camera[4*3];
                mrcal_compose_Rt(Rt_vehicle_camera,
                                 Rt_vehicle_lidar0, &Rt_lidar0_camera_solve[i*4*3]);
                mrcal_rt_from_Rt(rt_vehicle_camera[i].r.xyz, NULL,
                                 Rt_vehicle_camera);
            }
        }


        if(isvisible_per_sensor_per_sector != NULL)
        {
            if(!get_isvisible_per_sensor_per_sector(// out
                                                    // A dense array of shape (Nsensors,Nsectors); may be NULL
                                                    isvisible_per_sensor_per_sector,
                                                    // in
                                                    Nlidars,
                                                    Ncameras,
                                                    Nsectors,
                                                    // Nlidars-1 (4,3) arrays
                                                    Rt_lidar0_lidar_solve,
                                                    // Ncameras (4,3) arrays
                                                    Rt_lidar0_camera_solve,
                                                    // may be NULL,
                                                    rt_vehicle_lidar0 == NULL ? NULL : Rt_vehicle_lidar0,
                                                    // used for isvisible_per_sensor_per_sector
                                                    threshold_valid_lidar_range,
                                                    threshold_valid_lidar_Npoints,
                                                    uncertainty_quantification_range,
                                                    lidar_scans_for_isvisible,
                                                    lidar_packet_stride,
                                                    // Ncameras of these
                                                    models))
                return false;
        }

        if(stdev_worst_per_sector != NULL)
        {
            const double sector_width_rad = 2.*M_PI/(double)Nsectors;
            const double c = cos(sector_width_rad);
            const double s = sin(sector_width_rad);

            // I will rotate this by sector_width_rad with each step
            mrcal_point3_t pquery_vehicle =
                {.x = uncertainty_quantification_range * cos(0.5 * sector_width_rad),
                 .y = uncertainty_quantification_range * sin(0.5 * sector_width_rad),
                 .z = 0};
            for(int isector=0; isector<Nsectors; isector++)
            {
                // point at the center of the sector
                mrcal_point3_t pquery_lidar0;
                if(rt_vehicle_lidar0 != NULL)
                    mrcal_transform_point_Rt_inverted(pquery_lidar0.xyz,NULL,NULL,
                                                      Rt_vehicle_lidar0, pquery_vehicle.xyz);
                else
                    pquery_lidar0 = pquery_vehicle;

                if(!transformation_uncertainty_in_sector(// out
                                                         &stdev_worst_per_sector[isector],
                                                         &isensors_pair_stdev_worst[2*isector],
                                                         // in
                                                         &pquery_lidar0,
                                                         rt_lidar0_lidar__optimization,
                                                         rt_lidar0_camera__optimization,
                                                         isector,
                                                         Nsectors,
                                                         isvisible_per_sensor_per_sector,

                                                         Var_rt_lidar0_sensor,
                                                         Nlidars,
                                                         Ncameras))
                    return false;

                const double x = pquery_vehicle.x;
                const double y = pquery_vehicle.y;
                pquery_vehicle.x = x*c - y*s;
                pquery_vehicle.y = y*c + x*s;
            }

        }
    }

    result = true;

 done:
    free(pool);
    return result;
}
