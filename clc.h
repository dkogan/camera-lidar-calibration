#pragma once

#include <stdbool.h>
#include <stdint.h>
#include <mrcal/mrcal.h>

typedef union
{
    struct
    {
        float x,y,z;
    };
    float xyz[3];
} clc_point3f_t;

typedef struct
{
    // These are both defined in the sensor coordinate system. The sensor is at
    // the origin


    // A point in the plane region. The observed points might be very unevenly
    // distributed, pushing this mean far off-center
    clc_point3f_t p_mean;

    // A unit normal to the plane
    clc_point3f_t n;
} clc_plane_t;

typedef struct
{
    unsigned int n;
    uint32_t ipoint[8192 - 7]; // -7 to make each clc_points_and_plane_t fit evenly
                               // into a round-sized chunk of memory
    clc_plane_t plane;
} clc_points_and_plane_t;

typedef struct
{
    unsigned int n;
    clc_plane_t  plane;
} clc_points_and_plane_dense_t;

// I can't find a single static assertion invocation that works in both C++ and
// C. The below is ugly, but works
#ifdef __cplusplus
static_assert (sizeof(clc_points_and_plane_t) == 8192*4, "clc_points_and_plane_t has expected size");
#else
_Static_assert(sizeof(clc_points_and_plane_t) == 8192*4, "clc_points_and_plane_t has expected size");
#endif


#define clc_Nlidars_max  16
#define clc_Ncameras_max 16

typedef uint32_t clc_is_bgr_mask_t;
// I can't find a single static assertion invocation that works in both C++ and
// C. The below is ugly, but works
#ifdef __cplusplus
static_assert ((int)sizeof(clc_is_bgr_mask_t)*8 >= clc_Ncameras_max,
               "is_bgr_mask should be large-enough to index all the possible cameras: need at least one bit per camera");
#else
_Static_assert((int)sizeof(clc_is_bgr_mask_t)*8 >= clc_Ncameras_max,
               "is_bgr_mask should be large-enough to index all the possible cameras: need at least one bit per camera");
#endif


typedef struct
{
    // These point to the FIRST point,ring and are NOT stored densely. The
    // stride for both of these is given in lidar_packet_stride
    clc_point3f_t* points;      // 3D points, in the lidar frame
    uint16_t*      rings;       // For each point, which laser observed the point

    unsigned int   Npoints;     // How many {point,ring} tuples are stored here
} clc_lidar_scan_unsorted_t;

typedef struct
{
    // length sum(Npoints). Sorted by ring and then by
    // azimuth. Use clc_lidar_sort() to do this
    clc_point3f_t* points;
    // length ctx->Nrings
    unsigned int* Npoints;
} clc_lidar_scan_sorted_t;

typedef struct
{
    // The segmented point cloud, as indices into points[]
    clc_points_and_plane_t points_and_plane;
    // assumed stored densely (no lidar_packet_stride)
    clc_point3f_t* points;
} clc_lidar_scan_segmented_t;

typedef struct
{
    // The segmented point cloud, as indices into points[]
    clc_points_and_plane_dense_t points_and_plane;
    // assumed stored densely
    clc_point3f_t* points;
} clc_lidar_scan_segmented_dense_t;

typedef struct
{
    // The caller has to know which camera is grayscale and which is color. This
    // would be indicated by a bit array on a higher level
    union
    {
        mrcal_image_uint8_t uint8;
        mrcal_image_bgr_t   bgr;
    } images[clc_Ncameras_max];

    clc_lidar_scan_unsorted_t lidar_scans[clc_Nlidars_max];

} clc_sensor_snapshot_unsorted_t;

typedef struct
{
    // The caller has to know which camera is grayscale and which is color. This
    // would be indicated by a bit array on a higher level
    union
    {
        mrcal_image_uint8_t uint8;
        mrcal_image_bgr_t   bgr;
    } images[clc_Ncameras_max];

    clc_lidar_scan_sorted_t lidar_scans[clc_Nlidars_max];

} clc_sensor_snapshot_sorted_t;

typedef struct
{
    // The caller has to know which camera is grayscale and which is color. This
    // would be indicated by a bit array on a higher level
    union
    {
        mrcal_image_uint8_t uint8;
        mrcal_image_bgr_t   bgr;
    } images[clc_Ncameras_max];

    clc_lidar_scan_segmented_t lidar_scans[clc_Nlidars_max];

} clc_sensor_snapshot_segmented_t;

typedef struct
{
    // The caller has to know which camera is grayscale and which is color. This
    // would be indicated by a bit array on a higher level
    union
    {
        mrcal_image_uint8_t uint8;
        mrcal_image_bgr_t   bgr;
    } images[clc_Ncameras_max];

    clc_lidar_scan_segmented_dense_t lidar_scans[clc_Nlidars_max];

} clc_sensor_snapshot_segmented_dense_t;



#define CLC_LIDAR_SEGMENTATION_LIST_CONTEXT(_)                                                 \
  /* bool, but PyArg_ParseTupleAndKeywords("p") wants an int */         \
  _(int,   dump,                                         (int)false,        "p","i") \
  _(int,   debug_iring,                                  -1,                "i","i") \
  _(float, debug_xmin,                                   FLT_MAX,           "f","f") \
  _(float, debug_xmax,                                   -FLT_MAX,          "f","f") \
  _(float, debug_ymin,                                   FLT_MAX,           "f","f") \
  _(float, debug_ymax,                                   -FLT_MAX,          "f","f") \
  _(int,   threshold_min_Npoints_in_segment,             10,                "i","i") \
  _(int,   threshold_max_Npoints_invalid_segment,        5,                 "i","i") \
  _(float, threshold_max_range,                          9.f,               "f","f") \
                                                                        \
  /* This is unnaturally high. I'm comparing it to p-mean(p), but if the points aren't */ \
  /* distributed evenly, mean(p) won't be at the center */ \
  _(float, threshold_max_plane_size,                     1.9f,              "f","f") \
  _(float, threshold_max_rms_fit_error,                  0.02f,            "f","f") \
  _(float, threshold_min_rms_point_cloud_2nd_dimension,  0.1f,              "f","f") \
  /* found empirically in dump-lidar-scan.py */                         \
  _(int,   Npoints_per_rotation,                         1809,              "i","i") \
  _(int,   Npoints_per_segment,                          20,                "i","i") \
  _(int,   threshold_max_Ngap,                           2,                 "i","i") \
  _(float, threshold_max_deviation_off_segment_line,     0.05f,             "f","f") \
                                                                        \
  /* should be a factor of threshold_max_plane_size */                  \
  _(float, threshold_max_distance_across_rings,          0.4f,              "f","f") \
  _(int,   Nrings,                                       32,                "i","i") \
  /* cos(90-5deg) */                                                    \
  _(float, threshold_max_cos_angle_error_normal,         0.15,   "f","f") \
  /* cos(5deg) */                                                       \
  _(float, threshold_min_cos_angle_error_same_direction, 0.996194698092f,   "f","f") \
  _(float, threshold_max_plane_point_error_stage2,       0.3,               "f","f") \
  _(float, threshold_max_plane_point_error_stage3,       0.05,              "f","f") \
  _(float, threshold_min_plane_point_error_isolation,    0.3,               "f","f") \
  _(int,   threshold_max_Nsegments_in_cluster,           150,               "i","i") \
  _(int,   threshold_min_Nsegments_in_cluster,           5,                 "i","i") \
  _(int,   threshold_min_Nrings_in_cluster,              3,                 "i","i") \
  /* used in refinement */                                              \
  _(float, threshold_max_gap_th_rad,                     0.5f * M_PI/180.f, "f","f")




typedef struct
{
#define CLC_LIDAR_SEGMENTATION_LIST_CONTEXT_DECLARE_C(type,name, ...) \
    type name;

    CLC_LIDAR_SEGMENTATION_LIST_CONTEXT(CLC_LIDAR_SEGMENTATION_LIST_CONTEXT_DECLARE_C)
#undef CLC_LIDAR_SEGMENTATION_LIST_CONTEXT_DECLARE_C
} clc_lidar_segmentation_context_t;


// Sorts the lidar data by ring and azimuth, to be passable to
// clc_lidar_segmentation_sorted()
void clc_lidar_sort(// out
                    //
                    // These buffers must be pre-allocated
                    // length sum(Npoints). Sorted by ring and then by azimuth
                    clc_point3f_t* points,
                    // indices; length(sum(Npoints))
                    uint32_t* ipoint_unsorted_in_sorted_order,
                    // length Nrings
                    unsigned int* Npoints,

                    // in
                    int Nrings,
                    // The stride, in bytes, between each successive points or
                    // rings value in clc_lidar_scan_unsorted_t
                    const unsigned int      lidar_packet_stride,
                    const clc_lidar_scan_unsorted_t* scan);


// Returns how many planes were found or <0 on error
int8_t clc_lidar_segmentation_unsorted(// out
                          clc_points_and_plane_t* points_and_plane,
                          // in
                          const int8_t Nplanes_max, // buffer length of points_and_plane[]
                          const clc_lidar_scan_unsorted_t* scan,
                          // The stride, in bytes, between each successive points or rings value
                          // in clc_lidar_scan_unsorted_t
                          const unsigned int lidar_packet_stride,
                          const clc_lidar_segmentation_context_t* ctx);

// Returns how many planes were found or <0 on error
int8_t clc_lidar_segmentation_sorted(// out
                          clc_points_and_plane_t* points_and_plane,
                          // in
                          const int8_t Nplanes_max, // buffer length of points_and_plane[]
                          const clc_lidar_scan_sorted_t* scan,
                          const clc_lidar_segmentation_context_t* ctx);

void clc_lidar_segmentation_default_context(clc_lidar_segmentation_context_t* ctx);




// Each sensor is uniquely identified by its position in the
// sensor_snapshots[].lidar_scans[] or .images[] arrays. An unobserved sensor in
// some sensor snapshot should be indicated by lidar_scans[] = {} or images[] =
// {}
//
// On output, the rt_ref_lidar[] and rt_ref_camera[] arrays will be filled-in.
// If solving for ALL the sensor geometry wasn't possible, we return false. On
// success, we return true
bool clc_unsorted(// out
         mrcal_pose_t* rt_ref_lidar,  // Nlidars  of these to fill
         mrcal_pose_t* rt_ref_camera, // Ncameras of these to fill

         // Covariance of the output. Symmetric matrix of shape
         // (Nstate_sensor_poses,Nstate_sensor_poses) stored densely, written on
         // output. Nstate_sensor_poses = (Nlidars-1 + Ncameras)*6
         double*       Var_rt_lidar0_sensor,

         // Pass non-NULL to get the fit-inputs dump. On success, these encode
         // the data buffer. The caller must free(*buf_inputs_dump) when done.
         char**  buf_inputs_dump,
         size_t* size_inputs_dump,

         // in
         const clc_sensor_snapshot_unsorted_t* sensor_snapshots,
         const unsigned int                    Nsensor_snapshots,
         // The stride, in bytes, between each successive points or rings value
         // in clc_lidar_scan_unsorted_t
         const unsigned int           lidar_packet_stride,

         // These apply to ALL the sensor_snapshots[]
         const unsigned int Nlidars,
         const unsigned int Ncameras,
         const mrcal_cameramodel_t*const* models, // Ncameras of these
         // The dimensions of the chessboard grid being detected in the images
         const int object_height_n,
         const int object_width_n,
         const double object_spacing,

         // bits indicating whether a camera in
         // sensor_snapshots.images[] is color or not
         const clc_is_bgr_mask_t is_bgr_mask,

         const clc_lidar_segmentation_context_t* ctx,

         bool check_gradient__use_distance_to_plane,
         bool check_gradient);


bool clc_sorted(// out
         mrcal_pose_t* rt_ref_lidar,  // Nlidars  of these to fill
         mrcal_pose_t* rt_ref_camera, // Ncameras of these to fill

         // Covariance of the output. Symmetric matrix of shape
         // (Nstate_sensor_poses,Nstate_sensor_poses) stored densely, written on
         // output. Nstate_sensor_poses = (Nlidars-1 + Ncameras)*6
         double*       Var_rt_lidar0_sensor,

         // Pass non-NULL to get the fit-inputs dump. On success, these encode
         // the data buffer. The caller must free(*buf_inputs_dump) when done.
         char**  buf_inputs_dump,
         size_t* size_inputs_dump,

         // in
         const clc_sensor_snapshot_sorted_t* sensor_snapshots,
         const unsigned int                  Nsensor_snapshots,

         // These apply to ALL the sensor_snapshots[]
         const unsigned int Nlidars,
         const unsigned int Ncameras,
         const mrcal_cameramodel_t*const* models, // Ncameras of these
         // The dimensions of the chessboard grid being detected in the images
         const int object_height_n,
         const int object_width_n,
         const double object_spacing,

         // bits indicating whether a camera in
         // sensor_snapshots.images[] is color or not
         const clc_is_bgr_mask_t is_bgr_mask,

         const clc_lidar_segmentation_context_t* ctx,

         bool check_gradient__use_distance_to_plane,
         bool check_gradient);


bool clc_lidar_segmented(// out
         mrcal_pose_t* rt_ref_lidar,  // Nlidars  of these to fill
         mrcal_pose_t* rt_ref_camera, // Ncameras of these to fill

         // Covariance of the output. Symmetric matrix of shape
         // (Nstate_sensor_poses,Nstate_sensor_poses) stored densely, written on
         // output. Nstate_sensor_poses = (Nlidars-1 + Ncameras)*6
         double*       Var_rt_lidar0_sensor,

         // Pass non-NULL to get the fit-inputs dump. On success, these encode
         // the data buffer. The caller must free(*buf_inputs_dump) when done.
         char**  buf_inputs_dump,
         size_t* size_inputs_dump,

         // in
         const clc_sensor_snapshot_segmented_t* sensor_snapshots,
         const unsigned int                     Nsensor_snapshots,

         // These apply to ALL the sensor_snapshots[]
         const unsigned int Nlidars,
         const unsigned int Ncameras,
         const mrcal_cameramodel_t*const* models, // Ncameras of these
         // The dimensions of the chessboard grid being detected in the images
         const int object_height_n,
         const int object_width_n,
         const double object_spacing,

         // bits indicating whether a camera in
         // sensor_snapshots.images[] is color or not
         const clc_is_bgr_mask_t is_bgr_mask,

         bool check_gradient__use_distance_to_plane,
         bool check_gradient);

bool clc_lidar_segmented_dense(// out
         mrcal_pose_t* rt_ref_lidar,  // Nlidars  of these to fill
         mrcal_pose_t* rt_ref_camera, // Ncameras of these to fill

         // Covariance of the output. Symmetric matrix of shape
         // (Nstate_sensor_poses,Nstate_sensor_poses) stored densely, written on
         // output. Nstate_sensor_poses = (Nlidars-1 + Ncameras)*6
         double*       Var_rt_lidar0_sensor,

         // Pass non-NULL to get the fit-inputs dump. On success, these encode
         // the data buffer. The caller must free(*buf_inputs_dump) when done.
         char**  buf_inputs_dump,
         size_t* size_inputs_dump,

         // in
         const clc_sensor_snapshot_segmented_dense_t* sensor_snapshots,
         const unsigned int                           Nsensor_snapshots,

         // These apply to ALL the sensor_snapshots[]
         const unsigned int Nlidars,
         const unsigned int Ncameras,
         const mrcal_cameramodel_t*const* models, // Ncameras of these
         // The dimensions of the chessboard grid being detected in the images
         const int object_height_n,
         const int object_width_n,
         const double object_spacing,

         // bits indicating whether a camera in
         // sensor_snapshots.images[] is color or not
         const clc_is_bgr_mask_t is_bgr_mask,

         bool check_gradient__use_distance_to_plane,
         bool check_gradient);

bool clc_post_solve_statistics( // out
                                // A dense array of shape (Nsensors,Nsectors)
                                uint8_t* isvisible_per_sensor_per_sector,
                                // array of shape (Nsectors,)
                                double* stdev_worst,
                                // dense array of shape (Nsectors,2); corresponds to stdev_worst
                                uint16_t* isensors_pair_stdev_worst,
                                const int Nsectors,
                                const double threshold_valid_lidar_range,
                                const int    threshold_valid_lidar_Npoints,
                                const double uncertainty_quantification_range,

                                // out,in
                                // On input:  the ref frame is lidar-0
                                // On output: the ref frame is the vehicle frame, as defined in rt_vehicle_lidar0
                                mrcal_pose_t* rt_ref_lidar,  // Nlidars  of these to fill
                                mrcal_pose_t* rt_ref_camera, // Ncameras of these to fill

                                // in
                                // Covariance of the output. Symmetric matrix of shape
                                // (Nstate_sensor_poses,Nstate_sensor_poses) stored densely, written
                                // on output. Nstate_sensor_poses = (Nlidars-1 + Ncameras)*6
                                const double*       Var_rt_lidar0_sensor,
                                const mrcal_pose_t* rt_vehicle_lidar0,
                                const clc_lidar_scan_unsorted_t* lidar_scans, // Nlidars of these
                                // The stride, in bytes, between each successive points or rings value
                                // in clc_lidar_scan_unsorted_t
                                const unsigned int           lidar_packet_stride,
                                const int Nlidars,
                                const int Ncameras,
                                const mrcal_cameramodel_t*const* models // Ncameras of these
                                );

bool clc_fit_from_inputs_dump(// out
                              int* Nlidars,
                              int* Ncameras,
                              // Allocated by the function on success.
                              // It's the caller's responsibility to
                              // free() these
                              mrcal_pose_t** rt_ref_lidar,
                              mrcal_pose_t** rt_ref_camera,
                              // in
                              const char* buf_inputs_dump,
                              size_t      size_inputs_dump,
                              // if(!do_fit_seed && !do_inject_noise) { fit(previous fit_seed() result)     }
                              // if(!do_fit_seed &&  do_inject_noise) { fit(previous fit() result)          }
                              // if(do_fit_seed)                      { fit( fit_seed() )                   }
                              bool do_fit_seed,
                              // if true, the observations are noised; regardless of do_fit_seed
                              bool do_inject_noise,
                              bool do_skip_prints);
