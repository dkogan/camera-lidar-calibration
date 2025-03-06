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
    mrcal_point2_t* chessboard_corners[clc_Ncameras_max];
    clc_lidar_scan_segmented_t lidar_scans[clc_Nlidars_max];
} clc_sensor_snapshot_segmented_t;

typedef struct
{
    mrcal_point2_t* chessboard_corners[clc_Ncameras_max];
    clc_lidar_scan_segmented_dense_t lidar_scans[clc_Nlidars_max];
} clc_sensor_snapshot_segmented_dense_t;



#define CLC_LIDAR_SEGMENTATION_LIST_CONTEXT(_)                          \
  /* bool, but PyArg_ParseTupleAndKeywords("p") wants an int */         \
  _(int,   dump,                                         (int)false,        "p","i", \
      "if true, diagnostic detector data meant for plotting is output on stdout. The intended use is\n" \
      "  ./lidar-segmentation-test.py --dump TOPIC BAG |\n"             \
      "  | feedgnuplot \\\n"                                            \
      "      --style label 'with labels' \\\n"                          \
      "      --style ACCEPTED \"with points pt 2 ps 2 lw 2 lc \\\"red\\\"\"\\\n" \
      "      --tuplesize label 4 \\\n"                                  \
      "      --style all 'with points pt 7 ps 0.5' \\\n"                \
      "      --style stage1-segment \"with vectors lc \\\"green\\\"\"\\\n" \
      "      --style plane-normal   \"with vectors lc \\\"black\\\"\"\\\n" \
      "      --tuplesize stage1-segment,plane-normal 6\\\n"             \
      "      --3d \\\n"                                                 \
      "      --domain \\\n"                                             \
      "      --dataid \\\n"                                             \
      "      --square \\\n"                                             \
      "      --points \\\n"                                             \
      "      --tuplesizeall 3 \\\n"                                     \
      "      --autolegend \\\n"                                         \
      "      --xlabel x \\\n"                                           \
      "      --ylabel y \\\n"                                           \
      "      --zlabel z\\n")                                            \
  _(int,   debug_iring,                                  -1,                "i","i", \
      "stage1: report diagnostic information on stderr, ONLY for this ring" ) \
  _(float, debug_xmin,                                   FLT_MAX,           "f","f", \
      "report diagnostic information on stderr, ONLY for the region within the given xy bounds" ) \
  _(float, debug_xmax,                                   -FLT_MAX,          "f","f", \
      "report diagnostic information on stderr, ONLY for the region within the given xy bounds" ) \
  _(float, debug_ymin,                                   FLT_MAX,           "f","f", \
      "report diagnostic information on stderr, ONLY for the region within the given xy bounds" ) \
  _(float, debug_ymax,                                   -FLT_MAX,          "f","f", \
      "report diagnostic information on stderr, ONLY for the region within the given xy bounds" ) \
  _(int,   threshold_min_Npoints_in_segment,             10,                "i","i", \
      "stage1: segments are accepted only if they contain at least this many points" ) \
  _(int,   threshold_max_Npoints_invalid_segment,        5,                 "i","i", \
      "stage1: segments are accepted only if they contain at most this many invalid points" ) \
  _(float, threshold_max_range,                          13.f,               "f","f", \
      "stage2: discard all segment clusters that lie COMPLETELY past the given range" ) \
  _(float, threshold_distance_adjacent_points_cross_segment,                          .1f,               "f","f", \
      "stage2: adjacent cross-segment points in the same ring must be at most this far apart" ) \
  /* cos(10deg) */                                                       \
  _(float, threshold_min_cos_angle_error_same_direction_intra_ring,                          0.984807753012f,               "f","f", \
      "stage2: cos threshold used to accumulate a segment to an adjacent one in the same ring" ) \
                                                                        \
  /* This is unnaturally high. I'm comparing it to p-mean(p), but if the points aren't */ \
  /* distributed evenly, mean(p) won't be at the center */              \
  _(float, threshold_max_plane_size,                     1.9f,              "f","f", \
      "Post-processing: high limit on the linear size of the reported plane.\n" \
      "In a square board this is roughly compared to the side length") \
  _(float, threshold_max_rms_fit_error,                  0.02f,            "f","f", \
      "Post-processing: high limit on the RMS plane fit residual. Lower values will demand flatter planes" ) \
  _(float, threshold_min_rms_point_cloud_2nd_dimension,  0.1f,              "f","f", \
      "Post-processing: low limit on the short length of the found plane. Too-skinny planes are rejected" ) \
  /* found empirically in dump-lidar-scan.py */                         \
  _(int,   Npoints_per_rotation,                         1809,              "i","i", \
      "How many points are reported by the LIDAR in a rotation.\n" \
      "This is hardware-dependent, and needs to be revisited for different LIDAR units" ) \
  _(int,   Npoints_per_segment,                          15,                "i","i", \
      "stage1: length of segments we're looking for" ) \
  _(int,   threshold_max_Ngap,                           2,                 "i","i", \
      "The maximum number of consecutive missing points in a ring" ) \
  _(float, threshold_max_deviation_off_segment_line,     0.05f,             "f","f", \
      "stage1: maximum allowed deviation off a segment line fit.\n" \
      "If any points violate this, the entire segment is rejected" ) \
                                                                        \
  /* should be a factor of threshold_max_plane_size */                  \
  _(float, threshold_max_distance_across_rings,          0.5f,              "f","f", \
      "stage2: max ring-ring distance allowed to join two segments into a cluster" ) \
  _(int,   Nrings,                                       32,                "i","i", \
      "How many rings are present in the LIDAR data.\n" \
      "This is hardware-dependent, and needs to be revisited for different LIDAR units" ) \
  /* cos(90-5deg) */                                                    \
  _(float, threshold_max_cos_angle_error_normal,         0.15,   "f","f", \
      "stage2: cos(v,n) threshold to accept a segment (and its direction v) into an existing cluster (and its normal n)" ) \
  /* cos(5deg) */                                                       \
  _(float, threshold_min_cos_angle_error_same_direction_cross_ring, 0.996194698092f,   "f","f", \
      "stage2: cos threshold used to construct a cluster from two cross-ring segments.\n" \
      "Non fitting pairs are not used to create a new cluster" ) \
  _(float, threshold_max_plane_point_error_stage2,       0.3,               "f","f", \
      "stage2: distance threshold to make sure each segment center lies in plane\n" \
      "Non-fitting segments are not added to the cluster") \
  _(float, threshold_max_plane_point_error_stage3,       0.05,              "f","f", \
      "stage3: distance threshold to make sure each point lies in the plane\n" \
      "Non-fitting points are culled from the reported plane") \
  _(float, threshold_min_plane_point_error_isolation,    0.3,               "f","f", \
      "stage3: points just off the edge of the detected board must fit AT LEAST this badly" ) \
  _(int,   threshold_max_Nsegments_in_cluster,           150,               "i","i", \
      "stage2: clusters with more than this many segments are rejected" ) \
  _(int,   threshold_min_Nsegments_in_cluster,           4,                 "i","i", \
      "stage2: clusters with fewer than this many segments are rejected" ) \
  _(int,   threshold_min_Nrings_in_cluster,              3,                 "i","i", \
      "stage2: clusters with date from fewer than this many rings are rejected" ) \
  /* used in refinement */                                              \
  _(float, threshold_max_gap_th_rad,                     0.5f * M_PI/180.f, "f","f", \
      "stage3: moving from the center, we stop accumulating points when we encounter\n" \
      "an angular gap of this many radians" )




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

bool
clc_camera_chessboard_detection(// out
                               mrcal_point2_t* chessboard_corners,
                               // in
                               const mrcal_image_uint8_t* image, // might be color
                               const bool is_image_bgr,
                               const int object_height_n,
                               const int object_width_n);



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

         // may be NULL. Will attempt to report this even if clc() fails; -1
         // means it could not be computed
         int* isector_of_last_snapshot,

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
         bool verbose);

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
                              const int*  exclude_isnapshot, // NULL to not exclude any
                              const int   Nexclude_isnapshot,

                              const double fit_seed_position_err_threshold,
                              const double fit_seed_cos_angle_err_threshold,

                              // if(!do_fit_seed && !do_inject_noise) { fit(previous fit_seed() result)     }
                              // if(!do_fit_seed &&  do_inject_noise) { fit(previous fit() result)          }
                              // if(do_fit_seed)                      { fit( fit_seed() )                   }
                              bool do_fit_seed,
                              // if true, the observations are noised; regardless of do_fit_seed
                              bool do_inject_noise,
                              bool do_skip_plots,
                              bool verbose);

bool
clc_estimate_camera_pose_from_fixed_point_observations(// out
                                                    double* Rt_cam_points,
                                                    // in
                                                    const mrcal_lensmodel_t* lensmodel,
                                                    const double*            intrinsics,
                                                    const mrcal_point2_t*    observations,
                                                    const mrcal_point3_t*    points_ref,
                                                    const int                N);

bool clc_fit_Rt_camera_board(// out
                 double*                    Rt_camera_board,
                 // in
                 const mrcal_cameramodel_t* model,
                 const mrcal_point2_t*      observations,
                 const int                  object_height_n,
                 const int                  object_width_n,
                 const double               object_spacing);

void
clc_ref_calibration_object(// out
                       mrcal_point3_t*            points_ref,
                       // in
                       const int                  object_height_n,
                       const int                  object_width_n,
                       const double               object_spacing);
