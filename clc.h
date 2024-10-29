#pragma once

#include <stdbool.h>
#include <stdint.h>
#include <mrcal/mrcal-image.h>
#include <mrcal/basic-geometry.h>

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
    clc_point3f_t p; // A point somewhere on the plane
    clc_point3f_t n; // A unit normal to the plane
} clc_plane_t;

typedef struct
{
    uint32_t ipoint[8192 - 7]; // -7 to make each clc_points_and_plane_t fit evenly
                               // into a round-sized chunk of memory
    unsigned int n;
} clc_ipoint_set_t;

typedef struct
{
    clc_ipoint_set_t ipoint_set;
    clc_plane_t      plane;
} clc_points_and_plane_t;
_Static_assert(sizeof(clc_points_and_plane_t) == 8192*4, "clc_points_and_plane_t should hhas expected size");



#define clc_Nlidars_max  16
#define clc_Ncameras_max 16

typedef uint32_t clc_is_bgr_mask_t;
_Static_assert((int)sizeof(clc_is_bgr_mask_t)*8 >= clc_Ncameras_max,
               "is_bgr_mask should be large-enough to index all the possible cameras: need at least one bit per camera");

typedef struct
{
    // These point to the FIRST point,ring and are NOT stored densely. The
    // stride for both of these is given in lidar_packet_stride
    clc_point3f_t* points;      // 3D points, in the lidar frame
    uint16_t*      rings;       // For each point, which laser observed the point

    unsigned int   Npoints;     // How many {point,ring} tuples are stored here
} clc_lidar_scan_t;

typedef struct
{
    uint64_t time_us_since_epoch;

    clc_lidar_scan_t lidar_scans[clc_Nlidars_max];

    // The caller has to know which camera is grayscale and which is color. This
    // would be indicated by a bit array on a higher level
    union
    {
        mrcal_image_uint8_t uint8;
        mrcal_image_bgr_t   bgr;
    } images[clc_Ncameras_max];
} clc_sensor_snapshot_t;

typedef struct
{
    // Wrapping is assumed to make sure that az_rad01[1] > az_rad01[0] and that
    // az_rad01[1]-az_rad01[0] < 2*pi
    double az_rad01[2];
} clc_yaw_sector_t;



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


// Returns how many planes were found or <0 on error
int8_t clc_lidar_segmentation(// out
                          clc_points_and_plane_t* points_and_plane,
                          // in
                          const int8_t Nplanes_max, // buffer length of points_and_plane[]
                          const clc_point3f_t* points,  // length sum(Npoints)
                          const unsigned int* Npoints,
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
bool clc(// out
         mrcal_pose_t* rt_ref_lidar,  // Nlidars  of these to fill
         mrcal_pose_t* rt_ref_camera, // Ncameras of these to fill

         // in
         const clc_sensor_snapshot_t* sensor_snapshots,
         const unsigned int           Nsensor_snapshots,
         // The stride, in bytes, between each successive points or rings value
         // in clc_lidar_scan_t
         const unsigned int           lidar_packet_stride,

         // These apply to ALL the sensor_snapshots[]
         const unsigned int Nlidars,
         const unsigned int Ncameras,

         // bits indicating whether a camera in
         // sensor_snapshots.images[] is color or not
         const clc_is_bgr_mask_t is_bgr_mask);


bool
clc_overlapping_regions(// out
                        clc_yaw_sector_t* yaw_sectors,
                        unsigned int*     Nyaw_sectors,

                        // in
                        const unsigned int  Nyaw_sectors_max,
                        const mrcal_pose_t* rt_ref_lidar,  // Nlidars  of these
                        const mrcal_pose_t* rt_ref_camera, // Ncameras of these
                        const unsigned int  Nlidars,
                        const unsigned int  Ncameras);
