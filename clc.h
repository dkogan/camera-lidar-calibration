#pragma once

#include <stdbool.h>
#include <stdint.h>

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
    int n;
} clc_ipoint_set_t;

typedef struct
{
    clc_ipoint_set_t ipoint_set;
    clc_plane_t      plane;
} clc_points_and_plane_t;
_Static_assert(sizeof(clc_points_and_plane_t) == 8192*4, "clc_points_and_plane_t has expected size");



#define CLC_LIST_CONTEXT(_)                                                 \
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
#define CLC_LIST_CONTEXT_DECLARE_C(type,name, ...) \
    type name;

    CLC_LIST_CONTEXT(CLC_LIST_CONTEXT_DECLARE_C)
#undef CLC_LIST_CONTEXT_DECLARE_C
} context_t;


// Returns how many planes were found or <0 on error
int8_t clc_lidar_segmentation(// out
                          clc_points_and_plane_t* points_and_plane,
                          // in
                          const int8_t Nplanes_max, // buffer length of points_and_plane[]
                          const clc_point3f_t* points,  // length sum(Npoints)
                          const int* Npoints,
                          const context_t* ctx);

void clc_default_context(context_t* ctx);
