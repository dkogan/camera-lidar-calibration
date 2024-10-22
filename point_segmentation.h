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
} point3f_t;

typedef struct
{
    point3f_t p; // A point somewhere on the plane
    point3f_t n; // A unit normal to the plane
} plane_t;

typedef struct
{
    uint32_t ipoint[8192 - 7]; // -7 to make each of these fit evenly into a
                               // round-sized chunk of memory
    int n;

    plane_t plane;
} points_and_plane_t;
_Static_assert(sizeof(points_and_plane_t) == 8192*4, "points_and_plane_t has expected size");



#define LIST_CONTEXT(_)                                                 \
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
  _(float, threshold_max_plane_size,                     1.7f,              "f","f") \
  _(float, threshold_max_rms_fit_error,                  0.015f,            "f","f") \
  _(float, threshold_min_rms_point_cloud_2nd_dimension,  0.1f,              "f","f") \
  /* found empirically in dump-lidar-scan.py */                         \
  _(int,   Npoints_per_rotation,                         1809,              "i","i") \
  _(int,   Npoints_per_segment,                          20,                "i","i") \
  _(int,   threshold_max_Ngap,                           2,                 "i","i") \
  _(float, threshold_max_deviation_off_segment_line,     0.05f,             "f","f") \
  _(int,   Nrings,                                       32,                "i","i") \
  /* cos(90-5deg) */                                                    \
  _(float, threshold_max_cos_angle_error_normal,         0.087155742747f,   "f","f") \
  /* cos(5deg) */                                                       \
  _(float, threshold_min_cos_angle_error_same_direction, 0.996194698092f,   "f","f") \
  _(float, threshold_max_plane_point_error,              0.15,              "f","f") \
  _(int,   threshold_max_Nsegments_in_cluster,           150,               "i","i") \
  _(int,   threshold_min_Nsegments_in_cluster,           5,                 "i","i") \
  _(int,   threshold_min_Nrings_in_cluster,              3,                 "i","i") \
  /* used in refinement */                                              \
  _(float, threshold_max_gap_th_rad,                     0.5f * M_PI/180.f, "f","f")




typedef struct
{
#define LIST_CONTEXT_DECLARE_C(type,name, ...) \
    type name;

    LIST_CONTEXT(LIST_CONTEXT_DECLARE_C)
#undef LIST_CONTEXT_DECLARE_C
} context_t;


// Returns how many planes were found or <0 on error
int8_t point_segmentation(// out
                          points_and_plane_t* points_and_plane,
                          // in
                          const int8_t Nplanes_max, // buffer length of points_and_plane[]
                          const point3f_t* points,  // length sum(Npoints)
                          const int* Npoints,
                          const context_t* ctx);

void default_context(context_t* ctx);
