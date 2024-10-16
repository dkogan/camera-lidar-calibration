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


bool point_segmentation__parse_input_file( // out
                                           const point3f_t** points, // Nrings of these
                                           int* Npoints,             // Nrings of these
                                           // in
                                           const char* filename
                                           );

// Returns how many planes were found or <0 on error
int8_t point_segmentation(// out
                          points_and_plane_t* points_and_plane,
                          // in
                          const int8_t Nplanes_max, // buffer length of points_and_plane[]
                          const point3f_t* points,  // length sum(Npoints)
                          const int* Npoints);
