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


bool point_segmentation__parse_input_file( // out
                                           const point3f_t** points, // Nrings of these
                                           int* Npoints,             // Nrings of these
                                           // in
                                           const char* filename
                                           );

// Returns how many planes were found and reported in plane_idx[] and plane[].
// Or <0 on error
int8_t point_segmentation(// out
                          int8_t* plane_idx, // which plane each point belongs
                                             // to; <0 for "none". 1-1
                                             // correspondence to elements of
                                             // points[]
                          plane_t* plane,
                          // in
                          const int8_t Nplanes_max, // buffer length of plane[]
                          const point3f_t* points,  // length sum(Npoints)
                          const int* Npoints);
