#pragma once

#include <stdbool.h>

typedef union
{
    struct
    {
        float x,y,z;
    };
    float xyz[3];
} point3f_t;


bool point_segmentation__parse_input_file( // out
                                           const point3f_t** points, // Nrings of these
                                           int* Npoints,             // Nrings of these
                                           // in
                                           const char* filename
                                           );

void point_segmentation(const point3f_t** points,
                        const int* Npoints);
