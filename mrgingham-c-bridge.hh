#pragma once

#include <stdbool.h>
#include <mrcal/mrcal-image.h>
#include <mrcal/basic-geometry.h>

#ifdef __cplusplus
extern "C"
#endif
bool
chessboard_detection_mrgingham(// out
                     mrcal_point2_t* chessboard_corners,
                     // in
                     const mrcal_image_uint8_t* image, // might be color
                     const bool is_image_bgr,
                     const int object_height_n,
                     const int object_width_n);
