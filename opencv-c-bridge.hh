#pragma once

#include <stdbool.h>
#include <mrcal/mrcal-image.h>
#include <mrcal/basic-geometry.h>

#ifdef __cplusplus
extern "C"
#endif
bool
cv_solvePnP(// in/out
            // output; if(useExtrinsicGuess) { use these as input as well }
            mrcal_point3_t* rvec,
            mrcal_point3_t* tvec,
            // in
            const mrcal_point3_t* p,
            const mrcal_point2_t* q,
            const int N,
            const double* camera_matrix,
            bool useExtrinsicGuess);
