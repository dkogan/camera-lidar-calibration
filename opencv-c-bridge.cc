#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <opencv2/calib3d.hpp>

#include "util.h"
#include "opencv-c-bridge.hh"

extern "C"
__attribute__((visibility("hidden")))
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
            bool useExtrinsicGuess)
{
    cv::Mat cv_p(N, 3,
                 CV_64FC1,
                 (void*)&p->x);
    cv::Mat cv_q(N, 2,
                 CV_64FC1,
                 (void*)&q->x);
    cv::Mat cv_camera_matrix(3, 3,
                 CV_64FC1,
                 (void*)camera_matrix);

    cv::Mat cv_rvec(1,3, CV_64FC1);
    cv::Mat cv_tvec(1,3, CV_64FC1);
    if(useExtrinsicGuess)
    {
        memcpy((char*)&cv_rvec.at<double>(0,0),
               (char*)&(rvec->x),
               sizeof(*rvec));
        memcpy((char*)&cv_tvec.at<double>(0,0),
               (char*)&(tvec->x),
               sizeof(*tvec));
    }
    if(!cv::solvePnP(cv_p,
                     cv_q,
                     cv_camera_matrix,
                     cv::Mat(),
                     cv_rvec,
                     cv_tvec,
                     useExtrinsicGuess))
        return false;

    memcpy((char*)&(rvec->x),
           (char*)&cv_rvec.at<double>(0,0),
           sizeof(*rvec));
    memcpy((char*)&(tvec->x),
           (char*)&cv_tvec.at<double>(0,0),
           sizeof(*tvec));
    return true;
}
