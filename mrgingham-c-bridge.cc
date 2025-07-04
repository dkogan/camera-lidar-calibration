#include <stdio.h>
#include <stdlib.h>
#include <mrgingham/mrgingham.hh>
#include <assert.h>
#include <opencv2/imgproc/imgproc.hpp>

#include <mrcal/mrcal-image.h>
#include <mrcal/basic-geometry.h>

#include "util.h"


// This was an attempt to use an existing buffer when invoking mrgingham, but it
// didn't work
#if 0
// "Allocator" to pass a pre-existing pointer to a std::vector
template <typename T>
class StaticAllocator
{
private:
    T*  buf;
    int size;

public:
    typedef int size_type;
    typedef T*  pointer;
    typedef T   value_type;

    StaticAllocator(T* _buf, int _size) :
        buf(_buf), size(_size)
    {
    }
    ~StaticAllocator() {}

    T* allocate(int n)
    {
        return buf;
    }
    void deallocate(T* ptr, int n)
    {
    }

    int max_size() const
    {
        return size;
    }
};
#endif


extern "C"
bool
clc_camera_chessboard_detection(// out
                               mrcal_point2_t* chessboard_corners,
                               // in
                               const mrcal_image_uint8_t* image, // might be color
                               const bool is_image_bgr,
                               const int object_height_n,
                               const int object_width_n)
{
    if(object_height_n != object_width_n)
    {
        MSG("mrgingham requires that object_width_n == object_height_n");
        return false;
    }

    // This was an attempt to use an existing buffer when invoking mrgingham,
    // but it didn't work: passing the custom vector with a custom allocator to
    // mrgingham wasn't possible: the compiler didn't see this as a compatible
    // type
    //
    //   std::vector<mrgingham::PointDouble, StaticAllocator<mrgingham::PointDouble> > points_out(0,
    //                                                                                            StaticAllocator<mrgingham::PointDouble>((mrgingham::PointDouble*)chessboard_corners, object_height_n*object_width_n));
    //
    // So I allocate a new thing, copy and deallocate each time
    std::vector<mrgingham::PointDouble> points_out;

    cv::Mat mat(image->rows, image->cols,
                is_image_bgr ? CV_8UC3 : CV_8UC1,
                image->data, image->stride );

    if(is_image_bgr)
        cv::cvtColor(mat,mat, cv::COLOR_BGR2GRAY);


    // cv::Ptr<cv::CLAHE> clahe;
    // if(doclahe)
    // {
    //     clahe = cv::createCLAHE();
    //     clahe->setClipLimit(8);
    // }

    // if( doclahe )
    // {
    //     // CLAHE doesn't by itself use the full dynamic range all the time,
    //     // so I explicitly normalize the image and then CLAHE
    //     cv::normalize(mat, mat, 0, 65535, cv::NORM_MINMAX);
    //     clahe->apply(mat, mat);
    // }
    // if( blur_radius > 0 )
    // {
    //     cv::blur( mat, mat,
    //               cv::Size(1 + 2*blur_radius,
    //                        1 + 2*blur_radius));
    // }









    if(0 > mrgingham::find_chessboard_from_image_array( points_out,
                                                        NULL,
                                                        object_width_n,
                                                        mat ) )
        return false;

    static_assert(sizeof(chessboard_corners[0]) == sizeof(mrgingham::PointDouble));
    memcpy((char*)chessboard_corners,
           (char*)&points_out[0],
           object_height_n*object_width_n*sizeof(chessboard_corners[0]));
    return true;
}
