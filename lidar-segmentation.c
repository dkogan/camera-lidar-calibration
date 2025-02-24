#define _GNU_SOURCE // for qsort_r

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <limits.h>
#include <float.h>

#include "clc.h"
#include "util.h"
#include "bitarray.h"

#define DEBUG_ON_TRUE_POINT(what, p, fmt, ...)                          \
    ({  if(debug && (what))                                             \
        {                                                               \
            MSG("REJECTED (%.2f %.2f %.2f) because " #what ": " fmt,    \
                (p)->x,(p)->y,(p)->z,                                   \
                ##__VA_ARGS__);                                         \
        }                                                               \
        what;                                                           \
    })

#define DEBUG_ON_TRUE_SEGMENT(what, iring, isegment, fmt, ...)          \
    ({  if(debug && (what))                                             \
        {                                                               \
            MSG("REJECTED %d-%d because " #what ": " fmt,               \
                iring,isegment,                                         \
                ##__VA_ARGS__);                                         \
        }                                                               \
        what;                                                           \
    })




/* round up */
#define Nsegments_per_rotation                                          \
    (int)((ctx->Npoints_per_rotation + ctx->Npoints_per_segment-1) / ctx->Npoints_per_segment)




__attribute__((unused))
static clc_point3f_t point3f_from_double(const double* p)
{
    return (clc_point3f_t){ .x = (float)p[0],
                        .y = (float)p[1],
                        .z = (float)p[2]};
}
__attribute__((unused))
static float inner(const clc_point3f_t a, const clc_point3f_t b)
{
    return
        a.x*b.x +
        a.y*b.y +
        a.z*b.z;
}
__attribute__((unused))
static float norm2(const clc_point3f_t a)
{
    return inner(a,a);
}
__attribute__((unused))
static float mag(const clc_point3f_t a)
{
    return sqrtf(norm2(a));
}
__attribute__((unused))
static clc_point3f_t add(const clc_point3f_t a, const clc_point3f_t b)
{
    return (clc_point3f_t){ .x = a.x + b.x,
                        .y = a.y + b.y,
                        .z = a.z + b.z };
}
__attribute__((unused))
static clc_point3f_t mean(const clc_point3f_t a, const clc_point3f_t b)
{
    return (clc_point3f_t){ .x = (a.x + b.x)/2.,
                        .y = (a.y + b.y)/2.,
                        .z = (a.z + b.z)/2. };
}
__attribute__((unused))
static clc_point3f_t scale(const clc_point3f_t a, const float s)
{
    return (clc_point3f_t){ .x = a.x * s,
                            .y = a.y * s,
                            .z = a.z * s };
}
static clc_point3f_t sub(const clc_point3f_t a, const clc_point3f_t b)
{
    return (clc_point3f_t){ .x = a.x - b.x,
                        .y = a.y - b.y,
                        .z = a.z - b.z };
}
static clc_point3f_t cross(const clc_point3f_t a, const clc_point3f_t b)
{
    return (clc_point3f_t){ .x = a.y*b.z - a.z*b.y,
                        .y = a.z*b.x - a.x*b.z,
                        .z = a.x*b.y - a.y*b.x };
}



typedef struct
{
    clc_point3f_t p; // the center
    clc_point3f_t v; // a direction vector in the plane; may not be normalized

    // point indices inside each ring
    int ipoint0;
    int ipoint1  : sizeof(int)*8-1; // leave one bit for "visited"
    bool visited : 1;
} segment_t;

typedef struct
{
    clc_point3f_t p; // A point somewhere on the plane

    // A normal to the plane direction vector in the plane; not necessarily
    // normalized
    clc_point3f_t n_unnormalized;
} plane_unnormalized_t;

typedef struct
{
    union
    {
        clc_plane_t              plane;
        plane_unnormalized_t plane_unnormalized;
    };

    int16_t irings[2]; // first and last ring
    // first and last segment in each successive ring, starting with irings[0]
    int16_t isegments[32][2];

} segment_cluster_t;


static void eigenvector(// out
                        double* v,
                        // in
                        const double l,
                        const double* M,
                        const bool normalize_v)
{
    // Now to find the corresponding eigenvectors. Following:
    //   https://en.wikipedia.org/wiki/Eigenvalue_algorithm#Eigenvectors_of_normal_3%C3%973_matrices


    // l is an eigenvalue of M (I assume with multiplicity 1). As described on
    // that wikipedia page, the cross-product of any two linearly-independent
    // columns of M-eye(3)*l is parallel to the eigenvector. The eigenvector is
    // in the null space, so rank(M - eye(3)*l) = 2. So you MIGHT have two
    // columns that aren't linearly independent. I explicitly try two sets of
    // columns, and pick the larger cross-product.


    const double v0[] = {M[0] - l,
                         M[1],
                         M[2]};
    const double v1[] = {M[1],
                         M[3] - l,
                         M[4]};
    const double v2[] = {M[2],
                         M[4],
                         M[5] - l};

    double* vcross1 = v;
    double vcross2[3];

    vcross1[0] = v0[1]*v1[2] - v0[2]*v1[1];
    vcross1[1] = v0[2]*v1[0] - v0[0]*v1[2];
    vcross1[2] = v0[0]*v1[1] - v0[1]*v1[0];

    vcross2[0] = v0[1]*v2[2] - v0[2]*v2[1];
    vcross2[1] = v0[2]*v2[0] - v0[0]*v2[2];
    vcross2[2] = v0[0]*v2[1] - v0[1]*v2[0];

    const double norm2_vcross1 = vcross1[0]*vcross1[0] + vcross1[1]*vcross1[1] + vcross1[2]*vcross1[2];
    const double norm2_vcross2 = vcross2[0]*vcross2[0] + vcross2[1]*vcross2[1] + vcross2[2]*vcross2[2];

    if(norm2_vcross1 > norm2_vcross2)
    {
        // we take vcross1. This is already in v
        if(normalize_v)
        {
            const double mag = sqrt(norm2_vcross1);
            for(int i=0; i<3; i++)
                v[i] /= mag;
        }
    }
    else
    {
        // we take vcross2
        if(normalize_v)
        {
            const double mag = sqrt(norm2_vcross2);
            for(int i=0; i<3; i++)
                v[i] = vcross2[i] / mag;
        }
        else
            for(int i=0; i<3; i++)
                v[i] = vcross2[i];

    }
}

__attribute__((visibility("hidden")))
void eig_real_symmetric_3x3( // out
                             double* vsmallest, // the smallest-eigenvalue eigenvector; may be NULL
                             double* vlargest,  // the largest-eigenvalue  eigenvector; may be NULL
                             double* l,         // ALL the eigenvalues, in ascending order
                             // in
                             const double* M, // shape (6,); packed storage; row-first
                             const bool normalize_v )
{
    // I have a symmetric 3x3 matrix M. So the eigenvalues are real and >= 0.
    // The eigenvectors are orthonormal.

    // This implements
    // https://en.wikipedia.org/wiki/Eigenvalue_algorithm#Symmetric_3%C3%973_matrices
    const double p1    = M[1]*M[1] + M[2]*M[2] + M[4]*M[4];
    const double trace = M[0] + M[3] + M[5];

    const double q  = trace / 3.;
    const double dq[] = {M[0]-q,
                         M[3]-q,
                         M[5]-q};
    const double p2 =
        dq[0]*dq[0] + dq[1]*dq[1] + dq[2]*dq[2]
        + 2. * p1;
    const double p  = sqrt(p2 / 6.);
    const double r  =
        (q*(q*(-q + trace) +
            p1 - M[0]*M[3] - M[0]*M[5] - M[3]*M[5]) +
         M[0]*M[3]*M[5] - M[0]*M[4]*M[4] - M[1]*M[1]*M[5] + 2*M[1]*M[2]*M[4] - M[2]*M[2]*M[3])
        / (2.*p*p*p);

    // To handle round-off errors
    double phi;
    if (r <= -1)
        phi = M_PI / 3;
    else if(r >= 1)
        phi = 0.;
    else
        phi = acos(r) / 3.;

    l[0] = q + 2. * p * cos(phi + (2.*M_PI/3.)); // smallest; "eig3" in wikipedia
    l[2] = q + 2. * p * cos(phi);                // largest;  "eig1" in wikipedia
    l[1] = 3.*q - l[2] - l[0];


    if(vsmallest != NULL)
        eigenvector(vsmallest,
                    l[0], M, normalize_v);
    if(vlargest != NULL)
        eigenvector(vlargest,
                    l[2], M, normalize_v);
}

static void pca_preprocess_from_ipoint_set( // out
                                           double*              M,
                                           clc_point3f_t*       pmean,
                                           float*               max_norm2_dp, // may be NULL
                                           // in
                                           const clc_point3f_t* points,
                                           unsigned int         n,
                                           const uint32_t*      ipoint)
{
    *pmean = (clc_point3f_t){};
    for(unsigned int i=0; i<n; i++)
        for(int j=0; j<3; j++)
            pmean->xyz[j] += points[ipoint[i]].xyz[j];
    for(int j=0; j<3; j++)
        pmean->xyz[j] /= (float)(n);


    if(max_norm2_dp != NULL) *max_norm2_dp = 0.0f;

    for(unsigned int i=0; i<n; i++)
    {
        const clc_point3f_t dp = sub(points[ipoint[i]], *pmean);

        const float norm2_dp = norm2(dp);
        if(max_norm2_dp != NULL &&
           *max_norm2_dp < norm2_dp)
            *max_norm2_dp = norm2_dp;

        M[0] += (double)(dp.xyz[0]*dp.xyz[0]);
        M[1] += (double)(dp.xyz[0]*dp.xyz[1]);
        M[2] += (double)(dp.xyz[0]*dp.xyz[2]);
        M[3] += (double)(dp.xyz[1]*dp.xyz[1]);
        M[4] += (double)(dp.xyz[1]*dp.xyz[2]);
        M[5] += (double)(dp.xyz[2]*dp.xyz[2]);
    }
}

static void pca_preprocess_from_ipoint0_ipoint1( // out
                                                double*    M,
                                                clc_point3f_t* pmean,
                                                float*     max_norm2_dp, // may be NULL
                                                // in
                                                const clc_point3f_t* points,
                                                const int ipoint0, const int ipoint1)
{
    *pmean = (clc_point3f_t){};
    for(int i=ipoint0; i<=ipoint1; i++)
        for(int j=0; j<3; j++)
            pmean->xyz[j] += points[i].xyz[j];
    for(int j=0; j<3; j++)
        pmean->xyz[j] /= (float)(ipoint1-ipoint0+1);


    if(max_norm2_dp != NULL) *max_norm2_dp = 0.0f;

    for(int i=ipoint0; i<=ipoint1; i++)
    {
        const clc_point3f_t dp = sub(points[i], *pmean);

        const float norm2_dp = norm2(dp);
        if(max_norm2_dp != NULL &&
           *max_norm2_dp < norm2_dp)
            *max_norm2_dp = norm2_dp;

        M[0] += (double)(dp.xyz[0]*dp.xyz[0]);
        M[1] += (double)(dp.xyz[0]*dp.xyz[1]);
        M[2] += (double)(dp.xyz[0]*dp.xyz[2]);
        M[3] += (double)(dp.xyz[1]*dp.xyz[1]);
        M[4] += (double)(dp.xyz[1]*dp.xyz[2]);
        M[5] += (double)(dp.xyz[2]*dp.xyz[2]);
    }
}

static void pca( // out
                clc_point3f_t* pmean,
                clc_point3f_t* vsmallest,    // the smallest-eigenvalue eigenvector; may be NULL
                clc_point3f_t* vlargest,     // the largest-eigenvalue  eigenvector; may be NULL
                float*     max_norm2_dp, // may be NULL
                float*     eigenvalues_ascending, // 3 of these; may be NULL
                // in
                const clc_point3f_t* points,
                unsigned int         n,
                const uint32_t*      ipoint, // if NULL, we use [ipoint0,ipoint1]
                const int ipoint0, const int ipoint1,
                const bool normalize_v )
{
    /*
      I fit a plane to a set of points. The most accurate way to do this is to
      minimize the observation errors (ranges; what the fit ingesting this data
      will be doing). But that's a nonlinear solve, and I avoid that here. I
      simply minimize the norm2 off-plane error instead:

      - compute pmean
      - compute M = sum(outer(p[i]-pmean,p[i]-pmean))
      - n = eigenvector of M corresponding to the smallest eigenvalue

      Derivation: plane is (p,n); points are in a (3,N) matrix P.
      ei = nt (Pi - p)
      E = sum(inner(ei,ei))
      dE/dp ~ sum(eit nt)
            = sum( (Pit-pt) n nt )
      dE/dp = 0 -> sum(Pit) n nt = sum(pt) n nt
      -> p = mean(P)

      Let Q = P-mean(P)
      ei = nt Qi
      E = sum(inner(ei,ei)) = nt Q Qt n

      Lagrange multipliers to constrain norm2(n) = 1:
      L = nt Q Qt n - l nt n
      dL/dn = 2 nt Q Qt - 2 l nt
      dL/dn = 0 -> Q Qt n = l n

      This is an eigenvalue problem: l,n are eigen(values,vectors) of Q Qt
      E = nt Q Qt n = l nt n = l

     */

    // packed storage; row-first
    // double-precision because this is potentially inaccurate
    double M[3+2+1] = {};
    if(ipoint != NULL)
        pca_preprocess_from_ipoint_set( // out
                                       M,
                                       pmean,
                                       max_norm2_dp,
                                       // in
                                       points, n, ipoint);
    else
        pca_preprocess_from_ipoint0_ipoint1( // out
                                            M,
                                            pmean,
                                            max_norm2_dp,
                                            // in
                                            points,
                                            ipoint0,ipoint1);

    double l[3]; // all the eigenvalues, in ascending order
    double _vsmallest[3];
    double _vlargest [3];
    eig_real_symmetric_3x3(vsmallest != NULL ? _vsmallest : NULL,
                           vlargest  != NULL ? _vlargest  : NULL,
                           l,M,
                           normalize_v);
    if(vsmallest != NULL) *vsmallest = point3f_from_double(_vsmallest);
    if(vlargest  != NULL) *vlargest  = point3f_from_double(_vlargest);

    if(eigenvalues_ascending != NULL)
        for(int i=0; i<3; i++)
            eigenvalues_ascending[i] = (float)l[i];
}

static void fit_plane_into_points__normalized( // out
                                               clc_plane_t*         plane,
                                               float*               max_norm2_dp,
                                               float*               eigenvalues_ascending, // 3 of these
                                               // in
                                               const clc_point3f_t* points,
                                               unsigned int         n,
                                               const uint32_t*      ipoint)

{
    pca(&plane->p_mean,
        &plane->n,
        NULL,
        max_norm2_dp,
        eigenvalues_ascending,
        points, n, ipoint,
        -1,-1,
        true);
}

static void fit_plane_into_points__unnormalized( // out
                                                 plane_unnormalized_t* plane_unnormalized,
                                                 float*                max_norm2_dp,
                                                 float*                eigenvalues_ascending, // 3 of these
                                                 // in
                                                 const clc_point3f_t*  points,
                                                 unsigned int          n,
                                                 const uint32_t*       ipoint)
{
    pca(&plane_unnormalized->p,
        &plane_unnormalized->n_unnormalized,
        NULL,
        max_norm2_dp,
        eigenvalues_ascending,
        points, n, ipoint,
        -1,-1,
        false);
}




static
float th_from_point(const clc_point3f_t* p)
{
    return atan2f(p->y, p->x);
}


// ASSUMES th_rad CAME FROM atan2, SO IT'S IN [-pi,pi]
static
int isegment_from_th(const float th_rad,
                     const clc_lidar_segmentation_context_t* ctx)
{
    const float segment_width_rad = 2.0f*M_PI * (float)ctx->Npoints_per_segment / (float)ctx->Npoints_per_rotation;

    const int i = (int)((th_rad + M_PI) / segment_width_rad);
    // Should be in range EXCEPT if th_rad == +pi. Just in case and for good
    // hygiene, I check both cases
    if( i < 0 )
        return 0;
    if(i >= Nsegments_per_rotation)
        return Nsegments_per_rotation-1;
    return i;
}
static
bool point_is_valid__presolve(const clc_point3f_t* p,
                              const float dth_rad,
                              const bool debug,
                              const clc_lidar_segmentation_context_t* ctx)
{
    const int Ngap = (int)( 0.5f + fabsf(dth_rad) * (float)ctx->Npoints_per_rotation / (2.0f*M_PI) );

    // Ngap==1 is the expected, normal value. Anything larger is a gap
    if( DEBUG_ON_TRUE_POINT((Ngap-1) > ctx->threshold_max_Ngap,
                      p,
                      "%d > %d", Ngap-1, ctx->threshold_max_Ngap))
        return false;

    return true;
}

static
bool segment_is_valid(const segment_t* segment)
{
    return !(segment->v.x == 0 &&
             segment->v.y == 0 &&
             segment->v.z == 0);
}

static
bool is_point_segment_planar(const clc_point3f_t* p,
                             const int ipoint0,
                             const int ipoint1,
                             const uint64_t* bitarray_invalid,
                             const clc_lidar_segmentation_context_t* ctx)
{
    const clc_point3f_t* p0 = &p[ipoint0];
    const clc_point3f_t* p1 = &p[ipoint1];

    const clc_point3f_t v01 = sub(*p1,*p0);

    const float recip_norm2_v01 = 1.f / norm2(v01);

    // do all the points in the chunk lie along v? If so, this is a linear
    // feature
    for(int ipoint=ipoint0+1; ipoint<=ipoint1; ipoint++)
    {
        if(bitarray64_check(bitarray_invalid,ipoint-ipoint0))
            continue;

        const clc_point3f_t v = sub(p[ipoint], *p0);

        // I'm trying hard to avoid using anything other that +,-,*. Even /
        // is used just once, in the outer loop

        // v ~ k v01 -> E = norm2(k v01 - v) -> dE/dk = 0 ~ (k v01 - v)t v01
        // -> k norm2(v01) = inner(v01,v) -> k = inner(v01,v) / norm2(v01)
        // e = k v01 - v
        // norm2(e) = norm2(v) + inner(v01,v)^2 / norm2(v01)^2 norm2(v01) - 2 inner(v01,v)^2 / norm2(v01)
        //          = norm2(v) + inner(v01,v)^2 / norm2(v01) - 2 inner(v01,v)^2 / norm2(v01)
        //          = norm2(v) - inner(v01,v)^2 / norm2(v01)
        const float norm2_e = norm2(v) - inner(v,v01)*inner(v,v01) * recip_norm2_v01;

        if(norm2_e > ctx->threshold_max_deviation_off_segment_line*ctx->threshold_max_deviation_off_segment_line)
            return false;
    }
    return true;
}


static
void stage1_finish_segment(// out
                           segment_t* segments,
                           // in
                           const int iring, const int isegment,
                           const int Npoints_invalid_in_segment,
                           const uint64_t* bitarray_invalid,
                           const clc_point3f_t* p,
                           const int ipoint0,
                           const int ipoint1,
                           const bool debug_this_ring,
                           const clc_lidar_segmentation_context_t* ctx)
{
    segment_t* segment = &segments[isegment];

    const bool debug_region =
        (ctx->debug_xmin < p[ipoint0].x && p[ipoint0].x < ctx->debug_xmax &&
         ctx->debug_ymin < p[ipoint0].y && p[ipoint0].y < ctx->debug_ymax) ||
        (ctx->debug_xmin < p[ipoint1].x && p[ipoint1].x < ctx->debug_xmax &&
         ctx->debug_ymin < p[ipoint1].y && p[ipoint1].y < ctx->debug_ymax);
    const bool debug = debug_this_ring && debug_region;


    const int Npoints = ipoint1 - ipoint0 + 1 - Npoints_invalid_in_segment;

    if(DEBUG_ON_TRUE_SEGMENT(Npoints_invalid_in_segment > ctx->threshold_max_Npoints_invalid_segment,
                             iring,isegment,
                             "%d > %d", Npoints_invalid_in_segment, ctx->threshold_max_Npoints_invalid_segment) ||
       DEBUG_ON_TRUE_SEGMENT(Npoints < ctx->threshold_min_Npoints_in_segment,
                             iring,isegment,
                             "%d < %d", Npoints, ctx->threshold_min_Npoints_in_segment) ||
       DEBUG_ON_TRUE_SEGMENT(!is_point_segment_planar(p,ipoint0,ipoint1,bitarray_invalid, ctx),
                             iring,isegment,
                             ""))
    {
        *segment = (segment_t){};
        return;
    }

    // We passed the crude test: is_point_segment_planar() says that all the
    // points lie on my line. I now refine this segment's p,v estimate to make
    // this all work better with the downstream logic


    // This will be a conic section that I'm trying to represent as a line.

    pca(&segment->p,
        NULL,
        &segment->v,
        NULL,
        NULL,
        p,
        0, NULL,
        ipoint0, ipoint1,
        false);

    segment->ipoint0 = ipoint0;
    segment->ipoint1 = ipoint1;

    if(ctx->dump)
        printf("%f %f label %f %d-%d\n",
               segment->p.x,segment->p.y,segment->p.z,
               iring, isegment);
}


static void
stage1_segment_from_ring(// out
                         segment_t* segments_thisring,

                         // in
                         const int iring,
                         const clc_point3f_t* points_thisring,
                         const int Npoints_thisring,
                         const clc_lidar_segmentation_context_t* ctx)
{
    if(Npoints_thisring <= 0)
        return;

    // I want this to be fast, and I'm looking for very clear planes, so I do a
    // crude thing here:
    //
    // 1. I look for long plane-y (or line-y) sections, then expand them later
    //
    // 2. I have ordered data, and I know that each planar segment will be a
    //    very squashed conic section segment. In many cases, it'll be so
    //    squashed to appear linear (i.e. its plane would be ill-defined)
    //
    // So I check for linearity first. And then for a curve (conic section
    // slice). NO; TODAY I LOOK FOR LINEAR SEGMENTS ONLY

    // bit-field to indicate which points are valid/invalid
    const int Nwords_bitarray_invalid = bitarray64_nwords(ctx->Npoints_per_segment);
    uint64_t bitarray_invalid[Nwords_bitarray_invalid];


    const float th_rad0 = th_from_point(&points_thisring[0]);

    int ipoint0   = 0;
    int isegment0 = isegment_from_th(th_rad0, ctx);
    int Npoints_invalid_in_segment = 0;
    float th_rad_prev = th_rad0;

    memset(bitarray_invalid, 0, Nwords_bitarray_invalid*sizeof(uint64_t));

    for(int ipoint=1; ipoint<Npoints_thisring; ipoint++)
    {
        const float th_rad = th_from_point(&points_thisring[ipoint]);
        const int isegment = isegment_from_th(th_rad, ctx);
        if(isegment != isegment0)
        {
            stage1_finish_segment(segments_thisring,
                                  iring, isegment0,
                                  Npoints_invalid_in_segment,
                                  bitarray_invalid,
                                  points_thisring, ipoint0, ipoint-1,
                                  iring == ctx->debug_iring,
                                  ctx);

            ipoint0   = ipoint;
            isegment0 = isegment;
            Npoints_invalid_in_segment = 0;
            memset(bitarray_invalid, 0, Nwords_bitarray_invalid*sizeof(uint64_t));
        }

        // This should ALWAYS be true. But some datasets are weird, and the
        // azimuths don't change as quickly as expected, and we have extra
        // points in each segment. I ignore those; hopefully they're not
        // important
        if(ipoint-ipoint0 < ctx->Npoints_per_segment)
        {
            if(!point_is_valid__presolve(&points_thisring[ipoint], th_rad - th_rad_prev,
                                         (iring == ctx->debug_iring) &&
                                         (ctx->debug_xmin < points_thisring[ipoint].x && points_thisring[ipoint].x < ctx->debug_xmax &&
                                          ctx->debug_ymin < points_thisring[ipoint].y && points_thisring[ipoint].y < ctx->debug_ymax),
                                         ctx))
            {
                Npoints_invalid_in_segment++;
                bitarray64_set(bitarray_invalid, ipoint-ipoint0);
            }
        }

        th_rad_prev = th_rad;
    }

    stage1_finish_segment(segments_thisring,
                          iring, isegment0,
                          Npoints_invalid_in_segment,
                          bitarray_invalid,
                          points_thisring, ipoint0, Npoints_thisring-1,
                          iring == ctx->debug_iring,
                          ctx);
}

static bool is_normal(const clc_point3f_t v,
                      const clc_point3f_t n,
                      const clc_lidar_segmentation_context_t* ctx)
{
    // inner(v,n) = cos magv magn ->
    // cos = inner / (magv magn) ->
    // cos^2 = inner^2 / (norm2v norm2n) ->
    // cos^2 norm2v norm2n = inner^2
    float cos_mag_mag = inner(n,v);

    return
        cos_mag_mag*cos_mag_mag <
        ctx->threshold_max_cos_angle_error_normal*ctx->threshold_max_cos_angle_error_normal*norm2(n)*norm2(v);
}

static bool is_same_direction(const clc_point3f_t a,
                              const clc_point3f_t b,
                              const clc_lidar_segmentation_context_t* ctx)
{
    // inner(a,b) = cos maga magb ->
    // cos = inner / (maga magb) ->
    // cos^2 = inner^2 / (norm2a norm2b) ->
    // cos^2 norm2a norm2b = inner^2
    float cos_mag_mag = inner(a,b);

    return
        cos_mag_mag > 0.0f &&
        cos_mag_mag*cos_mag_mag >
        ctx->threshold_min_cos_angle_error_same_direction*ctx->threshold_min_cos_angle_error_same_direction*norm2(a)*norm2(b);
}

static bool segment_segment_across_rings_close_enough(const clc_point3f_t* dp,
                                                      const bool debug,
                                                      const clc_lidar_segmentation_context_t* ctx,
                                                      // for diagnostics only
                                                      const int iring1, const int isegment)
{
    return
        !DEBUG_ON_TRUE_SEGMENT(norm2(*dp) > ctx->threshold_max_distance_across_rings*ctx->threshold_max_distance_across_rings,
                               iring1,isegment,
                               "cross-ring segments are too far apart: %f > %f",
                               mag(*dp), ctx->threshold_max_distance_across_rings);
}

static bool plane_from_segment_segment(// out
                                       plane_unnormalized_t* plane_unnormalized,
                                       // in
                                       const segment_t* s0,
                                       const segment_t* s1,
                                       const clc_lidar_segmentation_context_t* ctx,
                                       // for diagnostics only
                                       const int iring1, const int isegment)
{
    // I want:
    //   inner(p1-p0, n=cross(v0,v1)) = 0
    clc_point3f_t dp = sub(s1->p, s0->p);

    const bool debug =
        ctx->debug_xmin < s1->p.x && s1->p.x < ctx->debug_xmax &&
        ctx->debug_ymin < s1->p.y && s1->p.y < ctx->debug_ymax;

    if(!segment_segment_across_rings_close_enough(&dp,
                                                  debug,
                                                  ctx,
                                                  iring1,isegment))
        return false;


    // The two normal estimates must be close
    clc_point3f_t n0 = cross(dp,s0->v);
    clc_point3f_t n1 = cross(dp,s1->v);

    if(!is_same_direction(n0,n1,ctx))
        return false;

    plane_unnormalized->n_unnormalized = mean(n0,n1);
    plane_unnormalized->p              = mean(s0->p, s1->p);
    return true;
}

static float plane_point_error_stage3_normalized(const clc_plane_t*   plane,
                                                 const clc_point3f_t* point)
{
    // I want (point - p) to be perpendicular to n. I want this in terms of
    // "distance-off-plane" so err = inner( (point - p), n) / mag(n)
    //
    // Accept if threshold > inner( (point - p), n) / mag(n)
    // n is normalized here, so I omit the /magn
    const clc_point3f_t dp = sub(*point, plane->p_mean);

    return inner(dp, plane->n);
}

static bool plane_point_compatible_stage2_unnormalized(const plane_unnormalized_t* plane_unnormalized,
                                                       const clc_point3f_t* point,
                                                       const clc_lidar_segmentation_context_t* ctx)
{
    // I want (point - p) to be perpendicular to n. I want this in terms of
    // "distance-off-plane" so err = inner( (point - p), n) / mag(n)
    //
    // Accept if threshold > inner( (point - p), n) / mag(n)
    // n is normalized here, so I omit the /magn
    const clc_point3f_t dp = sub(*point, plane_unnormalized->p);
    const float proj = inner(dp, plane_unnormalized->n_unnormalized);
    return ctx->threshold_max_plane_point_error_stage2*norm2(plane_unnormalized->n_unnormalized) > proj*proj;
}


static int
count_segments_in_cluster(const segment_cluster_t* cluster)
{
    const int Nrings = cluster->irings[1] - cluster->irings[0] + 1;
    int N = 0;
    for(int i=0; i<Nrings; i++)
    {
        N +=
            cluster->isegments[i][1] -
            cluster->isegments[i][0] + 1;
    }
    return N;
}


static bool fit_plane_into_cluster(// out
                                   plane_unnormalized_t* plane_unnormalized,
                                   // in
                                   const segment_cluster_t* cluster,

                                   // extra segment to consider in addition to the segments in the cluster
                                   const segment_t*   segment0,
                                   const int          iring0,

                                   const segment_t*   segments,
                                   const clc_point3f_t*   points,
                                   const int*         ipoint0_in_ring,
                                   const clc_lidar_segmentation_context_t*   ctx)
{
    // Storing the left/right ends of the existing segments and the one
    // candidate new segment
    const int Nsegments_max =
        ctx->threshold_max_Nsegments_in_cluster +
        1; // because of segment0

    const int       ipoint_size = 2*Nsegments_max;
    uint32_t        ipoint[ipoint_size];

    int npoints = 0;
    if(segment0 != NULL)
    {
        ipoint[npoints++] = ipoint0_in_ring[iring0] + segment0->ipoint0;
        ipoint[npoints++] = ipoint0_in_ring[iring0] + segment0->ipoint1;
    }

    for(int iring = cluster->irings[0];
        iring    <= cluster->irings[1];
        iring++)
    {
        for(int isegment = cluster->isegments[iring-cluster->irings[0]][0];
            isegment    <= cluster->isegments[iring-cluster->irings[0]][1];
            isegment++)
        {
            const segment_t* segment =
                &segments[iring*Nsegments_per_rotation + isegment];

            // Cluster is too big; it doesn't matter if it fits into the plane;
            // I return false
            if(npoints == ipoint_size)
                return false;

            ipoint[npoints++] = ipoint0_in_ring[iring] + segment->ipoint0;
            ipoint[npoints++] = ipoint0_in_ring[iring] + segment->ipoint1;
        }
    }

    float max_norm2_dp;
    float eigenvalues_ascending[3];
    fit_plane_into_points__unnormalized( // out
                                         plane_unnormalized,
                                         &max_norm2_dp,
                                         eigenvalues_ascending, // 3 of these
                                         // in
                                         points, npoints,ipoint);
    return true;
}

static bool stage2_plane_segment_compatible(// The initial plane estimate in
                                            // cluster->plane_unnormalized may be updated by
                                            // this call, if we return true
                                            segment_cluster_t* cluster,
                                            const int          iring, const int isegment,
                                            const segment_t*   segments,
                                            // for diagnostics only
                                            const int          icluster,
                                            const clc_point3f_t*   points,
                                            const int*         ipoint0_in_ring,
                                            const clc_lidar_segmentation_context_t*   ctx)
{
    const segment_t* segment = &segments[iring*Nsegments_per_rotation + isegment];

    // both segment->p and segment->v must lie in the plane

    // I want:
    //   inner(segv,   n) = 0
    //   inner(segp-p, n) = 0
    const bool debug =
        ctx->debug_xmin < segment->p.x && segment->p.x < ctx->debug_xmax &&
        ctx->debug_ymin < segment->p.y && segment->p.y < ctx->debug_ymax;
    if(!( DEBUG_ON_TRUE_SEGMENT( !is_normal(segment->v, cluster->plane_unnormalized.n_unnormalized, ctx),
                                 iring,isegment,
                                 "icluster=%d: segment isn't plane-consistent during accumulation: the direction isn't in-plane",
                                 icluster) ||
          DEBUG_ON_TRUE_SEGMENT( !plane_point_compatible_stage2_unnormalized(&cluster->plane_unnormalized, &segment->p, ctx),
                                 iring,isegment,
                                 "icluster=%d: segment isn't plane-consistent during accumulation: the point isn't in-plane",
                                 icluster)))
        return true;


    // Segment doesn't fit. I try to update the estimate using this new segment,
    // and check again. I don't do this in all paths, because it's potentially
    // slow, and I'd like to avoid doing this as much as possible.
    plane_unnormalized_t plane_unnormalized;
    if(!fit_plane_into_cluster(// out
                               &plane_unnormalized,
                               // in
                               cluster,
                               segment, iring,
                               segments,
                               points,
                               ipoint0_in_ring,
                               ctx))
        return false;


    // same check as above, but for all the extant segments and a with a new,
    // fitted plane. If the new plane doesn't fit any of the current segments, I
    // fail the test
    if(DEBUG_ON_TRUE_SEGMENT(!is_normal(segment->v, plane_unnormalized.n_unnormalized, ctx),
                             iring,isegment,
                             "icluster=%d: segment isn't plane-consistent during the re-fit check: the direction isn't in-plane",
                             icluster) ||
       DEBUG_ON_TRUE_SEGMENT( !plane_point_compatible_stage2_unnormalized(&plane_unnormalized, &segment->p, ctx),
                              iring,isegment,
                              "icluster=%d: segment isn't plane-consistent during the re-fit check: the point isn't in-plane",
                              icluster))
        return false;


    for(int iring_here = cluster->irings[0];
        iring_here    <= cluster->irings[1];
        iring_here++)
    {
        for(int isegment_here = cluster->isegments[iring_here-cluster->irings[0]][0];
            isegment_here    <= cluster->isegments[iring_here-cluster->irings[0]][1];
            isegment_here++)
        {
            const segment_t* segment_here =
                &segments[iring_here*Nsegments_per_rotation + isegment_here];

            if(DEBUG_ON_TRUE_SEGMENT(!is_normal(segment_here->v, plane_unnormalized.n_unnormalized, ctx),
                                     iring_here,isegment_here,
                                     "icluster=%d: segment isn't plane-consistent during the re-fit check: the direction isn't in-plane",
                                     icluster) ||
               DEBUG_ON_TRUE_SEGMENT(!plane_point_compatible_stage2_unnormalized(&plane_unnormalized, &segment_here->p, ctx),
                                     iring_here,isegment_here,
                                     "icluster=%d: segment isn't plane-consistent during the re-fit check: the isn't in-plane",
                                     icluster))
                return false;
        }
    }


    // new plane fits well-enough
    cluster->plane_unnormalized = plane_unnormalized;
    return true;
}


static void stage2_accumulate_segments_samering( // out
                                                int16_t*                                isegment,
                                                // in
                                                const int16_t                           isegment_increment,
                                                const int16_t                           isegment_limit, // first invalid
                                                segment_cluster_t*                      cluster,
                                                const int                               iring,
                                                segment_t*                              segments, // not const to set visited
                                                const int                               icluster,
                                                const clc_point3f_t*                    points,
                                                const int*                              ipoint0_in_ring,
                                                const clc_lidar_segmentation_context_t* ctx)
{
    while(true)
    {
        if(*isegment == isegment_limit)
            return;

        int16_t isegment_next = *isegment + isegment_increment;

        segment_t* segment = &segments[iring*Nsegments_per_rotation + isegment_next];

        if(segment->visited)
            return;

        if(!segment_is_valid(segment))
            return;

        if(!stage2_plane_segment_compatible(// The initial plane estimate in
                                            // cluster->plane_unnormalized may be updated by
                                            // this call, if we return true
                                            cluster,
                                            iring, isegment_next,
                                            segments,
                                            // for diagnostics only
                                            icluster, points, ipoint0_in_ring, ctx))
            return;

        segment->visited = true;
        *isegment = isegment_next;
    }
}

static
bool stage2_next_ring_segment_compatible(// The initial plane estimate in
                                         // cluster->plane_unnormalized may be
                                         // updated by this call, if we return
                                         // true
                                         segment_cluster_t*                      cluster,
                                         const int                               iring1,
                                         const int                               isegment,
                                         // context
                                         const segment_t*                        segments,
                                         const int                               icluster,
                                         const clc_point3f_t*                    points,
                                         const int*                              ipoint0_in_ring,
                                         const clc_lidar_segmentation_context_t* ctx)
{
    const segment_t*     segment  = &segments[iring1    *Nsegments_per_rotation + isegment];
    const segment_t*     segment0 = &segments[(iring1-1)*Nsegments_per_rotation + isegment];
    const clc_point3f_t* p0       = &segment0->p;
    const clc_point3f_t* p        = &segment->p;
    const clc_point3f_t  dp       = sub(*p0, *p);
    const bool debug              =
        ctx->debug_xmin < p->x && p->x < ctx->debug_xmax &&
        ctx->debug_ymin < p->y && p->y < ctx->debug_ymax;

    return
        !segment->visited &&
        segment_is_valid(segment) &&
        segment_segment_across_rings_close_enough(&dp,
                                                  debug,
                                                  ctx,
                                                  // for diagnostics only
                                                  iring1,isegment) &&
        stage2_plane_segment_compatible(// The initial plane estimate in
                                        // cluster->plane_unnormalized may be updated by
                                        // this call, if we return true
                                        cluster,
                                        iring1, isegment,
                                        segments,
                                        // for diagnostics only
                                        icluster,
                                        points, ipoint0_in_ring,
                                        ctx);

}

static void
stage2_grow_cluster(// out
                    segment_cluster_t* cluster,
                    // in
                    const int        icluster,
                    // start from here
                    const int iring0, const int isegment,
                    segment_t*       segments, // non-const to be able to set "visited"
                    const clc_point3f_t* points,
                    const int*       ipoint0_in_ring,
                    const clc_lidar_segmentation_context_t* ctx)
{
    *cluster = (segment_cluster_t){.irings       = {iring0,iring0},
                                   .isegments    =
                                   { [0] = {isegment,isegment} }};

    int iring = iring0;
    while(true)
    {
        stage2_accumulate_segments_samering( // out
                                             &cluster->isegments[iring-iring0][0],
                                             // in
                                             -1,
                                             -1,
                                             cluster,
                                             iring,
                                             segments,
                                             icluster,
                                             points,
                                             ipoint0_in_ring,
                                             ctx);
        stage2_accumulate_segments_samering( // out
                                             &cluster->isegments[iring-iring0][1],
                                             // in
                                             1,
                                             Nsegments_per_rotation,
                                             cluster,
                                             iring,
                                             segments,
                                             icluster,
                                             points,
                                             ipoint0_in_ring,
                                             ctx);

        // We have the expanded set of segments in iring. I try to find a
        // matching segment in the next ring, and propagate that
        const int isegment0 = cluster->isegments[iring-iring0][0];
        const int isegment1 = cluster->isegments[iring-iring0][1];

        iring++;

        if(iring-iring0 >= sizeof(cluster->isegments)/sizeof(cluster->isegments[0]))
        {
            MSG("Ring overflow. Increase sizeof(cluster->isegments)");
            return;
        }

        if(iring >= ctx->Nrings)
            return;

        int isegment_nextring_offcenter0 = -1;
        int isegment_nextring_offcenter1 = -1;
        const int isegment_center = (isegment0 + isegment1)/2;
        // I start searching in the middle of the previous ring
        for( int i = 0;
             isegment_center-i >= isegment0;
             i++ )
        {
            if(stage2_next_ring_segment_compatible(cluster,
                                                   iring,
                                                   isegment_center-i,
                                                   // context
                                                   segments, icluster, points, ipoint0_in_ring, ctx))
            {
                isegment_nextring_offcenter0 = i;
                break;
            }
        }
        for( int i = 1;
             isegment_center+i <= isegment1;
             i++ )
        {
            if(stage2_next_ring_segment_compatible(cluster,
                                                   iring,
                                                   isegment_center+i,
                                                   // context
                                                   segments, icluster, points, ipoint0_in_ring, ctx))
            {
                isegment_nextring_offcenter1 = i;
                break;
            }
        }

        // The nearest (to the center) matching segments in the next ring are
        // isegment_nextring_offcenter0,1. >=0 if defined
        int isegment_nextring;
        if(isegment_nextring_offcenter0 < 0 &&
           isegment_nextring_offcenter1 < 0)
        {
            // No matching segments in the next ring. We're done
            return;
        }
        if(isegment_nextring_offcenter0 < 0)
            // The matching segment appears on only one side. Take it
            isegment_nextring = isegment_center+isegment_nextring_offcenter1;
        else if(isegment_nextring_offcenter1 < 0)
            // The matching segment appears on only one side. Take it
            isegment_nextring = isegment_center-isegment_nextring_offcenter0;
        else
        {
            // The matching segment appears on both sides. Take the one closer
            // to the center
            if(isegment_nextring_offcenter0 < isegment_nextring_offcenter1)
                isegment_nextring = isegment_center-isegment_nextring_offcenter0;
            else
                isegment_nextring = isegment_center+isegment_nextring_offcenter1;
        }

        // I have the next-ring segment. Add it to the cluster, and expand it
        cluster->irings[1] = iring;
        cluster->isegments[iring-iring0][0] = isegment_nextring;
        cluster->isegments[iring-iring0][1] = isegment_nextring;

        segment_t* segment = &segments[iring*Nsegments_per_rotation + isegment_nextring];
        segment->visited = true;
    }
}


static void stage2_cluster_segments(// out
                                    segment_cluster_t* clusters,
                                    int* Nclusters,
                                    const int Nclusters_max, // size of clusters[]

                                    // in
                                    segment_t*       segments, // non-const to be able to set "visited"
                                    const clc_point3f_t* points,
                                    const int*       ipoint0_in_ring,
                                    const clc_lidar_segmentation_context_t* ctx)
{
    *Nclusters = 0;

    for(int iring0 = 0; iring0 < ctx->Nrings-1; iring0++)
    {
        const int iring1 = iring0+1;

        for(int isegment = 0; isegment < Nsegments_per_rotation; isegment++)
        {
            const int icluster = *Nclusters;

            if(icluster == Nclusters_max)
            {
                MSG("Too many flat objects in scene, exceeded Nclusters_max. Not reporting any more candidate planes. Bump Nclusters_max");
                return;
            }

            segment_t* segment0 = &segments[iring0*Nsegments_per_rotation + isegment];
            segment_t* segment1 = &segments[iring1*Nsegments_per_rotation + isegment];
            if(!(segment_is_valid(segment0) && !segment0->visited))
                continue;
            if(!(segment_is_valid(segment1) && !segment1->visited))
                continue;


            // A single segment has only a direction. To define a plane I need
            // another non-colinear segment, and I can get it from one of the
            // two adjacent rings. Once I get a plane I find the biggest
            // connected component of plane-consistent segments around this one
            segment_cluster_t* cluster = &clusters[icluster];
            const bool debug =
                ctx->debug_xmin < segment0->p.x && segment0->p.x < ctx->debug_xmax &&
                ctx->debug_ymin < segment0->p.y && segment0->p.y < ctx->debug_ymax;
            if(DEBUG_ON_TRUE_SEGMENT(!plane_from_segment_segment(&cluster->plane_unnormalized,
                                                                 segment0,segment1,
                                                                 ctx,
                                                                 // for diagnostics only
                                                                 iring1, isegment),
                                     iring0,isegment,
                                     "icluster=%d: segment isn't plane-consistent with %d-%d",
                                     icluster,
                                     iring1,isegment))
                continue;

            segment0->visited = true;

            stage2_grow_cluster(// out
                                cluster,

                                // in
                                icluster,
                                iring0, isegment,
                                segments,
                                points,
                                ipoint0_in_ring,
                                ctx);

            const int Nsegments_in_cluster = count_segments_in_cluster(cluster);

            if(DEBUG_ON_TRUE_SEGMENT(Nsegments_in_cluster == 2,
                                     iring0,isegment,
                                     "icluster=%d only contains the seed segments",
                                     icluster))
            {
                // This hypothetical ring-ring component is too small. The
                // next-ring segment might still be valid in another component,
                // with a different plane, without segment. So I allow it again.
                segment1->visited = false;
                continue;
            }

            if(DEBUG_ON_TRUE_SEGMENT(Nsegments_in_cluster < ctx->threshold_min_Nsegments_in_cluster,
                                     iring0,isegment,
                                     "icluster=%d too small: %d < %d",
                                     icluster,
                                     Nsegments_in_cluster, ctx->threshold_min_Nsegments_in_cluster))
            {
                continue;
            }

            if(DEBUG_ON_TRUE_SEGMENT(Nsegments_in_cluster > ctx->threshold_max_Nsegments_in_cluster,
                                     iring0,isegment,
                                     "icluster=%d too big: %d > %d",
                                     icluster,
                                     Nsegments_in_cluster, ctx->threshold_max_Nsegments_in_cluster))
            {
                continue;
            }

            {
                const int iring0 = cluster->irings[0];
                const int iring1 = cluster->irings[1];
                if(DEBUG_ON_TRUE_SEGMENT(iring1-iring0+1 < ctx->threshold_min_Nrings_in_cluster,
                                         iring0,isegment,
                                         "icluster=%d contains too-few rings: %d < %d",
                                         icluster,
                                         iring1-iring0+1, ctx->threshold_min_Nrings_in_cluster))
                    continue;
            }

            // I throw out any cluster that's entirely too far. I only do this
            // now because it's possible to see far-too-large planes that are
            // partially too far (walls, ground), and I want to detect this
            // far-too-large-ness, and throw them out
            bool keep = false;
            for(int iring = cluster->irings[0];
                iring    <= cluster->irings[1];
                iring++)
            {
                for(int isegment = cluster->isegments[iring-cluster->irings[0]][0];
                    isegment    <= cluster->isegments[iring-cluster->irings[0]][1];
                    isegment++)
                {
                    const segment_t* segment = &segments[iring*Nsegments_per_rotation + isegment];

                    if( norm2(segment->p) < ctx->threshold_max_range*ctx->threshold_max_range )
                    {
                        keep = true;
                        break;
                    }
                }
                if(keep) break;
            }
            if(DEBUG_ON_TRUE_SEGMENT(!keep,
                                     iring0,isegment,
                                     "icluster=%d is completely past the threshold_max_range=%f",
                                     icluster, ctx->threshold_max_range))
                continue;



            // We're accepting this cluster. To prepare for the next stage I
            // refine the plane estimate and I normalize the normal vector
            if(!fit_plane_into_cluster(// out
                                       &cluster->plane_unnormalized,
                                       // in
                                       cluster,
                                       NULL, -1,
                                       segments,
                                       points,
                                       ipoint0_in_ring,
                                       ctx))
                continue;

            const float magn = mag(cluster->plane_unnormalized.n_unnormalized);
            for(int i=0; i<3;i++)
                cluster->plane.n.xyz[i] = cluster->plane_unnormalized.n_unnormalized.xyz[i] / magn;

            (*Nclusters)++;
        }
    }
}


static void stage3_accumulate_points(// out
                                     unsigned int* n, // in,out
                                     uint32_t* ipoints,
                                     unsigned int max_num_ipoints,
                                     // in,out
                                     uint64_t* bitarray_visited, // indexed by IN-RING points
                                     // in
                                     const int ipoint0, const int ipoint_increment, const int ipoint_limit, // in-ring
                                     const clc_plane_t* plane,
                                     const clc_point3f_t* points,
                                     const int ipoint0_in_ring, // start of this ring in the full points[] array
                                     const int ipoint_segment_limit,
                                     // for diagnostics
                                     const int icluster, const int iring, const int isegment,
                                     const bool debug __attribute__((unused)),
                                     const clc_lidar_segmentation_context_t* ctx)
{

    float th_rad_last = FLT_MAX; // indicate an invalid value initially

    for(int ipoint=ipoint0;
        ipoint != ipoint_limit;
        ipoint += ipoint_increment)
    {
        if( DEBUG_ON_TRUE_POINT( bitarray64_check(bitarray_visited, ipoint),
                                 &points[ipoint0_in_ring + ipoint],
                                 "%d-%d: we already processed this point; accumulation stopped",
                                 iring,isegment))
        {
            // We already processed this point, presumably from the other side.
            // There's no reason to keep going, since we already approached from the
            // other side
            break;
        }

        const float th_rad = th_from_point(&points[ipoint0_in_ring + ipoint]);
        float abs_dth_rad = 0.0f;
        if(th_rad_last < FLT_MAX)
        {
            // we have a valid th_rad_last
            abs_dth_rad = fabsf(th_rad - th_rad_last);
            if( DEBUG_ON_TRUE_POINT( abs_dth_rad > ctx->threshold_max_gap_th_rad,
                                     &points[ipoint0_in_ring + ipoint],
                                     "%d-%d: gap too large; accumulation stopped. Have ~ %f > %f",
                                     iring,isegment,
                                     abs_dth_rad, ctx->threshold_max_gap_th_rad))
                break;
        }
        else
        {
            // we do not have a valid th_rad_last. Stop when we reach the segment
            // limit
            if( DEBUG_ON_TRUE_POINT( ipoint == ipoint_segment_limit,
                                     &points[ipoint0_in_ring + ipoint],
                                     "%d-%d: reached end of point sequence; accumulation stopped. Have %d == %d",
                                     iring,isegment,
                                     ipoint, ipoint_segment_limit))
                break;
        }

        // no threshold_max_range check here. This was already checked when
        // constructing the candidate segments. So if we got this far, I assume it's
        // good
        if( DEBUG_ON_TRUE_POINT( ctx->threshold_max_plane_point_error_stage3 <
                                 fabsf(plane_point_error_stage3_normalized(plane,
                                                                           &points[ipoint0_in_ring + ipoint])),
                                 &points[ipoint0_in_ring + ipoint],
                                 "%d-%d: point too far off-plane; skipping point, but continuing. Have %f < fabsf(%f)",
                                 iring,isegment,
                                 ctx->threshold_max_plane_point_error_stage3,
                                 plane_point_error_stage3_normalized(plane,
                                                                     &points[ipoint0_in_ring + ipoint])))
        {
            // Not accepting this point, but also not updating th_rad_last. So too
            // many successive invalid points will create a too-large gap, failing
            // the threshold_max_gap_th_rad check above
            continue;
        }

        // accept the point



        // I will be fitting a plane to a set of points. The most accurate way
        // to do this is to minimize the observation errors (ranges; what the
        // fit ingesting this data will be doing). But that's a nonlinear solve,
        // and I avoid that here. I simply minimize the norm2 off-plane error
        // instead:
        //
        // - compute pmean
        // - compute M = sum(outer(p[i]-pmean,p[i]-pmean))
        // - n = eigenvector of M corresponding to the smallest eigenvalue
        //
        // So to accumulate a point I would need two passes over the data: to
        // compute pmean and then M. I might be able to expand the sum() in M to
        // make it work in one pass, but that's very likely to produce high
        // floating point round-off errors. So I actually accumulate the full
        // points for now, and might do something more efficient later
        if(*n == max_num_ipoints)
        {
            MSG("clc_points_and_plane_t->ipoint overflow. Skipping the rest of the points");
            break;
        }
        ipoints[(*n)++] = ipoint0_in_ring + ipoint;

        bitarray64_set(bitarray_visited, ipoint);
        th_rad_last = th_rad;
    }
}


// Not doing this yet. It's complex, and doesn't obviously work well to make
// things better
// I'm turning this off for now. It doesn't work right. I'm at ddc519d. This
// fails:
//
//   ./lidar-segmentation-test.py --dump --debug -1 ${^x0y0x1y1} \
//     /vl_points_1 \
//     2023-11-01/images-and-lidar-24.bag \
//   | awk " $x0y0x1y1[1] < \$1 && \$1 < $x0y0x1y1[3] && $x0y0x1y1[2] < \$2 && \$2 < $x0y0x1y1[4]" \
//   | feedgnuplot \
//       --style label "with labels" \
//       --style ACCEPTED "with points pt 2 ps 2 lw 2 lc \"red\"" \
//       --tuplesize label 4 \
//       --style all "with points pt 7 ps 0.5" \
//       --style stage1-segment "with vectors lc \"green\"" \
//       --style plane-normal   "with vectors lc \"black\"" \
//       --tuplesize stage1-segment,plane-normal 6 \
//       --3d \
//       --domain \
//       --dataid \
//       --square \
//       --points \
//       --tuplesizeall 3 \
//       --autolegend \
//       --xlabel x \
//       --ylabel y \
//       --zlabel z
#if 0
static int
stage3_cull_bloom_and_count_non_isolated(// out
                                         unsigned int* n, // in,out
                                         uint32_t* ipoints,
                                         // in
                                         const int ipoint_increment, const int ipoint_limit, // in-ring
                                         const clc_plane_t* plane,
                                         const clc_point3f_t* points,
                                         const int ipoint_set_start_this_ring,
                                         const int ipoint0_in_ring, // start of this ring in the full points[] array
                                         const int icluster, const int iring,
                                         const bool final_iteration,
                                         const bool debug __attribute__((unused)),
                                         const clc_lidar_segmentation_context_t* ctx)
{
    /*
      Ideally, the board I'm looking for should be floating in space:
      immediately around the board I should see either invalid data or points
      far away from the board. If this is violated, then it's likely I'm looking
      not at a board, but at a flat region of a larger object.

      Furthermore, lidar observations exhibit a blooming effect: just past the
      edge of an object in space we might observe extra points that might be a
      bit too far or a bit too near, but this is usually consistent. I want to
      identify and throw out these bloomed regions, but retain the full
      observation.

      So the full logic at the observation edge is:

      - Compute the plane residual (a signed scalar). At the edge of each ring, this
      projection will lie on one side of 0, and eventually cross over. All points
      before this crossover are potentially a part of the bloom, and should be
      discarded

      - Now that we have thrown out the bloom, we check for past-the-edge
      detections that would invalidate this whole plane. I require at least X
      radians of empty-or-far-away-from-board space on either side, any
      blooming excepted (X should be larger than any expected bloom size)
    */



    const float threshold_bloom = 0.02f;

    const float th_bloom_allowed_rad     = 1.0f * M_PI/180.f;
    const float th_deadzone_required_rad = 2.0f * M_PI/180.f;

    const float threshold_isolation = 0.2;




    const int ipoint_set_n_before_cull_bloom = *n;


    float err_previous =
        plane_point_error_stage3_normalized(plane,
                                            &points[ ipoints[*n - 1] ]);
    if(fabsf(err_previous) < threshold_bloom)
    {
        // The edge point is very near the plane. There's no blooming I can
        // reliably see, and I'm done with the bloom-removal stuff
    }
    else
    {
        bool err_positive_first = err_previous > 0.0f;


        for( int i = *n - 2;
             i >= ipoint_set_start_this_ring;
             i-- )
        {
            const float err =
                plane_point_error_stage3_normalized(plane,
                                                    &points[ ipoints[i] ]);

            // I stop the cull when the error turns around at the other side of 0
            bool err_positive_now        = err > 0.0f;
            bool err_positive_derivative = err > err_previous;

            if( fabsf(err) < threshold_bloom ||
                ((err_positive_now ^ err_positive_first) &&
                 (err_positive_now ^ err_positive_derivative)) )
            {
                // We're at the first "good" point. Cull everything else
                const int ipoint_set_n_new = i+1;
                if(final_iteration && ctx->dump)
                    for(unsigned int j=ipoint_set_n_new; j < *n; j++)
                        printf("%f %f stage3-culled-bloom %f\n",
                               points[ ipoints[j] ].x,
                               points[ ipoints[j] ].y,
                               points[ ipoints[j] ].z);



                *n = ipoint_set_n_new;
                break;
            }

            err_previous = err;
        }

        /* if(too few points remaining) */
        /*  complain and exit; */

        // if we get here without ever finding a "good" point, do something
    }



    if(!final_iteration)
        return 0;


    ////////////////////////////////////////////////////////





    // I start at the edge, and move outwards. I want to see
    //
    // - Maybe a small angular region where this more-or-less fit (bloom)
    // - A minimum-angle region of no data or points that fit very badly
    //
    // If I see anything else, this isn't an isolated ring, and I conclude that
    // I'm not looking at a valid board, and I reject the whole thing

    // last accepted point
    const int ipoint0 = ipoints[ipoint_set_n_before_cull_bloom-1] - ipoint0_in_ring;
    float th0_rad = th_from_point(&points[ipoint0_in_ring + ipoint0]);

    float th0_deadzone_start_rad;

    bool in_bloom_region = true;

    if(debug)
    {
        fprintf(stderr, "\n");
        MSG("iring=%d ipoint_increment=%d: starting at edge: (%f,%f,%f)",
            iring, ipoint_increment,
            points[ipoint0_in_ring + ipoint0].x,
            points[ipoint0_in_ring + ipoint0].y,
            points[ipoint0_in_ring + ipoint0].z);
    }


    int Npoints_non_isolated = 0;

    for(int ipoint = ipoint0 + ipoint_increment;
        ipoint != ipoint_limit;
        ipoint += ipoint_increment)
    {
        if(debug)
        {
            MSG("iring=%d ipoint_increment=%d: looking at (%f,%f,%f)",
                iring, ipoint_increment,
                points[ipoint0_in_ring + ipoint].x,
                points[ipoint0_in_ring + ipoint].y,
                points[ipoint0_in_ring + ipoint].z);
        }

        const float th_rad = th_from_point(&points[ipoint0_in_ring + ipoint]);

        if(in_bloom_region)
        {
            const float dth = th_rad - th0_rad;
            if(fabsf(dth) <= th_bloom_allowed_rad)
            {
                const float err =
                    fabsf(plane_point_error_stage3_normalized(plane,
                                                              &points[ipoint0_in_ring + ipoint]));
                if(DEBUG_ON_TRUE_POINT(threshold_isolation > err,
                                       &points[ipoint0_in_ring + ipoint],
                                       "iring=%d: bloom; point is in-plane; continuing; %f > %f",
                                       iring,
                                       threshold_isolation, err))
                {
                    continue;
                }

                if(debug)
                {
                    MSG("iring=%d ipoint_increment=%d: bloom; point is NOT in-plane; looking for the dead-zone early: %f < %f",
                        iring, ipoint_increment,
                        threshold_isolation, err);
                }
                // We start the deadzone early. Right at this badly-fitting
                // point
                th0_deadzone_start_rad = th_rad;
            }
            else
            {
                th0_deadzone_start_rad = th0_rad + copysignf(th_bloom_allowed_rad, dth);

                if(debug)
                {
                    MSG("iring=%d ipoint_increment=%d: looking for the dead-zone",
                        iring, ipoint_increment);
                }

            }

            // We're past where blooming is expected or the point doesn't fit. This is not a bloom anymore, and we change
            // modes, and fall through
            in_bloom_region = false;
        }

        const float dth = th_rad - th0_deadzone_start_rad;
        if(fabsf(dth) > th_deadzone_required_rad)
        {
            // We're past the deadzone. Done!
            if(debug)
            {
                MSG("iring=%d ipoint_increment=%d: past the dead zone; done",
                    iring, ipoint_increment);
            }
            return Npoints_non_isolated;
        }

        const float err =
            fabsf(plane_point_error_stage3_normalized(plane,
                                                      &points[ipoint0_in_ring + ipoint]));
        if(DEBUG_ON_TRUE_POINT(ctx->threshold_min_plane_point_error_isolation > err,
                               &points[ipoint0_in_ring + ipoint],
                               "iring=%d: looking for the dead zone, but point is too close. Plane NOT isolated: %f > %f",
                               iring,
                               ctx->threshold_min_plane_point_error_isolation,
                               err))
        {
            if(ctx->dump)
                printf("%f %f stage3-non-isolated %f\n",
                       points[ipoint0_in_ring + ipoint].x,
                       points[ipoint0_in_ring + ipoint].y,
                       points[ipoint0_in_ring + ipoint].z);

            // Too close. This plane isn't isolated. Throw it out
            Npoints_non_isolated++;
        }
        if(debug)
            MSG("iring=%d ipoint_increment=%d: looking for the dead zone, and point is far-enough. Continuing",
                iring, ipoint_increment);
    }

    if(debug)
        MSG("iring=%d ipoint_increment=%d: reached the end of the scan. We reject the whole plane",
            iring, ipoint_increment);
    return INT_MAX;
}
#endif

static bool stage3_refine_cluster(// out
                                  clc_points_and_plane_t* points_and_plane,
                                  float*              max_norm2_dp,
                                  float*              eigenvalues_ascending, // 3 of these
                                  // in
                                  const int icluster,
                                  const segment_cluster_t* cluster,
                                  const segment_t* segments,
                                  const clc_point3f_t* points,
                                  const int* ipoint0_in_ring,
                                  const unsigned int* Npoints,
                                  const clc_lidar_segmentation_context_t* ctx)
{
    /* I have an approximate plane estimate.

       while(...)
       {
         gather set of neighborhood points that match the current plane estimate
         update plane estimate using this set of points
       }
     */

    const int iring0 = cluster->irings[0];
    const int iring1 = cluster->irings[1];

    const int Nrings_considered = iring1-iring0+1;

    // I keep track of the already-visited points
    const int Nwords_bitarray_visited = bitarray64_nwords(ctx->Npoints_per_rotation); // largest-possible size
    uint64_t bitarray_visited[Nrings_considered][Nwords_bitarray_visited];

    // Start with the best-available plane estimate. This should be pretty good
    // already.
    clc_plane_t plane_out = cluster->plane;


    const bool debug =
        ctx->debug_xmin < cluster->plane.p_mean.x && cluster->plane.p_mean.x < ctx->debug_xmax &&
        ctx->debug_ymin < cluster->plane.p_mean.y && cluster->plane.p_mean.y < ctx->debug_ymax;

    // will only be modified on the final iteration
    int Npoints_non_isolated = 0;

#warning FOR NOW I JUST RUN A SINGLE ITERATION
    const int Niterations = 1;
    for(int iteration=0; iteration<Niterations; iteration++)
    {
        const bool final_iteration = (iteration == Niterations-1);

        points_and_plane->n = 0;
        int ipoint_set_n_prev = 0;

        for(int i=0; i<Nrings_considered; i++)
            memset(bitarray_visited[i], 0, Nwords_bitarray_visited*sizeof(uint64_t));





        if(Nrings_considered > 64)
        {
            MSG("Too many rings");
            return false;
        }
        // all the rings fit into one uint64_t word
        uint64_t bitarray_ring_visited = 0;



        for(int iring = cluster->irings[0];
            iring    <= cluster->irings[1];
            iring++)
        {
            for(int isegment = cluster->isegments[iring-cluster->irings[0]][0];
                isegment    <= cluster->isegments[iring-cluster->irings[0]][1];
                isegment++)
            {
                /////////// This is temporary, until I reimplement the way data is
                /////////// passed to this function. It should just be a list of
                /////////// rings and a single seed point for each
                if(bitarray_ring_visited & (1U << (iring-iring0)))
                    continue;
                bitarray_ring_visited |= 1U << (iring-iring0);



                const segment_t* segment =
                    &segments[iring*Nsegments_per_rotation + isegment];

                // I start in the center of each segment, and expand outwards to
                // capture all the matching points
                const int ipoint0 = (segment->ipoint0 + segment->ipoint1) / 2;

                unsigned int ipoint_set_start_this_ring __attribute__((unused)); // for the currently-disabled bloom_cull logic

                ipoint_set_start_this_ring = points_and_plane->n;
                stage3_accumulate_points(// out
                                         &points_and_plane->n,
                                         points_and_plane->ipoint,
                                         (int)(sizeof(points_and_plane->ipoint)/sizeof(points_and_plane->ipoint[0])),
                                         bitarray_visited[iring-iring0],
                                         // in
                                         ipoint0, +1, Npoints[iring],
                                         &plane_out,
                                         points,
                                         ipoint0_in_ring[iring],
                                         segment->ipoint1,
                                         // for diagnostics
                                         icluster, iring, isegment,
                                         debug,
                                         ctx);

                // disabling this for now; see comment at stage3_cull_bloom_and_count_non_isolated() above
#if 0
                if(points_and_plane->n > ipoint_set_start_this_ring)
                {
                    // some points were added

                    // I cull the bloom points at the edges. This will need to
                    // un-accumulate some points. stage3_accumulate_points() does:
                    //
                    //   points_and_plane->ipoint[points_and_plane->n++] = ipoint0_in_ring + ipoint;
                    //   bitarray64_set(bitarray_visited, ipoint);
                    //
                    // I can easily update the ipoint_set: I pull some points off the
                    // end. The bitarray_visited doesn't matter, since I will never
                    // process this ring again
                    //
                    // After we get a final-ish fit. I check to see if this board
                    // isn't isolated in space. And if it isn't, I reject it

                    // This will be non-zero ONLY if final_iteration
                    int Npoints_non_isolated_here =
                        stage3_cull_bloom_and_count_non_isolated(// out
                                                                 &points_and_plane->n,
                                                                 points_and_plane->ipoint,
                                                                 // in
                                                                 +1, Npoints[iring],
                                                                 &plane_out,
                                                                 points,
                                                                 ipoint_set_start_this_ring,
                                                                 ipoint0_in_ring[iring],
                                                                 icluster, iring,
                                                                 final_iteration,
                                                                 debug,
                                                                 ctx);
                    if(Npoints_non_isolated_here == INT_MAX)
                        Npoints_non_isolated = Npoints_non_isolated_here;
                    else
                        Npoints_non_isolated += Npoints_non_isolated_here;
                }
#endif


                ipoint_set_start_this_ring = points_and_plane->n;
                stage3_accumulate_points(// out
                                         &points_and_plane->n,
                                         points_and_plane->ipoint,
                                         (int)(sizeof(points_and_plane->ipoint)/sizeof(points_and_plane->ipoint[0])),
                                         bitarray_visited[iring-iring0],
                                         // in
                                         ipoint0-1, -1, -1,
                                         &plane_out,
                                         points,
                                         ipoint0_in_ring[iring],
                                         segment->ipoint0,
                                         // for diagnostics
                                         icluster, iring, isegment,
                                         debug,
                                         ctx);
                // disabling this for now; see comment at stage3_cull_bloom_and_count_non_isolated() above
#if 0
                if(points_and_plane->n > ipoint_set_start_this_ring)
                {
                    // This will be non-zero ONLY if final_iteration
                    int Npoints_non_isolated_here =
                        stage3_cull_bloom_and_count_non_isolated(// out
                                                                 &points_and_plane->n,
                                                                 points_and_plane->ipoint,
                                                                 // in
                                                                 -1, -1,
                                                                 &plane_out,
                                                                 points,
                                                                 ipoint_set_start_this_ring,
                                                                 ipoint0_in_ring[iring],
                                                                 icluster, iring,
                                                                 final_iteration,
                                                                 debug,
                                                                 ctx);
                    if(Npoints_non_isolated_here == INT_MAX)
                        Npoints_non_isolated = Npoints_non_isolated_here;
                    else
                        Npoints_non_isolated += Npoints_non_isolated_here;
                }
#endif

                // I don't bother to look in rings that don't appear in the
                // cluster. This will by contain not very much data (because
                // the pre-solve didn't find it), and won't be of much value
                if(debug)
                {
                    MSG("%d-%d at icluster=%d: refinement gathered %d points",
                        iring, isegment,
                        icluster,
                        points_and_plane->n - ipoint_set_n_prev);
                    ipoint_set_n_prev = points_and_plane->n;
                }
            }
        }

        if(DEBUG_ON_TRUE_POINT(points_and_plane->n == 0,
                                &cluster->plane.p_mean,
                               "All points thrown out during refinement"))
            return false;

        // Got a set of points. Fit a plane
        fit_plane_into_points__normalized(&plane_out,
                                          max_norm2_dp,
                                          eigenvalues_ascending,
                                          points, points_and_plane->n, points_and_plane->ipoint);
    }

    points_and_plane->plane = plane_out;

    // I want the normals need to be consistent here. I point them away from the
    // sensor, to match the coordinate system of chessboards
    if(inner(points_and_plane->plane.p_mean,
             points_and_plane->plane.n)
       < 0)
    {
        points_and_plane->plane.n = scale(points_and_plane->plane.n, -1.f);
    }

    const int threshold_non_isolated = 25;

    return !DEBUG_ON_TRUE_POINT(Npoints_non_isolated >= threshold_non_isolated,
                                &cluster->plane.p_mean,
                                "Too many non-isolated points around the plane: %d >= %d",
                                Npoints_non_isolated, threshold_non_isolated);
}

void clc_lidar_segmentation_default_context(clc_lidar_segmentation_context_t* ctx)
{
#define CLC_LIDAR_SEGMENTATION_LIST_CONTEXT_SET_DEFAULT(type,name,default,...) \
    .name = default,

    *ctx = (clc_lidar_segmentation_context_t)
        { CLC_LIDAR_SEGMENTATION_LIST_CONTEXT(CLC_LIDAR_SEGMENTATION_LIST_CONTEXT_SET_DEFAULT) };

#undef CLC_LIDAR_SEGMENTATION_LIST_CONTEXT_SET_DEFAULT
}

// Returns how many planes were found or <0 on error
int8_t clc_lidar_segmentation_sorted(// out
                          clc_points_and_plane_t* points_and_plane,
                          // in
                          const int8_t Nplanes_max, // buffer length of points_and_plane[]
                          const clc_lidar_scan_sorted_t* scan,
                          const clc_lidar_segmentation_context_t* ctx)
{
    if(!(ctx->Nrings > 0 && ctx->Nrings <= 1024))
    {
        MSG("Unexpected value of Nrings=%d. Does your LIDAR really have this many lasers?",
            ctx->Nrings);
        return -1;
    }

    int ipoint0_in_ring[ctx->Nrings];
    ipoint0_in_ring[0] = 0;
    for(int i=1; i<ctx->Nrings; i++)
        ipoint0_in_ring[i] = ipoint0_in_ring[i-1] + scan->Npoints[i-1];

    segment_t segments[ctx->Nrings*Nsegments_per_rotation];
    memset(segments, 0, ctx->Nrings*Nsegments_per_rotation*sizeof(segments[0]));

    for(int iring=0; iring<ctx->Nrings; iring++)
    {
        stage1_segment_from_ring(// out
                                 &segments[Nsegments_per_rotation*iring],
                                 // in
                                 iring,
                                 &scan->points[ipoint0_in_ring[iring]], scan->Npoints[iring],
                                 ctx);

        if(ctx->dump)
        {
            for(unsigned int i=0; i<scan->Npoints[iring]; i++)
                printf("%f %f all %f\n",
                       scan->points[ipoint0_in_ring[iring] + i].x,
                       scan->points[ipoint0_in_ring[iring] + i].y,
                       scan->points[ipoint0_in_ring[iring] + i].z);

            for(int i=0; i<Nsegments_per_rotation; i++)
            {
                // Breaks the "vnl" part of this, since there are too many
                // columns here
                if(!(segments[iring*Nsegments_per_rotation + i].v.x == 0 &&
                     segments[iring*Nsegments_per_rotation + i].v.y == 0 &&
                     segments[iring*Nsegments_per_rotation + i].v.z == 0))
                {
                    const float magv = mag(segments[iring*Nsegments_per_rotation + i].v);
                    printf("%f %f stage1-segment %f %f %f %f\n",
                           segments[iring*Nsegments_per_rotation + i].p.x,
                           segments[iring*Nsegments_per_rotation + i].p.y,
                           segments[iring*Nsegments_per_rotation + i].p.z,
                           segments[iring*Nsegments_per_rotation + i].v.x/magv * .3,
                           segments[iring*Nsegments_per_rotation + i].v.y/magv * .3,
                           segments[iring*Nsegments_per_rotation + i].v.z/magv * .3 );
                }
            }
        }
    }

    // plane_clusters_from_segments() will return only clusters of an acceptable size,
    // so there will not be a huge number of candidates
    const int Nmax_planes = 20;
    segment_cluster_t segment_clusters[Nmax_planes];
    int Nclusters;
    stage2_cluster_segments(segment_clusters,
                            &Nclusters,
                            (int)(sizeof(segment_clusters)/sizeof(segment_clusters[0])),
                            segments,
                            scan->points, ipoint0_in_ring,
                            ctx);

    if(ctx->dump)
        for(int icluster=0; icluster<Nclusters; icluster++)
        {
            const segment_cluster_t* cluster = &segment_clusters[icluster];
            for(int iring = cluster->irings[0];
                iring    <= cluster->irings[1];
                iring++)
            {
                for(int isegment = cluster->isegments[iring-cluster->irings[0]][0];
                    isegment    <= cluster->isegments[iring-cluster->irings[0]][1];
                    isegment++)
                {

                    segment_t* segment = &segments[iring*Nsegments_per_rotation + isegment];

                    printf("%f %f stage2-cluster-kernels-%d %f\n",
                           segment->p.x,
                           segment->p.y,
                           icluster,
                           segment->p.z);


                    for(int ipoint=segment->ipoint0;
                        ipoint <= segment->ipoint1;
                        ipoint++)
                    {
                        printf("%f %f stage2-cluster-points-%d %f\n",
                               scan->points[ipoint0_in_ring[iring] + ipoint].x,
                               scan->points[ipoint0_in_ring[iring] + ipoint].y,
                               icluster,
                               scan->points[ipoint0_in_ring[iring] + ipoint].z);
                    }
                }
            }
        }


    int8_t iplane_out = 0;
    for(int icluster=0; icluster<Nclusters; icluster++)
    {
        if(iplane_out == Nplanes_max)
        {
            MSG("Nplanes_max=%d exceeded. Ignoring not-yet-processed planes",
                Nplanes_max);
            return Nplanes_max;
        }

        const segment_cluster_t* cluster = &segment_clusters[icluster];

        float max_norm2_dp;
        float eigenvalues_ascending[3];
        bool not_rejected =
            stage3_refine_cluster(&points_and_plane[iplane_out],
                                  &max_norm2_dp,
                                  eigenvalues_ascending,
                                  icluster,
                                  cluster,
                                  segments,
                                  scan->points, ipoint0_in_ring, scan->Npoints,
                                  ctx);

        const int Npoints_in_plane = points_and_plane[iplane_out].n;

        const bool debug =
            ctx->debug_xmin < points_and_plane[iplane_out].plane.p_mean.x && points_and_plane[iplane_out].plane.p_mean.x < ctx->debug_xmax &&
            ctx->debug_ymin < points_and_plane[iplane_out].plane.p_mean.y && points_and_plane[iplane_out].plane.p_mean.y < ctx->debug_ymax;

        const bool rejected =

            !not_rejected ||

            // Each eigenvalue is a 1-sigma ellipse of our point cloud. It
            // represents the sum-of-squares of deviations from the mean. The
            // RMS is the useful value: RMS = sqrt(sum_of_squares/N)
            DEBUG_ON_TRUE_POINT(eigenvalues_ascending[0] > ctx->threshold_max_rms_fit_error*ctx->threshold_max_rms_fit_error*(float)Npoints_in_plane,
                                &points_and_plane[iplane_out].plane.p_mean,
                                "icluster=%d: refined plane doesn't fit the constituent points well-enough: %f > %f",
                                icluster,
                                sqrt(eigenvalues_ascending[0]/(float)Npoints_in_plane), ctx->threshold_max_rms_fit_error) ||

            // Some point clouds are degenerate, and we throw them away. The
            // first eigenvalue is 0-ish: we're looking at a plane, and the data
            // must be squished in that way. But the next eigenvalue should be a
            // decent size. Otherwise the data is linear-y instead of plane-y
            DEBUG_ON_TRUE_POINT(eigenvalues_ascending[1] < ctx->threshold_min_rms_point_cloud_2nd_dimension*ctx->threshold_min_rms_point_cloud_2nd_dimension*(float)Npoints_in_plane,
                                &points_and_plane[iplane_out].plane.p_mean,
                                "icluster=%d: refined plane is degenerate (2nd eigenvalue of point cloud dispersion is too small): %f < %f",
                                icluster,
                                sqrt(eigenvalues_ascending[1]/(float)Npoints_in_plane), ctx->threshold_min_rms_point_cloud_2nd_dimension) ||

            DEBUG_ON_TRUE_POINT(max_norm2_dp*2.*2. > ctx->threshold_max_plane_size*ctx->threshold_max_plane_size,
                                &points_and_plane[iplane_out].plane.p_mean,
                                "icluster=%d: refined plane is too big: max_mag_dp*2 > threshold: %f > %f",
                                icluster,
                                sqrtf(max_norm2_dp)*2., ctx->threshold_max_plane_size);

        if(ctx->dump)
        {
            for(unsigned int i=0; i<points_and_plane[iplane_out].n; i++)
                if(!rejected)
                    printf("%f %f ACCEPTED %f\n",
                           scan->points[points_and_plane[iplane_out].ipoint[i]].x,
                           scan->points[points_and_plane[iplane_out].ipoint[i]].y,
                           scan->points[points_and_plane[iplane_out].ipoint[i]].z);
                else
                    printf("%f %f stage3-refined-points-%d-rejected %f\n",
                           scan->points[points_and_plane[iplane_out].ipoint[i]].x,
                           scan->points[points_and_plane[iplane_out].ipoint[i]].y,
                           icluster,
                           scan->points[points_and_plane[iplane_out].ipoint[i]].z);

            printf("%f %f plane-normal %f %f %f %f\n",
                   points_and_plane[iplane_out].plane.p_mean.x,
                   points_and_plane[iplane_out].plane.p_mean.y,
                   points_and_plane[iplane_out].plane.p_mean.z,
                   points_and_plane[iplane_out].plane.n.x * 0.2,
                   points_and_plane[iplane_out].plane.n.y * 0.2,
                   points_and_plane[iplane_out].plane.n.z * 0.2);
        }

        if(rejected)
            continue;

        iplane_out++;
    }

    return iplane_out;
}

int8_t clc_lidar_segmentation_unsorted(// out
                          clc_points_and_plane_t* points_and_plane,
                          // in
                          const int8_t Nplanes_max, // buffer length of points_and_plane[]
                          const clc_lidar_scan_unsorted_t* scan,
                          // The stride, in bytes, between each successive points or rings value
                          // in clc_lidar_scan_unsorted_t
                          const unsigned int lidar_packet_stride,
                          const clc_lidar_segmentation_context_t* ctx)
{
    clc_point3f_t points[scan->Npoints];
    unsigned int Npoints[ctx->Nrings];

    uint32_t ipoint_unsorted_in_sorted_order[scan->Npoints];

    clc_lidar_sort(// out
                   //
                   // These buffers must be pre-allocated
                   // length sum(Npoints). Sorted by ring and then by azimuth
                   points,
                   // indices; length(sum(Npoints)); may be NULL
                   ipoint_unsorted_in_sorted_order,
                   // length Nrings
                   Npoints,

                   // in
                   ctx->Nrings,
                   // The stride, in bytes, between each successive points or
                   // rings value in clc_lidar_scan_unsorted_t
                   lidar_packet_stride,
                   scan);

    int8_t Nplanes =
        clc_lidar_segmentation_sorted(// out
                                      points_and_plane,
                                      // in
                                      Nplanes_max,
                                      &(clc_lidar_scan_sorted_t){.points  = points,
                                                                 .Npoints = Npoints},
                                      ctx);
    if(Nplanes <= 0)
        return Nplanes;


    // Update ipoints to point to the unsorted points
    for(int i=0; i<Nplanes; i++)
        for(unsigned int j=0; j<points_and_plane[i].n; j++)
            points_and_plane[i].ipoint[j] =
                ipoint_unsorted_in_sorted_order[ points_and_plane[i].ipoint[j] ];

    return Nplanes;
}

typedef struct
{
    // applies to rings, not to az. If lidar_packet_stride==0, dense storage is
    // assumed
    unsigned int lidar_packet_stride;
    uint16_t*    rings;

    float*       az;
} compare_ring_az_ctx_t;
static int compare_ring_az(const void* _a, const void* _b, void* cookie)
{
    const compare_ring_az_ctx_t* ctx = (const compare_ring_az_ctx_t*)cookie;

    const uint32_t a = *(const uint32_t*)_a;
    const uint32_t b = *(const uint32_t*)_b;

    if(ctx->lidar_packet_stride > 0)
    {
        const uint16_t* ring_a = (uint16_t*)&((uint8_t*)ctx->rings )[ctx->lidar_packet_stride*a];
        const uint16_t* ring_b = (uint16_t*)&((uint8_t*)ctx->rings )[ctx->lidar_packet_stride*b];
        if(*ring_a < *ring_b) return -1;
        if(*ring_a > *ring_b) return  1;
    }
    else
    {
        // dense storage
        const uint16_t ring_a = ctx->rings[a];
        const uint16_t ring_b = ctx->rings[b];
        if(ring_a < ring_b) return -1;
        if(ring_a > ring_b) return  1;
    }
    if(ctx->az[a] < ctx->az[b]) return -1;
    if(ctx->az[a] > ctx->az[b]) return  1;

    return 0;
}
// Sorts the lidar data by ring and azimuth, to be passable to
// clc_lidar_segmentation_sorted()
void clc_lidar_sort(// out
                    //
                    // These buffers must be pre-allocated
                    // length sum(Npoints). Sorted by ring and then by azimuth
                    clc_point3f_t* points,
                    // indices; length(sum(Npoints))
                    uint32_t* ipoint_unsorted_in_sorted_order,
                    // length Nrings
                    unsigned int* Npoints,

                    // in
                    int Nrings,
                    // The stride, in bytes, between each successive points or
                    // rings value in clc_lidar_scan_t. If
                    // lidar_packet_stride==0, dense storage is assumed
                    const unsigned int      lidar_packet_stride,
                    const clc_lidar_scan_unsorted_t* scan)
{
    float    az    [scan->Npoints];
    for(unsigned int i=0; i<scan->Npoints; i++)
    {
        ipoint_unsorted_in_sorted_order[i] = i;

        if(lidar_packet_stride > 0)
        {
            clc_point3f_t* p = (clc_point3f_t*)&((uint8_t*)scan->points)[lidar_packet_stride*i];
            az[i] = atan2f( p->y, p->x );
        }
        else
        {
            // dense storage
            clc_point3f_t* p = &scan->points[i];
            az[i] = atan2f( p->y, p->x );
        }
    }

    compare_ring_az_ctx_t ctx = {.lidar_packet_stride = lidar_packet_stride,
                                 .rings               = scan->rings,
                                 .az                  = az};
    qsort_r(ipoint_unsorted_in_sorted_order, scan->Npoints, sizeof(ipoint_unsorted_in_sorted_order[0]),
            &compare_ring_az, (void*)&ctx);

    // I now have the sorted indices in ipoint_unsorted_in_sorted_order. Copy everything
    int      i_ring_prev_start = 0;
    uint16_t ring_prev         = UINT16_MAX;
    for(unsigned int i=0; i<scan->Npoints; i++)
    {
        uint16_t ring_here;
        if(lidar_packet_stride > 0)
        {
            points[i] = *((clc_point3f_t*)&((uint8_t*)scan->points)[lidar_packet_stride*ipoint_unsorted_in_sorted_order[i]]);
            ring_here = *((uint16_t*)     &((uint8_t*)scan->rings )[lidar_packet_stride*ipoint_unsorted_in_sorted_order[i]]);
        }
        else
        {
            // dense storage
            points[i] = scan->points[ipoint_unsorted_in_sorted_order[i]];
            ring_here = scan->rings [ipoint_unsorted_in_sorted_order[i]];
        }

        if(ring_prev != ring_here)
        {
            // see new ring. Update Npoints[]
            if(ring_prev < ring_here)
            {
                // normal path
                Npoints[ring_prev] = i - i_ring_prev_start;
            }
            else
            {
                // Initial condition. No points for ring 0
                // Nothing to do. this for() loop will roll around and take care
                // of everything
            }
            // account for any rings we didn't see at all
            for(ring_prev++; ring_prev<ring_here; ring_prev++)
                Npoints[ring_prev] = 0;
            i_ring_prev_start = i;
            ring_prev = ring_here;
        }
    }

    // handle last ring
    uint16_t ring_here = Nrings;
    Npoints[ring_prev] = scan->Npoints - i_ring_prev_start;
    // account for any rings we didn't see at all
    for(ring_prev++; ring_prev<ring_here; ring_prev++)
        Npoints[ring_prev] = 0;
}

/*
The segment finder should use missing points or too-long ranges as breaks. The
current implementation doesn't do this right: it throws out the point after a
too-large gap, but then continues adding subsequent points to the segment

check for conic sections, not just line segments

I already flag too many invalid points total in a segment. I should have a
separate metric to flag too many in a contiguous block

Make a note that the initial segment finder is crude. It does not try to work on
the edges at all: it sees a gap or a switch to another object, and it throws out
the entire segment

handle wraparound at th=180. All th - th math should be modulo 360

For each ring segment, make sure that the local gradient is in-plane. This
should throw out points at the edges. Or even better: if off-plane points exist
at the edges, throw out the whole ring: we're looking at a wall instead of a
plane floating in space

*/
