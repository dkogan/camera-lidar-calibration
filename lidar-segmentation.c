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




/* assumed to fit exactly; validate_ctx() checks that */
#define Nsegments_per_rotation                                          \
    (int)(ctx->Npoints_per_rotation / ctx->Npoints_per_segment)




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
    int ipoint1 /* last point */ : sizeof(int)*8-1; // leave one bit for "visited"
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
    int16_t isegments[clc_Nrings_max][2];

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
float az_from_point(const clc_point3f_t* p)
{
    return atan2f(p->y, p->x);
}


static
void isegment_az_from_point(// out
                            int*   isegment,
                            float* az_rad,
                            // in
                            const clc_point3f_t* p,
                            const clc_lidar_segmentation_context_t* ctx)
{
    // I want the points in a ring to end up ordered by isegment.
    // clc_lidar_preprocess() orders them by az = atan2(). So I want the segment
    // starting at -pi to have isegment=0
    *az_rad = az_from_point(p);

    const float segment_width_rad = 2.0f*M_PI * (float)ctx->Npoints_per_segment / (float)ctx->Npoints_per_rotation;

    const int i = (int)(((*az_rad) + M_PI) / segment_width_rad);
    // Should be in range EXCEPT if az_rad == +pi. Just in case and for good
    // hygiene, I check both cases
    if( i < 0 )
        *isegment = 0;
    else if(i >= Nsegments_per_rotation)
        *isegment = Nsegments_per_rotation-1;
    else
        *isegment = i;
}
static
bool point_is_valid__presolve(const clc_point3f_t* p,
                              const float daz_rad,
                              const bool debug,
                              const clc_lidar_segmentation_context_t* ctx)
{
    const int Ngap = (int)( 0.5f + fabsf(daz_rad) * (float)ctx->Npoints_per_rotation / (2.0f*M_PI) );

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
float stage1_worst_deviationsq_off_segment(const clc_point3f_t* p,
                                           const int ipoint0,
                                           const int ipoint1,
                                           const uint64_t* bitarray_invalid)
{
    const clc_point3f_t* p0 = &p[ipoint0];
    const clc_point3f_t* p1 = &p[ipoint1];

    const clc_point3f_t v01 = sub(*p1,*p0);

    const float recip_norm2_v01 = 1.f / norm2(v01);

    float norm2_e_worst = 0.f;
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
        if(norm2_e_worst < norm2_e) norm2_e_worst = norm2_e;
    }
    return norm2_e_worst;
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
       DEBUG_ON_TRUE_SEGMENT(stage1_worst_deviationsq_off_segment(p,ipoint0,ipoint1,bitarray_invalid) >
                             ctx->threshold_max_deviation_off_segment_line*ctx->threshold_max_deviation_off_segment_line,
                             iring,isegment,
                             "%f > %f; segment is (%f,%f,%f) - (%f,%f,%f)",
                             sqrt(stage1_worst_deviationsq_off_segment(p,ipoint0,ipoint1,bitarray_invalid)),
                             ctx->threshold_max_deviation_off_segment_line,
                             p[ipoint0].x,p[ipoint0].y,p[ipoint0].z,
                             p[ipoint1].x,p[ipoint1].y,p[ipoint1].z))
    {
        *segment = (segment_t){};
        return;
    }

    // We passed the crude test:
    // sqrt(stage1_worst_deviationsq_off_segment())<threshold_max_deviation_off_segment_line
    // says that all the points lie on my line. I now refine this segment's p,v
    // estimate to make this all work better with the downstream logic


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


    int ipoint0   = 0;
    int isegment0;
    float az_rad0;
    isegment_az_from_point(&isegment0, &az_rad0,
                           &points_thisring[0], ctx);
    int Npoints_invalid_in_segment = 0;
    float az_rad_prev = az_rad0;

    memset(bitarray_invalid, 0, Nwords_bitarray_invalid*sizeof(uint64_t));

    for(int ipoint=1; ipoint<Npoints_thisring; ipoint++)
    {
        int isegment;
        float az_rad;
        isegment_az_from_point(&isegment, &az_rad,
                               &points_thisring[ipoint], ctx);
        if(isegment != isegment0)
        {
            stage1_finish_segment(segments_thisring,
                                  iring, isegment0,
                                  Npoints_invalid_in_segment,
                                  bitarray_invalid,
                                  // ipoint-1 will not wrap around because I'm
                                  // in the same ring, and inside each segment
                                  // the seam is right before ipoint=0
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
            if(!point_is_valid__presolve(&points_thisring[ipoint], az_rad - az_rad_prev,
                                         (iring == ctx->debug_iring) &&
                                         (ctx->debug_xmin < points_thisring[ipoint].x && points_thisring[ipoint].x < ctx->debug_xmax &&
                                          ctx->debug_ymin < points_thisring[ipoint].y && points_thisring[ipoint].y < ctx->debug_ymax),
                                         ctx))
            {
                Npoints_invalid_in_segment++;
                bitarray64_set(bitarray_invalid, ipoint-ipoint0);
            }
        }

        az_rad_prev = az_rad;
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
        ctx->threshold_min_cos_angle_error_same_direction_cross_ring*
        ctx->threshold_min_cos_angle_error_same_direction_cross_ring*
        norm2(a)*norm2(b);
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

static bool stage2_plane_not_too_tilted(const clc_point3f_t* p,
                                        const clc_point3f_t* n_unnormalized,
                                        // for diagnostics only
                                        const bool debug,
                                        const clc_lidar_segmentation_context_t* ctx,
                                        const int iring, const int isegment)
{
    // I want angle(p,n) < threshold ->
    // cos > cos_threshold ->
    // inner/magp/magn > cos_threshold ->
    // inner > cos_threshold*magp*magn ->
    // inner*inner > cos_threshold^2*norm2p*norm2n
    const float inner_p_n = inner(*p, *n_unnormalized);
    const float norm2p    = norm2(*p);
    const float norm2n    = norm2(*n_unnormalized);
    return
        !DEBUG_ON_TRUE_SEGMENT(inner_p_n*inner_p_n < ctx->threshold_min_cos_plane_tilt_stage2*ctx->threshold_min_cos_plane_tilt_stage2*norm2n*norm2p,
                               iring,isegment,
                               "cross-ring segments implies a too-tilted plane; have th=%fdeg > threshold=%fdeg",
                               180./M_PI*acos(fabs(inner_p_n/sqrt(norm2p*norm2n))),
                               180./M_PI*acos(ctx->threshold_min_cos_plane_tilt_stage2));
}

static bool stage2_plane_from_segment_segment(// out
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

    if(DEBUG_ON_TRUE_SEGMENT(!segment_segment_across_rings_close_enough(&dp,
                                                                        debug,
                                                                        ctx,
                                                                        iring1,isegment),
                             iring1,isegment,
                             ""))
        return false;


    // The two normal estimates must be close
    clc_point3f_t n0 = cross(dp,s0->v);
    clc_point3f_t n1 = cross(dp,s1->v);

    if(DEBUG_ON_TRUE_SEGMENT(!is_same_direction(n0,n1,ctx),
                              iring1,isegment,
                              "cross-ring segments have unaligned normals") )
        return false;

    plane_unnormalized->n_unnormalized = mean(n0,n1);
    plane_unnormalized->p              = mean(s0->p, s1->p);

    if(DEBUG_ON_TRUE_SEGMENT(!stage2_plane_not_too_tilted(&plane_unnormalized->p, &plane_unnormalized->n_unnormalized,
                                                          debug, ctx, iring1, isegment),
                             iring1,isegment,
                             ""))
        return false;

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
    const clc_point3f_t dp = sub(*point, plane_unnormalized->p);
    const float proj = inner(dp, plane_unnormalized->n_unnormalized);
    return
        ctx->threshold_max_plane_point_error_stage2*ctx->threshold_max_plane_point_error_stage2*
        norm2(plane_unnormalized->n_unnormalized)
        > proj*proj;
}

static int isegment_add(const int isegment, const int dsegment,
                        const clc_lidar_segmentation_context_t* ctx)
{
    int i = isegment + dsegment;
    if(i >= Nsegments_per_rotation)
        i -= Nsegments_per_rotation;
    return i;
}

static int isegment_add_signed(const int isegment, const int dsegment,
                               const clc_lidar_segmentation_context_t* ctx)
{
    int i = isegment + dsegment;
    if     (i >= Nsegments_per_rotation)
        i -= Nsegments_per_rotation;
    else if(i < 0)
        i += Nsegments_per_rotation;
    return i;
}

static int isegment_sub(const int isegment, const int dsegment,
                        const clc_lidar_segmentation_context_t* ctx)
{
    int i = isegment - dsegment;
    if(i < 0)
        i += Nsegments_per_rotation;
    return i;
}

static int
count_segments_in_range(const int isegment0, const int isegment1,
                        const clc_lidar_segmentation_context_t* ctx)
{
    return isegment_sub(isegment1, isegment0, ctx) + 1;
}

static int
count_segments_in_cluster(const segment_cluster_t* cluster,
                          const clc_lidar_segmentation_context_t* ctx)
{
    const int Nrings = cluster->irings[1] - cluster->irings[0] + 1;
    int N = 0;
    for(int i=0; i<Nrings; i++)
        N += count_segments_in_range(cluster->isegments[i][0], cluster->isegments[i][1], ctx);
    return N;
}

static int
isegment_center_from_range(const int isegment0,
                           const int isegment1,
                           const clc_lidar_segmentation_context_t* ctx)
{
    const int Nsegments_in_range = count_segments_in_range(isegment0,isegment1, ctx);
    return isegment_add(isegment0, Nsegments_in_range/2,
                        ctx);
}





static int ipoint_add(const int ipoint, const int dpoint,
                      const int N)
{
    int i = ipoint + dpoint;
    if(i >= N)
        i -= N;
    return i;
}

static int ipoint_add_signed(const int ipoint, const int dpoint,
                             const int N)
{
    int i = ipoint + dpoint;
    if     (i >= N)
        i -= N;
    else if(i < 0)
        i += N;
    return i;
}

static int ipoint_sub(const int ipoint, const int dpoint,
                      const int N)
{
    int i = ipoint - dpoint;
    if(i < 0)
        i += N;
    return i;
}

static int
count_points_in_range(const int ipoint0, const int ipoint1,
                      const int N)
{
    return ipoint_sub(ipoint1, ipoint0, N) + 1;
}

static int
ipoint_center_from_range(const int ipoint0,
                         const int ipoint1,
                         const int N)
{
    const int Npoints_in_range = count_points_in_range(ipoint0,ipoint1, N);
    return ipoint_add(ipoint0, Npoints_in_range/2,
                      N);
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
        const int Nsegments_in_range = count_segments_in_range(cluster->isegments[iring-cluster->irings[0]][0],
                                                               cluster->isegments[iring-cluster->irings[0]][1],
                                                               ctx);
        for(int dsegment = 0; dsegment < Nsegments_in_range; dsegment++)
        {
            const int isegment =
                isegment_add(cluster->isegments[iring-cluster->irings[0]][0],
                             dsegment,
                             ctx);

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
    if(DEBUG_ON_TRUE_SEGMENT(!is_normal(segment->v, cluster->plane_unnormalized.n_unnormalized, ctx),
                             iring,isegment,
                             "icluster=%d: segment isn't plane-consistent during accumulation: the direction isn't in-plane",
                             icluster) ||
       DEBUG_ON_TRUE_SEGMENT( !plane_point_compatible_stage2_unnormalized(&cluster->plane_unnormalized, &segment->p, ctx),
                              iring,isegment,
                              "icluster=%d: segment isn't plane-consistent during accumulation: the point isn't in-plane",
                              icluster))
    {
        // This new segment does not fit the plane found so far in this cluster.
        // I don't reject it just yet. I will try to fit a new plane to see if
        // that works; the current plane estimate might just not be good-enough
    }
    else
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


    // same check as above, but with a new, fitted plane
    if(DEBUG_ON_TRUE_SEGMENT(!is_normal(segment->v, plane_unnormalized.n_unnormalized, ctx),
                             iring,isegment,
                             "icluster=%d: segment isn't plane-consistent during the re-fit check: the direction isn't in-plane",
                             icluster) ||
       DEBUG_ON_TRUE_SEGMENT( !plane_point_compatible_stage2_unnormalized(&plane_unnormalized, &segment->p, ctx),
                              iring,isegment,
                              "icluster=%d: segment isn't plane-consistent during the re-fit check: the point isn't in-plane",
                              icluster))
        return false;

    // I make sure the refitted plane fits all the segments in the cluster
    for(int iring_here = cluster->irings[0];
        iring_here    <= cluster->irings[1];
        iring_here++)
    {
        const int Nsegments_in_range = count_segments_in_range(cluster->isegments[iring_here-cluster->irings[0]][0],
                                                               cluster->isegments[iring_here-cluster->irings[0]][1],
                                                               ctx);
        for(int dsegment = 0; dsegment < Nsegments_in_range; dsegment++)
        {
            const int isegment_here =
                isegment_add(cluster->isegments[iring_here-cluster->irings[0]][0],
                             dsegment,
                             ctx);
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


static void stage2_accumulate_segments_samering( // out, in
                                                int16_t*                                isegment,
                                                // in
                                                const int16_t                           isegment_increment,
                                                segment_cluster_t*                      cluster,
                                                const int                               iring,
                                                segment_t*                              segments, // not const to set visited
                                                const int                               icluster,
                                                const clc_point3f_t*                    points,
                                                const int*                              ipoint0_in_ring,
                                                const clc_lidar_segmentation_context_t* ctx)
{
    const int16_t isegment_start = *isegment;

    do
    {
        const int16_t isegment0 = *isegment;
        const int16_t isegment1 = isegment_add_signed(*isegment, isegment_increment, ctx);

        const segment_t* segment0 = &segments[iring*Nsegments_per_rotation + isegment0];
        segment_t*       segment1 = &segments[iring*Nsegments_per_rotation + isegment1];

#warning "stage2_plane_segment_compatible() will be called many times, since if it's false we won't set visited; need separate tried_visit array I can reset at once"

        if(segment1->visited)
            return;

        if(!segment_is_valid(segment1))
            return;

        const bool debug =
            ctx->debug_xmin < segment1->p.x && segment1->p.x < ctx->debug_xmax &&
            ctx->debug_ymin < segment1->p.y && segment1->p.y < ctx->debug_ymax;


        // Before I make sure that this next segment works with the so-far found
        // plane, I want to make sure that it makes sense with the previous,
        // adjacent segment: the two endpoints should be close together, and the
        // orientations should be similar
        clc_point3f_t dp;
        if(isegment_increment > 0)
        {
            // Moving forward; should compare the LAST point of segment0 with
            // the FIRST point of segment1
            dp =
                sub( points[ipoint0_in_ring[iring] + segment1->ipoint0],
                     points[ipoint0_in_ring[iring] + segment0->ipoint1] );
        }
        else
        {
            // Moving backward; should compare the FIRST point of segment0 with
            // the LAST point of segment1
            dp =
                sub( points[ipoint0_in_ring[iring] + segment1->ipoint1],
                     points[ipoint0_in_ring[iring] + segment0->ipoint0] );
        }
        if(DEBUG_ON_TRUE_SEGMENT(norm2(dp) >
                                 ctx->threshold_distance_adjacent_points_cross_segment*ctx->threshold_distance_adjacent_points_cross_segment,
                                 iring,isegment1,
                                 "icluster=%d: next segment in the same ring is too far to lie in this cluster. Have %f > %f",
                                 icluster,
                                 mag(dp), ctx->threshold_distance_adjacent_points_cross_segment))
            return;
        const float inner01 = inner(segment0->v,segment1->v);
        if(DEBUG_ON_TRUE_SEGMENT(inner01*inner01 <
                                 ctx->threshold_min_cos_angle_error_same_direction_intra_ring*
                                 ctx->threshold_min_cos_angle_error_same_direction_intra_ring*
                                 norm2(segment0->v)*norm2(segment1->v),
                                 iring,isegment1,
                                 "icluster=%d: next segment in the same ring has a too-inconsistent direction vector. Have %fdeg > %fdeg",
                                 icluster,
                                 acos(inner01/(mag(segment0->v)*mag(segment1->v))) * 180./M_PI,
                                 acos(ctx->threshold_min_cos_angle_error_same_direction_intra_ring) * 180./M_PI))
            return;


        if(!stage2_plane_segment_compatible(// The initial plane estimate in
                                            // cluster->plane_unnormalized may be updated by
                                            // this call, if we return true
                                            cluster,
                                            iring, isegment1,
                                            segments,
                                            // for diagnostics only
                                            icluster, points, ipoint0_in_ring, ctx))
            return;

        segment1->visited = true;
        *isegment = isegment1;

    } while(*isegment != isegment_start);
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

    int iring = iring0;
    while(true)
    {
        stage2_accumulate_segments_samering( // out,in
                                             &cluster->isegments[iring-iring0][0],
                                             // in
                                             -1,
                                             cluster,
                                             iring,
                                             segments,
                                             icluster,
                                             points,
                                             ipoint0_in_ring,
                                             ctx);
        stage2_accumulate_segments_samering( // out,in
                                             &cluster->isegments[iring-iring0][1],
                                             // in
                                             1,
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

        // just in case
        if(iring >= clc_Nrings_max)
            return;

        int dsegment_nextring_offcenter0 = -1;
        int dsegment_nextring_offcenter1 = -1;

        const int isegment_center = isegment_center_from_range(isegment0, isegment1, ctx);

        // I start searching in the middle of the previous ring
        int dsegment;

        dsegment = 0;
        while(true)
        {
            const int isegment_next = isegment_sub(isegment_center, dsegment, ctx);

            if(stage2_next_ring_segment_compatible(cluster,
                                                   iring,
                                                   isegment_next,
                                                   // context
                                                   segments, icluster, points, ipoint0_in_ring, ctx))
            {
                dsegment_nextring_offcenter0 = dsegment;
                break;
            }

            if(isegment_next == isegment0)
                break;

            dsegment++;
        }

        dsegment = 1;
        if(isegment_center != isegment1)
            while(true)
            {
                const int isegment_next = isegment_add(isegment_center, dsegment, ctx);

                if(stage2_next_ring_segment_compatible(cluster,
                                                       iring,
                                                       isegment_next,
                                                       // context
                                                       segments, icluster, points, ipoint0_in_ring, ctx))
                {
                    dsegment_nextring_offcenter1 = dsegment;
                    break;
                }

                if(isegment_next == isegment1)
                    break;

                dsegment++;
            }

        // The nearest (to the center) matching segments in the next ring are
        // dsegment_nextring_offcenter0,1. >=0 if defined
        int isegment_nextring;
        if(dsegment_nextring_offcenter0 < 0 &&
           dsegment_nextring_offcenter1 < 0)
        {
            // No matching segments in the next ring. We're done
            return;
        }
        if(dsegment_nextring_offcenter0 < 0)
            // The matching segment appears on only one side. Take it
            isegment_nextring = isegment_add(isegment_center,dsegment_nextring_offcenter1, ctx);
        else if(dsegment_nextring_offcenter1 < 0)
            // The matching segment appears on only one side. Take it
            isegment_nextring = isegment_sub(isegment_center,dsegment_nextring_offcenter0, ctx);
        else
        {
            // The matching segment appears on both sides. Take the one closer
            // to the center
            if(dsegment_nextring_offcenter0 < dsegment_nextring_offcenter1)
                isegment_nextring = isegment_sub(isegment_center,dsegment_nextring_offcenter0, ctx);
            else
                isegment_nextring = isegment_add(isegment_center,dsegment_nextring_offcenter1, ctx);
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
#define DEBUG_ON_TRUE_STAGE2(what, fmt, ...)          \
    ({  if(debug && (what))                                             \
        {                                                               \
            MSG("icluster=%d (seed segment %d-%d): " #what ": " fmt,    \
                icluster, cluster->irings[0], cluster->isegments[0][0], \
                ##__VA_ARGS__);                                         \
        }                                                               \
        what;                                                           \
    })



    *Nclusters = 0;

    for(int iring0 = 0; iring0 < clc_Nrings_max-1; iring0++)
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
            *cluster = (segment_cluster_t){.irings       = {iring0,iring0},
                                           .isegments    =
                                           { [0] = {isegment,isegment} }};

            const bool debug =
                ctx->debug_xmin < segment0->p.x && segment0->p.x < ctx->debug_xmax &&
                ctx->debug_ymin < segment0->p.y && segment0->p.y < ctx->debug_ymax;

            DEBUG_ON_TRUE_STAGE2(true, "maybe starting with candidate 2nd segment %d-%d",
                                 iring1, isegment);
            if(DEBUG_ON_TRUE_STAGE2(!stage2_plane_from_segment_segment(&cluster->plane_unnormalized,
                                                                       segment0,segment1,
                                                                       ctx,
                                                                       // for diagnostics only
                                                                       iring1, isegment),
                                     "abandoning; %d-%d isn't plane-consistent",
                                     iring1, isegment))
                continue;
            DEBUG_ON_TRUE_STAGE2(true, "continuing");

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

            const int Nsegments_in_cluster = count_segments_in_cluster(cluster, ctx);

            if(DEBUG_ON_TRUE_STAGE2(Nsegments_in_cluster == 2,
                                    "abandoning; only contains the seed segments"))
            {
                // This hypothetical ring-ring component is too small. The
                // next-ring segment might still be valid in another component,
                // with a different plane, without segment. So I allow it again.
                segment1->visited = false;
                continue;
            }

            if(DEBUG_ON_TRUE_STAGE2(Nsegments_in_cluster < ctx->threshold_min_Nsegments_in_cluster,
                                    "abandoning; too small: %d < %d",
                                    Nsegments_in_cluster, ctx->threshold_min_Nsegments_in_cluster))
            {
                continue;
            }

            if(DEBUG_ON_TRUE_STAGE2(Nsegments_in_cluster > ctx->threshold_max_Nsegments_in_cluster,
                                     "abandoning; too big: %d > %d",
                                     Nsegments_in_cluster, ctx->threshold_max_Nsegments_in_cluster))
            {
                continue;
            }

            {
                const int iring0 = cluster->irings[0];
                const int iring1 = cluster->irings[1];
                if(DEBUG_ON_TRUE_STAGE2(iring1-iring0+1 < ctx->threshold_min_Nrings_in_cluster,
                                         "abandoning; too-few rings: %d < %d",
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
                const int Nsegments_in_range = count_segments_in_range(cluster->isegments[iring-cluster->irings[0]][0],
                                                                       cluster->isegments[iring-cluster->irings[0]][1],
                                                                       ctx);
                for(int dsegment = 0; dsegment < Nsegments_in_range; dsegment++)
                {
                    const int isegment =
                        isegment_add(cluster->isegments[iring-cluster->irings[0]][0],
                                     dsegment,
                                     ctx);
                    const segment_t* segment = &segments[iring*Nsegments_per_rotation + isegment];

                    if( norm2(segment->p) < ctx->threshold_max_range*ctx->threshold_max_range )
                    {
                        keep = true;
                        break;
                    }
                }
                if(keep) break;
            }
            if(DEBUG_ON_TRUE_STAGE2(!keep,
                                    "abandoning; completely past the threshold_max_range=%f",
                                    ctx->threshold_max_range))
                continue;


            // Cluster looks mostly good. I refine the plane...
            if(DEBUG_ON_TRUE_STAGE2(!fit_plane_into_cluster(// out
                                       &cluster->plane_unnormalized,
                                       // in
                                       cluster,
                                       NULL, -1,
                                       segments,
                                       points,
                                       ipoint0_in_ring,
                                       ctx),
                                    "abandoning"))
                continue;

            const int iring_mid = (cluster->irings[0] + cluster->irings[1])/2;
            if(DEBUG_ON_TRUE_STAGE2(
                 !stage2_plane_not_too_tilted(&cluster->plane_unnormalized.p, &cluster->plane_unnormalized.n_unnormalized,
                                              debug, ctx,
                                              iring_mid,
                                              isegment_center_from_range(cluster->isegments[iring_mid][0], cluster->isegments[iring_mid][0], ctx)),
                 "abandoning"))
                continue;


            // WE'RE ACCEPTING THIS CLUSTER
            DEBUG_ON_TRUE_STAGE2(true, "accepting!");

            // To prepare for the next stage I normalize the normal vector
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
                                     uint64_t* bitarray_visited, // indexed by Nrings*Npoints_per_rotation points
                                     // in
                                     const int ipoint0, const int ipoint_increment, // in-ring
                                     const clc_plane_t* plane,
                                     const clc_point3f_t* points,
                                     const int ipoint0_in_ring, // start of this ring in the full points[] array
                                     const int Npoints_this_ring,
                                     const int iring,
                                     // for diagnostics
                                     const int icluster, const int isegment,
                                     const bool debug __attribute__((unused)),
                                     const clc_lidar_segmentation_context_t* ctx)
{
    float az_rad_last = FLT_MAX; // indicate an invalid value initially

    clc_point3f_t dp_last = {}; // init to pacify compiler
    float norm2_dp_last = FLT_MAX; // indicate an invalid value initially

    bool first = true;
    int Npoint_steps = 0;
    for(int ipoint=ipoint0;
        first || (ipoint != ipoint0);
        ipoint = ipoint_add_signed(ipoint, ipoint_increment, Npoints_this_ring), Npoint_steps++)
    {
        first = false;

        if(ipoint >= ctx->Npoints_per_rotation)
        {
            MSG("WARNING: ipoint indexing past Npoints_per_rotation=%d; perhaps Npoints_per_rotation is too small? Ignoring the remaining points in this call",
                ctx->Npoints_per_rotation);
            break;
        }

        const int ibit = ctx->Npoints_per_rotation * iring + ipoint;
        if( DEBUG_ON_TRUE_POINT( bitarray64_check(bitarray_visited, ibit),
                                 &points[ipoint0_in_ring + ipoint],
                                 "%d-%d: we already processed this point; accumulation stopped",
                                 iring,isegment))
        {
            // We already processed this point, presumably from the other side.
            // There's no reason to keep going, since we already approached from the
            // other side
            break;
        }

        const float az_rad = az_from_point(&points[ipoint0_in_ring + ipoint]);

        int Ngap = -1;
        if(az_rad_last < FLT_MAX)
        {
            // we have a valid az_rad_last
            float daz_rad = (float)ipoint_increment*(az_rad - az_rad_last);
            if(daz_rad < 0) daz_rad += 2.0f*M_PI;

            Ngap =
                (int)( 0.5f + daz_rad * (float)ctx->Npoints_per_rotation / (2.0f*M_PI) );

            if( DEBUG_ON_TRUE_POINT( Ngap >= ctx->threshold_max_gap_Npoints,
                                     &points[ipoint0_in_ring + ipoint],
                                     "%d-%d: gap too large; accumulation stopped. Have ~ %d >= %d",
                                     iring,isegment,
                                     Ngap, ctx->threshold_max_gap_Npoints))
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
            // Not accepting this point, but also not updating az_rad_last. So too
            // many successive invalid points will create a too-large gap, failing
            // the threshold_max_gap_Npoints check above
            continue;
        }

        // The local direction of the points in the ring being accumulated here
        // should be constant, or at worst, changing slowly. It can only change
        // quickly if we hit a corner. I compute the local direction by looking
        // at the difference between two points N points apart; where larger N
        // serve to reduce sensitivity to noise.

        const int Npoints_check_direction_step = 6;

        const int ipoint_lagging =
            ipoint_add_signed(ipoint, -Npoints_check_direction_step*ipoint_increment, Npoints_this_ring);
        const int ibit_lagging = ctx->Npoints_per_rotation * iring + ipoint_lagging;
        if(Npoint_steps >= Npoints_check_direction_step &&
           bitarray64_check(bitarray_visited, ibit_lagging))
        {
            // some bitarray64_check() succeeded here, so bitarray64_set() was
            // called at some point, so az_rad_last is valid, so Ngap is valid
            // too. In case my thought process is wrong, I confirm here
            if(Ngap < 0)
            {
                MSG("This is a bug. Ngap must be >= 0 here");
                break;
            }

            // This lagging point is in range and was accepted, so I check the
            // direction consistency

            clc_point3f_t dp =
                sub( points[ipoint0_in_ring + ipoint],
                     points[ipoint0_in_ring + ipoint_lagging] );

            // I compare directions. I want threshold < cos(th) =
            // inner(dp,dp_last) / sqrt( norm2(dp) * norm2(dp_last))
            // -> I want
            //    threshold^2* norm2(dp) * norm2(dp_last) < inner(dp,dp_last)^2
            float norm2_dp    = norm2(dp);
            if(norm2_dp_last < FLT_MAX)
            {
                // have dp_last, norm2_dp_last
                const float cos_threshold_baseline = cosf(10.0f*M_PI/180.f);

                // the cos_threshold_baseline is intended for a single gap. For
                // bigger gaps I want to allow bigger errors
                // cos(x) ~ 1 - x^2/2. -> cos(x*N) ~ 1 - (x*N)^2/2.
                // -> cos(x*N) ~ 1 - (1-cos(x))*N^2
                //             = 1-N^2 + N^2 cos(x)
                //             = 1 + N^2 (cos(x) - 1.)
                const float cos_threshold = 1.f + (float)(Ngap*Ngap)*(cos_threshold_baseline - 1.f);

                const float inner_dp_dp = inner(dp,dp_last);


                if(DEBUG_ON_TRUE_POINT(cos_threshold*cos_threshold*norm2_dp*norm2_dp_last > inner_dp_dp*inner_dp_dp,
                                       &(points[ipoint0_in_ring + ipoint]),
                                       "angle changed too quickly in %d-%d (icluster=%d); culling last %d points; cos_threshold_baseline=%f; I see threshold > inner(dp,dp_last) / (mag(dp) mag(dp_last)) ~ %f > %f/(%f * %f)",
                                       iring, isegment, icluster,
                                       Npoints_check_direction_step+1,
                                       cos_threshold_baseline,
                                       cos_threshold, inner_dp_dp, sqrtf(norm2_dp), sqrtf(norm2_dp_last)))
                {
                    continue;
                }
            }

            dp_last       = dp;
            norm2_dp_last = norm2_dp;
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

        bitarray64_set(bitarray_visited, ibit);
        az_rad_last = az_rad;
    }
}


// Not doing this yet. It's complex, and doesn't obviously work well to make
// things better
// I'm turning this off for now. It doesn't work right. I'm at ddc519d. This
// fails:
//
//   ./lidar-segmentation.py --dump --debug -1 ${^x0y0x1y1} \
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
                if(ctx->dump)
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
    float th0_rad = az_from_point(&points[ipoint0_in_ring + ipoint0]);

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

        const float az_rad = az_from_point(&points[ipoint0_in_ring + ipoint]);

        if(in_bloom_region)
        {
            const float dth = az_rad - th0_rad;
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
                th0_deadzone_start_rad = az_rad;
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

        const float dth = az_rad - th0_deadzone_start_rad;
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
                                  // out,in
                                  uint64_t* bitarray_visited,
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

    // Start with the best-available plane estimate. This should be pretty good
    // already.
    clc_plane_t plane_out = cluster->plane;


    const bool debug =
        ctx->debug_xmin < cluster->plane.p_mean.x && cluster->plane.p_mean.x < ctx->debug_xmax &&
        ctx->debug_ymin < cluster->plane.p_mean.y && cluster->plane.p_mean.y < ctx->debug_ymax;

    int Npoints_non_isolated = 0;

    points_and_plane->n = 0;
    int ipoint_set_n_prev = 0;

    for(int iring = cluster->irings[0];
        iring    <= cluster->irings[1];
        iring++)
    {
        const int isegment0 = cluster->isegments[iring-cluster->irings[0]] [0];
        const int isegment1 = cluster->isegments[iring-cluster->irings[0]] [1];

        const segment_t* segment0 = &segments[iring*Nsegments_per_rotation + isegment0];
        const segment_t* segment1 = &segments[iring*Nsegments_per_rotation + isegment1];

        const int isegment_mid =
            isegment_center_from_range(isegment0,isegment1,
                                       ctx);
#warning "at this point I'm already wrong; I see segment0->ipoint=0. Why is it exactly 0???"
        // I start in the center, and expand outwards to capture all the
        // matching points
        const int ipoint0 = ipoint_center_from_range(segment0->ipoint0,
                                                     segment1->ipoint1,
                                                     Npoints[iring]);

        unsigned int ipoint_set_start_this_ring __attribute__((unused)); // for the currently-disabled bloom_cull logic

        ipoint_set_start_this_ring = points_and_plane->n;
        stage3_accumulate_points(// out
                                 &points_and_plane->n,
                                 points_and_plane->ipoint,
                                 (int)(sizeof(points_and_plane->ipoint)/sizeof(points_and_plane->ipoint[0])),
                                 bitarray_visited,
                                 // in
                                 ipoint0, +1,
                                 &plane_out,
                                 points,
                                 ipoint0_in_ring[iring],
                                 Npoints[iring],
                                 iring,
                                 // for diagnostics
                                 icluster, isegment_mid,
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
                                 bitarray_visited,
                                 // in
                                 ipoint_sub(ipoint0,1,Npoints[iring]), -1,
                                 &plane_out,
                                 points,
                                 ipoint0_in_ring[iring],
                                 Npoints[iring],
                                 iring,
                                 // for diagnostics
                                 icluster, isegment_mid,
                                 debug,
                                 ctx);
        // disabling this for now; see comment at stage3_cull_bloom_and_count_non_isolated() above
#if 0
        if(points_and_plane->n > ipoint_set_start_this_ring)
        {
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
                                                         debug,
                                                         ctx);
            if(Npoints_non_isolated_here == INT_MAX)
                Npoints_non_isolated = Npoints_non_isolated_here;
            else
                Npoints_non_isolated += Npoints_non_isolated_here;
        }
#endif

        const int Npoints_thisring = points_and_plane->n - ipoint_set_n_prev;

        // I don't bother to look in rings that don't appear in the
        // cluster. This will by contain not very much data (because
        // the pre-solve didn't find it), and won't be of much value
        if(debug)
        {
            MSG("%d-%d at icluster=%d: refinement gathered %d points",
                iring, isegment_mid,
                icluster,
                Npoints_thisring);
            ipoint_set_n_prev = points_and_plane->n;
        }

        const int threshold_min_points = ctx->threshold_min_points_per_ring__multiple_Npoints_per_segment * ctx->Npoints_per_segment;
        if(DEBUG_ON_TRUE_SEGMENT(Npoints_thisring < threshold_min_points,
                                 iring,isegment_mid,
                                 "Ring contains too few points N=%d <= threshold=%d; giving up on the whole cluster",
                                 Npoints_thisring, threshold_min_points))
            return false;

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


static uint32_t ceil_power_of2(uint32_t v)
{
    // from https://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}

static int infer_Npoints_per_rotation( const float* az_unsorted,
                                       const uint32_t* ipoint_unsorted_in_sorted_order,
                                       const unsigned int* Npoints_per_ring,
                                       const int Nrings)
{
    const int Npoints_per_rotation_min = 1;
    const int Npoints_per_rotation_max = 65536;
    const int Nbins = Npoints_per_rotation_max - Npoints_per_rotation_min + 1;
    int counts[Nbins];
    for(int i=0; i<Nbins; i++) counts[i] = 0;

    int ipoint0_ring = 0;
    for(int iring=0; iring<Nrings; iring++)
    {
        const int Npoints_here = Npoints_per_ring[iring];
        for(int ipoint_here=1; ipoint_here<Npoints_here; ipoint_here++)
        {
            const int ipoint0 = ipoint_unsorted_in_sorted_order[ipoint0_ring + ipoint_here-1];
            const int ipoint1 = ipoint_unsorted_in_sorted_order[ipoint0_ring + ipoint_here  ];
            float daz = az_unsorted[ipoint1] - az_unsorted[ipoint0];
            const int Npoints_per_rotation_here = (int)(2.0f*M_PI / fabsf(daz) + 0.5f);
            const int ibin = Npoints_per_rotation_here - Npoints_per_rotation_min;
            if(!(ibin >= 0 && ibin < Nbins))
                continue;

            counts[ibin]++;
        }
        ipoint0_ring += Npoints_here;
    }

    int count_max = -1, ibin_max = -1;
    for(int i=0; i<Nbins; i++)
        if(counts[i] > count_max)
        {
            count_max = counts[i];
            ibin_max  = i;
        }

    counts[ibin_max] = 0;
    int count_2nd = -1;
    for(int i=0; i<Nbins; i++)
        if(counts[i] > count_2nd)
            count_2nd = counts[i];

    const float ratio = (float)count_max / (float)count_2nd;
    const int Npoints_per_rotation_mode = ibin_max + Npoints_per_rotation_min;
    if( ratio < 5.f)
        MSG("WARNING: the delta-az histogram doesn't have a single 5x spike. count(first)/count(second) = %.1f; Npoints_per_rotation_mode=%d might be wrong",
            ratio, Npoints_per_rotation_mode);

    // nearest power-of-2
    const int Npoints_per_rotation_ceil_power_of2  = ceil_power_of2(Npoints_per_rotation_mode);
    const int Npoints_per_rotation_floor_power_of2 = Npoints_per_rotation_ceil_power_of2 >> 1;
    const int Npoints_per_rotation =
        ( Npoints_per_rotation_ceil_power_of2 - Npoints_per_rotation_mode <
          Npoints_per_rotation_mode - Npoints_per_rotation_floor_power_of2) ?
        Npoints_per_rotation_ceil_power_of2 : Npoints_per_rotation_floor_power_of2;

    MSG("Inferred Npoints_per_rotation mode: %d, using nearest-power-of-2: %d",
        Npoints_per_rotation_mode, Npoints_per_rotation);
    return Npoints_per_rotation;
}

static bool validate_ctx(const clc_lidar_segmentation_context_t* ctx)
{
    if(ctx->Npoints_per_rotation <= 0)
    {
        MSG("Invalid Npoints_per_rotation. This is hardware-dependent, and must be set by the caller. If using clc_..._unsorted(), clc will try to infer this automatically");
        return false;
    }

    if( Nsegments_per_rotation * ctx->Npoints_per_segment != ctx->Npoints_per_rotation)
    {
        MSG("Npoints_per_segment must fit into Npoints_per_rotation exactly");
        return false;
    }


    return true;
}

// Returns how many planes were found or <0 on error
int8_t clc_lidar_segmentation_sorted(// out
                          clc_points_and_plane_t* points_and_plane,
                          // in
                          const int8_t Nplanes_max, // buffer length of points_and_plane[]
                          const clc_lidar_scan_sorted_t* scan,
                          const clc_lidar_segmentation_context_t* ctx)
{
    if(!validate_ctx(ctx))
        return -1;

    int ipoint0_in_ring[clc_Nrings_max];
    ipoint0_in_ring[0] = 0;
    for(int iring=1; iring<clc_Nrings_max; iring++)
        ipoint0_in_ring[iring] = ipoint0_in_ring[iring-1] + scan->Npoints_per_ring[iring-1];

    segment_t segments[clc_Nrings_max*Nsegments_per_rotation];
    memset(segments, 0, clc_Nrings_max*Nsegments_per_rotation*sizeof(segments[0]));

    for(int iring=0; iring<clc_Nrings_max; iring++)
    {
        stage1_segment_from_ring(// out
                                 &segments[Nsegments_per_rotation*iring],
                                 // in
                                 iring,
                                 &scan->points[ipoint0_in_ring[iring]], scan->Npoints_per_ring[iring],
                                 ctx);

        if(ctx->dump)
        {
            for(unsigned int i=0; i<scan->Npoints_per_ring[iring]; i++)
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
    const int Nmax_planes = 50;
    segment_cluster_t clusters[Nmax_planes];
    int Nclusters;
    stage2_cluster_segments(clusters,
                            &Nclusters,
                            (int)(sizeof(clusters)/sizeof(clusters[0])),
                            segments,
                            scan->points, ipoint0_in_ring,
                            ctx);

    if(ctx->dump)
        for(int icluster=0; icluster<Nclusters; icluster++)
        {
            const segment_cluster_t* cluster = &clusters[icluster];
            for(int iring = cluster->irings[0];
                iring    <= cluster->irings[1];
                iring++)
            {
                const int Nsegments_in_range = count_segments_in_range(cluster->isegments[iring-cluster->irings[0]][0],
                                                                       cluster->isegments[iring-cluster->irings[0]][1],
                                                                       ctx);
                for(int dsegment = 0; dsegment < Nsegments_in_range; dsegment++)
                {
                    const int isegment =
                        isegment_add(cluster->isegments[iring-cluster->irings[0]][0],
                                     dsegment,
                                     ctx);
                    segment_t* segment = &segments[iring*Nsegments_per_rotation + isegment];

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


    // I keep track of the already-visited points
    const int Nwords_bitarray_visited = bitarray64_nwords(clc_Nrings_max * ctx->Npoints_per_rotation); // largest-possible size
    uint64_t bitarray_visited[Nwords_bitarray_visited];
    memset(bitarray_visited, 0, Nwords_bitarray_visited*sizeof(uint64_t));

    int8_t iplane_out = 0;
    for(int icluster=0; icluster<Nclusters; icluster++)
    {
        if(iplane_out == Nplanes_max)
        {
            MSG("Nplanes_max=%d exceeded. Ignoring not-yet-processed planes",
                Nplanes_max);
            return Nplanes_max;
        }

        const segment_cluster_t* cluster = &clusters[icluster];

        float max_norm2_dp;
        float eigenvalues_ascending[3];
        bool not_rejected =
            stage3_refine_cluster(// out
                                  &points_and_plane[iplane_out],
                                  &max_norm2_dp,
                                  eigenvalues_ascending,
                                  // out,in
                                  bitarray_visited,
                                  // in
                                  icluster,
                                  cluster,
                                  segments,
                                  scan->points, ipoint0_in_ring, scan->Npoints_per_ring,
                                  ctx);

        const int Npoints_in_plane = points_and_plane[iplane_out].n;

        const bool debug =
            ctx->debug_xmin < points_and_plane[iplane_out].plane.p_mean.x && points_and_plane[iplane_out].plane.p_mean.x < ctx->debug_xmax &&
            ctx->debug_ymin < points_and_plane[iplane_out].plane.p_mean.y && points_and_plane[iplane_out].plane.p_mean.y < ctx->debug_ymax;

        const float threshold_min_rms_point_cloud_2nd_dimension =
            ctx->threshold_min_rms_point_cloud_2nd_dimension__multiple_max_plane_size * ctx->threshold_max_plane_size;


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
            DEBUG_ON_TRUE_POINT(eigenvalues_ascending[1] < threshold_min_rms_point_cloud_2nd_dimension*threshold_min_rms_point_cloud_2nd_dimension*(float)Npoints_in_plane,
                                &points_and_plane[iplane_out].plane.p_mean,
                                "icluster=%d: refined plane is degenerate (2nd eigenvalue of point cloud dispersion is too small): %f < %f",
                                icluster,
                                sqrt(eigenvalues_ascending[1]/(float)Npoints_in_plane), threshold_min_rms_point_cloud_2nd_dimension) ||

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
                          // not const to be able to compute ctx->Npoints_per_rotation
                          clc_lidar_segmentation_context_t* ctx)
{
    clc_point3f_t points[scan->Npoints];
    unsigned int Npoints_per_ring[clc_Nrings_max];

    uint32_t ipoint_unsorted_in_sorted_order[scan->Npoints];
    int* Npoints_per_rotation_estimate;
    if(ctx->Npoints_per_rotation > 0)
        // already have Npoints_per_rotation; don't need to estimate it
        Npoints_per_rotation_estimate = NULL;
    else
        Npoints_per_rotation_estimate = &ctx->Npoints_per_rotation;
    // Sort and cull invalid points
    clc_lidar_preprocess(// out
                         //
                         // These buffers must be pre-allocated
                         // length scan->Npoints = sum(Npoints_per_ring). Sorted by ring and then by azimuth
                         points,
                         // indices; length(sum(Npoints_per_ring)); may be NULL
                         ipoint_unsorted_in_sorted_order,
                         // length Nrings
                         Npoints_per_ring,
                         // length scan->Npoints = sum(Npoints_per_ring)
                         Npoints_per_rotation_estimate,

                         // in
                         clc_Nrings_max,
                         // The stride, in bytes, between each successive points or
                         // rings value in clc_lidar_scan_unsorted_t
                         lidar_packet_stride,
                         scan);

    if(!validate_ctx(ctx))
        return -1;

    int8_t Nplanes =
        clc_lidar_segmentation_sorted(// out
                                      points_and_plane,
                                      // in
                                      Nplanes_max,
                                      &(clc_lidar_scan_sorted_t){.points  = points,
                                                                 .Npoints_per_ring = Npoints_per_ring},
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
    unsigned int stride_rings;
    uint16_t*    rings;

    float*       az;
} compare_ring_az_ctx_t;
static int compare_ring_az(const void* _a, const void* _b, void* cookie)
{
    const compare_ring_az_ctx_t* ctx = (const compare_ring_az_ctx_t*)cookie;

    const uint32_t a = *(const uint32_t*)_a;
    const uint32_t b = *(const uint32_t*)_b;

    // sort the invalid points to the end
    if(ctx->az[a] == FLT_MAX) return 1;
    if(ctx->az[b] == FLT_MAX) return -1;

    const uint16_t ring_a = *(uint16_t*)&((uint8_t*)ctx->rings )[ctx->stride_rings*a];
    const uint16_t ring_b = *(uint16_t*)&((uint8_t*)ctx->rings )[ctx->stride_rings*b];
    if(ring_a < ring_b) return -1;
    if(ring_a > ring_b) return  1;

    if(ctx->az[a] < ctx->az[b]) return -1;
    if(ctx->az[a] > ctx->az[b]) return  1;

    return 0;
}
// Sorts the lidar data by ring and azimuth, and removes invalid points. To be
// passable to clc_lidar_segmentation_sorted()
void clc_lidar_preprocess(// out
                          //
                          // These buffers must be pre-allocated
                          // length scan->Npoints = sum(Npoints). Sorted by ring and then by azimuth
                          clc_point3f_t* points,
                          // indices; length(sum(Npoints))
                          uint32_t* ipoint_unsorted_in_sorted_order,
                          // length Nrings
                          unsigned int* Npoints_per_ring,
                          // an estimate for Npoints_per_rotation; NULL if we
                          // don't need it
                          int* Npoints_per_rotation,

                          // in
                          const int Nrings,
                          // The stride, in bytes, between each successive points or
                          // rings value in clc_lidar_scan_t. If
                          // lidar_packet_stride==0, dense storage is assumed
                          const unsigned int      lidar_packet_stride,
                          const clc_lidar_scan_unsorted_t* scan)
{
    unsigned int stride_points, stride_rings;
    if(lidar_packet_stride <= 0)
    {
        // dense storage
        stride_points = sizeof(scan->points[0]);
        stride_rings  = sizeof(scan->rings [0]);
    }
    else
    {
        stride_points = lidar_packet_stride;
        stride_rings  = lidar_packet_stride;
    }

    float az_unsorted[scan->Npoints];
    for(unsigned int i=0; i<scan->Npoints; i++)
    {
        ipoint_unsorted_in_sorted_order[i] = i;

        clc_point3f_t* p = (clc_point3f_t*)&((uint8_t*)scan->points)[stride_points*i];

        if(p->x == 0.0f && p->y == 0.0f)
            // Invalid point. Indicate that
            az_unsorted[i] = FLT_MAX;
        else
            az_unsorted[i] = atan2f( p->y, p->x );
    }

    compare_ring_az_ctx_t ctx = {.stride_rings = stride_rings,
                                 .rings        = scan->rings,
                                 .az           = az_unsorted};
    qsort_r(ipoint_unsorted_in_sorted_order, scan->Npoints, sizeof(ipoint_unsorted_in_sorted_order[0]),
            &compare_ring_az, (void*)&ctx);

    // I now have the sorted indices in ipoint_unsorted_in_sorted_order. The
    // invalid points are all sorted to the end. Copy everything up to the first
    // invalid point.
    int      i_ring_prev_start = 0;
    uint16_t ring_prev         = UINT16_MAX;
    unsigned int i;
    for(i=0;
        i<scan->Npoints && az_unsorted[ipoint_unsorted_in_sorted_order[i]] != FLT_MAX;
        i++)
    {
        points[i]          = *((clc_point3f_t*)&((uint8_t*)scan->points)[stride_points*ipoint_unsorted_in_sorted_order[i]]);
        uint16_t ring_here = *((uint16_t*)     &((uint8_t*)scan->rings )[stride_rings *ipoint_unsorted_in_sorted_order[i]]);

        if(ring_prev != ring_here)
        {
            // see new ring. Update Npoints[]
            if(ring_prev < ring_here)
            {
                // normal path
                Npoints_per_ring[ring_prev] = i - i_ring_prev_start;
            }
            else
            {
                // Initial condition. No points for ring 0
                // Nothing to do. this for() loop will roll around and take care
                // of everything
            }
            // account for any rings we didn't see at all
            for(ring_prev++; ring_prev<ring_here; ring_prev++)
                Npoints_per_ring[ring_prev] = 0;
            i_ring_prev_start = i;
            ring_prev = ring_here;
        }
    }
    unsigned int Npoints_valid = i;

    // handle last ring
    uint16_t ring_here = Nrings;
    Npoints_per_ring[ring_prev] = Npoints_valid - i_ring_prev_start;
    // account for any rings we didn't see at all
    for(ring_prev++; ring_prev<ring_here; ring_prev++)
        Npoints_per_ring[ring_prev] = 0;

    if(Npoints_per_rotation != NULL)
    {
        *Npoints_per_rotation =
            infer_Npoints_per_rotation( az_unsorted,
                                        ipoint_unsorted_in_sorted_order,
                                        Npoints_per_ring,
                                        Nrings);
    }
}

static int compar_uint32(const void* _a, const void* _b)
{
    const uint32_t a = *(uint32_t*)_a;
    const uint32_t b = *(uint32_t*)_b;
    if(a < b) return -1;
    if(a > b) return  1;
    return 0;
}

static float mode_over_lastdim_ignoring0__oneslice(// in
                                                   // shape (Ndatasets,Nsamples)
                                                   const float* x,
                                                   const int Nsamples,
                                                   const float quantum,
                                                   const int report_mode_if_N_atleast)

{
    uint32_t xint[Nsamples];
    int i1=0;
    for(int i0=0; i0<Nsamples; i0++)
    {
        if(x[i0] == 0)
            continue;
        xint[i1] = (uint32_t)(x[i0] / quantum);
        i1++;
    }
    if(i1 == 0)
        return 0.;

    const int Nsamples1 = i1;

    if(Nsamples1 < report_mode_if_N_atleast)
        // Too few nonzero returns. No valid mode to return
        return 0.;

    qsort(xint, Nsamples1, sizeof(xint[0]), &compar_uint32);

    uint32_t dxint[Nsamples1-1];
    for(int i=1; i<Nsamples1; i++)
        dxint[i-1] = xint[i]-xint[i-1];

    // I want to find the biggest adjacent histogram bins. So I look for the
    // longest sequence of [0,0,0,...,0,1,0,0,0...,0] in dxint[]
    const int Nmax_block0 = 256;
    uint32_t block0_start[Nmax_block0];
    uint32_t block0_size [Nmax_block0];
    const int Nwords_bitarray_block0_follows_block0_single1 = bitarray64_nwords(Nmax_block0);
    uint64_t bitarray_block0_follows_block0_single1[Nwords_bitarray_block0_follows_block0_single1];
    bitarray64_clear_all(bitarray_block0_follows_block0_single1, Nwords_bitarray_block0_follows_block0_single1);

    uint32_t Nblocks0 = 0;
    bool in_block0 = false;
    for(int i=0;
        i<Nsamples1; // including 1-past-the-end
        i++)
    {
        if(!in_block0)
        {
            // not looking at a block of 0s
            if(i >= Nsamples1-1)
                break; // we're at the end

            if(dxint[i] != 0)
                continue;

            // starting a new block of 0s
            if(Nblocks0 >= Nmax_block0)
            {
                MSG("WARNING: Exceeded expected Nmax_block0. Ignoring all ranges past this one. Maybe bump up Nmax_block0");
                break;
            }
            block0_start[Nblocks0] = i;
            in_block0 = true;

            if(Nblocks0 > 0 &&
               block0_start[Nblocks0-1]+block0_size[Nblocks0-1] + 1 == i &&
               dxint[i-1] == 1)
                bitarray64_set(bitarray_block0_follows_block0_single1, Nblocks0);
        }
        else
        {
            // currently looking at a block of 0s
            if(i < Nsamples1-1 && dxint[i] == 0)
                continue;

            // finishing block of 0s
            block0_size[Nblocks0] = i - block0_start[Nblocks0];
            in_block0 = false;
            // accept only blocks of 0 longer than a certain size. I COULD use
            // report_mode_if_N_atleast as the threshold, but I want to find
            // 000010000 sequences because the true mode might be on the
            // boundary of my bins. So I use a smaller threshold here. Throwing
            // away the REALLY tiny blocks reduces accuracy a bit, but it's
            // close enough
            if(block0_size[Nblocks0] > 2)
                Nblocks0++;
        }
    }

    int j0max = -1; // the start of the biggest range
    int Nmax = 0;

    // I now look through all the candidates, and find the longest one. The
    // candidates are either blocksof0 or two blocksof0 separated by exactly
    // one 1
    for(int j=0; j<Nblocks0; j++)
    {
        int N  = block0_size[j];
        if(j+1 < Nblocks0 &&
           bitarray64_check(bitarray_block0_follows_block0_single1, j+1))
        {
            // I have 0001000
            N += 1 + block0_size[j+1];
        }

        if(N > Nmax)
        {
            Nmax = N;
            j0max = block0_start[j];
        }
    }

    // The longest sequence of dxint is Nmax. This measures intervals, so
    // the sequence of xint is one longer
    Nmax++;

    if(Nmax < report_mode_if_N_atleast)
        return 0;

    if(j0max < 0)
        // shouldn't happen, but just in case
        return 0;

    return (0.5f + (float)(xint[j0max + Nmax/2])) * quantum;
}

void _clc_mode_over_lastdim_ignoring0(// out
                                      // shape (Ndatasets)
                                      float* mode,
                                      // in
                                      // shape (Ndatasets,Nsamples)
                                      const float* x,
                                      const int Nsamples,
                                      const int Ndatasets,
                                      const float quantum,
                                      const int report_mode_if_N_atleast)
{
    for(int i=0; i<Ndatasets; i++)
         mode[i] = mode_over_lastdim_ignoring0__oneslice(&x[i*Nsamples],
                                                        Nsamples,
                                                        quantum,
                                                        report_mode_if_N_atleast);
}
