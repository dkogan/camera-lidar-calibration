#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <limits.h>
#include <float.h>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>

#include "point_segmentation.h"


#define MSG(fmt, ...) fprintf(stderr, "%s(%d): " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__)


#define DEBUG_ON_TRUE(what, p, fmt, ...)                                \
    ({  if(debug && (what))                                             \
        {                                                               \
            MSG("REJECTED (%.2f %.2f %.2f) at %s():%d because " #what ": " fmt, \
                (p)->x,(p)->y,(p)->z,                                   \
                __func__, __LINE__, ##__VA_ARGS__);                     \
        }                                                               \
        what;                                                           \
    })




/* round up */
#define Nsegments_per_rotation                                          \
    (int)((ctx->Npoints_per_rotation + ctx->Npoints_per_segment-1) / ctx->Npoints_per_segment)




__attribute__((unused))
static point3f_t point3f_from_double(const double* p)
{
    return (point3f_t){ .x = (float)p[0],
                        .y = (float)p[1],
                        .z = (float)p[2]};
}
__attribute__((unused))
static float inner(const point3f_t a, const point3f_t b)
{
    return
        a.x*b.x +
        a.y*b.y +
        a.z*b.z;
}
__attribute__((unused))
static float norm2(const point3f_t a)
{
    return inner(a,a);
}
__attribute__((unused))
static float mag(const point3f_t a)
{
    return sqrtf(norm2(a));
}
__attribute__((unused))
static point3f_t add(const point3f_t a, const point3f_t b)
{
    return (point3f_t){ .x = a.x + b.x,
                        .y = a.y + b.y,
                        .z = a.z + b.z };
}
__attribute__((unused))
static point3f_t mean(const point3f_t a, const point3f_t b)
{
    return (point3f_t){ .x = (a.x + b.x)/2.,
                        .y = (a.y + b.y)/2.,
                        .z = (a.z + b.z)/2. };
}
__attribute__((unused))
static point3f_t sub(const point3f_t a, const point3f_t b)
{
    return (point3f_t){ .x = a.x - b.x,
                        .y = a.y - b.y,
                        .z = a.z - b.z };
}
static point3f_t cross(const point3f_t a, const point3f_t b)
{
    return (point3f_t){ .x = a.y*b.z - a.z*b.y,
                        .y = a.z*b.x - a.x*b.z,
                        .z = a.x*b.y - a.y*b.x };
}



typedef struct
{
    point3f_t p; // the center
    point3f_t v; // a direction vector in the plane; may not be normalized

    // point indices inside each ring
    int ipoint0;
    int ipoint1  : sizeof(int)*8-1; // leave one bit for "visited"
    bool visited : 1;
} segment_t;

typedef struct
{
    point3f_t p; // A point somewhere on the plane

    // A normal to the plane direction vector in the plane; not necessarily
    // normalized
    point3f_t n_unnormalized;
} plane_unnormalized_t;

typedef struct
{
    uint16_t isegment, iring;
} segmentref_t;

typedef struct
{
    segmentref_t nodes[255];
    int n;
} stack_t;

typedef struct
{
    union
    {
        plane_t              plane;
        plane_unnormalized_t plane_unnormalized;
    };

    segmentref_t segments[512-7];
    int n;
} segment_cluster_t;


static
void eig_real_symmetric_3x3( // out
                             double* vsmallest, // the smallest-eigenvalue eigenvector
                             double* l,         // ALL the eigenvalues, in ascending order
                             // in
                             const double* M // shape (6,); packed storage; row-first
                             )
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


    // Now to find the corresponding eigenvector. Following:
    //   https://en.wikipedia.org/wiki/Eigenvalue_algorithm#Eigenvectors_of_normal_3%C3%973_matrices
    //
    // I expect a well-behaved point cloud. Only one
    const double v0[] = {M[0] - l[0],
                         M[1],
                         M[2]};
    const double v1[] = {M[1],
                         M[3] - l[0],
                         M[4]};
    vsmallest[0] = v0[1]*v1[2] - v0[2]*v1[1];
    vsmallest[1] = v0[2]*v1[0] - v0[0]*v1[2];
    vsmallest[2] = v0[0]*v1[1] - v0[1]*v1[0];
    const double mag = sqrt(vsmallest[0]*vsmallest[0] + vsmallest[1]*vsmallest[1] + vsmallest[2]*vsmallest[2]);
    for(int i=0; i<3; i++)
        vsmallest[i] /= mag;
}

static void fit_plane_into_points( // out
                                   plane_t*         plane,
                                   float*           max_norm2_dp,
                                   float*           eigenvalues_ascending, // 3 of these
                                   // in
                                   const point3f_t* points,
                                   const points_and_plane_t* points_and_plane
                                   )
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

    point3f_t pmean = {};
    for(int i=0; i<points_and_plane->n; i++)
        for(int j=0; j<3; j++)
            pmean.xyz[j] += points[points_and_plane->ipoint[i]].xyz[j];
    for(int j=0; j<3; j++)
        pmean.xyz[j] /= (float)(points_and_plane->n);


    *max_norm2_dp = 0.0f;

    // packed storage; row-first
    // double-precision because this is potentially inaccurate
    double M[3+2+1] = {};
    for(int i=0; i<points_and_plane->n; i++)
    {
        const point3f_t dp = sub(points[points_and_plane->ipoint[i]], pmean);

        const float norm2_dp = norm2(dp);
        if(*max_norm2_dp < norm2_dp)
            *max_norm2_dp = norm2_dp;

        M[0] += (double)(dp.xyz[0]*dp.xyz[0]);
        M[1] += (double)(dp.xyz[0]*dp.xyz[1]);
        M[2] += (double)(dp.xyz[0]*dp.xyz[2]);
        M[3] += (double)(dp.xyz[1]*dp.xyz[1]);
        M[4] += (double)(dp.xyz[1]*dp.xyz[2]);
        M[5] += (double)(dp.xyz[2]*dp.xyz[2]);
    }

    double l[3];         // all the eigenvalues, in ascending order
    double vsmallest[3]; // eigenvector for l[0]
    eig_real_symmetric_3x3(vsmallest,l,M);
    plane->p = pmean;
    plane->n = point3f_from_double(vsmallest);

    for(int i=0; i<3; i++)
        eigenvalues_ascending[i] = (float)l[i];
}



static
float th_from_point(const point3f_t* p)
{
    return atan2f(p->y, p->x);
}


// ASSUMES th_rad CAME FROM atan2, SO IT'S IN [-pi,pi]
static
int isegment_from_th(const float th_rad,
                     const context_t* ctx)
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
bool point_is_valid__presolve(const point3f_t* p,
                              const float dth_rad,
                              const bool debug,
                              const context_t* ctx)
{
    const int Ngap = (int)( 0.5f + fabsf(dth_rad) * (float)ctx->Npoints_per_rotation / (2.0f*M_PI) );

    // Ngap==1 is the expected, normal value. Anything larger is a gap
    if( DEBUG_ON_TRUE((Ngap-1) > ctx->threshold_max_Ngap,
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


/* bitarray. Test it like this:

int main(int argc, char* argv[])
{
    const int Nbits = 350;

    const int Nwords = bitarray64_nwords(Nbits);
    uint64_t bitarray[Nwords];

    if(Nwords != 6)
    {
        printf("Mismatched Nwords\n");
        return 1;
    }
    uint64_t ref[6] = {};

    memset(bitarray, 0, Nwords*sizeof(uint64_t));

    bitarray64_set(      bitarray, 1);
    bitarray64_set_range(bitarray, 5, 30);
    bitarray64_clear(    bitarray, 6);
    bitarray64_set_range(bitarray, 60,4);
    ref[0] = 0xf0000007ffffffa2;

    bitarray64_set_range(bitarray, 64*1 + 60, 7);
    ref[1] = 0xf000000000000000;

    bitarray64_set_range(bitarray, 64*2 + 50, 100);
    ref[2] = 0xfffc000000000007;
    ref[3] = 0xffffffffffffffff;
    ref[4] = 0x00000000003fffff;

    bitarray64_set_range(bitarray, 64*5 + 0,  20);
    ref[5] = 0x00000000000fffff;

    for(int i=0; i<Nwords; i++)
    {
        printf("word %d ref/computed/xor:\n0x%016"PRIx64"\n0x%016"PRIx64"\n0x%016"PRIx64"\n\n",
               i,
               ref[i],
               bitarray[i],
               ref[i] ^ bitarray[i]);
    }
    if(0 != memcmp(ref, bitarray, Nwords*sizeof(uint64_t)))
    {
        printf("Mismatched data\n");
        return 1;
    }
    printf("All OK\n");
    return 0;
}

*/
__attribute__((unused))
static int bitarray64_nwords(const int Nbits)
{
    // round up the number of 64-bit words required
    return (Nbits+63)/64;
}
__attribute__((unused))
static void bitarray64_set(uint64_t* bitarray, int ibit)
{
    bitarray[ibit/64] |= (1ul << (ibit % 64));
}
__attribute__((unused))
static void bitarray64_clear(uint64_t* bitarray, int ibit)
{
    bitarray[ibit/64] &= ~(1ul << (ibit % 64));
}
__attribute__((unused))
static bool bitarray64_check(const uint64_t* bitarray, int ibit)
{
    return bitarray[ibit/64] & (1ul << (ibit % 64));
}
__attribute__((unused))
static void bitarray64_set_range_oneword(uint64_t* word,
                                         int ibit0, int Nbits)
{
    *word |= ((1ul << Nbits) - 1) << ibit0;
}
__attribute__((unused))
static void bitarray64_set_range(uint64_t* bitarray,
                                 int ibit0, int Nbits)
{
    // The first chunk, up to the first word boundary
    int ibit_next_start_of_word = 64*(int)((ibit0+63)/64);

    int Nbits_remaining_in_word = ibit_next_start_of_word - ibit0;
    if(Nbits_remaining_in_word)
    {
        if(Nbits <= Nbits_remaining_in_word)
        {
            bitarray64_set_range_oneword(&bitarray[ibit0/64],
                                         ibit0%64,
                                         Nbits);
            return;
        }

        bitarray64_set_range_oneword(&bitarray[ibit0/64],
                                     ibit0%64,
                                     Nbits_remaining_in_word);

        ibit0 = ibit_next_start_of_word;
        Nbits -= Nbits_remaining_in_word;
    }

    // Next chunk starts at an even word boundary

    // Process any whole words
    while(Nbits >= 64)
    {
        bitarray[ibit0/64] = ~0ul;
        ibit0 += 64;
        Nbits -= 64;
    }

    // Last little bit
    if(Nbits)
        bitarray64_set_range_oneword(&bitarray[ibit0/64],
                                     0,
                                     Nbits);
}



static
bool is_point_segment_planar(const point3f_t* p,
                             const int ipoint0,
                             const int ipoint1,
                             const uint64_t* bitarray_invalid,
                             const context_t* ctx)
{
    const point3f_t* p0 = &p[ipoint0];
    const point3f_t* p1 = &p[ipoint1];

    const point3f_t v01 = sub(*p1,*p0);

    const float recip_norm2_v01 = 1.f / norm2(v01);

    // do all the points in the chunk lie along v? If so, this is a linear
    // feature
    for(int ipoint=ipoint0+1; ipoint<=ipoint1; ipoint++)
    {
        if(bitarray64_check(bitarray_invalid,ipoint-ipoint0))
            continue;

        const point3f_t v = sub(p[ipoint], *p0);

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
                           const int isegment, const int iring,
                           const int Npoints_invalid_in_segment,
                           const uint64_t* bitarray_invalid,
                           const point3f_t* p,
                           const int ipoint0,
                           const int ipoint1,
                           const bool debug_this_ring,
                           const context_t* ctx)
{
    segment_t* segment = &segments[isegment];

    const bool debug_region =
        (ctx->debug_xmin < p[ipoint0].x && p[ipoint0].x < ctx->debug_xmax &&
         ctx->debug_ymin < p[ipoint0].y && p[ipoint0].y < ctx->debug_ymax) ||
        (ctx->debug_xmin < p[ipoint1].x && p[ipoint1].x < ctx->debug_xmax &&
         ctx->debug_ymin < p[ipoint1].y && p[ipoint1].y < ctx->debug_ymax);
    const bool debug = debug_this_ring && debug_region;


    const int Npoints = ipoint1 - ipoint0 + 1 - Npoints_invalid_in_segment;

    if(DEBUG_ON_TRUE(Npoints_invalid_in_segment > ctx->threshold_max_Npoints_invalid_segment,
                     &p[ipoint0],
                     "%d > %d", Npoints_invalid_in_segment, ctx->threshold_max_Npoints_invalid_segment) ||
       DEBUG_ON_TRUE(Npoints < ctx->threshold_min_Npoints_in_segment,
                     &p[ipoint0],
                     "%d < %d", Npoints, ctx->threshold_min_Npoints_in_segment) ||
       DEBUG_ON_TRUE(!is_point_segment_planar(p,ipoint0,ipoint1,bitarray_invalid, ctx),
                     &p[ipoint0],
                     ""))
    {
        *segment = (segment_t){};
        return;
    }

    segment->p = mean(p[ipoint1], p[ipoint0]);
    segment->v = sub( p[ipoint1], p[ipoint0]);
    segment->ipoint0 = ipoint0;
    segment->ipoint1 = ipoint1;

    if(debug_region)
        MSG("isegment=%d iring=%d has p = (%.2f %.2f %.2f) at %s():%d",
            isegment, iring,
            segment->p.x,segment->p.y,segment->p.z,
            __func__, __LINE__);
}


static void
stage1_segment_from_ring(// out
                         segment_t* segments_thisring,

                         // in
                         const int iring,
                         const point3f_t* points_thisring,
                         const int Npoints_thisring,
                         const context_t* ctx)
{
    // I want this to be fast, and I'm looking for very clear planes, so I do a
    // crude thing here:
    //
    // 1. I look for long plane-y (or line-y) sections, then expand them later
    //
    // 2. I have ordered data, and I know that each planar segment will be a
    //    very squashed conic section segment. In many cases, it'll be so
    //    squashed to appear linear (i.e.) its plane would be ill-defined
    //
    // So I check for linearity first. And then for a curve (conic section
    // slice)

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
                                  isegment0, iring,
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
        if(ipoint-ipoint0 <= ctx->Npoints_per_segment)
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
                          isegment0, iring,
                          Npoints_invalid_in_segment,
                          bitarray_invalid,
                          points_thisring, ipoint0, Npoints_thisring-1,
                          iring == ctx->debug_iring,
                          ctx);
}

static void ring_minmax_from_segment_cluster(// out
                                             int* iring0,
                                             int* iring1,
                                             // int
                                             const segment_cluster_t* segment_cluster)
{
    *iring0 = INT_MAX;
    *iring1 = INT_MIN;
    for(int isegment=0; isegment<segment_cluster->n; isegment++)
    {
        const segmentref_t* segmentref = &segment_cluster->segments[isegment];
        const int           iring      = segmentref->iring;
        if(iring < *iring0) *iring0 = iring;
        if(iring > *iring1) *iring1 = iring;
    }
}

static bool stack_empty(stack_t* stack)
{
    return (stack->n == 0);
}
// returns the node to fill in, or NULL if full
static segmentref_t* stack_push(stack_t* stack)
{
    if(stack->n == (int)(sizeof(stack->nodes)/sizeof(stack->nodes[0])))
        return NULL;
    return &stack->nodes[stack->n++];
}
// returns the node, or NULL if empty
static segmentref_t* stack_pop(stack_t* stack)
{
    if(stack->n == 0)
        return NULL;
    return &stack->nodes[--stack->n];
}

static bool is_normal(const point3f_t v,
                      const point3f_t n,
                      const context_t* ctx)
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

static bool is_same_direction(const point3f_t a,
                              const point3f_t b,
                              const context_t* ctx)
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

static bool plane_from_segment_segment(// out
                                       plane_unnormalized_t* plane_unnormalized,
                                       // in
                                       const segment_t* s0,
                                       const segment_t* s1,
                                       const context_t* ctx)
{
    // I want:
    //   inner(p1-p0, n=cross(v0,v1)) = 0
    point3f_t dp = sub(s1->p, s0->p);

    // The two normal estimates must be close
    point3f_t n0 = cross(dp,s0->v);
    point3f_t n1 = cross(dp,s1->v);

    if(!is_same_direction(n0,n1,ctx))
        return false;

    plane_unnormalized->n_unnormalized = mean(n0,n1);
    plane_unnormalized->p              = mean(s0->p, s1->p);
    return true;
}

static bool plane_point_compatible(const plane_t*   plane,
                                   const point3f_t* point,
                                   const context_t* ctx)
{
    // I want (point - p) to be perpendicular to n. I want this in terms of
    // "distance-off-plane" so err = inner( (point - p), n) / mag(n)
    //
    // Accept if threshold > inner( (point - p), n) / mag(n)
    // n is normalized here, so I omit the /magn
    const point3f_t dp = sub(*point, plane->p);

    return ctx->threshold_max_plane_point_error > fabsf(inner(dp, plane->n));
}

static bool plane_point_compatible_unnormalized(const plane_unnormalized_t* plane_unnormalized,
                                                const point3f_t* point,
                                                const context_t* ctx)
{
    // I want (point - p) to be perpendicular to n. I want this in terms of
    // "distance-off-plane" so err = inner( (point - p), n) / mag(n)
    //
    // Accept if threshold > inner( (point - p), n) / mag(n)
    // n is normalized here, so I omit the /magn
    const point3f_t dp = sub(*point, plane_unnormalized->p);
    const float proj = inner(dp, plane_unnormalized->n_unnormalized);
    return ctx->threshold_max_plane_point_error*norm2(plane_unnormalized->n_unnormalized) > proj*proj;
}

static bool plane_segment_compatible(const plane_unnormalized_t* plane_unnormalized,
                                     const segment_t*            segment,
                                     const int iring, const int isegment,
                                     const context_t* ctx)
{
    // both segment->p and segment->v must lie in the plane

    // I want:
    //   inner(segv,   n) = 0
    //   inner(segp-p, n) = 0
    const bool debug =
        ctx->debug_xmin < segment->p.x && segment->p.x < ctx->debug_xmax &&
        ctx->debug_ymin < segment->p.y && segment->p.y < ctx->debug_ymax;
    return
        !( DEBUG_ON_TRUE(!is_normal(segment->v, plane_unnormalized->n_unnormalized, ctx),
                         &segment->p,
                         "segment iring=%d isegment=%d isn't plane-consistent during accumulation: the segment direction isn't in-plane",
                         iring,isegment) ||
           DEBUG_ON_TRUE(!plane_point_compatible_unnormalized(plane_unnormalized, &segment->p, ctx),
                         &segment->p,
                         "segment iring=%d isegment=%d isn't plane-consistent during accumulation: the segment direction isn't in-plane",
                         iring,isegment));
}


static void try_visit(stack_t* stack,
                      // out
                      segment_cluster_t* cluster,
                      // what we're trying
                      const int iring, const int isegment,
                      // context
                      const plane_unnormalized_t* plane_unnormalized,
                      segment_t* segments, // non-const to be able to set "visited"
                      const int Nrings,
                      const context_t* ctx)
{
    if(iring    < 0 || iring    >= Nrings                ) return;
    if(isegment < 0 || isegment >= Nsegments_per_rotation) return;

    segment_t* segment = &segments[iring*Nsegments_per_rotation + isegment];

    if(segment_is_valid(segment) &&
       !segment->visited &&
       plane_segment_compatible(plane_unnormalized, segment,
                                iring, isegment,
                                ctx))
    {
        segmentref_t* node = stack_push(stack);

        // Do this before the error checking; otherwise the error conditions may
        // go into an infinite loop
        segment->visited = true;

        if(node == NULL)
        {
            MSG("Connected component too large. Ignoring the rest of it. Please bump up the size of 'stack_t.nodes'");
            return;
        }
        if(cluster->n == (int)(sizeof(cluster->segments)/sizeof(cluster->segments[0])))
        {
            MSG("Connected component too large. Ignoring the rest of it. Please bump up the size of 'cluster_t.segments'");
            return;
        }

        node->isegment = isegment;
        node->iring    = iring;

        cluster->segments[cluster->n++] = *node;
    }
}

static void stage2_cluster_segments(// out
                                    segment_cluster_t* clusters,
                                    int* Nclusters,
                                    const int Nclusters_max, // size of clusters[]

                                    // in
                                    segment_t* segments, // non-const to be able to set "visited"
                                    const int Nrings,
                                    const context_t* ctx)
{
    *Nclusters = 0;

    for(int iring = 0; iring < Nrings-1; iring++)
    {
        for(int isegment = 0; isegment < Nsegments_per_rotation; isegment++)
        {
            if(*Nclusters == Nclusters_max)
            {
                MSG("Too many flat objects in scene, exceeded Nclusters_max. Not reporting any more candidate planes. Bump Nclusters_max");
                return;
            }
            segment_cluster_t* cluster = &clusters[*Nclusters];

            segment_t* segment = &segments[iring*Nsegments_per_rotation + isegment];
            if(!(segment_is_valid(segment) && !segment->visited))
                continue;

            // A single segment has only a direction. To define a plane I need
            // another non-colinear segment, and I can get it from one of the
            // two adjacent rings. Once I get a plane I find the biggest
            // connected component of plane-consistent segments around this one
            //
            // The biggest-connected-component routine is traditionally done
            // with recursion, but setting it up manually allows me to better
            // control exactly what is pushed onto the stack, and minimize it


            const int iring1 = iring+1;

            *cluster = (segment_cluster_t){.n = 2,
                                           .segments = {[0] = {.isegment = isegment,
                                                               .iring    = iring},
                                                        [1] = {.isegment = isegment,
                                                               .iring    = iring1}}};

            segment_t* segment1 = &segments[iring1*Nsegments_per_rotation + isegment];
            if(!(segment_is_valid(segment1) && !segment1->visited))
                continue;

            const bool debug =
                ctx->debug_xmin < segment->p.x && segment->p.x < ctx->debug_xmax &&
                ctx->debug_ymin < segment->p.y && segment->p.y < ctx->debug_ymax;
            if(DEBUG_ON_TRUE(!plane_from_segment_segment(&cluster->plane_unnormalized,
                                                         segment,segment1,
                                                         ctx),
                             &segment->p,
                             "segment iring=%d isegment=%d isn't plane-consistent with segment iring=%d isegment=%d",
                             iring,isegment,
                             iring1,isegment))
                continue;

            stack_t stack = {};

            segmentref_t* node0 = stack_push(&stack);
            node0->iring    = iring;
            node0->isegment = isegment;

            segmentref_t* node1 = stack_push(&stack);
            node1->iring    = iring1;
            node1->isegment = isegment;

            segment ->visited = true;
            segment1->visited = true;

            while(!stack_empty(&stack))
            {
                segmentref_t* node = stack_pop(&stack);
                try_visit(&stack,
                          cluster,
                          node->iring-1, node->isegment, &cluster->plane_unnormalized,
                          segments, Nrings,
                          ctx);
                try_visit(&stack,
                          cluster,
                          node->iring+1, node->isegment, &cluster->plane_unnormalized,
                          segments, Nrings,
                          ctx);
                try_visit(&stack,
                          cluster,
                          node->iring, node->isegment-1, &cluster->plane_unnormalized,
                          segments, Nrings,
                          ctx);
                try_visit(&stack,
                          cluster,
                          node->iring, node->isegment+1, &cluster->plane_unnormalized,
                          segments, Nrings,
                          ctx);
            }

            if(DEBUG_ON_TRUE(cluster->n == 2,
                             &segment->p,
                             "cluster starting with iring=%d isegment=%d only contains the seed segments", iring,isegment))
            {
                // This hypothetical ring-ring component is too small. The
                // next-ring segment might still be valid in another component,
                // with a different plane, without segment. So I allow it again.
                segment1->visited = false;
                continue;
            }

            if(DEBUG_ON_TRUE(cluster->n < ctx->threshold_min_Nsegments_in_cluster,
                             &segment->p,
                             "cluster starting with iring=%d isegment=%d too small: %d < %d",
                             iring,isegment,
                             cluster->n, ctx->threshold_min_Nsegments_in_cluster))
            {
                continue;
            }

            if(DEBUG_ON_TRUE(cluster->n > ctx->threshold_max_Nsegments_in_cluster,
                             &segment->p,
                             "cluster starting with iring=%d isegment=%d too big: %d > %d",
                             iring,isegment,
                             cluster->n, ctx->threshold_max_Nsegments_in_cluster))
            {
                continue;
            }

            {
                int iring0,iring1;
                ring_minmax_from_segment_cluster(&iring0, &iring1, cluster);
                if(DEBUG_ON_TRUE(iring1-iring0+1 < ctx->threshold_min_Nrings_in_cluster,
                                 &segment->p,
                                 "cluster starting with iring=%d isegment=%d only contains too-few rings: %d < %d",
                                 iring,isegment,
                                 iring1-iring0+1, ctx->threshold_min_Nrings_in_cluster))
                    continue;
            }

            // I throw out any cluster that's entirely too far. I only do this
            // now because it's possible to see far-too-large planes that are
            // partially too far (walls, ground), and I want to detect this
            // far-too-large-ness, and throw them out
            bool keep = false;
            for(int i=0; i<cluster->n; i++)
            {
                const segmentref_t* node = &cluster->segments[i];
                const segment_t* segment = &segments[node->iring*Nsegments_per_rotation + node->isegment];

                if( norm2(segment->p) < ctx->threshold_max_range*ctx->threshold_max_range )
                {
                    keep = true;
                    break;
                }
            }
            if(DEBUG_ON_TRUE(!keep,
                             &segment->p,
                             "cluster starting with iring=%d isegment=%d is completely past the threshold_max_range=%f",
                             iring,isegment, ctx->threshold_max_range))
                continue;




            // We're accepting this cluster. There won't be a lot of these, and
            // we will be accessing the plane normal a lot, so I normalize the
            // normal vector
            const float magn = mag(cluster->plane_unnormalized.n_unnormalized);
            for(int i=0; i<3;i++)
                cluster->plane.n.xyz[i] = cluster->plane_unnormalized.n_unnormalized.xyz[i] / magn;

            (*Nclusters)++;
        }
    }
}


// Returns true if we processed this point (maybe by accumulating it) and we
// should keep going. Returns false if we should stop the iteration
static bool accumulate_point(// out
                             points_and_plane_t* points_and_plane,
                             // in,out
                             uint64_t* bitarray_visited, // indexed by IN-RING points
                             float* th_rad_last,
                             // in
                             const point3f_t* points,
                             const int ipoint0_in_ring, // start of this ring in the full points[] array
                             const int ipoint,          // IN-RING index
                             const int ipoint_segment_limit,
                             const context_t* ctx)
{
    if(bitarray64_check(bitarray_visited, ipoint))
        // We already processed this point, presumably from the other side.
        // There's no reason to keep going, since we already approached from the
        // other side
        return false;

    const float th_rad = th_from_point(&points[ipoint0_in_ring + ipoint]);
    if(*th_rad_last < FLT_MAX)
    {
        // we have a valid th_rad_last
        if(fabsf(th_rad - *th_rad_last) > ctx->threshold_max_gap_th_rad)
            return false;
    }
    else
    {
        // we do not have a valid th_rad_last. Stop when we reach the segment
        // limit
        if(ipoint == ipoint_segment_limit)
            return false;
    }

    // no threshold_max_range check here. This was already checked when
    // constructing the candidate segments. So if we got this far, I assume it's
    // good

    if( plane_point_compatible(&points_and_plane->plane,
                               &points[ipoint0_in_ring + ipoint],
                               ctx) )
    {
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
        if(points_and_plane->n == (int)(sizeof(points_and_plane->ipoint)/sizeof(points_and_plane->ipoint[0])))
        {
            MSG("points_and_plane->ipoint overflow. Skipping the reset of the points");
            return false;
        }
        points_and_plane->ipoint[points_and_plane->n++] = ipoint0_in_ring + ipoint;

        bitarray64_set(bitarray_visited, ipoint);
        *th_rad_last = th_rad;
    }
    else
    {
        // Not accepting this point, but also not updating th_rad_last. So too
        // many successive invalid points will create a too-large gap, failing
        // the threshold_max_gap_th_rad check above
    }

    return true;
}


static void stage3_refine_clusters(// out
                                   points_and_plane_t* points_and_plane,
                                   float*              max_norm2_dp,
                                   float*              eigenvalues_ascending, // 3 of these
                                   // in
                                   const segment_cluster_t* segment_cluster,
                                   const segment_t* segments,
                                   const point3f_t* points,
                                   const int* ipoint0_in_ring,
                                   const int* Npoints,
                                   const context_t* ctx)
{
    /* I have an approximate plane estimate.

       while(...)
       {
         gather set of neighborhood points that match the current plane estimate
         update plane estimate using this set of points
       }
     */


    int iring0,iring1;
    ring_minmax_from_segment_cluster(&iring0, &iring1, segment_cluster);

    const int Nrings_considered = iring1-iring0+1;

    // I keep track of the already-visited points
    const int Nwords_bitarray_visited = bitarray64_nwords(ctx->Npoints_per_rotation); // largest-possible size
    uint64_t bitarray_visited[Nrings_considered][Nwords_bitarray_visited];

    // Start with the best-available plane estimate
    points_and_plane->plane = segment_cluster->plane;

    while(true)
    {
        points_and_plane->n = 0;

        for(int i=0; i<Nrings_considered; i++)
            memset(bitarray_visited[i], 0, Nwords_bitarray_visited*sizeof(uint64_t));

        for(int isegment_in_cluster=0; isegment_in_cluster<segment_cluster->n; isegment_in_cluster++)
        {
            const segmentref_t* segmentref = &segment_cluster->segments[isegment_in_cluster];

            const int iring    = segmentref->iring;
            const int isegment = segmentref->isegment;

            const segment_t* segment =
                &segments[iring*Nsegments_per_rotation + isegment];

            // I start in the center of each segment, and expand outwards to
            // capture all the matching points
            const int ipoint0 = (segment->ipoint0 + segment->ipoint1) / 2;

            float th_rad_last;

            th_rad_last = FLT_MAX; // indicate an invalid value initially
            for(int ipoint = ipoint0;
                ipoint < Npoints[iring];
                ipoint++)
            {
                if(!accumulate_point(points_and_plane,
                                     bitarray_visited[iring-iring0],
                                     &th_rad_last,
                                     points,
                                     ipoint0_in_ring[iring],
                                     ipoint,
                                     segment->ipoint1,
                                     ctx))
                    break;
            }

            th_rad_last = FLT_MAX; // indicate an invalid value initially
            for(int ipoint = ipoint0-1;
                ipoint >= 0;
                ipoint--)
            {
                if(!accumulate_point(points_and_plane,
                                     bitarray_visited[iring-iring0],
                                     &th_rad_last,
                                     points,
                                     ipoint0_in_ring[iring],
                                     ipoint,
                                     segment->ipoint0,
                                     ctx))
                    break;
            }


            // I don't bother to look in rings that don't appear in the
            // segment_cluster. This will by contain not very much data (because
            // the pre-solve didn't find it), and won't be of much value
        }


        // Got a set of points. Fit a plane
        fit_plane_into_points(&points_and_plane->plane,
                              max_norm2_dp,
                              eigenvalues_ascending,
                              points,
                              points_and_plane);

#warning FOR NOW I JUST RUN A SINGLE ITERATION
        return;
    }
}

void default_context(context_t* ctx)
{
#define LIST_CONTEXT_SET_DEFAULT(type,name,default,...) \
    .name = default,

    *ctx = (context_t)
        { LIST_CONTEXT(LIST_CONTEXT_SET_DEFAULT) };

#undef LIST_CONTEXT_SET_DEFAULT
}

// Returns how many planes were found or <0 on error
int8_t point_segmentation(// out
                          points_and_plane_t* points_and_plane,
                          // in
                          const int8_t Nplanes_max, // buffer length of points_and_plane[]
                          const point3f_t* points,  // length sum(Npoints)
                          const int* Npoints,
                          const context_t* ctx)

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
        ipoint0_in_ring[i] = ipoint0_in_ring[i-1] + Npoints[i-1];

    segment_t segments[ctx->Nrings*Nsegments_per_rotation] = {};

    if(ctx->dump)
        printf("# x y what z\n");

    for(int iring=0; iring<ctx->Nrings; iring++)
    {
        stage1_segment_from_ring(// out
                                 &segments[Nsegments_per_rotation*iring],
                                 // in
                                 iring,
                                 &points[ipoint0_in_ring[iring]], Npoints[iring],
                                 ctx);

        if(ctx->dump)
        {
            for(int i=0; i<Npoints[iring]; i++)
                printf("%f %f all %f\n",
                       points[ipoint0_in_ring[iring] + i].x,
                       points[ipoint0_in_ring[iring] + i].y,
                       points[ipoint0_in_ring[iring] + i].z);

            for(int i=0; i<Nsegments_per_rotation; i++)
            {
                if(!(segments[iring*Nsegments_per_rotation + i].v.x == 0 &&
                     segments[iring*Nsegments_per_rotation + i].v.y == 0 &&
                     segments[iring*Nsegments_per_rotation + i].v.z == 0))
                    printf("%f %f stage1-segment-from-ring-%02d %f\n",
                           segments[iring*Nsegments_per_rotation + i].p.x,
                           segments[iring*Nsegments_per_rotation + i].p.y,
                           iring,
                           segments[iring*Nsegments_per_rotation + i].p.z);
            }
        }
    }

    // plane_clusters_from_segments() will return only clusters of an acceptable size,
    // so there will not be a huge number of candidates
    const int Nmax_planes = 10;
    segment_cluster_t segment_clusters[Nmax_planes];
    int Nclusters;
    stage2_cluster_segments(segment_clusters,
                            &Nclusters,
                            (int)(sizeof(segment_clusters)/sizeof(segment_clusters[0])),
                            segments,
                            ctx->Nrings,
                            ctx);

    if(ctx->dump)
        for(int icluster=0; icluster<Nclusters; icluster++)
        {
            segment_cluster_t* segment_cluster = &segment_clusters[icluster];
            for(int i=0; i<segment_cluster->n; i++)
            {
                const int iring    = segment_cluster->segments[i].iring;
                const int isegment = segment_cluster->segments[i].isegment;
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
                           points[ipoint0_in_ring[iring] + ipoint].x,
                           points[ipoint0_in_ring[iring] + ipoint].y,
                           icluster,
                           points[ipoint0_in_ring[iring] + ipoint].z);
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

        segment_cluster_t* segment_cluster = &segment_clusters[icluster];

        float max_norm2_dp;
        float eigenvalues_ascending[3];
        stage3_refine_clusters(&points_and_plane[iplane_out],
                               &max_norm2_dp,
                               eigenvalues_ascending,
                               segment_cluster,
                               segments,
                               points, ipoint0_in_ring, Npoints,
                               ctx);

        const int Npoints_in_plane = points_and_plane[iplane_out].n;

        const bool debug =
            ctx->debug_xmin < points_and_plane[iplane_out].plane.p.x && points_and_plane[iplane_out].plane.p.x < ctx->debug_xmax &&
            ctx->debug_ymin < points_and_plane[iplane_out].plane.p.y && points_and_plane[iplane_out].plane.p.y < ctx->debug_ymax;

        // Each eigenvalue is a 1-sigma ellipse of our point cloud. It
        // represents the sum-of-squares of deviations from the mean. The RMS is
        // the useful value: RMS = sqrt(sum_of_squares/N)
        if(DEBUG_ON_TRUE(eigenvalues_ascending[0] > ctx->threshold_max_rms_fit_error*ctx->threshold_max_rms_fit_error*(float)Npoints_in_plane,
                         &points_and_plane[iplane_out].plane.p,
                         "Refined plane doesn't fit the constituent points well-enough: %f > %f",
                         sqrt(eigenvalues_ascending[0]/(float)Npoints_in_plane), ctx->threshold_max_rms_fit_error))
            continue;

        // Some point clouds are degenerate, and we throw them away. The first
        // eigenvalue is 0-ish: we're looking at a plane, and the data must be
        // squished in that way. But the next eigenvalue should be a decent
        // size. Otherwise the data is linear-y instead of plane-y
        if(DEBUG_ON_TRUE(eigenvalues_ascending[1] < ctx->threshold_min_rms_point_cloud_2nd_dimension*ctx->threshold_min_rms_point_cloud_2nd_dimension*(float)Npoints_in_plane,
                         &points_and_plane[iplane_out].plane.p,
                         "Refined plane is degenerate (2nd eigenvalue of point cloud dispersion is too small): %f < %f",
                         sqrt(eigenvalues_ascending[1]/(float)Npoints_in_plane), ctx->threshold_min_rms_point_cloud_2nd_dimension))
            continue;

        if(DEBUG_ON_TRUE(max_norm2_dp*2.*2. > ctx->threshold_max_plane_size*ctx->threshold_max_plane_size,
                         &points_and_plane[iplane_out].plane.p,
                         "Refined plane is too big: max_mag_dp*2 > threshold: %f > %f",
                         sqrtf(max_norm2_dp)*2., ctx->threshold_max_plane_size))
            continue;


        // We're past all the filters. I accept this plane
        if(ctx->dump)
            for(int i=0; i<points_and_plane[iplane_out].n; i++)
                printf("%f %f stage3-refined-points-%d %f\n",
                       points[points_and_plane[iplane_out].ipoint[i]].x,
                       points[points_and_plane[iplane_out].ipoint[i]].y,
                       icluster,
                       points[points_and_plane[iplane_out].ipoint[i]].z);
        iplane_out++;
    }

    return iplane_out;
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
