#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <limits.h>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>

#define MSG(fmt, ...) fprintf(stderr, "%s(%d): " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__)


static bool dump = true;
static int debug_iring = 10;
static float debug_xmin = -1e6f;
static float debug_xmax =  1e6f;
static float debug_ymin = -1e6f;
static float debug_ymax =  -3.f;

#define DEBUG_ERR_ON_TRUE(what, p, fmt, ...)                            \
    ({  if(debug && (what))                                             \
        {                                                               \
            MSG("REJECTED (%.2f %.2f %.2f) at %s():%d because " #what ": " fmt, \
                (p)->x,(p)->y,(p)->z,                                   \
                __func__, __LINE__, ##__VA_ARGS__);                     \
        }                                                               \
        what;                                                           \
    })









static const int   threshold_min_Npoints_in_segment         = 10;
static const int   threshold_max_Npoints_invalid_segment    = 5;
static const float threshold_max_range                      = 5.f;
static const int   Npoints_per_rotation                     = 1809; // found empirically in dump-lidar-scan.py
static const int   Npoints_per_segment                      = 20;
static const int   threshold_max_Ngap                       = 2;
static const float threshold_max_deviation_off_segment_line = 0.05f;
static const int   Nrings                                   = 32;

static const float threshold_max_cos_angle_error_normal         = 0.087155742747f; // cos(90-5deg)
static const float threshold_min_cos_angle_error_same_direction = 0.996194698092f; // cos(5deg)
static const float threshold_max_plane_point_error              = 0.15;
// round up
static const int Nsegments_per_rotation = (int)((Npoints_per_rotation + Npoints_per_segment-1) / Npoints_per_segment);

static const int threshold_max_Nsegments_in_cluster = 40;
static const int threshold_min_Nsegments_in_cluster = 5;

// used in refinement
static const float threshold_max_gap_th_rad         = 0.5f * M_PI/180.f;

typedef union
{
    struct
    {
        float x,y,z;
    };
    float xyz[3];
} point3f_t;

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
    point3f_t p; // A point somewhere on the plane
    point3f_t n; // A normal to the plane
} plane_t;

typedef struct
{
    uint16_t isegment, iring;
} segmentref_t;

typedef struct
{
    segmentref_t nodes[128];
    int n;
} stack_t;

typedef struct
{
    union
    {
        plane_t              plane;
        plane_unnormalized_t plane_unnormalized;
    };

    segmentref_t segments[128];
    int n;
} segment_cluster_t;

typedef struct
{
    point3f_t p[4096];
    int n;

    plane_t plane;
} point_cluster_t;

static
float th_from_point(const point3f_t* p)
{
    return atan2f(p->y, p->x);
}


// ASSUMES th_rad CAME FROM atan2, SO IT'S IN [-pi,pi]
static
int isegment_from_th(const float th_rad)
{
    const float segment_width_rad = 2.0f*M_PI * (float)Npoints_per_segment / (float)Npoints_per_rotation;

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
                              const bool debug)
{
    if( DEBUG_ERR_ON_TRUE( norm2(*p) > threshold_max_range*threshold_max_range,
                           p,
                           "%f > %f", norm2(*p), threshold_max_range*threshold_max_range ))
        return false;

    const int Ngap = (int)( 0.5f + fabsf(dth_rad) * (float)Npoints_per_rotation / (2.0f*M_PI) );

    // Ngap==1 is the expected, normal value. Anything larger is a gap
    if( DEBUG_ERR_ON_TRUE((Ngap-1) > threshold_max_Ngap,
                          p,
                          "%d > %d", Ngap-1, threshold_max_Ngap))
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
                             const uint64_t* bitarray_invalid)
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

        if(norm2_e > threshold_max_deviation_off_segment_line*threshold_max_deviation_off_segment_line)
            return false;
    }
    return true;
}


static
void finish_segment(// out
                    segment_t* segment,
                    // in
                    const int Npoints_invalid_in_segment,
                    const uint64_t* bitarray_invalid,
                    const point3f_t* p,
                    const int ipoint0,
                    const int ipoint1,
                    const bool debug)
{
    const int Npoints = ipoint1 - ipoint0 + 1 - Npoints_invalid_in_segment;

    if(DEBUG_ERR_ON_TRUE(Npoints_invalid_in_segment > threshold_max_Npoints_invalid_segment,
                         &p[ipoint0],
                         "%d > %d", Npoints_invalid_in_segment, threshold_max_Npoints_invalid_segment) ||
       DEBUG_ERR_ON_TRUE(Npoints < threshold_min_Npoints_in_segment,
                         &p[ipoint0],
                         "%d < %d", Npoints, threshold_min_Npoints_in_segment) ||
       DEBUG_ERR_ON_TRUE(!is_point_segment_planar(p,ipoint0,ipoint1,bitarray_invalid),
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
}


static void
fit_plane_from_ring(// out
                    segment_t* segments,

                    // in
                    const point3f_t* points,
                    const int Npoints,
                    const bool debug_this_ring)
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
    const int Nwords_bitarray_invalid = bitarray64_nwords(Npoints_per_segment);
    uint64_t bitarray_invalid[Nwords_bitarray_invalid];


    const float th_rad0 = th_from_point(&points[0]);

    int ipoint0   = 0;
    int isegment0 = isegment_from_th(th_rad0);
    int Npoints_invalid_in_segment = 0;
    float th_rad_prev = th_rad0;

    memset(bitarray_invalid, 0, Nwords_bitarray_invalid*sizeof(uint64_t));

    for(int ipoint=1; ipoint<Npoints; ipoint++)
    {
        const bool debug =
            debug_this_ring &&
            ((debug_xmin < points[ipoint0 ].x && points[ipoint0 ].x < debug_xmax &&
              debug_ymin < points[ipoint0 ].y && points[ipoint0 ].y < debug_ymax) ||
             (debug_xmin < points[ipoint-1].x && points[ipoint-1].x < debug_xmax &&
              debug_ymin < points[ipoint-1].y && points[ipoint-1].y < debug_ymax));

        const float th_rad = th_from_point(&points[ipoint]);
        const int isegment = isegment_from_th(th_rad);
        if(isegment != isegment0)
        {
            finish_segment(&segments[isegment0],
                           Npoints_invalid_in_segment,
                           bitarray_invalid,
                           points, ipoint0, ipoint-1,
                           debug);

            ipoint0   = ipoint;
            isegment0 = isegment;
            Npoints_invalid_in_segment = 0;
            memset(bitarray_invalid, 0, Nwords_bitarray_invalid*sizeof(uint64_t));
        }

        // This should ALWAYS be true. But some datasets are weird, and the
        // azimuths don't changes as quickly as expected, and we have extra
        // points in each segment. I ignore those; hopefully they're not
        // important
        if(ipoint-ipoint0 <= Npoints_per_segment)
        {
            if(!point_is_valid__presolve(&points[ipoint], th_rad - th_rad_prev,
                                         debug))
            {
                Npoints_invalid_in_segment++;
                bitarray64_set(bitarray_invalid, ipoint-ipoint0);
            }
        }

        th_rad_prev = th_rad;
    }

    const bool debug =
        debug_this_ring &&
        ((debug_xmin < points[ipoint0  ].x && points[ipoint0  ].x < debug_xmax &&
          debug_ymin < points[ipoint0  ].y && points[ipoint0  ].y < debug_ymax) ||
         (debug_xmin < points[Npoints-1].x && points[Npoints-1].x < debug_xmax &&
          debug_ymin < points[Npoints-1].y && points[Npoints-1].y < debug_ymax));
    finish_segment(&segments[isegment0],
                   Npoints_invalid_in_segment,
                   bitarray_invalid,
                   points, ipoint0, Npoints-1,
                   debug);
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
                      const point3f_t n)
{
#warning "it is weird to do this with an angle threshold. should be distance threshold"
    // inner(v,n) = cos magv magn ->
    // cos = inner / (magv magn) ->
    // cos^2 = inner^2 / (norm2v norm2n) ->
    // cos^2 norm2v norm2n = inner^2
    float cos_mag_mag = inner(n,v);

    return
        cos_mag_mag*cos_mag_mag <
        threshold_max_cos_angle_error_normal*threshold_max_cos_angle_error_normal*norm2(n)*norm2(v);
}

static bool is_same_direction(const point3f_t a,
                              const point3f_t b)
{
#warning "it is weird to do this with an angle threshold. should be distance threshold"
    // inner(a,b) = cos maga magb ->
    // cos = inner / (maga magb) ->
    // cos^2 = inner^2 / (norm2a norm2b) ->
    // cos^2 norm2a norm2b = inner^2
    float cos_mag_mag = inner(a,b);

    return
        cos_mag_mag > 0.0f &&
        cos_mag_mag*cos_mag_mag >
        threshold_min_cos_angle_error_same_direction*threshold_min_cos_angle_error_same_direction*norm2(a)*norm2(b);
}

static bool plane_from_segment_segment(// out
                                       plane_unnormalized_t* plane_unnormalized,
                                       // in
                                       const segment_t* s0,
                                       const segment_t* s1)
{
    // I want:
    //   inner(p1-p0, n=cross(v0,v1)) = 0
    point3f_t dp = sub(s1->p, s0->p);

    // The two normal estimates must be close
    point3f_t n0 = cross(dp,s0->v);
    point3f_t n1 = cross(dp,s1->v);

    if(!is_same_direction(n0,n1))
        return false;

    plane_unnormalized->n_unnormalized = mean(n0,n1);
    plane_unnormalized->p              = mean(s0->p, s1->p);
    return true;
}


static bool plane_segment_compatible(const plane_unnormalized_t* plane_unnormalized,
                                     const segment_t*            segment)
{
    // both segment->p and segment->v must lie in the plane

    // I want:
    //   inner(segv,   n) = 0
    //   inner(segp-p, n) = 0
    return
        is_normal(segment->v, plane_unnormalized->n_unnormalized) &&
        is_normal(sub(segment->p, plane_unnormalized->p), plane_unnormalized->n_unnormalized);
}


static bool plane_point_compatible(const plane_t*   plane,
                                   const point3f_t* point)
{
    // I want (point - p) to be perpendicular to n. I want this in terms of
    // "distance-off-plane" so err = inner( (point - p), n) / mag(n)
    //
    // Accept if threshold > inner( (point - p), n) / mag(n)
    // n is normalized here, so I omit the /magn
    const point3f_t dp = sub(*point, plane->p);

    return threshold_max_plane_point_error > fabsf(inner(dp, plane->n));
}

static void try_visit(stack_t* stack,
                      // out
                      segment_cluster_t* cluster,
                      // what we're trying
                      const int iring, const int isegment,
                      // context
                      const plane_unnormalized_t* plane_unnormalized,
                      segment_t* segments, // non-const to be able to set "visited"
                      const int Nrings, const int Nsegments_per_rotation)
{
    if(iring    < 0 || iring    >= Nrings                ) return;
    if(isegment < 0 || isegment >= Nsegments_per_rotation) return;

    segment_t* segment = &segments[iring*Nsegments_per_rotation + isegment];

    if(segment_is_valid(segment) &&
       !segment->visited &&
       plane_segment_compatible(plane_unnormalized, segment))
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

static void segment_clusters_from_segments(// out
                                           segment_cluster_t* clusters,
                                           int* Nclusters,
                                           const int Nclusters_max, // size of clusters[]

                                           // in
                                           segment_t* segments, // non-const to be able to set "visited"
                                           const int Nrings, const int Nsegments_per_rotation)
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

            if(!plane_from_segment_segment(&cluster->plane_unnormalized,
                                           segment,segment1))
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
                          segments, Nrings, Nsegments_per_rotation);
                try_visit(&stack,
                          cluster,
                          node->iring+1, node->isegment, &cluster->plane_unnormalized,
                          segments, Nrings, Nsegments_per_rotation);
                try_visit(&stack,
                          cluster,
                          node->iring, node->isegment-1, &cluster->plane_unnormalized,
                          segments, Nrings, Nsegments_per_rotation);
                try_visit(&stack,
                          cluster,
                          node->iring, node->isegment+1, &cluster->plane_unnormalized,
                          segments, Nrings, Nsegments_per_rotation);
            }

            if(cluster->n == 2)
            {
                // This hypothetical ring-ring component is too small. The
                // next-ring segment might still be valid in another component,
                // with a different plane, without segment. So I allow it again.
                segment1->visited = false;
            }

            if(cluster->n < threshold_min_Nsegments_in_cluster ||
               cluster->n > threshold_max_Nsegments_in_cluster)
            {
                continue;
            }

            // We're accepting this cluster. There won't be a lot of these, and
            // we will be accessing the plane normal a lot, so I normalize the
            // normal vector
            const float magn = mag(cluster->plane_unnormalized.n_unnormalized);
            for(int i=0; i<3;i++)
                cluster->plane.n.xyz[i] = cluster->plane_unnormalized.n_unnormalized.xyz[i] / magn;

            (*Nclusters)++;

            if(dump)
            {
                for(int i=0; i<cluster->n; i++)
                {
                    const segmentref_t* node = &cluster->segments[i];
                    const segment_t* segment = &segments[node->iring*Nsegments_per_rotation + node->isegment];

                    printf("%f %f cluster-kernels-%02d %f\n",
                           segment->p.x,
                           segment->p.y,
                           *Nclusters - 1,
                           segment->p.z);
                }
            }
        }
    }
}


// Returns true if we processed this point (maybe by accumulating it) and we
// should keep going. Returns false if we should stop the iteration
static bool accumulate_point(// out
                             point_cluster_t* point_cluster,
                             // in,out
                             uint64_t* bitarray_visited,
                             float* th_rad_last,
                             // in
                             const point3f_t* points,
                             const plane_t* plane,
                             const int ipoint)
{
    if(bitarray64_check(bitarray_visited, ipoint))
        // We already processed this point, presumably from the other side.
        // There's no reason to keep going, since we already approached from the
        // other side
        return false;

    const float th_rad = th_from_point(&points[ipoint]);
    if(*th_rad_last < 1e6f && // if we have a valid th_rad_last
       fabsf(th_rad - *th_rad_last) > threshold_max_gap_th_rad)
        return false;

    // no threshold_max_range check here. This was already checked when
    // constructing the candidate segments. So if we got this far, I assume it's
    // good

    if( plane_point_compatible(plane,
                               &points[ipoint]) )
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
        if(point_cluster->n == (int)(sizeof(point_cluster->p)/sizeof(point_cluster->p[0])))
        {
            MSG("sizeof(point_cluster->p) exceeded. Skippping the reset of the point cluster. Increase the size of sizeof(point_cluster->p)");
            return false;
        }
        point_cluster->p[ point_cluster->n++ ] = points[ipoint];

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










// A drop-in-able public implementation is needed
void eig_smallest_real_symmetric_3x3( // out
                                      double* v,
                                      double* l,
                                      // in
                                      const double* M // shape (6,); packed storage; row-first
                                      );



















// Ingests a point set, and write the fitted plane into point_cluster->plane.
// Returns a fit cost.
static float fit_plane_into_points(// in,out
                                   point_cluster_t* point_cluster)
{
    // I fit a plane to a set of points. The most accurate way to do this is to
    // minimize the observation errors (ranges; what the fit ingesting this data
    // will be doing). But that's a nonlinear solve, and I avoid that here. I
    // simply minimize the norm2 off-plane error instead:
    //
    // - compute pmean
    // - compute M = sum(outer(p[i]-pmean,p[i]-pmean))
    // - n = eigenvector of M corresponding to the smallest eigenvalue
    //
    // Derivation: plane is (p,n); points are in a (3,N) matrix P. I minimize
    // E = sum(norm2( inner(n,Pi-p) ))
    //   = norm2((Pt-pt) n )
    //
    // dE/dp ~ nt (P-p) n
#warning "finish this. I believe the eigenvalue is the measure of tightness of fit; the derivation should show this"

    point3f_t pmean = {};
    for(int i=0; i<point_cluster->n; i++)
        for(int j=0; j<3; j++)
            pmean.xyz[j] += point_cluster->p[i].xyz[j];
    for(int j=0; j<3; j++)
        pmean.xyz[j] /= (float)(point_cluster->n);


    // packed storage; row-first
    // double-precision because this is potentially inaccurate
    double M[3+2+1] = {};
    for(int i=0; i<point_cluster->n; i++)
    {
        const point3f_t dp = sub(point_cluster->p[i], pmean);
        M[0] += (double)(dp.xyz[0]*dp.xyz[0]);
        M[1] += (double)(dp.xyz[0]*dp.xyz[1]);
        M[2] += (double)(dp.xyz[0]*dp.xyz[2]);
        M[3] += (double)(dp.xyz[1]*dp.xyz[1]);
        M[4] += (double)(dp.xyz[1]*dp.xyz[2]);
        M[5] += (double)(dp.xyz[2]*dp.xyz[2]);
    }

    double v[3];
    double l;
    eig_smallest_real_symmetric_3x3(v,&l,M);
    point_cluster->plane.p = pmean;
    point_cluster->plane.n = point3f_from_double(v);

    return (float)l;
}


static float refine_point_cluster_from_segment_cluster(// out
                                                       point_cluster_t* point_cluster,

                                                       // in
                                                       const segment_cluster_t* segment_cluster,
                                                       const segment_t* segments,
                                                       const int Nrings, const int Nsegments_per_rotation,
                                                       const point3f_t** points,
                                                       const int* Npoints)
{
    /* I have an approximate plane estimate.

       while(...)
       {
         gather set of neighborhood points that match the current plane estimate
         update plane estimate using this set of points
       }
     */

    while(true)
    {
        point_cluster->n = 0;

        // I find the min/max ring indices in the cluster
        int iring0 = INT_MAX, iring1 = INT_MIN;
        for(int isegment=0; isegment<segment_cluster->n; isegment++)
        {
            const segmentref_t* segmentref = &segment_cluster->segments[isegment];
            const int           iring      = segmentref->iring;
            if(iring < iring0) iring0 = iring;
            if(iring > iring1) iring1 = iring;
        }

        // I consider the neighboring rings in the cluster, so I add padding on
        // both sides
        const int Nrings_pad = 5;
        iring0 -= Nrings_pad;
        iring1 += Nrings_pad;
        const int Nrings_considered = iring1-iring0+1;

        // I keep track of the already-visited points
        const int Nwords_bitarray_visited = bitarray64_nwords(Npoints_per_rotation); // largest-possible size
        uint64_t bitarray_visited[Nrings_considered][Nwords_bitarray_visited];
        for(int i=0; i<Nrings_considered; i++)
            memset(bitarray_visited[i], 0, Nwords_bitarray_visited*sizeof(uint64_t));

        for(int isegment=0; isegment<segment_cluster->n; isegment++)
        {
            const segmentref_t* segmentref = &segment_cluster->segments[isegment];

            const int iring    = segmentref->iring;
            const int isegment = segmentref->isegment;

            const segment_t* segment =
                &segments[iring*Nsegments_per_rotation + isegment];

            // I start in the center of each segment, and expand outwards to
            // capture all the matching points
            const int ipoint0 = (segment->ipoint0 + segment->ipoint1) / 2;

            float th_rad_last;

            th_rad_last = 1e6f; // indicate an invalid value initially
            for(int ipoint = ipoint0;
                ipoint < Npoints[iring];
                ipoint++)
            {
                if(!accumulate_point(point_cluster,
                                     bitarray_visited[iring-iring0],
                                     &th_rad_last,
                                     points[iring],
                                     &segment_cluster->plane,
                                     ipoint))
                    break;
            }

            th_rad_last = 1e6f; // indicate an invalid value initially
            for(int ipoint = ipoint0-1;
                ipoint >= 0;
                ipoint--)
            {
                if(!accumulate_point(point_cluster,
                                     bitarray_visited[iring-iring0],
                                     &th_rad_last,
                                     points[iring],
                                     &segment_cluster->plane,
                                     ipoint))
                    break;
            }


            // I don't bother to look in rings that don't appear in the
            // segment_cluster. This will by contain not very much data (because
            // the pre-solve didn't find it), and won't be of much value
        }


        // Got a set of points. Fit a plane. This sets point_cluster->plane
        float fit_cost = fit_plane_into_points(point_cluster);
#warning FOR NOW I JUST RUN A SINGLE ITERATION
        return fit_cost;
    }

    return -1.0f;
}

static void point_segmentation(const point3f_t** points,
                               const int* Npoints)
{
    segment_t segments[Nrings*Nsegments_per_rotation] = {};

    if(dump)
        printf("# x y what z\n");


    // parsing complete. Do stuff
    for(int iring=0; iring<Nrings; iring++)
    {
        fit_plane_from_ring(// out
                            &segments[Nsegments_per_rotation*iring],
                            // in
                            points[iring], Npoints[iring],
                            iring == debug_iring);

        if(dump)
        {
            for(int i=0; i<Npoints[iring]; i++)
                printf("%f %f all %f\n",
                       points[iring][i].x, points[iring][i].y, points[iring][i].z);

            for(int i=0; i<Nsegments_per_rotation; i++)
            {
                if(!(segments[iring*Nsegments_per_rotation + i].v.x == 0 &&
                     segments[iring*Nsegments_per_rotation + i].v.y == 0 &&
                     segments[iring*Nsegments_per_rotation + i].v.z == 0))
                    printf("%f %f segment-ring-%02d %f\n",
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
    segment_clusters_from_segments(segment_clusters,
                                   &Nclusters,
                                   (int)(sizeof(segment_clusters)/sizeof(segment_clusters[0])),
                                   segments,
                                   Nrings, Nsegments_per_rotation);

    if(dump)
        for(int icluster=0; icluster<Nclusters; icluster++)
        {
            segment_cluster_t* segment_cluster = &segment_clusters[icluster];
            for(int i=0; i<segment_cluster->n; i++)
            {
                const int iring    = segment_cluster->segments[i].iring;
                const int isegment = segment_cluster->segments[i].isegment;

                segment_t* segment = &segments[iring*Nsegments_per_rotation + isegment];

                for(int ipoint=segment->ipoint0;
                    ipoint <= segment->ipoint1;
                    ipoint++)
                {
                    printf("%f %f cluster-points-raw-%d %f\n",
                           points[iring][ipoint].x,
                           points[iring][ipoint].y,
                           icluster,
                           points[iring][ipoint].z);
                }
            }
        }


    for(int icluster=0; icluster<Nclusters; icluster++)
    {
        segment_cluster_t* segment_cluster = &segment_clusters[icluster];

        point_cluster_t point_cluster;
        float fit_cost =
            refine_point_cluster_from_segment_cluster(&point_cluster,
                                                      segment_cluster,
                                                      segments,
                                                      Nrings, Nsegments_per_rotation,
                                                      points, Npoints);
        if(fit_cost >= 0.0f && // acceptable plane
#warning "made-up threshold"
           fit_cost < 1.0)
        {
            if(dump)
                for(int ipoint=0;
                    ipoint < point_cluster.n;
                    ipoint++)
                {
                    printf("%f %f cluster-points-refined-%d %f\n",
                                   point_cluster.p[ipoint].x,
                                   point_cluster.p[ipoint].y,
                                   icluster,
                                   point_cluster.p[ipoint].z);
                }
        }
    }
}

int main(void)
{
    // from dump-lidar-scan.py
    const char* filename = "/tmp/tst.dat";

    int fd = open(filename, O_RDONLY);
    if(fd <= 0)
    {
        MSG("Error opening '%s'\n", filename);
        return 1;
    }

    struct stat sb;
    int res = fstat(fd, &sb);
    if(res)
    {
        MSG("Error stat('%s')\n", filename);
        return 1;
    }
    if(sb.st_size == 0)
    {
        MSG("stat('%s') says that st_size == 0\n", filename);
        return 1;
    }

    char* data = mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if(data == MAP_FAILED)
    {
        MSG("mmap('%s') failed\n", filename);
        return 1;
    }

    int ibyte_data   = 0;
    int version_read = -1;
    int Nrings_read  = -1;
    while(version_read < 0 || Nrings_read < 0)
    {
        // ignore comments
        if(data[ibyte_data] == '#')
        {
            const char* newline = strchr(&data[ibyte_data], '\n');
            if(newline == NULL)
            {
                MSG("Data parsing failed");
                return 1;
            }
            ibyte_data = (int)(newline-data) + 1;
            continue;
        }

#define READ_INTEGER_FIELD(x,  key)                             \
        ({ int nbytes_read_here;                                \
           bool result = ( 1 == sscanf(&data[ibyte_data],       \
                                       key " = %d%n",           \
                                       x, &nbytes_read_here)) ; \
           if(result) {                                         \
               ibyte_data += nbytes_read_here;                  \
               ibyte_data += strspn(&data[ibyte_data], " "); /* ignore block of spaces */ \
               if(data[ibyte_data] != '\n') result = false;     \
               else ibyte_data++;                               \
           }                                                    \
           result;                                              \
        })

        if( !READ_INTEGER_FIELD(&version_read, "version") &&
            !READ_INTEGER_FIELD(&Nrings_read,  "Nrings") )
        {
            MSG("ERROR: not a comment or any of the known fields, but some known fields are still incomplete, so we still need to parse the header");
            return 1;
        }

#warning "INCOMPLETE PARSING. NEED TO CHECK THAT ibyte_data<sb.st_size always"
    }

    if(1 != version_read)
    {
        MSG("Currently I'm only accepting version=1 data, but this datafile has version = %d",
            version_read);
        return 1;
    }
    if(Nrings != Nrings_read)
    {
        MSG("Currently we're assuming a specific value of Nrings = %d, but this datafile has Nrings = %d",
            Nrings, Nrings_read);
        return 1;
    }
    // Done with the header. The file should be set up such that we're now at an
    // aligned location
    if(ibyte_data % 16)
    {
        MSG("ERROR: after reading the header we're not aligned at a multiple of 16. The data-file writing or parsing are buggy");
        return 1;
    }
    if((uintptr_t)data % 16)
    {
        MSG("ERROR: the data buffer isn't aligned. mmap() SHOULD return aligned pointers. Something is wrong buggy");
        return 1;
    }



    const point3f_t* points[Nrings];
    int Npoints[Nrings];


    for(int iring=0; iring<Nrings; iring++)
    {
#warning "INCOMPLETE PARSING. NEED TO CHECK THAT ibyte_data<sb.st_size always"


        // We extract an ascii string representing Npoints in this ring. This
        // lives in a 16-byte block
        char buf[17] = {};
        memcpy(buf, &data[ibyte_data], 16);
        ibyte_data += 16;

        if(1 != sscanf(buf, "%d", &Npoints[iring]))
        {
            MSG("Couldn't parse Npoints for iring=%d", iring);
            return 1;
        }

        const int Nbytes_remaining = sb.st_size - ibyte_data;
        if(Npoints[iring] * (int)sizeof(point3f_t) > Nbytes_remaining)
        {
            MSG("Not enough bytes in file for iring=%d", iring);
            return 1;
        }

        points[iring] = (const point3f_t*)&data[ibyte_data];

        ibyte_data += Npoints[iring] * sizeof(point3f_t);
    }
    if(sb.st_size != ibyte_data)
    {
        MSG("File has trailing bytes");
        return 1;
    }


    point_segmentation(points, Npoints);

    return 0;
}


/*
The plan:

report line segments: (p,v)

I subdivide space into az chunks and el

Then I find the largest connected component from each object

*/




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

*/
