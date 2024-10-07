#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>

#define MSG(fmt, ...) fprintf(stderr, "%s(%d): " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__)



static const int   threshold_min_Npoints_in_segment         = 10;
static const int   threshold_max_Npoints_invalid_segment    = 5;
static const float threshold_max_range                      = 5.f;
static const int   Npoints_per_rotation                     = 1809; // found empirically in dump-lidar-scan.py
static const int   Nsegments_per_rotation                   = 10;
static const int   threshold_max_Ngap                       = 2;
static const float threshold_max_deviation_off_segment_line = 0.05f;
static const int   Nrings                                   = 32;

typedef union
{
    struct
    {
        float x,y,z;
    };
    float xyz[3];
} point3f_t;

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



typedef struct
{
    point3f_t p; // the center
    point3f_t v; // a direction vector in the plane; may not be normalized
} plane_segment_t;


static void
dump_points(const point3f_t* p,
            const int Npoints,
            bool  dump_xyz // if false, dump xy only
            )
{
    for(int i=0; i<Npoints; i++)
        if(dump_xyz)
            printf("%f %f %f\n",
                   p[i].x, p[i].y, p[i].z);
        else
            printf("%f %f\n",
                   p[i].x, p[i].y);

}

static
float th_from_point(const point3f_t* p)
{
    return atan2f(p->y, p->x);
}


// ASSUMES th_rad CAME FROM atan2, SO IT'S IN [-pi,pi]
static
int isegment_from_th(const float th_rad)
{
    const float segment_width_rad = 2.0f*M_PI / (float)Nsegments_per_rotation;

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
bool point_is_valid(const point3f_t* p,
                    const float dth_rad)
{
    if(norm2(*p) > threshold_max_range*threshold_max_range)
        return false;


    const int Ngap = (int)( 0.5f + fabsf(dth_rad) * (float)Npoints_per_rotation / (2.0f*M_PI) );

    // Ngap==1 is the expected, normal value. Anything larger is a gap
    if((Ngap-1) > threshold_max_Ngap)
        return false;

    return true;
}

static
bool planar(const point3f_t* p,
            const int ipoint0,
            const int ipoint1)
{
    const point3f_t* p0 = &p[ipoint0];
    const point3f_t* p1 = &p[ipoint1];

    const point3f_t v01 = sub(*p1,*p0);

    const float recip_norm2_v01 = 1.f / norm2(v01);

    // do all the points in the chunk lie along v? If so, this is a linear
    // feature
    for(int ipoint=ipoint0+1; ipoint<=ipoint1; ipoint++)
    {
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
                    plane_segment_t* segment,
                    // in
                    const int Npoints_invalid_in_segment,
                    const point3f_t* p,
                    const int ipoint0,
                    const int ipoint1)
{
    const int Npoints = ipoint1 - ipoint0 + 1 - Npoints_invalid_in_segment;

    if(Npoints_invalid_in_segment > threshold_max_Npoints_invalid_segment ||
       Npoints < threshold_min_Npoints_in_segment ||
       !planar(p,ipoint0,ipoint1))
    {
        *segment = (plane_segment_t){};
        return;
    }

    MSG("finishing");
    segment->p = mean(p[ipoint1], p[ipoint0]);
    segment->v = sub( p[ipoint1], p[ipoint0]);
}


static void
fit_plane_from_ring(// out
                    plane_segment_t* segments,

                    // in
                    const point3f_t* p,
                    const int Npoints)
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
    float th_rad0 = th_from_point(p);

    int ipoint0   = 0;
    int isegment0 = isegment_from_th(th_rad0);
    int Npoints_invalid_in_segment = 0;
    float th_rad_prev = th_rad0;
    for(int ipoint=1; ipoint<Npoints; ipoint++)
    {
        const float th_rad = th_from_point(&p[ipoint]);
        const int isegment = isegment_from_th(th_rad);
        if(isegment != isegment0)
        {
            finish_segment(&segments[isegment0], Npoints_invalid_in_segment,
                           p, ipoint0, ipoint-1);

            ipoint0   = ipoint;
            isegment0 = isegment;
            Npoints_invalid_in_segment = 0;
        }

        if(!point_is_valid(&p[ipoint], th_rad - th_rad_prev))
            Npoints_invalid_in_segment++;

        th_rad_prev = th_rad;
    }
    finish_segment(&segments[isegment0], Npoints_invalid_in_segment,
                   p, ipoint0, Npoints-1);
}





#if 0
segment()
{

    const int min_points_in_plane_ring = 20;

    for(rings)
    {
        for(th)
        {
            accumulate_segment();
        }
    }

    // I have a list of planar/linear segments in each ring

    return 0;
}
#endif

int main(void)
{
    // from dump-lidar-scan.py
    const char* filename_fmt = "/tmp/tst-%02d.dat";


    plane_segment_t segments[Nrings*Nsegments_per_rotation] = {};

    for(int iring=0; iring<Nrings; iring++)
    {
        char filename[32];
        if((int)sizeof(filename) <= snprintf(filename, sizeof(filename),
                                             filename_fmt, iring))
        {
            MSG("Error: increase the size of filename[]");
            return 1;
        }

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

        const int Npoints = (int)(sb.st_size / sizeof(point3f_t));
        if(sb.st_size != Npoints*(int)sizeof(point3f_t))
        {
            MSG("size('%s') isn't an integer number of 3D points\n", filename);
            return 1;
        }

        point3f_t* p = (point3f_t*)mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if(p == MAP_FAILED)
        {
            MSG("mmap('%s') failed\n", filename);
            return 1;
        }

        fit_plane_from_ring(// out
                            &segments[Nsegments_per_rotation*iring],
                            // in
                            p, Npoints);
    }

    for(int i=0; i<(int)(sizeof(segments)/sizeof(segments[0])); i++)
    {
        if(!(segments[i].v.x == 0 &&
             segments[i].v.y == 0 &&
             segments[i].v.z == 0))
            printf("%f %f %f\n",
                   segments[i].p.x,
                   segments[i].p.y,
                   segments[i].p.z);
    }

    return 0;
}


/*
The plan:

report line segments: (p,v)

I subdivide space into az chunks and el

Then I find the largest connected component from each object

*/




/*

segment finder use missing points or too-long ranges as breaks

check for conic sections, not just line segments

I already flag too many invalid points total in a segment. I should have a
separate metric to flag too many in a contiguous block

Make a note that the initial segment finder is crude. It does not try to work on
the edges at all: it sees a gap or a switch to another object, and it throws out
the entire segment
*/
