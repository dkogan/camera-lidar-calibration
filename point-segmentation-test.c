#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>

#include "point_segmentation.h"




static const int   Nrings = 32;





int main(int argc, char* argv[])
{
    const char* usage =
#include "point_segmentation.usage.h"
        ;

    struct option opts[] = {
        { "help",              no_argument,       NULL, 'h' },
        {}
    };

    int opt;
    do
    {
        // "h" means -h does something
        opt = getopt_long(argc, argv, "+h", opts, NULL);
        switch(opt)
        {
        case -1:
            break;

        case 'h':
            printf(usage, argv[0]);
            return 0;
        case '?':
            fprintf(stderr, "Unknown option\n\n");
            fprintf(stderr, usage, argv[0]);
            return 1;
        }
    } while( opt != -1 );

    int Nargs_remaining = argc-optind;
    if( Nargs_remaining != 1 )
    {
        fprintf(stderr, "Need exactly 1 non-option argument. Got %d\n\n",Nargs_remaining);
        fprintf(stderr, usage, argv[0]);
        return 1;
    }




#if 0
    // from dump-lidar-scan.py
    const char* filename = argv[optind+0];
    const point3f_t* points[Nrings];
    int Npoints            [Nrings];


    if(!point_segmentation__parse_input_file(points,Npoints,
                                             filename))
        return 1;

    point_segmentation(points, Npoints);
#endif
    printf("THIS TOOL DOESN'T WORK CURRENTLY. MAKE point_segmentation__parse_input_file() work again\n");

    return 0;
}
