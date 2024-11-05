#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stddef.h>
#include <limits.h>
#include <signal.h>
#include <sys/wait.h>

#include "clc.h"
#include "util.h"


static bool serialize( int fd_out,
                       // in
                       const clc_sensor_snapshot_t* sensor_snapshots,
                       const unsigned int           Nsensor_snapshots,
                       // The stride, in bytes, between each successive points or rings value
                       // in clc_lidar_scan_t
                       const unsigned int           lidar_packet_stride,

                       // These apply to ALL the sensor_snapshots[]
                       const unsigned int Ncameras,
                       const unsigned int Nlidars,

                       // bits indicating whether a camera in
                       // sensor_snapshots.images[] is color or not
                       const clc_is_bgr_mask_t is_bgr_mask)
{
    write(fd_out, &Ncameras, sizeof(Ncameras));
    write(fd_out, &Nlidars, sizeof(Nlidars));
    write(fd_out, &is_bgr_mask, sizeof(is_bgr_mask));

    for(unsigned int i_sensor_snapshot=0; i_sensor_snapshot<Nsensor_snapshots; i_sensor_snapshot++)
    {
        const clc_sensor_snapshot_t* sensor_snapshot = &sensor_snapshots[i_sensor_snapshot];

        write(fd_out,
              &sensor_snapshot->time_us_since_epoch,
              sizeof(sensor_snapshot->time_us_since_epoch));
        for(unsigned int i_camera=0; i_camera<Ncameras; i_camera++)
        {
            if(sensor_snapshot->images[i_camera].uint8.data != NULL)
            {
                write(fd_out,
                      &sensor_snapshot->images[i_camera].uint8,
                      3*sizeof(int));
                // 4 instead of 3 because of padding
                _Static_assert(sizeof(sensor_snapshot->images[i_camera].uint8) - sizeof(void*) == 4*sizeof(int),
                               "mrcal_image_..._t has expected structure");
                _Static_assert(offsetof(mrcal_image_uint8_t, data) == 4*sizeof(int),
                               "mrcal_image_..._t has expected structure");
                // any type. doesn't matter
                write(fd_out,
                      sensor_snapshot->images[i_camera].uint8.data,
                      sensor_snapshot->images[i_camera].uint8.stride *
                      sensor_snapshot->images[i_camera].uint8.height);
            }
            else
            {
                write(fd_out,
                      &(mrcal_image_uint8_t){},
                      3*sizeof(int));
            }
        }

        for(unsigned int i_lidar=0; i_lidar<Nlidars; i_lidar++)
        {
            const clc_lidar_scan_t* scan = &sensor_snapshot->lidar_scans[i_lidar];

            write(fd_out, &scan->Npoints, sizeof(scan->Npoints));

            for(unsigned int i_point=0; i_point<scan->Npoints; i_point++)
            {
                write(fd_out, &(((uint8_t*)(scan->points))[ lidar_packet_stride*i_point ]), sizeof(clc_point3f_t));
                write(fd_out, &(((uint8_t*)(scan->rings ))[ lidar_packet_stride*i_point ]), sizeof(uint16_t));
            }
        }
    }

    return true;
}

// Each sensor is uniquely identified by its position in the
// sensor_snapshots[].lidar_scans[] or .images[] arrays. An unobserved sensor in
// some sensor snapshot should be indicated by lidar_scans[] = {} or images[] =
// {}
//
// On output, the rt_ref_lidar[] and rt_ref_camera[] arrays will be filled-in.
// If solving for ALL the sensor geometry wasn't possible, we return false. On
// success, we return true
bool clc(// out
         mrcal_pose_t* rt_ref_lidar,  // Nlidars  of these to fill
         mrcal_pose_t* rt_ref_camera, // Ncameras of these to fill

         // in
         const clc_sensor_snapshot_t* sensor_snapshots,
         const unsigned int           Nsensor_snapshots,
         // The stride, in bytes, between each successive points or rings value
         // in clc_lidar_scan_t
         const unsigned int           lidar_packet_stride,

         // These apply to ALL the sensor_snapshots[]
         const unsigned int Ncameras,
         const unsigned int Nlidars,

         // bits indicating whether a camera in
         // sensor_snapshots.images[] is color or not
         const clc_is_bgr_mask_t is_bgr_mask)
{
    // Pass the data to fit.py via a pipe, read results from a different pipe

    bool result = false;
    int pipefd_in [2] = {-1,-1};
    int pipefd_out[2] = {-1,-1};

    pid_t pid = 0;


    if(0 != pipe(pipefd_in))
    {
        MSG("Couldn't create input pipe");
        goto done;
    }
    if(0 != pipe(pipefd_out))
    {
        MSG("Couldn't create output pipe");
        goto done;
    }

    pid = fork();
    if(pid < 0)
    {
        MSG("Couldn't fork()");
        goto done;
    }
    if(pid == 0)
    {
        // child
        close(STDOUT_FILENO);
        close(STDIN_FILENO);

        // read
        close(pipefd_in[1]);
        dup2(pipefd_in[0],STDIN_FILENO);
        close(pipefd_in[0]);

        // write
        close(pipefd_out[0]);
        dup2(pipefd_out[1],STDOUT_FILENO);
        close(pipefd_out[1]);





        ////////// test code: just echo the input
        execl("/bin/sh", "sh", "-c",
              "cat",
              NULL);
        _exit(0);







        execl("/bin/sh", "sh", "-c",
              "python3 fit.py ....",
              NULL);
        _exit(0);
    }
    else
    {
        // parent



        // read
        close(pipefd_in[0]);
        pipefd_in[0] = -1;

        close(pipefd_out[1]);
        pipefd_out[1] = -1;

        serialize(pipefd_in[1],
                  sensor_snapshots,
                  Nsensor_snapshots,
                  lidar_packet_stride,
                  Ncameras,
                  Nlidars,
                  is_bgr_mask);

        close(pipefd_in[1]);
        pipefd_in[1] = -1;



        //////// test code: echo the input
        char buf[PIPE_BUF];
        while(true)
        {
            int Nread = read(pipefd_out[0], buf, PIPE_BUF);
            if(Nread < 0)
            {
                fprintf(stderr, "ERROR!\n");
                exit(1);
            }
            if(Nread == 0)
            {
                // done
                break;
            }
            write(STDOUT_FILENO, buf, Nread);
        }
        close(pipefd_out[0]);
        pipefd_out[0] = -1;

        result = true;
        goto done;




        // wait();
        // deserialize(pipefd_out[0]);
        // close(pipefd_out[0]);


    }
 done:
    for(int i=0; i<2; i++)
    {
        if(pipefd_in [i] >= 0) close(pipefd_in [i]);
        if(pipefd_out[i] >= 0) close(pipefd_out[i]);
    }
    if(pid == 0)
    {
        // child
        _exit(0);
    }
    else if(pid > 0)
    {
        // parent
        kill(pid,9);
        wait(NULL);
    }
    return result;
}





#if 0
bool clc(// out
         mrcal_pose_t* rt_ref_lidar,  // Nlidars  of these to fill
         mrcal_pose_t* rt_ref_camera, // Ncameras of these to fill

         // in
         const clc_sensor_snapshot_t* sensor_snapshots,
         const unsigned int           Nsensor_snapshots,
         // The stride, in bytes, between each successive points or rings value
         // in clc_lidar_scan_t
         const unsigned int           lidar_packet_stride,

         // These apply to ALL the sensor_snapshots[]
         const unsigned int Ncameras,
         const unsigned int Nlidars,

         // bits indicating whether a camera in
         // sensor_snapshots.images[] is color or not
         const clc_is_bgr_mask_t is_bgr_mask)
{
    if(Ncameras != 0)
    {
        MSG("Only LIDAR-LIDAR calibrations implemented for now");
        return false;
    }

    for(int isnapshot=0; isnapshot < Nsensor_snapshots; isnapshot++)
    {
        const clc_sensor_snapshot_t* sensor_snapshot = &sensor_snapshots[isnapshot];

        for(int ilidar=0; ilidar<Nlidars; ilidar++)
        {
            const clc_lidar_scan_t* scan = &sensor_snapshot->lidar_scans[ilidar];

            // sort by ring and th
            scan->Npoints;
            scan->points;
            scan->rings;

            const clc_lidar_segmentation_context_t ctx = {};
            clc_points_and_plane_t points_and_plane[2];
            const int Nplanes_found =
                clc_lidar_segmentation(// out
                                       points_and_plane,
                                       // in
                                       (int)(sizeof(points_and_plane)/sizeof(points_and_plane[0])),
                                       // length sum(Npoints). Sorted by ring and then by
                                       // azimuth
                                       const clc_point3f_t* points,
                                       // length ctx->Nrings
                                       const unsigned int* Npoints,
                                       &ctx);



            // rings = np.unique(ring) # unique and sorted

            // # I need to sort by ring and then by th
            // th = np.arctan2( points[:,1], points[:,0] )
            // def points_from_rings():
            //     for iring in rings:
            //         idx = ring==iring
            //         yield points[idx][ np.argsort(th[idx]) ]

            // points_sorted = nps.glue( *points_from_rings(),
            //                           axis = -2 )

            // Npoints = np.array([np.count_nonzero(ring==iring) for iring in rings],
            //                    dtype = np.int32)

            // ipoint, plane_pn =
            //     _clc.lidar_segmentation(points  = points_sorted,
            //                             Npoints = Npoints,
            //                             Nrings  = len(Npoints),
            //                             **kwargs)
            // Nplanes = len(ipoint)






            // have sorted, segmented thing. Python has a list of segmented
            // point clouds



        }
    }
}
#endif
