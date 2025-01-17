#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include <stdbool.h>
#include <Python.h>
#include <structmember.h>
#include <numpy/arrayobject.h>
// Required for numpy 2. They now #include complex.h, so I is #defined to be the
// complex I, which conflicts with my usage here
#undef I

#include <signal.h>

#define IS_NULL(x) ((x) == NULL || (PyObject*)(x) == Py_None)

#define BARF(fmt, ...) PyErr_Format(PyExc_RuntimeError, "%s:%d %s(): "fmt, __FILE__, __LINE__, __func__, ## __VA_ARGS__)

// Python is silly. There's some nuance about signal handling where it sets a
// SIGINT (ctrl-c) handler to just set a flag, and the python layer then reads
// this flag and does the thing. Here I'm running C code, so SIGINT would set a
// flag, but not quit, so I can't interrupt the solver. Thus I reset the SIGINT
// handler to the default, and put it back to the python-specific version when
// I'm done
#define SET_SIGINT() struct sigaction sigaction_old;                    \
do {                                                                    \
    if( 0 != sigaction(SIGINT,                                          \
                       &(struct sigaction){ .sa_handler = SIG_DFL },    \
                       &sigaction_old) )                                \
    {                                                                   \
        BARF("sigaction() failed");      \
        goto done;                                                      \
    }                                                                   \
} while(0)
#define RESET_SIGINT() do {                                             \
    if( 0 != sigaction(SIGINT,                                          \
                       &sigaction_old, NULL ))                          \
        BARF("sigaction-restore failed"); \
} while(0)



#include <stddef.h>

#include "clc.h"

#define PYMETHODDEF_ENTRY(name, c_function_name, args) {#name,          \
                                                        (PyCFunction)c_function_name, \
                                                        args,           \
                                                        name ## _docstring}


static bool lidar_scan_from_points_rings(// out
                                         clc_lidar_scan_unsorted_t* scan,
                                         unsigned int* lidar_packet_stride,
                                         // in
                                         PyArrayObject* points,
                                         PyArrayObject* rings)
{
    if(! (PyArray_TYPE(points) == NPY_FLOAT32 &&
          PyArray_NDIM(points) == 2 &&
          PyArray_DIMS(points)[1] == 3 &&
          PyArray_STRIDES(points)[1] == sizeof(float)) )
    {
        PyErr_SetString(PyExc_RuntimeError, "'points' must be an array of shape (N,3) containing 32-bit floats, each xyz stored densely");
        return false;
    }

    const int Npoints_total = PyArray_DIMS   (points)[0];
    *lidar_packet_stride    = PyArray_STRIDES(points)[0];

    if(! (PyArray_TYPE(rings) == NPY_UINT16 &&
          PyArray_NDIM(rings) == 1) )
    {
        PyErr_Format(PyExc_RuntimeError,
                     "'rings' must be a 1-dimensional array containing uint16");
        return false;
    }

    if(! (PyArray_DIMS(rings)[0] == Npoints_total) )
    {
        PyErr_Format(PyExc_RuntimeError,
                     "'rings' and 'points' must describe the same number of points; instead len(points)=%d, but len(rings)=%d",
                     Npoints_total, PyArray_DIMS(rings)[0]);
        return false;
    }
    if(! (*lidar_packet_stride == PyArray_STRIDES(rings)[0]) )
    {
        PyErr_Format(PyExc_RuntimeError,
                     "'rings' and 'points' must have the same point stride; instead points.stride[0]=%d, but rings.stride[0]=%d",
                     *lidar_packet_stride,
                     PyArray_STRIDES(rings)[0]);
        return false;
    }


    *scan = (clc_lidar_scan_unsorted_t)
        {.points  = (clc_point3f_t*)PyArray_DATA(points),
         .rings   = (uint16_t     *)PyArray_DATA(rings),
         .Npoints = Npoints_total};

    return true;
}

static PyObject* py_lidar_segmentation(PyObject* NPY_UNUSED(self),
                                       PyObject* args,
                                       PyObject* kwargs)
{
    PyObject* result = NULL;

    PyArrayObject* points    = NULL;
    PyArrayObject* rings     = NULL;

    // output
    PyArrayObject* plane_pn  = NULL;
    PyObject*      py_ipoint = NULL;

    const int Nplanes_max = 16;
    PyArrayObject* ipoint[Nplanes_max];
    memset(ipoint, 0, Nplanes_max*sizeof(ipoint[0]));

#warning "I should define a complex dtype to pass points_and_plane from a preallocated numpy array. Instead I do this in C and then copy the results. For now"
#warning "possibly this is too large"
    clc_points_and_plane_t points_and_plane[Nplanes_max];

    clc_lidar_segmentation_context_t ctx;
    clc_lidar_segmentation_default_context(&ctx);

    SET_SIGINT();

#define CLC_LIDAR_SEGMENTATION_LIST_CONTEXT_KEYWORDS(   type,name,default,pyparse,...) #name,
#define CLC_LIDAR_SEGMENTATION_LIST_CONTEXT_PYPARSE(    type,name,default,pyparse,...) pyparse
#define CLC_LIDAR_SEGMENTATION_LIST_CONTEXT_ADDRESS_CTX(type,name,default,pyparse,...) &ctx.name,
    char* keywords[] = { "points",
                         "rings",
                         CLC_LIDAR_SEGMENTATION_LIST_CONTEXT(CLC_LIDAR_SEGMENTATION_LIST_CONTEXT_KEYWORDS)
                         NULL };
    if(!PyArg_ParseTupleAndKeywords( args, kwargs,
                                     "|$" "O&O&" CLC_LIDAR_SEGMENTATION_LIST_CONTEXT(CLC_LIDAR_SEGMENTATION_LIST_CONTEXT_PYPARSE)
                                     ,
                                     keywords,
                                     PyArray_Converter, &points,
                                     PyArray_Converter, &rings,
                                     CLC_LIDAR_SEGMENTATION_LIST_CONTEXT(CLC_LIDAR_SEGMENTATION_LIST_CONTEXT_ADDRESS_CTX)
                                     NULL))
        goto done;

#undef CLC_LIDAR_SEGMENTATION_LIST_CONTEXT_KEYWORDS
#undef CLC_LIDAR_SEGMENTATION_LIST_CONTEXT_PYPARSE
#undef CLC_LIDAR_SEGMENTATION_LIST_CONTEXT_ADDRESS_CTX

    clc_lidar_scan_unsorted_t scan;
    unsigned int lidar_packet_stride;
    if(!lidar_scan_from_points_rings(// out
                                     &scan,
                                     &lidar_packet_stride,
                                     // in
                                     points,
                                     rings))
        goto done;

    int8_t Nplanes =
        clc_lidar_segmentation_unsorted( // out
                                         points_and_plane,
                                         // in
                                         Nplanes_max,
                                         &scan,
                                         lidar_packet_stride,
                                         &ctx);
    if(Nplanes < 0)
        goto done;

    // Success. Allocate the output and copy
    for(int i=0; i<Nplanes; i++)
    {
        ipoint[i] = (PyArrayObject*)PyArray_SimpleNew(1, ((npy_intp[]){points_and_plane[i].n}), NPY_UINT32);
        if(ipoint[i] == NULL)
            goto done;

        memcpy(PyArray_DATA(ipoint[i]), points_and_plane[i].ipoint, sizeof(points_and_plane[i].ipoint[0])*points_and_plane[i].n);
    }

    plane_pn = (PyArrayObject*)PyArray_SimpleNew(2, ((npy_intp[]){Nplanes,6}), NPY_FLOAT32);
    if(plane_pn == NULL)
        goto done;
    static_assert(offsetof(clc_plane_t,p_mean) == 0 && offsetof(clc_plane_t,n) == sizeof(clc_point3f_t) && sizeof(clc_plane_t) == 2*sizeof(clc_point3f_t),
                  "clc_plane_t is expected to densely store p and then n");
    for(int i=0; i<Nplanes; i++)
        memcpy( &((float*)PyArray_DATA(plane_pn))[6*i], &points_and_plane[i].plane, sizeof(points_and_plane[i].plane));

    py_ipoint = PyTuple_New(Nplanes);
    if(py_ipoint == NULL) goto done;
    for(int i=0; i<Nplanes; i++)
    {
        PyTuple_SET_ITEM(py_ipoint, i, ipoint[i]);
        Py_INCREF(ipoint[i]);
    }

    result = Py_BuildValue("OO",
                           py_ipoint,
                           plane_pn);

 done:
    Py_XDECREF(points);
    Py_XDECREF(rings);
    for(int i=0; i<Nplanes_max; i++)
        Py_XDECREF(ipoint[i]);
    Py_XDECREF(py_ipoint);
    Py_XDECREF(plane_pn);

    RESET_SIGINT();

    return result;
}

static bool ingest_camera_snapshot(// out
                                   clc_sensor_snapshot_unsorted_t* snapshot,
                                   int* Ncameras,
                                   clc_is_bgr_mask_t* is_bgr_mask,
                                   // in
                                   const PyObject* py_snapshot)
{
    PyObject* py_images = PyTuple_GET_ITEM(py_snapshot, 1);

    if(py_images == Py_None)
    {
        memset(snapshot->images, 0, sizeof(snapshot->images));
        *Ncameras = 0;
        return true;
    }

    int Ncameras_here;
    if(!PyTuple_Check(py_images))
    {
        BARF("Each sensor_snapshot[0] should be a tuple of images for each camera, or None if that camera had no observations");
        return false;
    }
    if(0 > (Ncameras_here = PyTuple_Size(py_images)))
    {
        BARF("Couldn't get Ncameras_here");
        return false;
    }
    if(*Ncameras < 0)
        *Ncameras = Ncameras_here;
    else
    {
        if(*Ncameras != Ncameras_here)
        {
            BARF("Inconsistent Ncameras; saw %d and %d",
                 *Ncameras, Ncameras_here);
            return false;
        }
    }
    if(*Ncameras == 0)
        return true;

    if(*Ncameras > clc_Ncameras_max)
    {
        BARF("Number of cameras give exceeds the static limit of clc_Ncameras_max=%d. Raise clc_Ncameras_max",
             clc_Ncameras_max);
        return false;
    }

    for(int i=0; i<*Ncameras; i++)
    {
        PyArrayObject* py_image = (PyArrayObject*)PyTuple_GET_ITEM(py_images, i);
        if((PyObject*)py_image == Py_None)
        {
            // Zero out the image structure to indicate an error. The image data
            // type doesn't matter here
            snapshot->images[i].uint8 = (mrcal_image_uint8_t){};
            continue;
        }

        if(!(PyArray_TYPE(py_image) == NPY_UINT8 &&
             PyArray_STRIDES(py_image)[PyArray_NDIM(py_image) - 1] == sizeof(uint8_t)))
        {
            PyErr_SetString(PyExc_RuntimeError, "'py_image' must be an array containing 8-bit uint, each pixel stored densely");
            return false;
        }

        if(PyArray_NDIM(py_image) == 2)
        {
            // mono8
            (snapshot->images[i].uint8) =
                (mrcal_image_uint8_t){.width  = PyArray_DIM   (py_image,1),
                                      .height = PyArray_DIM   (py_image,0),
                                      .stride = PyArray_STRIDE(py_image,0),
                                      .data   = PyArray_DATA  (py_image) };

        }
        else if(PyArray_NDIM(py_image) == 3)
        {
            // bgr8
            if(!(PyArray_STRIDES(py_image)[PyArray_NDIM(py_image) - 2] == sizeof(uint8_t)*3 &&
                 PyArray_DIMS   (py_image)[PyArray_NDIM(py_image) - 1] == 3))
            {
                PyErr_SetString(PyExc_RuntimeError, "'py_image' looks like a color array, but those MUST have shape (H,W,3) with dense bgr tuples");
                return false;
            }

            (snapshot->images[i].bgr) =
                (mrcal_image_bgr_t){.width  = PyArray_DIM   (py_image,1),
                                    .height = PyArray_DIM   (py_image,0),
                                    .stride = PyArray_STRIDE(py_image,0),
                                    .data   = PyArray_DATA  (py_image) };

            *is_bgr_mask |= (1U << i);
        }
        else
        {
            PyErr_SetString(PyExc_RuntimeError, "The given 'py_image' is neither a known mono8 or bgr8 format: must have len(shape)==2 or 3");
            return false;
        }
    }
    return true;
}

static bool ingest_lidar_scans(// out
                               clc_lidar_scan_unsorted_t* lidar_scans,
                               int* Nlidars,
                               unsigned int* lidar_packet_stride,
                               // in
                               PyObject* py_lidar_scans)
{
    if(!PyTuple_Check(py_lidar_scans))
    {
        BARF("Each sensor_snapshot[1] should be a tuple of py_lidar_scans for each lidar, or None if that lidar had no observations");
        return false;
    }

    int Nlidars_here;
    if(0 > (Nlidars_here = PyTuple_Size(py_lidar_scans)))
    {
        BARF("Couldn't get Nlidars_here");
        return false;
    }
    if(*Nlidars < 0)
        *Nlidars = Nlidars_here;
    else
    {
        if(*Nlidars != Nlidars_here)
        {
            BARF("Inconsistent Nlidars; saw %d and %d",
                 *Nlidars, Nlidars_here);
            return false;
        }
    }
    if(*Nlidars == 0)
        return true;

    if(*Nlidars > clc_Nlidars_max)
    {
        BARF("Number of lidars give exceeds the static limit of clc_Nlidars_max=%d. Raise clc_Nlidars_max",
             clc_Nlidars_max);
        return false;
    }

    for(int i=0; i<*Nlidars; i++)
    {
        clc_lidar_scan_unsorted_t* scan = &lidar_scans[i];

        PyObject* py_lidar_scan = PyTuple_GET_ITEM(py_lidar_scans, i);
        if(py_lidar_scan == Py_None)
        {
            *scan = (clc_lidar_scan_unsorted_t){};
            continue;
        }

        if(!PyTuple_Check(py_lidar_scan))
        {
            BARF("Each sensor_snapshot[1][i] should be a tuple (points,rings), or None if that lidar had no observations");
            return false;
        }
        int tuple_size;
        if(0 > (tuple_size = PyTuple_Size(py_lidar_scan)))
        {
            BARF("Couldn't get len(py_lidar_scan)");
            return false;
        }
        if(tuple_size != 2)
        {
            BARF("Each sensor_snapshot[1][i] should be a tuple (points,rings), or None if that lidar had no observations; tuple has the wrong length");
            return false;
        }

        PyArrayObject* points = (PyArrayObject*)PyTuple_GET_ITEM(py_lidar_scan, 0);
        PyArrayObject* rings  = (PyArrayObject*)PyTuple_GET_ITEM(py_lidar_scan, 1);
        if(!PyArray_Check(points))
        {
            BARF("Each sensor_snapshot[1][i] should be a tuple (points,rings), or None if that lidar had no observations; points is not a numpy array");
            return false;
        }
        if(!PyArray_Check(rings))
        {
            BARF("Each sensor_snapshot[1][i] should be a tuple (points,rings), or None if that lidar had no observations; points is not a numpy array");
            return false;
        }

        unsigned int lidar_packet_stride_here;
        if(!lidar_scan_from_points_rings(// out
                                         scan,
                                         &lidar_packet_stride_here,
                                         // in
                                         points,
                                         rings))
        {
            return false;
        }

        if(*lidar_packet_stride == 0)
            *lidar_packet_stride = lidar_packet_stride_here;
        else
        {
            if(*lidar_packet_stride != lidar_packet_stride_here)
            {
                BARF("Inconsistent lidar_packet_stride; saw %u and %u",
                     *lidar_packet_stride, lidar_packet_stride_here);
                return false;
            }
        }
    }
    return true;
}

static bool ingest_lidar_snapshot(// out
                                  clc_sensor_snapshot_unsorted_t* snapshot,
                                  int* Nlidars,
                                  unsigned int* lidar_packet_stride,
                                  // in
                                  const PyObject* py_snapshot)
{
    return ingest_lidar_scans(// out
                              snapshot->lidar_scans,
                              Nlidars,
                              lidar_packet_stride,
                              // in
                              PyTuple_GET_ITEM(py_snapshot, 0));
}

static PyObject* py_calibrate(PyObject* NPY_UNUSED(self),
                              PyObject* args,
                              PyObject* kwargs)
{
    PyObject* result = NULL;

    PyTupleObject* py_sensor_snapshots                = NULL;
    PyObject*      py_models                          = NULL;
    PyArrayObject* rt_ref_lidar                       = NULL;
    PyArrayObject* rt_ref_camera                      = NULL;
    PyArrayObject* Var_rt_lidar0_sensor               = NULL;
    PyObject*      inputs_dump                        = NULL;

    // used if(dump_optimization_inputs)
    char*  buf_inputs_dump  = NULL;
    size_t size_inputs_dump = NULL;

    PyObject* py_model = NULL;

    SET_SIGINT();

    // uninitialized
    unsigned int lidar_packet_stride = 0;
    int          Ncameras            = -1;
    int          Nlidars             = -1;
    int          Nsensor_snapshots   = -1;

    mrcal_cameramodel_t* models[clc_Ncameras_max] = {};

    int    object_height_n = -1;
    int    object_width_n  = -1;
    double object_spacing  = -1.;

    int check_gradient__use_distance_to_plane = 0;
    int check_gradient                        = 0;
    int dump_optimization_inputs              = 0;

    // sensor_snapshots is a tuple. Each slice corresponds to
    // clc_sensor_snapshot_unsorted_t; it is a tuple:
    //
    // - lidar_scans (an tuple, each element corresponding to clc_lidar_scan_unsorted_t)
    // - images      (an tuple, each element corresponding to a mrcal_image_uint8_t)
    //
    // Ncameras and Nlidars must be consistent across all sensor snapshots. The
    // data stride inside each lidar scan is read into lidar_packet_stride, and
    // must be consistent across all snapshots


    clc_lidar_segmentation_context_t ctx;
    clc_lidar_segmentation_default_context(&ctx);

#define CLC_LIDAR_SEGMENTATION_LIST_CONTEXT_KEYWORDS(   type,name,default,pyparse,...) #name,
#define CLC_LIDAR_SEGMENTATION_LIST_CONTEXT_PYPARSE(    type,name,default,pyparse,...) pyparse
#define CLC_LIDAR_SEGMENTATION_LIST_CONTEXT_ADDRESS_CTX(type,name,default,pyparse,...) &ctx.name,
    char* keywords[] = { "sensor_snapshots",
                         "models",
                         "object_height_n",
                         "object_width_n",
                         "object_spacing",
                         "check_gradient__use_distance_to_plane",
                         "check_gradient",
                         "dump_optimization_inputs",
                         CLC_LIDAR_SEGMENTATION_LIST_CONTEXT(CLC_LIDAR_SEGMENTATION_LIST_CONTEXT_KEYWORDS)
                         NULL };
    if(!PyArg_ParseTupleAndKeywords( args, kwargs,
                                     "O" "|$" "Oiidppp" CLC_LIDAR_SEGMENTATION_LIST_CONTEXT(CLC_LIDAR_SEGMENTATION_LIST_CONTEXT_PYPARSE)
                                     ,
                                     keywords,
                                     (PyTupleObject*)&py_sensor_snapshots,
                                     &py_models,
                                     &object_height_n,
                                     &object_width_n,
                                     &object_spacing,
                                     &check_gradient__use_distance_to_plane,
                                     &check_gradient,
                                     &dump_optimization_inputs,
                                     CLC_LIDAR_SEGMENTATION_LIST_CONTEXT(CLC_LIDAR_SEGMENTATION_LIST_CONTEXT_ADDRESS_CTX)
                                     NULL))
        goto done;

#undef CLC_LIDAR_SEGMENTATION_LIST_CONTEXT_KEYWORDS
#undef CLC_LIDAR_SEGMENTATION_LIST_CONTEXT_PYPARSE
#undef CLC_LIDAR_SEGMENTATION_LIST_CONTEXT_ADDRESS_CTX

    if(!PyTuple_Check((PyObject*)py_sensor_snapshots))
    {
        BARF("sensor_snapshots should be a tuple");
        goto done;
    }
    if(0 > (Nsensor_snapshots = PyTuple_Size((PyObject*)py_sensor_snapshots)))
    {
        BARF("Couldn't get len(sensor_snapshots)");
        goto done;
    }

    clc_is_bgr_mask_t is_bgr_mask = 0;

    {
        clc_sensor_snapshot_unsorted_t sensor_snapshots[Nsensor_snapshots];

        for(int i=0; i<Nsensor_snapshots; i++)
        {
            PyObject*                       py_snapshot = PyTuple_GET_ITEM(py_sensor_snapshots, i);
            clc_sensor_snapshot_unsorted_t* snapshot    = &sensor_snapshots[i];

            if(!PyTuple_Check(py_snapshot))
            {
                BARF("Each sensor_snapshot should be a tuple");
                goto done;
            }
            if(2 != PyTuple_Size(py_snapshot))
            {
                BARF("Each sensor_snapshot should be a tuple (images,lidar_scans), but the given tuple has %d elements",
                     PyTuple_Size(py_snapshot));
                goto done;
            }

            if(!ingest_lidar_snapshot (snapshot, &Nlidars, &lidar_packet_stride,
                                       py_snapshot))
                goto done;
            if(!ingest_camera_snapshot(snapshot, &Ncameras, &is_bgr_mask,
                                       py_snapshot))
                goto done;
        }

        if(Ncameras > 0)
        {
            if(0 >= object_height_n || 0 >= object_width_n || 0. >= object_spacing)
            {
                BARF("We have Ncameras=%d, so object_height_n and object_width_n and object_spacing must all be given and be >0",
                     Ncameras);
                goto done;
            }

            if(!PySequence_Check(py_models))
            {
                BARF("We have Ncameras=%d, so models must be an iterable of corresponding models",
                     Ncameras);
                goto done;
            }
            if(Ncameras != PySequence_Size(py_models))
            {
                BARF("We have Ncameras=%d, so models must be an iterable of the same length containing corresponding models",
                     Ncameras);
                goto done;
            }

            for(int i=0; i<Ncameras; i++)
            {
                py_model = PySequence_GetItem(py_models, i);
                if(py_model == NULL)
                {
                    BARF("Couldn't get the %d-th model", i);
                    goto done;
                }
                const char* filename = PyUnicode_AsUTF8(py_model);
                if(filename == NULL)
                {
                    BARF("Couldn't get filename from models argument");
                    goto done;
                }
                models[i] = mrcal_read_cameramodel_file(filename);
                if(models[i] == NULL)
                {
                    BARF("Couldn't read mrcal_cameramodel_t from '%s'", filename);
                    goto done;
                }
                Py_DECREF(py_model);
                py_model = NULL;
            }
        }


        rt_ref_lidar = (PyArrayObject*)PyArray_SimpleNew(2, ((npy_intp[]){Nlidars,6}), NPY_FLOAT64);
        if(rt_ref_lidar == NULL) goto done;

        rt_ref_camera = (PyArrayObject*)PyArray_SimpleNew(2, ((npy_intp[]){Ncameras,6}), NPY_FLOAT64);
        if(rt_ref_camera == NULL) goto done;

        const int Nsensors_optimized = Nlidars-1 + Ncameras;
        Var_rt_lidar0_sensor = (PyArrayObject*)PyArray_SimpleNew(4,
                                                                 ((npy_intp[]){Nsensors_optimized, 6,
                                                                               Nsensors_optimized, 6,}),
                                                                 NPY_FLOAT64);
        if(Var_rt_lidar0_sensor == NULL) goto done;

        static_assert(offsetof(mrcal_pose_t,r) == 0 && offsetof(mrcal_pose_t,t) == sizeof(double)*3,
                      "mrcal_pose_t should be a dense rt transform");

        if(!clc_unsorted(// out
                         (mrcal_pose_t*)PyArray_DATA(rt_ref_lidar),
                         (mrcal_pose_t*)PyArray_DATA(rt_ref_camera),
                         (double      *)PyArray_DATA(Var_rt_lidar0_sensor),
                         dump_optimization_inputs ? &buf_inputs_dump  : NULL,
                         dump_optimization_inputs ? &size_inputs_dump : NULL,

                         // in
                         sensor_snapshots,
                         Nsensor_snapshots,
                         lidar_packet_stride,
                         Nlidars,
                         Ncameras,
                         (const mrcal_cameramodel_t*const*)models,
                         object_height_n,
                         object_width_n,
                         object_spacing,
                         is_bgr_mask,
                         &ctx,
                         check_gradient__use_distance_to_plane,
                         check_gradient))
        {
            BARF("clc_unsorted() failed");
            goto done;
        }
    }

    if(buf_inputs_dump == NULL)
        result = Py_BuildValue("{sOsOsO}",
                               "rt_ref_lidar",  rt_ref_lidar,
                               "rt_ref_camera", rt_ref_camera,
                               "Var",           Var_rt_lidar0_sensor);
    else
    {
        inputs_dump =
            PyBytes_FromStringAndSize(buf_inputs_dump, size_inputs_dump);
        if(inputs_dump == NULL)
        {
            BARF("PyBytes_FromStringAndSize(buf_inputs_dump) failed");
            goto done;
        }
        result = Py_BuildValue("{sOsOsOsO}",
                               "rt_ref_lidar",  rt_ref_lidar,
                               "rt_ref_camera", rt_ref_camera,
                               "Var",           Var_rt_lidar0_sensor,
                               "inputs_dump",   inputs_dump);
    }

 done:
    Py_XDECREF(rt_ref_lidar);
    Py_XDECREF(rt_ref_camera);
    Py_XDECREF(Var_rt_lidar0_sensor);
    Py_XDECREF(py_model);
    for(int i=0; i<Ncameras; i++)
        mrcal_free_cameramodel(&models[i]);
    if(buf_inputs_dump != NULL)
        free(buf_inputs_dump);
    Py_XDECREF(inputs_dump);
    RESET_SIGINT();

    return result;
}


// just like PyArray_Converter(), but leave None as None
static
int PyArray_Converter_leaveNone(PyObject* obj, PyObject** address)
{
    if(obj == Py_None)
    {
        *address = Py_None;
        Py_INCREF(Py_None);
        return 1;
    }
    return PyArray_Converter(obj,address);
}

static PyObject* py_post_solve_statistics(PyObject* NPY_UNUSED(self),
                                          PyObject* args,
                                          PyObject* kwargs)
{
    PyObject* result = NULL;

    // out
    PyArrayObject* isvisible_per_sensor_per_sector = NULL;
    PyArrayObject* stdev_worst                     = NULL;
    PyArrayObject* isensors_pair_stdev_worst       = NULL;

    // out,in
    PyArrayObject* rt_ref_lidar  = NULL;
    PyArrayObject* rt_ref_camera = NULL;

    // in
    PyArrayObject* Var_rt_lidar0_sensor             = NULL;
    PyArrayObject* rt_vehicle_lidar0                = NULL;
    PyObject*      py_lidar_scans                   = NULL;
    PyObject*      py_models                        = NULL;
    int            Nsectors                         = 36;
    double         threshold_valid_lidar_range      = 1.0;
    int            threshold_valid_lidar_Npoints    = 100;
    double         uncertainty_quantification_range = 10;

    PyObject* py_model = NULL;

    SET_SIGINT();

    // uninitialized
    unsigned int lidar_packet_stride = 0;
    int          Ncameras            = 0;
    int          Nlidars             = -1;

    mrcal_cameramodel_t*      models[clc_Ncameras_max] = {};
    clc_lidar_scan_unsorted_t lidar_scans[clc_Nlidars_max];

    // lidar_scans is tuple, each element corresponding to
    // clc_lidar_scan_unsorted_t
    char* keywords[] = { "lidar_scans",
                         "Nsectors",
                         "threshold_valid_lidar_range",
                         "threshold_valid_lidar_Npoints",
                         "uncertainty_quantification_range",
                         "Var",
                         "rt_ref_lidar",
                         "rt_vehicle_lidar0",
                         "rt_ref_camera",
                         "models",
                         NULL };
    if(!PyArg_ParseTupleAndKeywords( args, kwargs,
                                     "|$" "OididO&O&O&O&O",
                                     keywords,

                                     &py_lidar_scans,
                                     &Nsectors,
                                     &threshold_valid_lidar_range,
                                     &threshold_valid_lidar_Npoints,
                                     &uncertainty_quantification_range,
                                     PyArray_Converter,           &Var_rt_lidar0_sensor,
                                     PyArray_Converter,           &rt_ref_lidar,
                                     PyArray_Converter,           &rt_vehicle_lidar0,
                                     PyArray_Converter_leaveNone, &rt_ref_camera,
                                     &py_models,
                                     NULL))
    {
        goto done;
    }

    if(py_lidar_scans == NULL)
    {
        BARF("lidar_scans must be given");
        goto done;
    }
    if(Nsectors <= 0)
    {
        BARF("I must be given Nsectors>0");
        goto done;
    }
    if(Var_rt_lidar0_sensor == NULL)
    {
        BARF("Var must be given");
        goto done;
    }
    if(rt_ref_lidar == NULL)
    {
        BARF("rt_ref_lidar must be given");
        goto done;
    }
    if(rt_vehicle_lidar0 == NULL)
    {
        BARF("rt_vehicle_lidar0 must be given");
        goto done;
    }

    if(!ingest_lidar_scans(lidar_scans, &Nlidars, &lidar_packet_stride,
                           py_lidar_scans))
        goto done;

    if(!(py_models == NULL || py_models == Py_None))
    {
        Ncameras = PySequence_Size(py_models);
        if(!PySequence_Check(py_models))
        {
            BARF("The given models must be an iterable");
            goto done;
        }

        for(int i=0; i<Ncameras; i++)
        {
            py_model = PySequence_GetItem(py_models, i);
            if(py_model == NULL)
            {
                BARF("Couldn't get the %d-th model", i);
                goto done;
            }
            const char* filename = PyUnicode_AsUTF8(py_model);
            if(filename == NULL)
            {
                BARF("Couldn't get filename from models argument");
                goto done;
            }
            models[i] = mrcal_read_cameramodel_file(filename);
            if(models[i] == NULL)
            {
                BARF("Couldn't read mrcal_cameramodel_t from '%s'", filename);
                goto done;
            }
            Py_DECREF(py_model);
            py_model = NULL;
        }
    }

    if(! (PyArray_IS_C_CONTIGUOUS(rt_ref_lidar) &&
          PyArray_NDIM(rt_ref_lidar)    == 2 &&
          PyArray_DIMS(rt_ref_lidar)[0] == Nlidars &&
          PyArray_DIMS(rt_ref_lidar)[1] == 6 &&
          PyArray_TYPE(rt_ref_lidar)    == NPY_FLOAT64) )
    {
        BARF("rt_ref_lidar should have shape (Nlidars=%d,6), have dtype=float64 and be contiguous",
             Nlidars);
        goto done;
    }

    if(Ncameras == 0)
    {
        if(! (rt_ref_camera == NULL ||
              (PyObject*)rt_ref_camera == Py_None ||
              PyArray_SIZE(rt_ref_camera) == 0) )
        {
            BARF("models aren't given, so rt_ref_camera shouldn't be given (or be None) as well");
            goto done;
        }
    }
    else
    {
        if(! (PyArray_IS_C_CONTIGUOUS(rt_ref_camera) &&
              PyArray_NDIM(rt_ref_camera)    == 2 &&
              PyArray_DIMS(rt_ref_camera)[0] == Ncameras &&
              PyArray_DIMS(rt_ref_camera)[1] == 6 &&
              PyArray_TYPE(rt_ref_camera)    == NPY_FLOAT64) )
        {
            BARF("rt_ref_camera should have shape (Ncameras=%d,6), have dtype=float64 and be contiguous",
                 Ncameras);
            goto done;
        }
    }

    const int Nsensors = Nlidars + Ncameras;
    if(! (PyArray_IS_C_CONTIGUOUS(rt_ref_lidar) &&
          PyArray_NDIM(Var_rt_lidar0_sensor)    == 4          &&
          PyArray_DIMS(Var_rt_lidar0_sensor)[0] == Nsensors-1 &&
          PyArray_DIMS(Var_rt_lidar0_sensor)[1] == 6          &&
          PyArray_DIMS(Var_rt_lidar0_sensor)[2] == Nsensors-1 &&
          PyArray_DIMS(Var_rt_lidar0_sensor)[3] == 6          &&
          PyArray_TYPE(Var_rt_lidar0_sensor)    == NPY_FLOAT64) )
    {
        BARF("Var_rt_lidar0_sensor should have shape (Nsensors-1,6,Nsensors-1,6) where Nsensors=%d, have dtype=float64 and be contiguous",
             Nsensors);
        goto done;
    }

    if(! (PyArray_IS_C_CONTIGUOUS(rt_vehicle_lidar0) &&
 Nsensors-1 &&
          PyArray_DIMS(Var_rt_lidar0_sensor)[3] == 6          &&
          PyArray_TYPE(Var_rt_lidar0_sensor)    == NPY_FLOAT64) )
    {
        BARF("Var_rt_lidar0_sensor should have shape (Nsensors-1,6,Nsensors-1,6) where Nsensors=%d, have dtype=float64 and be contiguous",
             Nsensors);
        goto done;
    }

    if(! (PyArray_IS_C_CONTIGUOUS(rt_vehicle_lidar0) &&
          PyArray_NDIM(rt_vehicle_lidar0)    == 1 &&
          PyArray_DIMS(rt_vehicle_lidar0)[0] == 6 &&
          PyArray_TYPE(rt_vehicle_lidar0)    == NPY_FLOAT64) )
    {
        BARF("rt_vehicle_lidar0 should have shape (6,), have dtype=float64 and be contiguous");
        goto done;
    }

    static_assert(offsetof(mrcal_pose_t,r) == 0 && offsetof(mrcal_pose_t,t) == sizeof(double)*3,
                  "mrcal_pose_t should be a dense rt transform");


    isvisible_per_sensor_per_sector = (PyArrayObject*)PyArray_SimpleNew(2,
                                                                        ((npy_intp[]){Nsensors,Nsectors}),
                                                                        NPY_UINT8);
    if(isvisible_per_sensor_per_sector == NULL) goto done;

    stdev_worst = (PyArrayObject*)PyArray_SimpleNew(1,
                                                    ((npy_intp[]){Nsectors}),
                                                    NPY_FLOAT64);
    if(stdev_worst == NULL) goto done;

    isensors_pair_stdev_worst = (PyArrayObject*)PyArray_SimpleNew(2,
                                                                  ((npy_intp[]){Nsectors,2}),
                                                                  NPY_UINT16);
    if(isensors_pair_stdev_worst == NULL) goto done;


    if(!clc_post_solve_statistics( // out
                                   (uint8_t     *)PyArray_DATA(isvisible_per_sensor_per_sector),
                                   (double      *)PyArray_DATA(stdev_worst),
                                   (uint16_t    *)PyArray_DATA(isensors_pair_stdev_worst),
                                   Nsectors,
                                   threshold_valid_lidar_range,
                                   threshold_valid_lidar_Npoints,
                                   uncertainty_quantification_range,

                                   // out,in
                                   (mrcal_pose_t*)PyArray_DATA(rt_ref_lidar),
                                   (mrcal_pose_t*)PyArray_DATA(rt_ref_camera),

                                   // in
                                   (const double*)PyArray_DATA(Var_rt_lidar0_sensor),
                                   (mrcal_pose_t*)PyArray_DATA(rt_vehicle_lidar0),

                                   lidar_scans,
                                   lidar_packet_stride,
                                   Nlidars,
                                   Ncameras,
                                   (const mrcal_cameramodel_t*const*)models))
    {
        BARF("clc_post_solve_statistics() failed");
        goto done;
    }

    result = Py_BuildValue("{sOsOsO}",
                           "isvisible_per_sensor_per_sector",    isvisible_per_sensor_per_sector,
                           "stdev_worst",   stdev_worst,
                           "isensors_pair_stdev_worst", isensors_pair_stdev_worst);

 done:
    Py_XDECREF(isvisible_per_sensor_per_sector);
    Py_XDECREF(stdev_worst);
    Py_XDECREF(isensors_pair_stdev_worst);
    Py_XDECREF(py_model);

    Py_XDECREF(Var_rt_lidar0_sensor);
    Py_XDECREF(rt_ref_lidar);
    Py_XDECREF(rt_ref_camera);
    Py_XDECREF(rt_vehicle_lidar0);

    Py_XDECREF(py_model);

    RESET_SIGINT();

    return result;
}

static PyObject* py_fit_from_optimization_inputs(PyObject* NPY_UNUSED(self),
                                                 PyObject* args,
                                                 PyObject* kwargs)
{
    PyObject* result      = NULL;
    PyObject* inputs_dump = NULL;

    int            Nlidars          = 0;
    int            Ncameras         = 0;
    mrcal_pose_t*  rt_ref_lidar     = NULL;
    mrcal_pose_t*  rt_ref_camera    = NULL;
    PyArrayObject* py_rt_ref_lidar  = NULL;
    PyArrayObject* py_rt_ref_camera = NULL;
    int            do_inject_noise  = 0;
    int            do_fit_seed      = 0;

    SET_SIGINT();

    char* keywords[] = { "inputs_dump",
                         "do_inject_noise",
                         "do_fit_seed",
                         NULL };
    if(!PyArg_ParseTupleAndKeywords( args, kwargs,
                                     "O" "|$" "pp",
                                     keywords,
                                     &inputs_dump,
                                     &do_inject_noise,
                                     &do_fit_seed,
                                     NULL))
        goto done;

    if(!PyBytes_Check(inputs_dump))
    {
        BARF("inputs_dump should be a 'bytes' object");
        goto done;
    }
    if(!clc_fit_from_optimization_inputs(// out
                                         &Nlidars,
                                         &Ncameras,
                                         &rt_ref_lidar,
                                         &rt_ref_camera,
                                         // in
                                         PyBytes_AS_STRING(inputs_dump),
                                         PyBytes_GET_SIZE( inputs_dump),
                                         do_inject_noise,
                                         do_fit_seed))
    {
        BARF("clc_fit_from_optimization_inputs() failed");
        goto done;
    }

    py_rt_ref_lidar = (PyArrayObject*)PyArray_SimpleNew(2, ((npy_intp[]){Nlidars,6}), NPY_FLOAT64);
    if(py_rt_ref_lidar == NULL) goto done;

    py_rt_ref_camera = (PyArrayObject*)PyArray_SimpleNew(2, ((npy_intp[]){Ncameras,6}), NPY_FLOAT64);
    if(py_rt_ref_camera == NULL) goto done;

    memcpy( PyArray_DATA(py_rt_ref_lidar),
            rt_ref_lidar,
            Nlidars*6*sizeof(double));
    memcpy( PyArray_DATA(py_rt_ref_camera),
            rt_ref_camera,
            Ncameras*6*sizeof(double));

    result = Py_BuildValue("{sOsO}",
                           "rt_ref_lidar",  py_rt_ref_lidar,
                           "rt_ref_camera", py_rt_ref_camera);
 done:
    free(rt_ref_lidar);
    free(rt_ref_camera);
    Py_XDECREF(py_rt_ref_lidar);
    Py_XDECREF(py_rt_ref_camera);
    RESET_SIGINT();

    return result;
}

static PyObject* py_lidar_segmentation_default_context(PyObject* NPY_UNUSED(self),
                                    PyObject* NPY_UNUSED(args))
{
    PyObject* result = NULL;

    clc_lidar_segmentation_context_t ctx;
    clc_lidar_segmentation_default_context(&ctx);


#define CLC_LIDAR_SEGMENTATION_LIST_CONTEXT_PYBUILD_PATTERN( type,name,default,pyparse,pybuild) "s" pybuild
#define CLC_LIDAR_SEGMENTATION_LIST_CONTEXT_PYBUILD_KEYVALUE(type,name,default,pyparse,pybuild) ,#name, ctx.name
    result = Py_BuildValue("{" CLC_LIDAR_SEGMENTATION_LIST_CONTEXT(CLC_LIDAR_SEGMENTATION_LIST_CONTEXT_PYBUILD_PATTERN) "}"
                           CLC_LIDAR_SEGMENTATION_LIST_CONTEXT(CLC_LIDAR_SEGMENTATION_LIST_CONTEXT_PYBUILD_KEYVALUE));
#undef CLC_LIDAR_SEGMENTATION_LIST_CONTEXT_PYBUILD_PATTERN
#undef CLC_LIDAR_SEGMENTATION_LIST_CONTEXT_PYBUILD_KEYVALUE

    // If Py_BuildValue failed, this will already be NULL
    return result;
}

static const char lidar_segmentation_docstring[] =
#include "lidar_segmentation.docstring.h"
    ;
static const char calibrate_docstring[] =
#include "calibrate.docstring.h"
    ;
static const char post_solve_statistics_docstring[] =
#include "post_solve_statistics.docstring.h"
    ;
static const char fit_from_optimization_inputs_docstring[] =
#include "fit_from_optimization_inputs.docstring.h"
    ;
static const char lidar_segmentation_default_context_docstring[] =
#include "lidar_segmentation_default_context.docstring.h"
    ;

static PyMethodDef methods[] =
    {
     PYMETHODDEF_ENTRY(calibrate,                   py_calibrate,                    METH_VARARGS | METH_KEYWORDS),
     PYMETHODDEF_ENTRY(post_solve_statistics,       py_post_solve_statistics,        METH_VARARGS | METH_KEYWORDS),
     PYMETHODDEF_ENTRY(fit_from_optimization_inputs,py_fit_from_optimization_inputs, METH_VARARGS | METH_KEYWORDS),
     PYMETHODDEF_ENTRY(lidar_segmentation,          py_lidar_segmentation,           METH_VARARGS | METH_KEYWORDS),
     PYMETHODDEF_ENTRY(lidar_segmentation_default_context, py_lidar_segmentation_default_context, METH_NOARGS),
     {}
    };

static struct PyModuleDef module_def =
    {
     PyModuleDef_HEAD_INIT,
     "_clc",
     "Geometric alignment of camera and LIDAR sensors",
     -1,
     methods
    };

PyMODINIT_FUNC PyInit__clc(void)
{
    import_array();
    return PyModule_Create(&module_def);
}
