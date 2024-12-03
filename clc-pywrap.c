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
        ipoint[i] = (PyArrayObject*)PyArray_SimpleNew(1, ((npy_intp[]){points_and_plane[i].ipoint_set.n}), NPY_UINT32);
        if(ipoint[i] == NULL)
            goto done;

        memcpy(PyArray_DATA(ipoint[i]), points_and_plane[i].ipoint_set.ipoint, sizeof(points_and_plane[i].ipoint_set.ipoint[0])*points_and_plane[i].ipoint_set.n);
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
                                   // in
                                   const PyObject* py_snapshot)
{
    PyObject* images = PyTuple_GET_ITEM(py_snapshot, 0);

    if(images == Py_None)
    {
        memset(snapshot->images, 0, sizeof(snapshot->images));
        *Ncameras = 0;
        return true;
    }

    int Ncameras_here;
    if(!PyTuple_Check(images))
    {
        BARF("Each sensor_snapshot[0] should be a tuple of images for each camera, or None if that camera had no observations");
        return false;
    }
    if(0 > (Ncameras_here = PyTuple_Size(images)))
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

    // write stuff to snapshot->images[i]


    BARF("Camera stuff not done yet");
    return false;
}

static bool ingest_lidar_snapshot(// out
                                  clc_sensor_snapshot_unsorted_t* snapshot,
                                  int* Nlidars,
                                  unsigned int* lidar_packet_stride,
                                  // in
                                  const PyObject* py_snapshot)
{
    PyObject* py_lidar_scans = PyTuple_GET_ITEM(py_snapshot, 1);

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

    for(int i=0; i<*Nlidars; i++)
    {
        clc_lidar_scan_unsorted_t* scan = &snapshot->lidar_scans[i];

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

static PyObject* py_calibrate(PyObject* NPY_UNUSED(self),
                              PyObject* args,
                              PyObject* kwargs)
{
    SET_SIGINT();

    PyObject* result = NULL;

    PyTupleObject* py_sensor_snapshots = NULL;
    PyArrayObject* rt_ref_lidar        = NULL;
    PyArrayObject* rt_ref_camera       = NULL;

    // uninitialized
    unsigned int lidar_packet_stride = 0;
    int          Ncameras            = -1;
    int          Nlidars             = -1;
    int          Nsensor_snapshots   = -1;

    int check_gradient__use_distance_to_plane = 0;
    int check_gradient                        = 0;
    // sensor_snapshots is a tuple. Each slice corresponds to
    // clc_sensor_snapshot_unsorted_t; it is a tuple:
    //
    // - images      (an tuple, each element corresponding to a mrcal_image_uint8_t)
    // - lidar_scans (an tuple, each element corresponding to clc_lidar_scan_unsorted_t)
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
                         "check_gradient__use_distance_to_plane",
                         "check_gradient",
                         CLC_LIDAR_SEGMENTATION_LIST_CONTEXT(CLC_LIDAR_SEGMENTATION_LIST_CONTEXT_KEYWORDS)
                         NULL };
    if(!PyArg_ParseTupleAndKeywords( args, kwargs,
                                     "O" "|$" "pp" CLC_LIDAR_SEGMENTATION_LIST_CONTEXT(CLC_LIDAR_SEGMENTATION_LIST_CONTEXT_PYPARSE)
                                     ,
                                     keywords,
                                     (PyTupleObject*)&py_sensor_snapshots,
                                     &check_gradient__use_distance_to_plane,
                                     &check_gradient,
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

            if(!ingest_camera_snapshot(snapshot, &Ncameras,
                                       py_snapshot))
                goto done;
            if(!ingest_lidar_snapshot (snapshot, &Nlidars, &lidar_packet_stride,
                                       py_snapshot))
                goto done;
        }


        rt_ref_lidar = (PyArrayObject*)PyArray_SimpleNew(2, ((npy_intp[]){Nlidars,6}), NPY_FLOAT64);
        if(rt_ref_lidar == NULL) goto done;

        rt_ref_camera = (PyArrayObject*)PyArray_SimpleNew(2, ((npy_intp[]){Ncameras,6}), NPY_FLOAT64);
        if(rt_ref_camera == NULL) goto done;

        static_assert(offsetof(mrcal_pose_t,r) == 0 && offsetof(mrcal_pose_t,t) == sizeof(double)*3,
                      "mrcal_pose_t should be a dense rt transform");


        if(!clc_unsorted(// out
                         (mrcal_pose_t*)PyArray_DATA(rt_ref_lidar),
                         (mrcal_pose_t*)PyArray_DATA(rt_ref_camera),
                         // in
                         sensor_snapshots,
                         Nsensor_snapshots,
                         lidar_packet_stride,
                         Ncameras,
                         Nlidars,
                         (clc_is_bgr_mask_t)0,
                         check_gradient__use_distance_to_plane,
                         check_gradient))
        {
            BARF("clc_unsorted() failed");
            goto done;
        }
    }

    result = Py_BuildValue("{sOsO}",
                           "rt_ref_lidar",  rt_ref_lidar,
                           "rt_ref_camera", rt_ref_camera);

 done:
    Py_XDECREF(rt_ref_lidar);
    Py_XDECREF(rt_ref_camera);
    Py_XDECREF(py_sensor_snapshots);

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
static const char lidar_segmentation_default_context_docstring[] =
#include "lidar_segmentation_default_context.docstring.h"
    ;

static PyMethodDef methods[] =
    {
     PYMETHODDEF_ENTRY(calibrate,               py_calibrate,               METH_VARARGS | METH_KEYWORDS),
     PYMETHODDEF_ENTRY(lidar_segmentation,      py_lidar_segmentation,      METH_VARARGS | METH_KEYWORDS),
     PYMETHODDEF_ENTRY(lidar_segmentation_default_context,         py_lidar_segmentation_default_context,         METH_NOARGS),
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
