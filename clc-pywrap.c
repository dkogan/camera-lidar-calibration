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


    if(! (PyArray_TYPE(points) == NPY_FLOAT32 &&
          PyArray_NDIM(points) == 2 &&
          PyArray_DIMS(points)[1] == 3 &&
          PyArray_STRIDES(points)[1] == sizeof(float)) )
    {
        PyErr_SetString(PyExc_RuntimeError, "'points' must be an array of shape (N,3) containing 32-bit floats, each xyz stored densely");
        goto done;
    }

    const int          Npoints_total       = PyArray_DIMS   (points)[0];
    const unsigned int lidar_packet_stride = PyArray_STRIDES(points)[0];

    if(! (PyArray_TYPE(rings) == NPY_UINT16 &&
          PyArray_NDIM(rings) == 1) )
    {
        PyErr_Format(PyExc_RuntimeError,
                     "'rings' must be a 1-dimensional array containing uint16");
        goto done;
    }

    if(! (PyArray_DIMS(rings)[0] == Npoints_total) )
    {
        PyErr_Format(PyExc_RuntimeError,
                     "'rings' and 'points' must describe the same number of points; instead len(points)=%d, but len(rings)=%d",
                     Npoints_total, PyArray_DIMS(rings)[0]);
        goto done;
    }
    if(! (lidar_packet_stride == PyArray_STRIDES(rings)[0]) )
    {
        PyErr_Format(PyExc_RuntimeError,
                     "'rings' and 'points' must have the same point stride; instead points.stride[0]=%d, but rings.stride[0]=%d",
                     lidar_packet_stride,
                     PyArray_STRIDES(rings)[0]);
        goto done;
    }


    {
        const
            clc_lidar_scan_t scan = {.points  = (clc_point3f_t*)PyArray_DATA(points),
                                     .rings   = (uint16_t     *)PyArray_DATA(rings),
                                     .Npoints = Npoints_total};

        clc_point3f_t points_sorted[Npoints_total];
        unsigned int  Npoints      [ctx.Nrings];

        clc_lidar_sort(// out
                       points_sorted,
                       Npoints,
                       // in
                       ctx.Nrings,
                       lidar_packet_stride,
                       &scan);

        int8_t Nplanes =
            clc_lidar_segmentation( // out
                                    points_and_plane,
                                    // in
                                    Nplanes_max,
                                    points_sorted,
                                    Npoints,
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
static const char lidar_segmentation_default_context_docstring[] =
#include "lidar_segmentation_default_context.docstring.h"
    ;

static PyMethodDef methods[] =
    {
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
