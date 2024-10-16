#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include <stdbool.h>
#include <Python.h>
#include <numpy/arrayobject.h>
#include <signal.h>
#include <stddef.h>

#include "point_segmentation.h"


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
        PyErr_SetString(PyExc_RuntimeError, "sigaction() failed");      \
        goto done;                                                      \
    }                                                                   \
} while(0)
#define RESET_SIGINT() do {                                             \
    if( 0 != sigaction(SIGINT,                                          \
                       &sigaction_old, NULL ))                          \
        PyErr_SetString(PyExc_RuntimeError, "sigaction-restore failed"); \
} while(0)

#define PYMETHODDEF_ENTRY(name, c_function_name, args) {#name,          \
                                                        (PyCFunction)c_function_name, \
                                                        args,           \
                                                        name ## _docstring}


static PyObject* py_point_segmentation(PyObject* NPY_UNUSED(self),
                                       PyObject* args,
                                       PyObject* kwargs)
{
    PyObject* result = NULL;

    // Each is an iterable of length Nrings
    PyArrayObject* points    = NULL;
    PyArrayObject* Npoints   = NULL;
    PyArrayObject* plane_pn  = NULL;
    PyObject* py_ipoint      = NULL;

    const int Nplanes_max = 16;
    PyArrayObject* ipoint[Nplanes_max] = {};

#warning "I should define a complex dtype to pass points_and_plane from a preallocated numpy array. Instead I do this in C and then copy the results. For now"
    points_and_plane_t points_and_plane[Nplanes_max];

    context_t ctx;
    default_context(&ctx);


#define LIST_CONTEXT_KEYWORDS(   type,name,default,pyparse) #name,
#define LIST_CONTEXT_PYPARSE(    type,name,default,pyparse) pyparse
#define LIST_CONTEXT_ADDRESS_CTX(type,name,default,pyparse) &ctx.name,
    char* keywords[] = { "points",
                         "Npoints",
                         LIST_CONTEXT(LIST_CONTEXT_KEYWORDS)
                         NULL };
    if(!PyArg_ParseTupleAndKeywords( args, kwargs,
                                     "O&O&" "|$" LIST_CONTEXT(LIST_CONTEXT_PYPARSE)
                                     ,
                                     keywords,
                                     PyArray_Converter, &points,
                                     PyArray_Converter, &Npoints,
                                     LIST_CONTEXT(LIST_CONTEXT_ADDRESS_CTX)
                                     NULL))
        goto done;

#undef LIST_CONTEXT_KEYWORDS
#undef LIST_CONTEXT_PYPARSE
#undef LIST_CONTEXT_ADDRESS_CTX


    if(! (PyArray_TYPE(points) == NPY_FLOAT32 &&
          PyArray_NDIM(points) == 2 &&
          PyArray_DIMS(points)[1] == 3 &&
          PyArray_STRIDES(points)[1] == sizeof(float) &&
          PyArray_STRIDES(points)[0] == sizeof(float)*3) )
    {
        PyErr_SetString(PyExc_RuntimeError, "'points' must be a densely-stored array of shape (N,3) containing 32-bit floats");
        goto done;
    }

    if(! (PyArray_TYPE(Npoints) == NPY_INT &&
          PyArray_NDIM(Npoints) == 1 &&
          PyArray_DIMS(Npoints)[0] == ctx.Nrings &&
          PyArray_STRIDES(Npoints)[0] == sizeof(int)) )
    {
        PyErr_Format(PyExc_RuntimeError,
                     "'Npoints' must be a densely-stored array of shape (Nrings=%d,) containing ints",
                     ctx.Nrings);
        goto done;
    }

    int Npoints_all = 0;
    for(int i=0; i<ctx.Nrings; i++)
        Npoints_all += ((int*)PyArray_DATA(Npoints))[i];
    if(Npoints_all != PyArray_DIMS(points)[0])
    {
        PyErr_Format(PyExc_RuntimeError, "'Npoints' says there are %d total points, but 'points' says %d. These must match",
                     Npoints_all,
                     (int)(PyArray_DIMS(points)[0]));
        goto done;
    }

    int8_t Nplanes =
        point_segmentation( // out
                            points_and_plane,
                            // in
                            Nplanes_max,
                            (const point3f_t*)PyArray_DATA(points),
                            (const int*)PyArray_DATA(Npoints),
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
    static_assert(offsetof(plane_t,p) == 0 && offsetof(plane_t,n) == sizeof(point3f_t) && sizeof(plane_t) == 2*sizeof(point3f_t),
                  "plane_t is expected to densely store p and then n");
    for(int i=0; i<Nplanes; i++)
        memcpy( &((float*)PyArray_DATA(plane_pn))[6*i], &points_and_plane[i].plane, sizeof(points_and_plane[i].plane));

    py_ipoint = PyTuple_New(Nplanes);
    if(py_ipoint == NULL) goto done;
    for(int i=0; i<Nplanes; i++)
    {
        PyTuple_SET_ITEM(py_ipoint, i, ipoint[i]);
        Py_INCREF(ipoint[i]);
    }

    result = Py_BuildValue("{sOsO}",
                           "ipoint",    py_ipoint,
                           "plane_pn",  plane_pn);

 done:
    Py_XDECREF(points);
    Py_XDECREF(Npoints);
    for(int i=0; i<Nplanes_max; i++)
        Py_XDECREF(ipoint[i]);
    Py_XDECREF(py_ipoint);
    Py_XDECREF(plane_pn);

    return result;
}

static const char point_segmentation_docstring[] =
#include "point_segmentation.docstring.h"
    ;

static PyMethodDef methods[] =
    {
     PYMETHODDEF_ENTRY(point_segmentation,      py_point_segmentation,         METH_VARARGS | METH_KEYWORDS),
     {}
    };

static struct PyModuleDef module_def =
    {
     PyModuleDef_HEAD_INIT,
     "camera_lidar_calibration",
     "Geometric alignment of camera and LIDAR sensors",
     -1,
     methods
    };

PyMODINIT_FUNC PyInit_camera_lidar_calibration(void)
{
    import_array();
    return PyModule_Create(&module_def);
}
