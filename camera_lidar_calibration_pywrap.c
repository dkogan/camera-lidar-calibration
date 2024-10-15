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

#warning hard-coded constant
    const int   Nrings = 32;


    PyObject* result = NULL;

    // Each is an iterable of length Nrings
    PyArrayObject* points    = NULL;
    PyArrayObject* Npoints   = NULL;
    PyArrayObject* plane_idx = NULL;
    PyArrayObject* plane_pn  = NULL;

    int ipoint0[Nrings];

    char* keywords[] = { "points",
                         "Npoints",
                         NULL };

    if(!PyArg_ParseTupleAndKeywords( args, kwargs,
                                     "$O&O&",
                                     keywords,
                                     PyArray_Converter, &points,
                                     PyArray_Converter, &Npoints,
                                     NULL))
        goto done;

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
          PyArray_DIMS(Npoints)[0] == Nrings &&
          PyArray_STRIDES(Npoints)[0] == sizeof(int)) )
    {
        PyErr_Format(PyExc_RuntimeError,
                     "'Npoints' must be a densely-stored array of shape (Nrings=%d,) containing ints",
                     Nrings);
        goto done;
    }

    ipoint0[0] = 0;
    for(int i=1; i<Nrings; i++)
        ipoint0[i] = ipoint0[i-1] + ((int*)PyArray_DATA(Npoints))[i-1];

    const int Npoints_all = ipoint0[Nrings-1] + ((int*)PyArray_DATA(Npoints))[Nrings-1];

    if(Npoints_all != PyArray_DIMS(points)[0])
    {
        PyErr_Format(PyExc_RuntimeError, "'Npoints' says there are %d total points, but 'points' says %d. These must match",
                     Npoints_all,
                     (int)(PyArray_DIMS(points)[0]));
        goto done;
    }

    plane_idx = (PyArrayObject*)PyArray_SimpleNew(1, ((npy_intp[]){Npoints_all}), NPY_INT8);
    if(plane_idx == NULL)
        goto done;

    const int Nplanes_max = 16;
    plane_pn = (PyArrayObject*)PyArray_SimpleNew(2, ((npy_intp[]){Nplanes_max,6}), NPY_FLOAT32);
    if(plane_pn == NULL)
        goto done;
    static_assert(offsetof(plane_t,p) == 0 && offsetof(plane_t,n) == sizeof(point3f_t) && sizeof(plane_t) == 2*sizeof(point3f_t),
                  "plane_t is expected to densely store p and then n");

    int8_t Nplanes =
        point_segmentation( // out
                            PyArray_DATA(plane_idx),
                            (plane_t*)PyArray_DATA(plane_pn),
                            // in
                            Nplanes_max,
                            (const point3f_t*)PyArray_DATA(points),
                            (const int*)PyArray_DATA(Npoints));
    if(Nplanes < 0)
        goto done;

    result = Py_BuildValue("{sOsO}",
                           "plane_idx", plane_idx,
                           "plane_pn",  plane_pn);

 done:
    Py_XDECREF(points);
    Py_XDECREF(Npoints);
    Py_XDECREF(plane_idx);
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
