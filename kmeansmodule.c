#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdlib.h>
#include <math.h>

/* פונקציית עזר לשחרור זיכרון בצורה בטוחה */
static void cleanup(double **points, double **centroids, double **sums, int *counts, int N, int K) {
    int i;
    if (points) {
        for (i = 0; i < N; i++) free(points[i]);
        free(points);
    }
    if (centroids) {
        for (i = 0; i < K; i++) free(centroids[i]);
        free(centroids);
    }
    if (sums) {
        for (i = 0; i < K; i++) free(sums[i]);
        free(sums);
    }
    if (counts) free(counts);
}

static PyObject* fit(PyObject *self, PyObject *args) {
    int K, iter, N, dim, i, j, it, p, c, d;
    double eps, eps_sq;
    PyObject *centroids_obj, *data_obj;
    double **points = NULL, **centroids = NULL, **sums = NULL;
    int *counts = NULL;
    PyObject *result, *centroid_row;

    if (!PyArg_ParseTuple(args, "iidOO", &K, &iter, &eps, &centroids_obj, &data_obj)) {
        return NULL;
    }

    N = (int)PyList_Size(data_obj);
    dim = (int)PyList_Size(PyList_GetItem(data_obj, 0));

    /* הקצאות זיכרון עם בדיקת תקינות */
    points = (double **)calloc(N, sizeof(double *));
    for (i = 0; i < N; i++) points[i] = (double *)malloc(dim * sizeof(double));
    
    centroids = (double **)calloc(K, sizeof(double *));
    for (i = 0; i < K; i++) centroids[i] = (double *)malloc(dim * sizeof(double));
    
    counts = (int *)calloc(K, sizeof(int));
    sums = (double **)calloc(K, sizeof(double *));
    for (i = 0; i < K; i++) sums[i] = (double *)calloc(dim, sizeof(double));

    /* המרת נתונים */
    for (i = 0; i < N; i++) {
        PyObject *row = PyList_GetItem(data_obj, i);
        for (j = 0; j < dim; j++) points[i][j] = PyFloat_AsDouble(PyList_GetItem(row, j));
    }
    for (i = 0; i < K; i++) {
        PyObject *row = PyList_GetItem(centroids_obj, i);
        for (j = 0; j < dim; j++) centroids[i][j] = PyFloat_AsDouble(PyList_GetItem(row, j));
    }

    eps_sq = eps * eps;

    /* לולאת K-Means */
    for (it = 0; it < iter; ++it) {
        for (i = 0; i < K; ++i) {
            counts[i] = 0;
            for (j = 0; j < dim; ++j) sums[i][j] = 0.0;
        }

        for (p = 0; p < N; ++p) {
            int best = 0;
            double best_d = 0.0;
            for (d = 0; d < dim; ++d) {
                double diff = points[p][d] - centroids[0][d];
                best_d += diff * diff;
            }
            for (c = 1; c < K; ++c) {
                double dist = 0.0;
                for (d = 0; d < dim; ++d) {
                    double diff = points[p][d] - centroids[c][d];
                    dist += diff * diff;
                }
                if (dist < best_d) { best_d = dist; best = c; }
            }
            counts[best]++;
            for (d = 0; d < dim; ++d) sums[best][d] += points[p][d];
        }

        double max_move_sq = 0.0;
        for (c = 0; c < K; ++c) {
            if (counts[c] > 0) {
                double inv = 1.0 / counts[c];
                double move_sq = 0.0;
                for (d = 0; d < dim; ++d) {
                    double new_val = sums[c][d] * inv;
                    double diff = new_val - centroids[c][d];
                    move_sq += diff * diff;
                    centroids[c][d] = new_val;
                }
                if (move_sq > max_move_sq) max_move_sq = move_sq;
            }
        }
        if (max_move_sq < eps_sq) break;
    }

    /* בניית התוצאה */
    result = PyList_New(K);
    for (i = 0; i < K; i++) {
        centroid_row = PyList_New(dim);
        for (j = 0; j < dim; j++) PyList_SetItem(centroid_row, j, PyFloat_FromDouble(centroids[i][j]));
        PyList_SetItem(result, i, centroid_row);
    }

    cleanup(points, centroids, sums, counts, N, K);
    return result;
}

static PyMethodDef mykmeanssp_Methods[] = {
    {"fit", (PyCFunction)fit, METH_VARARGS, "Runs K-Means clustering algorithm."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef mykmeanssp_module = {
    PyModuleDef_HEAD_INIT, "mykmeanssp", NULL, -1, mykmeanssp_Methods
};

PyMODINIT_FUNC PyInit_mykmeanssp(void) {
    return PyModule_Create(&mykmeanssp_module);
}