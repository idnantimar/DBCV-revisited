'''
Adapted from SciPy
[see https://github.com/scipy/scipy/blob/main/scipy/spatial/_ckdtree.pyx]

Authors: Anne M. Archibald
Copyright (c) 2001-2002 Enthought, Inc. 2003, SciPy Developers
License: 3-clause BSD


OUR CONTRIBUTION: @idnantimar
-----------------
    Here we keep SciPy's implementation for tree building as far as possible
    (for future upstream consistency), then customize the query routines as per our requirements.

    We introduce an additional exclude filter on Index during search traversal,
    rather than returning unfiltered results and performing post-processing.

    The combined time complexity of search + filtering remains the same in the worst case
    (as we are doing the same job with a different execution order),
    but the advantages are:

    [1.] Integer index comparison is almost always cheaper than floating-point distance calculations.
        When we know in advance which indices are not required,
        we can short-circuit the corresponding iterations in the brute-force loop of leaf nodes,
        thereby reducing average runtime.

    [2.] As we no longer need to store 'to be excluded' points in the output,
        peak memory usage is reduced.

'''


# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: initializedcheck=False


import numpy as np
cimport numpy as np

from cpython.mem cimport PyMem_Malloc, PyMem_Free
from libc.string cimport memcpy
from libcpp.mutex cimport py_safe_call_once, py_safe_once_flag
from libcpp.vector cimport vector
from libc.float cimport DBL_MAX

np.import_array()


__all__ = ['cKDTree']


# C++ implementations
# ===================    

cdef extern from "ckdtree_decl.h":

    struct ckdtreenode: # [[INDIVIDUAL NODES]]
        np.intp_t split_dim    # split axis
        np.intp_t children    # number of points under this node
        np.float64_t split    # split value along split_dim
        np.intp_t start_idx, end_idx    # [start, end) for this node subtree in raw_indices master array
        ckdtreenode *less    # child holding points < split
        ckdtreenode *greater    # child holding points >= split
        np.intp_t _less    # buffer offset of less, pre-pointer-fixup
        np.intp_t _greater    # buffer offset of greater, pre-pointer-fixup

    struct ckdtree: # [[TREE STRUCTURE]]
        vector[ckdtreenode] *tree_buffer    # flat storage for all nodes
        ckdtreenode *ctree    # pointer to the root node
        np.float64_t *raw_data    # point coordinates, n * m contiguous doubles
        np.intp_t n, m    # (n, m) shape of raw_data
        np.intp_t leafsize    # brute-force cutoff for leaf nodes
        np.float64_t *raw_maxes    # per-dimension global max
        np.float64_t *raw_mins    # per-dimension global min
        np.intp_t *raw_indices    # point indices master array, reordered by build()
        np.float64_t *raw_boxsize_data    # periodic box size; (not needed for plain geometry)
        np.intp_t size    # number of nodes in the tree


    int build_ckdtree( # [[THE ACTUAL BUILD ROUTINE]]
        ckdtree *self,
        np.intp_t start_idx,
        np.intp_t end_idx,
        np.float64_t *maxes,
        np.float64_t *mins,
        int _median,
        int _compact
    ) except + nogil


    void query_single_point_sqeuclidean_exact( # [[k-NN SEARCH]]
        const ckdtree *self,    # the underlying tree
        np.float64_t *result_distances_sqeuclidean,    # pointer to output distances array
        np.intp_t *result_indices,    # pointer to output indices
        const np.float64_t *x,    # pointer to query (single observation)
        const np.intp_t kmax,    # number of neighbors
        np.float64_t distance_upper_bound_sqeuclidean,    # distance upper cap
        const np.intp_t excludeIDx_start,        
        const np.intp_t excludeIDx_end 
        # only indices outside the range [excludeIDx_start, excludeIDx_end) will be consired as candidate neighbors.
        # default is [0,0) for no exclusion.          
    ) except + nogil

    void query_single_point_ball_sqeuclidean_exact( # [[RADIUS SEARCH]]
        const ckdtree *self,    # the underlying tree
        const np.float64_t *x,    # pointer to query (single observation)
        const np.float64_t r_sqeuclidean,    # square-Euclidean radius
        vector[np.intp_t] *result_indices,    # pointer to matched neighbor indices output
        const np.intp_t excludeIDx_start,       
        const np.intp_t excludeIDx_end
        # only indices outside [excludeIDx_start, excludeIDx_end) are candidates.
        # default is [0,0) for no exclusion.
    ) except + nogil




# Main cKDTree class
# ==================

cdef class cKDTree:
    r"""
    kd-tree for quick nearest-neighbor lookup.

    This class provides an index into a set of k-dimensional points 
    which can be used to rapidly look up the nearest neighbors of any point.

    ref: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html

    OUR CONTRIBUTION: 
    -----------------
        Here we keep SciPy's implementation for tree building as far as possible
        (for future upstream consistency), then customize the query routines as per our requirements.

        We introduce an additional exclude filter on Index during search traversal,
        rather than returning unfiltered results and performing post-processing.

        The combined time complexity of search + filtering remains the same in the worst case
        (as we are doing the same job with a different execution order),
        but the advantages are:

        [1.] Integer index comparison is almost always cheaper than floating-point distance calculations.
            When we know in advance which indices are not required,
            we can short-circuit the corresponding iterations in the brute-force loop of leaf nodes,
            thereby reducing average runtime.

        [2.] As we no longer need to store 'to be excluded' points in the output,
            peak memory usage is reduced.


    Parameters
    ----------
    data : array_like, shape (n,m)
        The n data points of dimension m to be indexed. 
        This array is not copied unless this is necessary to produce a contiguous array of doubles, 
        and so modifying this data will result in bogus results. 
        The data are also copied if the kd-tree is built with copy_data=True.
    leafsize : positive int, optional
        The number of points at which the algorithm switches over to brute-force. Default: 16.
    compact_nodes : bool, optional
        If True, the kd-tree is built to shrink the hyperrectangles to the actual data range. 
        This usually gives a more compact tree that is robust against degenerated input data 
        and gives faster queries at the expense of longer build time. 
        Default: True.
    copy_data : bool, optional
        If True the data is always copied to protect the kd-tree against data corruption. 
        Default: False.
    balanced_tree : bool, optional
        If True, the median is used to split the hyperrectangles instead of the midpoint. 
        This usually gives a more compact tree and faster queries at the expense of longer build time. 
        Default: True.
    boxsize : NotImplemented.

    Attributes
    ----------
    data : ndarray, shape (n,m)
        The n data points of dimension m to be indexed. 
        This array is not copied unless this is necessary to produce a contiguous array of doubles. 
        The data are also copied if the kd-tree is built with ``copy_data=True``. 
        Concurrently modifying the contents of the ``data`` array while the KDTree initializer is running 
        may lead to data corruption or crashes. 
        If the data are not copied, modifying the original ``data`` array after the tree is created 
        may lead to crashes or data corruption when searching the tree.
    leafsize : positive int
        The number of points at which the algorithm switches over to brute-force.
    m : int
        The dimension of a single data-point.
    n : int
        The number of data points.
    maxes : ndarray, shape (m,)
        The maximum value in each dimension of the n data points.
    mins : ndarray, shape (m,)
        The minimum value in each dimension of the n data points.
    size : int
        The number of nodes in the tree.
    tree : NA.
    boxsize : NA.

    """  
 
    # =================================================================
    # Build:

    # DO NOT MODIFY SciPy's implementation in this section.
    # [CRUCIAL FOR UPSTREAM CONSISTENCY IN FUTURE]
    # ==============================================================

    cdef:
        ckdtree * cself
        object                   _python_tree
        readonly np.ndarray      data
        readonly np.ndarray      maxes
        readonly np.ndarray      mins
        readonly np.ndarray      indices
        readonly object          boxsize
        np.ndarray               boxsize_data
        py_safe_once_flag        flag

    property n:
        def __get__(self): return self.cself.n

    property m:
        def __get__(self): return self.cself.m

    property leafsize:
        def __get__(self): return self.cself.leafsize

    property size:
        def __get__(self): return self.cself.size

    property tree:
        # make the tree viewable from Python
        def __get__(cKDTree self):
            # cast to void* is safe because it is either used to construct the
            # python tree and then discarded or immediately discarded
            cdef void *self_v = <void *>self
            py_safe_call_once(self.flag, init_pytree, self_v)
            return self._python_tree

    def __cinit__(cKDTree self):
        self.cself = <ckdtree * > PyMem_Malloc(sizeof(ckdtree))
        self.cself.tree_buffer = NULL

    def __init__(cKDTree self, data, np.intp_t leafsize=16, compact_nodes=True,
            copy_data=False, balanced_tree=True, boxsize=None):

        cdef:
            np.float64_t [::1] tmpmaxes, tmpmins
            np.float64_t *ptmpmaxes
            np.float64_t *ptmpmins
            ckdtree *cself = self.cself
            int compact, median

        self._python_tree = None

        if not copy_data: 
            copy_data = None 
        # [[REMOVED DEPENDANCY: scipy._lib._util copy_if_needed is replaced with native numpy equivalent.]]
        data = np.array(data, order='C', copy=copy_data, dtype=np.float64)

        # read-only view so ban people modifying tree.data after the tree is
        # constructed
        data = data.view(np.ndarray)
        data.flags.writeable = False

        if data.ndim != 2:
            raise ValueError("data must be of shape (n, m), where there are "
                             "n points of dimension m")

        if not np.isfinite(data).all():
            raise ValueError("data must be finite, check for nan or inf values")

        self.data = data
        cself.n = data.shape[0]
        cself.m = data.shape[1]
        cself.leafsize = leafsize

        if leafsize<1:
            raise ValueError("leafsize must be at least 1")

        if boxsize is None:
            self.boxsize = None
            self.boxsize_data = None
        else:
            # [THIS IS THE ONLY ALTERATION FROM ORIGINAL (SciPy) IN THIS SECTION.]
            # Early stopping before tree building;
            # so we don't need to maintain respective methods for toroidal geometry as placeholder.
            raise NotImplementedError(
                "Toroidal topology not implemented for DBCV."
            )

        self.maxes = np.ascontiguousarray(
            np.amax(self.data, axis=0) if self.n > 0 else np.zeros(self.m),
            dtype=np.float64)
        self.maxes.flags.writeable = False
        self.mins = np.ascontiguousarray(
            np.amin(self.data,axis=0) if self.n > 0 else np.zeros(self.m),
            dtype=np.float64)
        self.mins.flags.writeable = False
        self.indices = np.ascontiguousarray(np.arange(self.n,dtype=np.intp))
        self.indices.flags.writeable = False

        self._pre_init()

        compact = 1 if compact_nodes else 0
        median = 1 if balanced_tree else 0

        cself.tree_buffer = new vector[ckdtreenode]()

        tmpmaxes = np.copy(self.maxes)
        tmpmins = np.copy(self.mins)

        ptmpmaxes = &tmpmaxes[0]
        ptmpmins = &tmpmins[0]
        with nogil:
            build_ckdtree(cself, 0, cself.n, ptmpmaxes, ptmpmins, median, compact)

        # set up the tree structure pointers
        self._post_init()

    cdef _pre_init(cKDTree self):
        cself = self.cself

        # finalize the pointers from array attributes

        cself.raw_data = <np.float64_t*> np.PyArray_DATA(self.data)
        cself.raw_maxes = <np.float64_t*> np.PyArray_DATA(self.maxes)
        cself.raw_mins = <np.float64_t*> np.PyArray_DATA(self.mins)
        cself.raw_indices = <np.intp_t*> np.PyArray_DATA(self.indices)

        if self.boxsize_data is not None:
            cself.raw_boxsize_data = <np.float64_t*>np.PyArray_DATA(self.boxsize_data)
        else:
            cself.raw_boxsize_data = NULL

    cdef _post_init(cKDTree self):
        cself = self.cself
        # finalize the tree points, this calls _post_init_traverse

        cself.ctree = cself.tree_buffer.data()

        # set the size attribute after tree_buffer is built
        cself.size = cself.tree_buffer.size()

        self._post_init_traverse(cself.ctree)

    cdef _post_init_traverse(cKDTree self, ckdtreenode *node):
        cself = self.cself
        # recurse the tree and re-initialize
        # "less" and "greater" fields
        if node.split_dim == -1:
            # leafnode
            node.less = NULL
            node.greater = NULL
        else:
            node.less = cself.ctree + node._less
            node.greater = cself.ctree + node._greater
            self._post_init_traverse(node.less)
            self._post_init_traverse(node.greater)

    def __dealloc__(cKDTree self):
        cself = self.cself
        if cself.tree_buffer != NULL:
            del cself.tree_buffer
        PyMem_Free(cself)

    # ---------------------------------------


    # =================================================================
    # Query:

    # MODIFY to meet our custom requirements.
    # ==============================================================
    def query_sqeuclidean_exact(
            cKDTree self, 
            object x, int k = 1,
            np.double_t distance_upper_bound_sqeuclidean = DBL_MAX,
            tuple excludeIDx = (0, 0),
            object workers = None, 
        ):
        r"""
        Query the kd-tree for nearest neighbors.

        Parameters
        ----------
        x : array_like
            An array of points to query, where each row is an observation.
            NOTE: single query must be reshaped as ``(1, -1)`` format.
        k : integer
            The top k nearest neighbors to return. 
            NOTE: counting starts from 1.
        distance_upper_bound_sqeuclidean : nonnegative float, optional
            Return only neighbors within this square-Euclidean distance.  
        excludeIDx : tuple, optional
            Exclude a range of observation id ``[start, end)`` from search process.
        workers : int, optional
            Number of workers to use for parallel processing.

        Returns
        -------
        dd : ndarray of floats
            The square-Euclidean distances to the nearest neighbors.
            Missing neighbors are indicated with infinite distances.
        ii : ndarray of ints
            The index of each neighbor in ``self.data``.
            Missing neighbors are indicated with ``self.n``.

        NOTE: ``dd`` and ``ii`` will always be of shape ``(n_query, k)``

        """

        cdef:
            np.ndarray[np.double_t, ndim=2] x_arr = np.ascontiguousarray(x, dtype=np.float64)
            ckdtree *cself = self.cself

            np.intp_t kmax = max(k, 1) # ensure integer >= 1 

            np.double_t distance_upper_bound_sqeuclidean_ = np.clip(
                distance_upper_bound_sqeuclidean,
                0., DBL_MAX
            ) # clip +Inf to practical safe bound; ensure non-negative

            np.intp_t n, m

        if not np.isfinite(x_arr).all():
            raise ValueError("'x' must be finite, check for nan or inf values")

        n, m = validate_shape(x_arr, cself.m)

        # tuple unpacking for simpler integer comparisons
        cdef np.intp_t excludeIDx_start, excludeIDx_end
        excludeIDx_start, excludeIDx_end = excludeIDx
        if (
            excludeIDx_start < 0
            or excludeIDx_start > excludeIDx_end
            or excludeIDx_end > cself.n
        ):
            raise ValueError("excludeIDx must satisfy 0 <= start <= end <= n")

        cdef: # avoid cosmetic reshapings for sake of simplicity; output always of shape (n, kmax)
            const np.double_t [:, ::1] xx = x_arr 
            np.double_t [:, ::1] dd = np.empty((n, kmax), dtype=np.float64)
            np.intp_t [:, ::1] ii = np.empty((n, kmax), dtype=np.intp)

        def _batch_query(np.intp_t start, np.intp_t stop):
            cdef:
                const np.float64_t *pxx = &xx[start, 0]

                # output will ve written to allocated memory via pointer
                np.float64_t *pdd = &dd[start, 0]
                np.intp_t *pii = &ii[start, 0]
                
            cdef np.intp_t i
            with nogil:
                for i in range(stop - start):
                    query_single_point_sqeuclidean_exact(
                        cself,
                        pdd + i * kmax, pii + i * kmax, pxx + i * m,
                        kmax, distance_upper_bound_sqeuclidean_,
                        excludeIDx_start, excludeIDx_end
                    )

        _run_threads(_batch_query, n, workers) # [[THE ACTUAL QUERY HAPPENS HERE]]

        return np.asarray(dd), np.asarray(ii)


    def query_ball_point_sqeuclidean_exact(
            cKDTree self, 
            object x, 
            np.double_t r_sqeuclidean = DBL_MAX,
            tuple excludeIDx = (0, 0),
            object workers = None,
        ):
        r"""
        Find all points within distance r of point(s) x.

        Parameters
        ----------
        x : array_like
            An array of points to query, where each row is an observation.
            NOTE: single query must be reshaped as ``(1, -1)`` format.
        r_sqeuclidean : nonnegative float, optional
            Return neighbors within this square-Euclidean distance.  
        excludeIDx : tuple, optional
            Exclude a range of observation id ``[start, end)`` from search process.
        workers : int, optional
            Number of workers to use for parallel processing.

        Returns
        -------
        flat_indices : ndarray of intp, shape ``(total_matches,)``
            All matched neighbor indices, concatenated across every query point.
        offsets : ndarray of intp, shape ``(n_query + 1,)``
            Point i's matches are ``flat_indices[offsets[i]:offsets[i+1]]``.

        [see, CSR adjacency-list representation]

        """

        cdef:
            np.ndarray[np.double_t, ndim=2] x_arr = np.ascontiguousarray(x, dtype=np.float64)
            ckdtree *cself = self.cself

            np.double_t r_sqeuclidean_ = np.clip(
                r_sqeuclidean, 
                0., DBL_MAX
            ) # clip +Inf to practical safe bound; ensure non-negative

            np.intp_t n, m

        if not np.isfinite(x_arr).all():
            raise ValueError("'x' must be finite, check for nan or inf values")

        n, m = validate_shape(x_arr, cself.m)
        cdef const np.double_t [:, ::1] xx = x_arr

        # tuple unpacking for simpler integer comparisons
        cdef np.intp_t excludeIDx_start, excludeIDx_end
        excludeIDx_start, excludeIDx_end = excludeIDx
        if (
            excludeIDx_start < 0
            or excludeIDx_start > excludeIDx_end
            or excludeIDx_end > cself.n
        ):
            raise ValueError("excludeIDx must satisfy 0 <= start <= end <= n")

        # [[output for 1st query], [output for 2nd query], ..., [output for last query]]
        # we need jagged structure rather than 2D-array, as the number of outputs per query is not fixed.
        cdef vector[vector[np.intp_t]] vvres 
        vvres.resize(n) 

        def _batch_query(np.intp_t start, np.intp_t stop):
            cdef:
                np.intp_t i
                const np.float64_t *pxx = &xx[start, 0]
            with nogil:
                for i in range(stop - start):
                    query_single_point_ball_sqeuclidean_exact(
                        cself,
                        pxx + i * m,
                        r_sqeuclidean_,
                        &vvres[start + i], # fill outputs in corresponding inner list 
                        excludeIDx_start, excludeIDx_end
                    )

        _run_threads(_batch_query, n, workers) # [[THE ACTUAL QUERY HAPPENS HERE]]

        # pass 1: sizes -> offsets (running sum)
        cdef:
            np.intp_t [::1] offsets = np.empty(n + 1, dtype=np.intp)
            np.intp_t i, cnt, total = 0

        offsets[0] = 0
        for i in range(n):
            cnt = <np.intp_t> vvres[i].size()
            total += cnt
            offsets[i + 1] = total

        # pass 2: one bulk memcpy per point, no per-index boxing
        cdef:
            np.intp_t [::1] flat_indices = np.empty(total, dtype=np.intp)
            np.intp_t *pflat = NULL

        if total > 0:
            pflat = &flat_indices[0]

            for i in range(n):
                cnt = <np.intp_t> vvres[i].size()
                if cnt > 0:
                    memcpy(pflat + offsets[i], vvres[i].data(), cnt * sizeof(np.intp_t))

        return np.asarray(flat_indices), np.asarray(offsets)


    # ---------------------------------------





# Helpers:
# ========
import os
from joblib import Parallel, delayed
cdef _run_threads(_thread_func, np.intp_t n, workers: object):
    cdef np.intp_t n_jobs 
    if (not workers) or (workers == 1) or (not os.cpu_count()):
        _thread_func(0, n) # avoid parallel overhead
    else:
        n_jobs = min(
            n,
            workers if workers > 0 else (os.cpu_count() + workers + 1)
        )
        ranges = [
            (j * n // n_jobs, (j + 1) * n // n_jobs)
            for j in range(n_jobs)
        ]
        Parallel(n_jobs=n_jobs, backend='threading')(
            delayed(_thread_func)(start, end) for start, end in ranges
        )


cdef tuple validate_shape(np.ndarray x, np.intp_t pdim) except *:
    """
    Validates that x is 2D and each row has length pdim.
    """
    if x.ndim != 2 or x.shape[1] != pdim:
        raise ValueError("x must be 2D with rows of length {} but "
                         "has shape {}".format(pdim, np.shape(x)))
    return x.shape[0], x.shape[1]


cdef void init_pytree(void *void_tree):
    raise NotImplementedError(
        """
        The Python tree view (`cKDTree.tree`) is not implemented in this internal function.

        The underlying C++ KD-tree can be accessed through Cython:
            `vector[ckdtreenode] *buf = tree.cself.tree_buffer`

        """
    )