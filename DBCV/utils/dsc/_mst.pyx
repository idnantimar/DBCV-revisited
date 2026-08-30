'''
Adapted from HDBSCAN 
[see HDBSCAN commit 7b2f0e0dcff6ef99b0c976d1471cbeec99da49a9]

Authors: Leland McInnes, Steve Astels
Copyright (c) 2015, Leland McInnes
License: 3-clause BSD


OUR CONTRIBUTION: license: MIT Copyright (c) 2026 idnantimar
-----------------
    [1.] A dense distance matrix consumes n * n memory.
        A condensed array consumes n * (n - 1) / 2 memory, by storing only the upper-triangular block.

        The MST-building routine requires pairwise distances (i, j) as input,
        which are fully recoverable from a condensed array due to distance-matrix symmetry.

        We have modified the MST builder to use a condensed-distance array as input,
        and run Prim's method with additional index tricks.
        This can save ~50% of peak memory, at a cost of some relatively cheap index manipulation in the O(n^2) hot loop.

    [2.] Instead of returning (source, target, weight), we choose to return ((source, target), weight). 
        This apparently simple change avoids int ---> float conversion during return, 
        while downstream code can directly consume the indices without the float ---> int round-trip overhead and memory allocation.

'''

# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: initializedcheck=False



import numpy as np
cimport numpy as np

from libc.float cimport DBL_MAX
from libc.math cimport sqrt


cpdef tuple mst_linkage_core_condensed(
    np.ndarray[np.double_t, ndim=1] dist_matrix_condensed
):
    """
    Minimum Spanning Tree (MST) for a complete graph stored in scipy.spatial.distance.pdist(...) -style condensed ordering.

    ALGORITHM: Prim's method.

    OUR CONTRIBUTION: 
    -----------------
        [1.] A dense distance matrix consumes n * n memory.
            A condensed array consumes n * (n - 1) / 2 memory, by storing only the upper-triangular block.

            The MST-building routine requires pairwise distances (i, j) as input,
            which are fully recoverable from a condensed array due to distance-matrix symmetry.

            We have modified the MST builder to use a condensed-distance array as input,
            and run Prim's method with additional index tricks.
            This can save ~50% of peak memory, at a cost of some relatively cheap index manipulation in the O(n^2) hot loop.

        [2.] Instead of returning (source, target, weight), we choose to return ((source, target), weight). 
            This apparently simple change avoids int ---> float conversion during return, 
            while downstream code can directly consume the indices without the float ---> int round-trip overhead and memory allocation.


    Parameters
    ----------
    dist_matrix_condensed : ndarray, shape (n * (n - 1) / 2,)
        Upper-triangular pairwise distances in Scipy's `pdist(...)` style ordering.

    Returns
    -------
    Tuple:
        - ndarray, shape (n - 1, 2)
            MST edge endpoints as (source, target).

        - ndarray, shape (n - 1,)
            MST edge weights.

    The two arrays are row-wise aligned: endpoints[i] and weights[i] correspond to the same MST edge.
 
    """
    # solve the quadratic eqn:  m = n(n-1)/2
    cdef np.intp_t m = dist_matrix_condensed.shape[0]
    cdef np.intp_t n = <np.intp_t>((1.0 + sqrt(1.0 + 8.0 * m)) / 2.0)

    if n < 2: return np.empty((0, 2), dtype=np.intp), np.empty(0, dtype=np.float64)
    
    cdef np.intp_t n_1 = n - 1
    cdef np.intp_t n_2 = n - 2

    dist_matrix_condensed = np.ascontiguousarray(dist_matrix_condensed, dtype=np.float64)

    # graph with n nodes produces n - 1 mst edges: (from, to), weight
    cdef np.ndarray[np.intp_t, ndim=2] nodes_arr = np.empty(
        (n_1, 2), dtype=np.intp
    )
    cdef np.ndarray[np.double_t, ndim=1] edges_arr = np.empty(
        n_1, dtype=np.float64
    )

    cdef np.ndarray[np.npy_bool, ndim=1] not_in_tree_arr = np.ones(
        n, dtype=np.bool_
    )
    cdef np.ndarray[np.intp_t, ndim=1] current_sources_arr = np.empty(
        n, dtype=np.intp
    )
    # initializes distances as effectively +Inf
    cdef np.ndarray[np.double_t, ndim=1] current_distances_arr = np.full(
        n, DBL_MAX, dtype=np.float64
    )

    cdef np.double_t *distance_ptr = <np.double_t *> dist_matrix_condensed.data
    cdef np.double_t *current_distances = <np.double_t *> current_distances_arr.data
    cdef np.intp_t *current_sources = <np.intp_t *> current_sources_arr.data
    cdef np.npy_bool *not_in_tree = <np.npy_bool *> not_in_tree_arr.data

    cdef np.intp_t[:, ::1] nodes = nodes_arr
    cdef np.double_t *edges = <np.double_t *> edges_arr.data  

    # k is place-holder for current node 
    # Initialize the algorithm with the very first observation being inside Tree
    cdef np.intp_t k = 0 # current_node
    not_in_tree[k] = False

    cdef np.intp_t source_node, new_node, source_j
    cdef np.intp_t i, j, idx
    cdef np.double_t best_distance, d, dist_j

    with nogil:
        for i in range(1, n):
            best_distance = DBL_MAX
            new_node = 0
            source_node = 0

            # HEURISTIC :
            # pdist() stores the upper-triangular pairwise distances in the row-major style:
            #   (0,1), (0,2), ..., (0,n-1), (1,2), ..., (1,n-1), ..., (n-2,n-1).
            #
            # for any current_node k, 
            #   start from (0, k) position, 
            #   which lies on the very first row of upper-triangular block of distance matrix.
            #   - move down one step a time over column k
            #   - stop at main diagonal (k, k)
            #   - keep moving right side over row k 
            idx = k - 1

            # Case-1 : k > j ; available (j, current_node) in dist_matrix_condensed
            for j in range(k):
                if not_in_tree[j]: 
                    d = distance_ptr[idx]
                    
                    dist_j = current_distances[j]
                    source_j = current_sources[j]
                    if d < dist_j:
                        dist_j = d
                        source_j = k
                        current_distances[j] = dist_j
                        current_sources[j] = source_j

                    if dist_j < best_distance:
                        best_distance = dist_j
                        source_node = source_j
                        new_node = j

                idx += (n_2 - j)

            # Case-2 : (k, k) ; skip the main diagonal, turn right
            idx += 1

            # Case-3 : k < j ; available (current_node, j) in dist_matrix_condensed
            for j in range(k + 1, n):
                if not_in_tree[j]:
                    d = distance_ptr[idx]

                    dist_j = current_distances[j]
                    source_j = current_sources[j]
                    if d < dist_j:
                        dist_j = d
                        source_j = k
                        current_distances[j] = dist_j
                        current_sources[j] = source_j

                    if dist_j < best_distance:
                        best_distance = dist_j
                        source_node = source_j
                        new_node = j
                            
                idx += 1

            nodes[i - 1, 0] = source_node
            nodes[i - 1, 1] = new_node
            edges[i - 1] = best_distance

            k = new_node
            not_in_tree[k] = False

    return nodes_arr, edges_arr
