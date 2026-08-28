/*
Adapted from SciPy
[see https://github.com/scipy/scipy/tree/main/scipy/spatial]

Copyright (c) 2001-2002 Enthought, Inc. 2003, SciPy Developers
License: 3-clause BSD

*/

#ifndef CKDTREE_CPP_DECL
#define CKDTREE_CPP_DECL


#include <numpy/npy_common.h>
#include <cmath>
#include <vector>


#define ckdtree_intp_t npy_intp
#define ckdtree_fmin(x, y) fmin(x, y)
#define ckdtree_fmax(x, y) fmax(x, y)
#define ckdtree_fabs(x) fabs(x)





/* 
* =================================================================
OUR CONTRIBUTION @idnantimar
----------------

    We introduce an additional exclude filter on Index during search traversal,
    rather than returning unfiltered results and performing post-processing.

    The combined time complexity of search + filtering remains the same in the worst case
    (as we are doing the same job with a different execution order),
    but the advantages are:

    [1.] Integer index comparison is almost always cheaper than floating-point distance calculations.
        When we know in advance which indices are not required,
        we can short-circuit the corresponding iterations in the brute-force loop of leaf nodes,
        thereby reducing the average runtime.

    [2.] As we no longer need to store 'to be excluded' points in the output,
        peak memory usage is reduced.

* ==============================================================
*/
static inline bool
is_allowed(
    const ckdtree_intp_t idx,
    const ckdtree_intp_t excludeIDx_start = 0,
    const ckdtree_intp_t excludeIDx_end = 0
)
{
    return (idx >= excludeIDx_end) || (idx < excludeIDx_start);
}

// shared by both k-NN and radius search leaf-loop filtering.

// ---------------------------------------




/* 
* =================================================================
* Build methods in C++ for better speed and GIL release.

* DO NOT MODIFY SciPy's implementation in this section.
* [CRUCIAL FOR UPSTREAM CONSISTENCY IN FUTURE]
* ==============================================================
*/

struct ckdtreenode { // [[INDIVIDUAL NODES]]
    
    ckdtree_intp_t split_dim;    // split axis
    ckdtree_intp_t children;    // number of points under this node
    double split;    // split value along split_dim
    ckdtree_intp_t start_idx, end_idx;    // [start, end) for this node sub-tree in raw_indices master array
    ckdtreenode *less;    // child holding points < split
    ckdtreenode *greater;    // child holding points >= split
    ckdtree_intp_t _less;    // buffer offset of less, pre-pointer-fixup
    ckdtree_intp_t _greater;    // buffer offset of greater, pre-pointer-fixup

};

struct ckdtree {

    // [[TREE STRUCTURE]]
    std::vector<ckdtreenode> *tree_buffer;    // flat storage for all nodes
    ckdtreenode *ctree;    // pointer to the root node

    // [[META DATA]]
    double *raw_data;    // point coordinates, n * m contiguous doubles
    ckdtree_intp_t n, m;    // (n, m) shape of raw_data
    ckdtree_intp_t leafsize;    // brute-force cutoff for leaf nodes
    double *raw_maxes;    // per-dimension global max
    double *raw_mins;    // per-dimension global min
    ckdtree_intp_t *raw_indices;    // point indices master array, reordered by build()
    double *raw_boxsize_data;    // periodic box size; (not needed for plain geometry)
    ckdtree_intp_t size;    // number of nodes in the tree
    
};


int
build_ckdtree( // [[THE ACTUAL BUILD ROUTINE C++]]

    ckdtree *self, 
    ckdtree_intp_t start_idx, 
    ckdtree_intp_t end_idx,
    double *maxes, 
    double *mins, 
    int _median, 
    int _compact
);


// ---------------------------------------

/* 
* =================================================================
* Query methods in C++ for better speed and GIL release.

* MODIFY to meet our custom requirements.
* ==============================================================
*/

void
query_single_point_sqeuclidean_exact( // [[k-NN SEARCH]]
    const ckdtree *self, // the underlying tree
    double *result_distances_sqeuclidean, // pointer to output distances
    ckdtree_intp_t *result_indices, // pointer to output indices
    const double *x, // pointer to query (single observation)
    const ckdtree_intp_t kmax, // number of neighbors
    double distance_upper_bound_sqeuclidean, // distance upper cap
    const ckdtree_intp_t excludeIDx_start = 0,
    const ckdtree_intp_t excludeIDx_end = 0
    // only indices outside the range [excludeIDx_start, excludeIDx_end) will be consired as candidate neighbors.
    // default is [0,0) for no exclusion.
);


void
query_single_point_ball_sqeuclidean_exact( // [[RADIUS SEARCH]]
    const ckdtree *self, // the underlying tree
    const double *x, // pointer to query (single observation)
    const double r_sqeuclidean, // square-Euclidean radius
    std::vector<ckdtree_intp_t> *result_indices, // pointer to matched neighbor indices output
    const ckdtree_intp_t excludeIDx_start = 0,
    const ckdtree_intp_t excludeIDx_end = 0
    // only indices outside the range [excludeIDx_start, excludeIDx_end) will be considered as candidate neighbors.
    // default is [0,0) for no exclusion.
);

// ---------------------------------------


#endif