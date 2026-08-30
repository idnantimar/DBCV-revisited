/*
Adapted from SciPy
[see https://github.com/scipy/scipy/blob/main/scipy/spatial/ckdtree/src/query_ball_point.cxx]

Copyright (c) 2001-2002 Enthought, Inc. 2003, SciPy Developers
License: 3-clause BSD


HEURISTIC: every node in a kd-tree has a bounding hyperbox.
----------
Our goal is to find all the points in the tree lying within a given radius of a given query point.
* Case-1 If the closest point of a hyperbox is outside the search radius of the query point,
we can safely skip the hyperbox without checking individual points within it.
* Case-2 If the farthest point of a hyperbox is within the search radius of the query point,
we can surely include the entire hyperbox without checking individual points within it.
Until we can decide Case-1 or Case-2, we keep splitting the box into smaller sub-boxes.


OUR CONTRIBUTION license: MIT Copyright (c) 2026 idnantimar
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

*/


#include "ckdtree_decl.h"
#include "distance.h"
 
#include <vector>
#include <stdexcept>



struct qR_stack_item {
    /*
    * The logical unit that repeats over and over
    * is to keep track of the maximum and minimum distances
    * between a fixed query point
    * and a hyperrectangle, as that rectangle is successively split.
    * 
    * [unlike a general rect-rect tracker, there is no "which rectangle is being split?"
    * -- the query point is a degenerate rectangle; 
    * only the bounding rectangle of the tree node can be split.]
    */

    ckdtree_intp_t split_dim; // split axis of the current node

    double min_along_dim, max_along_dim; // lower and upper boundary along split_dim 
    double min_distance, max_distance; // minimum and maximum distance between the point and the rectangle
    
};
 

// in DFS,
//  - when we backtrack from `LESS` child of a node, we are yet to explore its `GREATER` child.
//  - when we backtrack from `GREATER` child of a node, we are done with its subtree; we ascend one step up to its parent.
const ckdtree_intp_t LESS = 1;
const ckdtree_intp_t GREATER = 2;



/* ==================================
 * Distance Tracker

 * TODO: SciPy suggested an incremental update rule for tree traversal,
 * but their own implementation initializes inaccurate_distance_limit = max_distance;
 *
 * since the distance along any dimension to any subtree can never exceed the global maximum,
 * the incremental rule never fires and always performs distance recomputation.
 *
 * [ we have simply removed that dead branch,
 *   unless we decide on some heuristic for the inaccurate_distance_limit threshold,
 *   at the cost of a small floating-point deviation from SciPy's behaviour. ]
 * ==============================
*/


struct query_to_Rectangle_Tracker_sqeuclidean_exact {
    /* CUSTOM STACK IMPLEMENTATION FOR THE PERFORMANCE-CRITICAL SEARCH PATH
     * (instead of generic std::stack)
    
     * unlike k-nn search (where we need k best neighbors),
     * radius search is an exhaustive search; the order of the output does not matter.
     * we implement a depth-first search here,
     * and use a stack for descend/backtrack bookkeeping.
    */
 
    const Rectangle& q; // the query point, represented as a zero-volume rectangle (mins == maxes); fixed
    Rectangle current_node; // the tree node's bounding box; clipped and restored as we descend/backtrack
 
    double min_distance; 
    double max_distance;
 
    ckdtree_intp_t stack_size; // number of items in stack currently
    ckdtree_intp_t stack_max_size; // buffer size (used + unused)
    std::vector<qR_stack_item> stack_arr; // flat storage for all stack_items
    qR_stack_item *stack; // pointer to the base of the stack

    double upper_bound_sqeuclidean; // [[SEARCH RADIUS (`<=` TYPE)]]
 
    void _resize_stack() {
        // helps extending the stack, when it is full
        stack_arr.resize(2*stack_max_size);
        stack = &stack_arr[0];
        stack_max_size = stack_arr.size();
    };
 
    query_to_Rectangle_Tracker_sqeuclidean_exact(
        const Rectangle& _q, const Rectangle& _current_node,
        const double _upper_bound_sqeuclidean
    ) : 
        q(_q), current_node(_current_node), 
        stack_arr(16), // stack of size 16 is more than enough for DFS in almost all practical data
        upper_bound_sqeuclidean(_upper_bound_sqeuclidean)
    {
 
        if (q.m != current_node.m) {
            const char *msg = "query and current_node have different dimensions";
            throw std::invalid_argument(msg); 
        }
 
        stack_max_size = stack_arr.size(); 
        stack_size = 0; // STACK IS STILL EMPTY AT INITIALIZATION
        stack = &stack_arr[0]; // stack is ready to store data at its base
  
        // AT INITIALIZATION, COMPUTE GLOBAL MINIMUM & MAXIMUM DISTANCE BETWEEN THE TREE AND GIVEN QUERY
        MinkowskiDistP2sq::rect_rect_p2(q, current_node, &min_distance, &max_distance);

        /* input data is already ensured to be finite, 
         * and for a realistic data used for density based clustering,
         * float-64 overflow (~1.8e+308) is practically not possible from sum-squares;
         * overflow check is removed intentionally to enable `-ffinite-math-only`.
        */
    };

    // [[BY ARCHIVING THE RELEVANT INFO BEFORE DOING A DESCEND IN DFS, 
    // WE CAN UNDO THE CHANGES WITHOUT RECOMPUTATION DURING BACKTRACK]]
    void push(
        const ckdtree_intp_t direction, // bookkeeping: are we descending `LESS` or `GREATER`?
        const ckdtree_intp_t split_dim, const double split_val
        // bounding hyperbox of a child node overlaps with its parent's at every axis except split_dim.
    ) {
 
        // before push, ensure there is enough room
        if (stack_size == stack_max_size) _resize_stack();
 
        // state at the moment of split;
        // min_distance and max_distance here still refer to the distance observed from last split.
        qR_stack_item *item = &stack[stack_size++];
        item->min_distance = min_distance; 
        item->max_distance = max_distance; 
        item->split_dim = split_dim;
        item->min_along_dim = current_node.mins()[split_dim];
        item->max_along_dim = current_node.maxes()[split_dim];

        /* [UPDATE DISTANCE TRACKER USING split_val AS THE HYPERBOX CHANGES]
        
         * suppose our hyperbox along split_dim was [a,b] at the moment of split : [a|b]
         *  - if we are descending towards `LESS`, updated hyperbox is [a|
         *  - if we are descending towards `GREATER`, updated hyperbox is |b]
         * there is no changes in axis other than split_dim
        */ 
        (direction == LESS ? current_node.maxes() : current_node.mins())[split_dim] = split_val;
  
        MinkowskiDistP2sq::rect_rect_p2(q, current_node, &min_distance, &max_distance);

    };
 
    // [[UNDO THE CHANGES DONE BY `push()`]]
    inline void pop() {
        
        qR_stack_item* item = &stack[--stack_size];

        min_distance = item->min_distance;
        max_distance = item->max_distance;
 
        current_node.mins()[item->split_dim] = item->min_along_dim;
        current_node.maxes()[item->split_dim] = item->max_along_dim;

    };

    // just two convenience wrappers around push
    inline void push_less_of(const ckdtreenode *node) {
        push(LESS, node->split_dim, node->split);
    };
    inline void push_greater_of(const ckdtreenode *node) {
        push(GREATER, node->split_dim, node->split);
    };
 
};
 


/* ==================================
 * radius search (for a single query)
 * ==============================
*/


/* HEURISTIC: every node in a kd-tree has a bounding hyperbox.
 * Our goal is to find all the points in the tree lying within a given radius of a given query point.
 *  - Case-1 If the closest point of a hyperbox is outside the search radius of the query point,
    we can safely skip the hyperbox without checking individual points within it.
 *  - Case-2 If the farthest point of a hyperbox is within the search radius of the query point,
    we can surely include the entire hyperbox without checking individual points within it.
 * Until we can decide Case-1 or Case-2, we keep splitting the box into smaller sub-boxes.
*/

static void
traverse_checking(
    const ckdtree *self, // the underlying tree
    const ckdtreenode *node, // the node to start with
    std::vector<ckdtree_intp_t> &results, // reference to output array
    query_to_Rectangle_Tracker_sqeuclidean_exact *tracker,
    const ckdtree_intp_t excludeIDx_start,
    const ckdtree_intp_t excludeIDx_end
)
{
    double d;
    const ckdtree_intp_t *indices = self->raw_indices;
    const ckdtree_intp_t start = node->start_idx;
    const ckdtree_intp_t end = node->end_idx;
    const double radius = tracker->upper_bound_sqeuclidean;

    if (tracker->min_distance > radius) { 
        // POINTS IN THIS NODE ARE ALREADY OUTSIDE THE SEARCH RADIUS,
        // EXIT; 
        // NO OUTPUT CANDIDATE FROM THIS NODE AND ITS SUB-TREE.
        return;
    }
    else if (tracker->max_distance <= radius) {
        // POINTS IN THIS NODE ARE ALREADY WITHIN THE SEARCH RADIUS,
        // BUT FILTER TO BE APPLIED BEFORE CONFIRMING CANDIDATURE IN OUTPUT.
        
        for (ckdtree_intp_t i = start; i < end; ++i) {
            if (is_allowed(indices[i], excludeIDx_start, excludeIDx_end)) results.push_back(indices[i]);
        }

        // Optimization: when cKDTree builder splits a node into two children,
        // the raw_indices master array is only reshuffled within [node->start_idx, node->end_idx),
        // but no element enters or exits the boundary.
        // so, a flat loop suffices rather than recursion over its children.

        // in brute-force loop of leaf nodes SciPy uses `<=` check.
        // so, in bulk-inclusion check also `<` is replaced safely with `<=` without any change in behaviour.

    }
    else {// INDECISIVE CALL

        if (node->split_dim == -1) {// BRUTE FORCE ON LEAF NODE

            const double *coords = tracker->q.mins();
            const double *data = self->raw_data;
            const ckdtree_intp_t m = self->m;

            for (ckdtree_intp_t i = start; i < end; ++i) {

                if (is_allowed(indices[i], excludeIDx_start, excludeIDx_end)) {
                    d = MinkowskiDistP2sq::point_point_p2(
                        data + indices[i] * m, coords, 
                        m, 
                        radius
                    );
                    if (d <= radius) results.push_back(indices[i]);
                }
            }
        }
        else {

            // DFS ON `LESS` BRANCH
            tracker->push_less_of(node);
            traverse_checking(self, node->less, results, tracker, excludeIDx_start, excludeIDx_end);
            tracker->pop();

            // DFS ON `GREATER` BRANCH
            tracker->push_greater_of(node);
            traverse_checking(self, node->greater, results, tracker, excludeIDx_start, excludeIDx_end);
            tracker->pop();

        }
    }
}
 

void
query_single_point_ball_sqeuclidean_exact(
    const ckdtree *self, // the underlying tree
    const double *x, // pointer to query (single observation)
    const double r_sqeuclidean, // square-Euclidean radius
    std::vector<ckdtree_intp_t> *result_indices, // pointer to matched neighbor indices output
    const ckdtree_intp_t excludeIDx_start,
    const ckdtree_intp_t excludeIDx_end
    // only indices outside the range [excludeIDx_start, excludeIDx_end) will be considered as candidate neighbors.
    // default as [0,0) for no exclusion.
)
{
    const ckdtree_intp_t m = self->m;
 
    // [intialize the rectangle bounding the whole tree]
    // [[CRITICAL: THE STARTING NODE AND HYPERBOX MUST BE INITIALIZED IN SYNC]]
    Rectangle rect(m, self->raw_mins, self->raw_maxes);
    const ckdtreenode *root_node = self->ctree;

    // [query point is considered as degenerate rectangle]
    Rectangle point(m, x, x);
 
    query_to_Rectangle_Tracker_sqeuclidean_exact tracker(point, rect, r_sqeuclidean);
    traverse_checking(
        self, root_node, 
        *result_indices, &tracker,
        excludeIDx_start, excludeIDx_end
    );

    // WE DO NOT RETURN ANY VALUE.
    // OUTPUT IS STORED IN PREDEFINED ARRAYS THROUGH POINTERS.

    // THE METHOD DOES NOT COMPUTE DISTANCES FROM INDIVUDUAL POINTS TO QUERY (UNLESS ON LEAF NODE);
    // HENCE IT IS NOT POSSIBLE TO RETURN DISTANCES ALONG WITH INDICES.

}
