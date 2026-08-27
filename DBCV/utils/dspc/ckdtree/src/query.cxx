/*
Adapted from SciPy
[see https://github.com/scipy/scipy/blob/main/scipy/spatial/ckdtree/src/query.cxx]

Copyright (c) 2001-2002 Enthought, Inc. 2003, SciPy Developers
License: 3-clause BSD


HEURISTIC: every node in a kd-tree has a bounding hyperbox;
----------
Our goal is to find the nearest neighbors in the tree for a given query point.
If a hyperbox is farther away than already visited points from the query point,
we can safely skip the hyperbox without checking individual points within it.


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
        thereby reducing average runtime.

    [2.] As we no longer need to store 'to be excluded' points in the output,
        peak memory usage is reduced.

*/


#include "ckdtree_decl.h"
#include "distance.h"

#include <vector>
#include <algorithm>
#include <limits>




union heapcontents {
    ckdtree_intp_t idx_data;
    void *ptr_data;
};

struct heapitem {
    /* Each item is designed to hold one of the following:
     *  - either an integer (e.g. index of an observation)
     *  - or a generic pointer (e.g. pointer to nodeinfo)
     * This improves code reusability.
    */
    double priority;
    heapcontents contents;
};

struct heap {
    /* CUSTOM MIN-HEAP IMPLEMENTATION FOR THE PERFORMANCE-CRITICAL SEARCH PATH
     * (instead of the generic std::priority_queue)
     *
     * The buffer grows as elements are inserted,
     * and avoids freeing memory after each removal to prevent unnecessary reallocations.
     *
     * The whole heap is deallocated at the end of each query.
    */

    std::vector<heapitem> _heap; // flat storage for all heapitems
    ckdtree_intp_t n; // number of active elements
    ckdtree_intp_t space; // buffer size (used + unused)

    heap (ckdtree_intp_t initial_size) : _heap(initial_size) {
        space = initial_size; 
        n = 0; 
    }

    inline void push(heapitem &item) {
        /*
         * put the new item at the end,
         * and sift it upward until the heap structure is restored.    
        */

        ckdtree_intp_t i = n; // current end position
        ckdtree_intp_t i_parent = (i - 1)>>1; // parent of i
        heapitem t;
        
        ++n; // ensure buffer is not full, before insert
        if (n > space) {
            _heap.resize(2*space+1);
            space = _heap.size();
        }
        _heap[i] = item; 

        // this block is only executed when at least one element exists.
        while ((i > 0) && (_heap[i].priority < _heap[i_parent].priority)) {
            t = _heap[i_parent];
            _heap[i_parent] = _heap[i];
            _heap[i] = t;
            i = i_parent;
            i_parent = (i - 1)>>1;
        }
    }

    inline heapitem peek() {return _heap[0];} // O(1) access of item with the lowest priority

    inline void remove() {
        /*
         * take the last element and put it at the top,
         * then sift it downward until the heap structure is restored.     
        */

        heapitem t;
        ckdtree_intp_t i = 0; // current position
        ckdtree_intp_t i_left = 1; // left child of i
        ckdtree_intp_t i_right; // right child of i
        ckdtree_intp_t i_new; // new position

        if (n > 0) {
            _heap[0] = _heap[--n];

            while (i_left < n) {// if i_left is at the end, no need to check i_right
                i_right = i_left + 1;
                i_new = ((i_right < n) && ( _heap[i_right].priority < _heap[i_left].priority)) ? i_right : i_left;

                if (_heap[i_new].priority >= _heap[i].priority) break;

                t = _heap[i_new];
                _heap[i_new] = _heap[i];
                _heap[i] = t;
                i = i_new;
                i_left = 2*i + 1;
            }
        }
    }

    inline heapitem pop() { 
        // peek + remove
        heapitem it = peek();
        remove();
        return it;
    }
};


/* ================================
 * nodeinfo 
 * (heapcontents candidate)
 * ============================
*/

struct nodeinfo {
    /* ZERO DATA DUPLICATION: 
     *  only store a pointer to the underlying tree node;
     *  tree-node attributes can be accessed through this pointer.
    */
    const ckdtreenode *node; 

    ckdtree_intp_t m; // dimension of data
    double min_distance; // current minimum possible distance
    double buf[1]; // [CRUCIAL: buf must be the last declaration]

    inline double * const side_distances() {
        return buf; // per dimension contribution to min_distance
    }

    // initialize a node's side_distances to those of another node
    // (e.g. assign a child's side_distances to its parent's)
    inline void init_plain(const struct nodeinfo * from) {
        std::copy(from->buf, from->buf + m, buf);
        min_distance = from->min_distance;
    }

    // update side_distances at a specific dimension
    inline void update_side_distance(
        const int d, 
        const double new_side_distance
    ) {
        min_distance += new_side_distance - side_distances()[d];
        side_distances()[d] = new_side_distance;
    }
};


struct nodeinfo_pool {/*THE GOOD OLD STRUCT HACK*/ 

    std::vector<char*> pool;

    // memory required to safely fit a node,
    // in multiples of 64
    // (standard cache-line size: 64 bytes)
    ckdtree_intp_t alloc_size; 

    // 64 nodes + some empty room
    // (standard OS memory page size: 64*64 bytes;
    // alloc_size already a multiple of 64)
    ckdtree_intp_t arena_size; 
    // unused bytes left in the current arena
    ckdtree_intp_t arena_remaining; 

    // dimension of data
    // (buf needs to hold m doubles as side_distances)  
    ckdtree_intp_t m;

    char *arena;
    char *arena_ptr;

    nodeinfo_pool(ckdtree_intp_t m) {
        // buf needs to hold side_distances (m doubles),
        // but one is already allocated
        alloc_size = sizeof(nodeinfo) + (m - 1)*sizeof(double);
        alloc_size = 64*(alloc_size/64)+64; 
        arena_size = (64*alloc_size)+4096; 

        // allocate an array of arena_size bytes in heap memory,
        // and store its starting address in arena
        arena = new char[arena_size];
        arena_ptr = arena;
        pool.push_back(arena);
        this->m = m;

        // full arena available at initialization
        arena_remaining = arena_size;
    }

    // delete entire arenas at once, 
    // rather than deleting individual nodeinfo objects.
    ~nodeinfo_pool() {
        for (ckdtree_intp_t i = pool.size()-1; i >= 0; --i)
            delete [] pool[i];
    }

    inline nodeinfo *allocate() {

        if (arena_remaining < alloc_size) {
            // create a new arena, if current one is full
            arena = new char[arena_size];
            arena_ptr = arena;
            arena_remaining = arena_size;
            pool.push_back(arena);
        }

        nodeinfo *node_ = (nodeinfo*)arena_ptr;
        node_->m = m;

        arena_ptr += alloc_size;
        arena_remaining -= alloc_size;

        return node_;
    }

};



/* ================================
 * k-NN search (for a single query)
 * ============================
*/


void
query_single_point_sqeuclidean_exact(
    const ckdtree *self, // the underlying tree
    double *result_distances_sqeuclidean, // pointer to output distances
    ckdtree_intp_t *result_indices, // pointer to output indices
    const double *x, // pointer to query (single observation)
    const ckdtree_intp_t kmax, // number of neighbors
    double distance_upper_bound_sqeuclidean, // distance upper cap
    const ckdtree_intp_t excludeIDx_start,
    const ckdtree_intp_t excludeIDx_end 
    // only indices outside the range [excludeIDx_start, excludeIDx_end) will be considered as candidate neighbors.
    // default is [0,0) for no exclusion.
)
{

    // memory pool to allocate and automatically reclaim nodeinfo structs 
    nodeinfo_pool pool_nodeinfo(self->m);

    /*
     * 'TO BE VISITED' NODES PRIORITY-QUEUE (MIN-HEAP)
     *
     * here priority = distance
     * i.e. the CLOSEST PENDING TREE NODE stays on TOP
     * (and is visited earliest when we need more candidates, by definition).
     *
     * the SIZE of this queue is NOT CAPPED.
    */
    heap waiting(16);

    /*
     * 'VISITED NEIGHBORS' PRIORITY-QUEUE (MIN-HEAP)
     *
     * here priority = -distance [this choice is made to reuse min-heap like a max-heap]
     * i.e. the FARTHEST VISITED NEIGHBOR POINT stays on TOP
     * (and is removed earliest when the queue is full, by definition).
     *
     * the SIZE of this queue is CAPPED at #neighbors;
     * this way, at any moment the queue contains the best up to k candidate neighbors visited so far.
    */ 
    heap neighbors(kmax);

    // loop-invariant variables for the whole query are hoisted out of the traversal loop
    const ckdtree_intp_t m = self->m;
    const double *data = self->raw_data;
    const ckdtree_intp_t *indices = self->raw_indices;
 
    nodeinfo *current_node;
    nodeinfo *far_node;
    double dist_sq, side_distance;
    heapitem queue_item, far_item, candidate_neighbor;
    const ckdtreenode *leaf_node;
    const ckdtreenode *internal_node;

    // [[BEGIN AT THE ROOT OF THE TREE]] 
    current_node = pool_nodeinfo.allocate();
    current_node->node = self->ctree;
    current_node->min_distance = 0;

    for (ckdtree_intp_t i=0; i<m; ++i) {// initialize side_distance one dimension at a time
        side_distance = PlainDist1D::side_distance_from_min_max(
            x, self->raw_mins, self->raw_maxes, i
        );
        side_distance = MinkowskiDistP2sq::distance_p2(side_distance);

        current_node->side_distances()[i] = 0;
        current_node->update_side_distance(i, side_distance);
    }

    for(;;) {// [[THE ACTUAL TRAVERSAL HAPPENS HERE]]

        if (current_node->node->split_dim == -1) {// BRUTE FORCE ON LEAF NODE
            leaf_node = current_node->node;
            const ckdtree_intp_t start_idx = leaf_node->start_idx;
            const ckdtree_intp_t end_idx = leaf_node->end_idx;

            for (ckdtree_intp_t i=start_idx; i<end_idx; ++i) {

                const ckdtree_intp_t idx = indices[i];

                // only indices outside the range [excludeIDx_start, excludeIDx_end) will be considered as candidate neighbors.
                // default is [0,0) for no exclusion.
                // filter using the cheap integer comparison first,
                // to avoid wasting relatively expensive distance calculations.
                if ((idx < excludeIDx_start) || (idx >= excludeIDx_end)) {

                    dist_sq = MinkowskiDistP2sq::point_point_p2(
                        data + m * idx, x,
                        m, 
                        distance_upper_bound_sqeuclidean
                    );

                    if (dist_sq < distance_upper_bound_sqeuclidean) {

                        // current_node is closer than the top of the neighbors queue,
                        // include current node in the neighbors queue;
                        // (if the queue is already full, remove the top before inserting current node.)                     
                        if (neighbors.n == kmax) neighbors.remove();
                        candidate_neighbor.priority = -dist_sq;
                        candidate_neighbor.contents.idx_data = idx;
                        neighbors.push(candidate_neighbor);

                        // update the distance bound, as it becomes tighter
                        if (neighbors.n == kmax)
                            distance_upper_bound_sqeuclidean = -neighbors.peek().priority;

                    }
                }
                
            }

            if (waiting.n > 0) {
                // prepare the next node to visit
                queue_item = waiting.pop();
                current_node = (nodeinfo*)(queue_item.contents.ptr_data);
            }
            else {
                // NO MORE PENDING nodes to visit 
                break; // TERMINATE.
            }
        }

        else {// TREE TRAVERSAL AT INTERNAL NODE

            if (current_node->min_distance > distance_upper_bound_sqeuclidean) {
                // scenario:
                //  the closest possible point of current_node from the query
                //  is already farther than the top of the neighbors queue.
                //
                //  the nodes in the waiting queue have even higher min_distance than current_node.
                //
                //  so we CAN NOT IMPROVE our search anymore
                break; // HENCE, TERMINATE.
            }

            internal_node = current_node->node;
            const ckdtree_intp_t split_dim = internal_node->split_dim;
            const double split = internal_node->split;

            // every internal node has a left and right child,
            // the one that is closer to the query will be dived deep further,
            // the farther one will be placed in the priority queue,
            // and only be searched later if we need more neighbors to meet our requirements.

            /*
             * suppose internal_node is split into `less` and `greater` by dimension j.
             * the bounding boxes of `less` and `greater` overlap with the bounding box of internal_node,
             * in all other dimensions except the newly split j.
             *
             * if x is on the `less` side of the split, then x has to first cross `less` to reach `greater` along axis j,
             * i.e. `greater` will be far_node.
             * if x is on the `greater` side of the split, then x has to first cross `greater` to reach `less` along axis j,
             * i.e. `less` will be far_node.
            */
            far_node = pool_nodeinfo.allocate();
            far_node->init_plain(current_node);

            side_distance = split - x[split_dim]; // signed
            far_node->node = (side_distance > 0) ? internal_node->greater : internal_node->less;

            // prepare the next node to visit;
            // min_distance DOES NOT CHANGE in this step since the query point
            // remains on the same side of the split
            current_node->node = (side_distance <= 0) ? internal_node->greater : internal_node->less;

            // we should place far_node in waiting only when there is a hope for improvement;
            // if far_node is already beyond the current worst-case distance,
            // we can safely discard it.
            side_distance = MinkowskiDistP2sq::distance_p2(side_distance); // squared
            far_node->update_side_distance(split_dim, side_distance);
            if (far_node->min_distance <= distance_upper_bound_sqeuclidean) {
                far_item.priority = far_node->min_distance;
                far_item.contents.ptr_data = (void*) far_node;
                waiting.push(far_item);
            }

        }
    }

    // [[HEAP SORT]]
    ckdtree_intp_t nnb = neighbors.n;
    
    // farthest neighbor is at the top of the queue
    for (ckdtree_intp_t i = nnb - 1; i >= 0; --i) { 
        candidate_neighbor = neighbors.pop();
        result_indices[i] = candidate_neighbor.contents.idx_data;
        result_distances_sqeuclidean[i] = -candidate_neighbor.priority;
    }

    // when the number of neighbors found is less than required,
    // fill with placeholder values to maintain the required shape.
    static const double inf = std::numeric_limits<double>::infinity();
    const ckdtree_intp_t n = self->n;

    for (ckdtree_intp_t i = nnb; i < kmax; ++i) {
        result_indices[i] = n;
        result_distances_sqeuclidean[i] = inf;
    }

    // WE DO NOT RETURN ANY VALUE.
    // OUTPUT IS STORED IN PREDEFINED ARRAYS THROUGH POINTERS.

}



