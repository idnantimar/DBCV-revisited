import numpy as np
from numba import njit
from math import sqrt
from itertools import compress
from scipy.spatial.distance import pdist
from hdbscan._hdbscan_linkage import mst_linkage_core
from scipy.spatial import cKDTree
from joblib import Parallel, delayed

import numpy.typing as npt
from typing import List, Tuple, Optional

import logging
logger = logging.getLogger(__name__)


# Flags indicating possible scoring outcomes
_NOT_ENOUGH_CLUSTERS = -2
_ALL_NOISE = -1
_SUCCESS = 0
EXIT = {
    _ALL_NOISE: "⚠ All points assigned to noise.",
    _NOT_ENOUGH_CLUSTERS: "⚠ Not enough clusters: must have at least two.",
}

# default dtypes
FLOAT_DTYPE = np.float64
IDX_DTYPE = np.intp



def DBCV_score(
        X: npt.NDArray[np.floating],
        labels: npt.NDArray[np.integer],
        *,
        ind_clust_scores: bool = False,
        n_jobs: int = 1,
    ) -> Optional[Tuple[float, Optional[npt.NDArray[np.floating]]]]:
    """
    Compute the aggregate DBCV score and, optionally, individual cluster scores for the given coordinates and cluster labels.
    [see, `intracluster_analysis()` & `intercluster_analysis()` for details.]

    Ref: "Moulavi, D., Jaskowiak, P. A., Campello, R. J. G. B., Zimek, A. & Sander, J. Density-based clustering validation. SIAM Int. Conf. Data Min. 2014, SDM 2014 2, 839-847 (2014)"
    https://www.researchgate.net/publication/260333211_Density-Based_Clustering_Validation
    

    Parameters
    ----------
    X : npt.NDArray[np.floating]
        2-D array of coordinates with shape (n, d), 
        where n is the total number of points (clustered + noise) and d is the dimensionality of the data.

    labels : npt.NDArray[np.integer]
        1-D array of cluster labels with shape (n,), corresponding to the rows of X. 
        NOTE : 
            Noise points must be labelled -1.
            Any cluster having size < 3 will be relabelled as noise (see, https://rdrr.io/cran/dbscan/src/R/dbcv.R).

    ind_clust_scores : bool, default=False
        If True, return the individual DBCV score for each cluster in addition to the aggregate DBCV score. 
        If False, only the aggregate DBCV score is returned and the second return value is None.

    n_jobs : parameters to pass to the `joblib.Parallel(...)`.

    
    Returns
    -------
    tuple or None
        On success, returns:
            - float: Aggregate DBCV score.
            - npt.NDArray[np.floating] or None: 
                If `ind_clust_scores=True`, individual cluster scores are returned.
                The scores are sorted by cluster labels, not in original observed order.
                If `ind_clust_scores=False`, returns None.

        Returns None if the data cannot be scored, due to not having enough non-noise clusters.

        
    EDGE CASES
    ----------
    For a cluster i,
    separation > 0, sparseness = 0 : cluster_scores[i] = +1 ; heuristically the best case
    separation = 0, sparseness > 0 : cluster_scores[i] = -1 ; heuristically the worst case

    separation = 0, sparseness = 0 : cluster_scores[i] not defined in the original paper by Moulavi et al.
        The existing HDBSCAN implementation also does not suggest any fallback.
        In such case, NumPy raises a RuntimeWarning and 
        we return None for the aggregate DBCV score and individual cluster scores are returned for downstream diagnostics.
              
    """

    X, labels = np.asarray(X), np.asarray(labels)

    ## Format the data to make clusters contiguous
    (
        status, 
        cluster_sort, cluster_groups, cluster_bounds, 
        n_samp, N_clust
    ) = _format_data(
        X, labels,
    )

    if status != _SUCCESS :
        logger.warning(EXIT[status])
        return None

    ## Sparseness calculation and find core points
    (
        sparseness,
        core_pts, core_dists_arr
    ) = intracluster_analysis(
        N_clust, cluster_groups,
        n_jobs=n_jobs,
    )
    logger.info("Intra-Cluster Analysis : SUCCESS ✓")
    
    ## Format core points for intercluster analysis        
    (
        core_cluster_sort, 
        core_cluster_groups, core_cluster_bounds
    ) = _format_core_points(
        cluster_sort, cluster_bounds, core_pts,
    )

    ## Separation calculation
    separation = intercluster_analysis(
        N_clust,
        core_cluster_sort, core_cluster_groups, core_cluster_bounds,
        core_dists_arr,
        n_jobs=n_jobs,
    )
    logger.info("Inter-Cluster Analysis : SUCCESS ✓")

    ## Compute individual and aggregate DBCV scores
    return _weighted_score(
        n_samp,
        sparseness, separation,
        cluster_bounds,
        ind_clust_scores,
    )


def _weighted_score(
        n_samp: int,
        sparseness: npt.NDArray[np.floating],
        separation: npt.NDArray[np.floating],
        cluster_bounds: npt.NDArray[np.integer],
        ind_clust_scores: bool = False,
    ) -> Tuple[float, Optional[npt.NDArray[np.floating]]]:
    """
    Performs weighted averaging of individual cluster scores to yield the aggregate DBCV score
    according to the definitions in Moulavi et al.

    Optionally returns individual scores if desired.

    """
    # Optimization: use NumPy vectorization instead of a Python loop over clusters.
    with np.errstate(invalid='warn'):
        cluster_scores = (
            (separation - sparseness) /
            np.maximum(separation, sparseness)
        )

    # For a cluster i,
    #   separation > 0, sparseness = 0 : cluster_scores_i = +1, heuristically the best case
    #   separation = 0, sparseness > 0 : cluster_scores_i = -1, heuristically the worst case
    #
    #   separation = 0, sparseness = 0 : cluster_scores_i not defined in the original paper by Moulavi et al.
    #       The existing HDBSCAN implementation also does not suggest any fallback.
    #       In such case, we raise a warning and return individual cluster scores for downstream diagnostics.
    nan_clusters = np.flatnonzero(np.isnan(cluster_scores))
    if len(nan_clusters):
        logger.warning(
            "⚠ %d cluster(s) %s have separation=sparseness=0; DBCV_val undefined, returning per-cluster diagnostics instead.",
            len(nan_clusters), nan_clusters.tolist(),
        )
        DBCV_val = None
        ind_clust_scores = True
    else: DBCV_val = np.sum(
        (np.diff(cluster_bounds) / n_samp) *
        cluster_scores
    ).item()

    return ( 
        DBCV_val, 
        cluster_scores if ind_clust_scores else None
    )




## =======================================================
# Density Sparseness of a Cluster (DSC)
## =======================================================
def _dsc_base(
        cluster: npt.NDArray[np.floating], 
    ) -> Tuple[
        float,
        npt.NDArray[np.integer],
        npt.NDArray[np.floating]
    ]: 
    """
    intracluster analysis base function for joblib.Parallel(...).
    """
    # pairwise distances 
    intra_clust_matrix_condensed = pdist(
        cluster,
        metric="euclidean",
    )

    # Optimization: operate directly on pdist()'s condensed distance vector of len n(n-1)/2 
    # instead of materializing an n×n distance matrix. 
    all_pts_core_dists = APCD(intra_clust_matrix_condensed, cluster.shape[1])

    # Optimization: applies the maximum operation only to the n(n-1)/2 distance-core_i-core_j tuples,
    # without first materializing an n×n distance matrix.
    intraclust_MRD_matrix = MRD(
        intra_clust_matrix_condensed,
        all_pts_core_dists,
    )

    # Optimization: generic Kruskal-based MST builder is replaced with 
    # Prim's-based implementation for dense MRD matrix.
    sparseness, core_pts = _MST_builder_HDBSCAN(intraclust_MRD_matrix)

    return sparseness, core_pts, all_pts_core_dists[core_pts]


def intracluster_analysis(
        N_clust: int, 
        cluster_groups: List[npt.NDArray[np.floating]],
        **kwargs
    ) -> Tuple[
        npt.NDArray[np.floating],
        List[npt.NDArray[np.integer]],
        npt.NDArray[np.floating]
    ]:
    """
    Computes the all-points core distance, identifies core points, and computes the sparseness value for each cluster 
    according to the definitions in Moulavi et al.

    
    Args:
    -----
    N_clust : int
        Number of non-noise clusters to be analyzed.

    cluster_groups : List[npt.NDArray[np.floating]]
        List of coordinates arrays containing the observations belonging to each non-noise cluster.
        [see, `_format_data()` for details.]


    Returns:
    --------
    Tuple containing:
        - npt.NDArray[np.floating]:
            Sparseness value for each cluster.

        - List[npt.NDArray[np.integer]]:
            Indices of the core points (i.e. degree >1)  in each cluster.

        - npt.NDArray[np.floating]:
            The all-points core distances for the core points, combined from all clusters.


    WORKHORSE: https://github.com/scikit-learn-contrib/hdbscan/blob/master/hdbscan/_hdbscan_linkage.pyx

    """

    sparseness = np.empty(N_clust, dtype=FLOAT_DTYPE)
    core_dists_arr = []
    core_pts = []

    results = Parallel(
        n_jobs=kwargs.get("n_jobs", 1),
        return_as="generator",
        prefer="threads",
    )(delayed(_dsc_base)(cluster) for cluster in cluster_groups)

    for id, val_pts_dists in enumerate(results):
        sparseness[id] = val_pts_dists[0]
        core_pts.append(val_pts_dists[1])
        core_dists_arr.append(val_pts_dists[2])

    # Combine the list of core-distance arrays for each cluster into a single array,
    # while maintaining the cluster order.
    core_dists_arr = np.concat(core_dists_arr, dtype=FLOAT_DTYPE)

    return (
        sparseness,
        core_pts, core_dists_arr
    )


@njit(cache=True)
def APCD(
        dist_matrix_condensed: npt.NDArray[np.floating],
        d: int,
    ) -> npt.NDArray[np.floating]:
    """
    Computes the all-points core distance according to the DBCV definition.

    The all-points core distance of point i is

        core(i) = [
            mean_{j != i} dist(i,j)^(-d)
        ]^(-1/d)

    where n is the number of points in the cluster and d is the
    dimensionality of the data.
    """
    # Optimization: operate directly on pdist()'s condensed distance vector of len n(n-1)/2 
    # instead of materializing an n×n distance matrix.
    p = - d
    n = int(0.5 * (1 + sqrt(1 + 8 * len(dist_matrix_condensed)))) # quadratic eqn: condensed_size = n(n-1)/2
    all_pts_core_dists = np.zeros(n, dtype=dist_matrix_condensed.dtype)

    # pdist() stores the upper-triangular pairwise distances in the row-major ordering:
    #   (0,1), (0,2), ..., (0,n-1), (1,2), ..., (1,n-1), ..., (n-2,n-1).
    # Each distance contributes to the core-distance sum of both endpoints.
    idx = 0
    with np.errstate(divide='ignore'): # duplicate point → APCD = 0 (see, https://github.com/scikit-learn-contrib/hdbscan/issues/127)
        for i in range(n - 1): # row id
            for j in range(i + 1, n):  # upper-triangular column id
                w = (dist_matrix_condensed[idx] ** p) 
                all_pts_core_dists[i] += w
                all_pts_core_dists[j] += w

                idx += 1

    all_pts_core_dists = (
        all_pts_core_dists / (n - 1)
    ) ** (1.0 / p) 

    return all_pts_core_dists 


@njit(cache=True)
def MRD(
        dist_matrix_condensed: npt.NDArray[np.floating],
        all_pts_core_dists: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
    """
    Constructs mutual-reachability distances.

    MRD(i,j) = max(
        dist(i,j),
        core(i),
        core(j)
    )
    """
    n = len(all_pts_core_dists)

    # Optimization: construct MRD(i, j) directly from the condensed pdist() ordering. 
    # This applies the maximum operation only to the n(n-1)/2 distance-core tuples,
    # without first materializing an n×n distance matrix.
    result = np.zeros(
        (n, n),
        dtype=dist_matrix_condensed.dtype,
    )

    idx = 0
    for i in range(n - 1): # row id
        for j in range(i + 1, n):  # upper-triangular column id
            mrd = max(
                dist_matrix_condensed[idx],
                all_pts_core_dists[i],
                all_pts_core_dists[j],
            )
            result[i, j] = mrd
            result[j, i] = mrd

            idx += 1

    return result


def _MST_builder_HDBSCAN(
        MRD_matrix: npt.NDArray[np.floating],
    ) -> Tuple[float, npt.NDArray[np.integer]]:
    """
    Helper function for intracluster_analysis() 
    that identifies core points based on the all points core distance, 
    and then computes the sparseness of the current cluster.
    """
    n = MRD_matrix.shape[0]

    # Optimization: use HDBSCAN's specialized implementation of Prim's algorithm 
    # rather than SciPy's generic Kruskal-based MST routine.
    # (MRD graph is complete and already represented as a dense distance matrix) 
    mst = mst_linkage_core(MRD_matrix)

    # MST has n - 1 rows and 3 columns: (from, to, weight). 
    # Source-node reconstruction is no longer required.
    # [see HDBSCAN commit 7b2f0e0dcff6ef99b0c976d1471cbeec99da49a9]
    nodes = mst[:, :-1].astype(IDX_DTYPE)
    edges = mst[:, -1]

    # Core points are defined as vertices with more than one incident edge in the MST.
    # Heuristically, these internal vertices better represent the cluster's internal structure 
    # than boundary points and are therefore used for sparseness/separation.
    core_pts = (np.bincount(nodes.ravel(), minlength=n,) > 1) 
    internal_edges = core_pts[nodes].all(axis=1)

    # Density sparseness is not well defined if there are no internal edges.
    # Following the original authors' MATLAB implementation, as adopted by HDBSCAN also,
    # use the maximum of all MST edge weights as a fallback.
    if internal_edges.any():
        sparseness = edges[internal_edges].max()
    else: sparseness = edges.max()

    core_pts = np.flatnonzero(
        core_pts
    )

    return sparseness, core_pts



## ------------ What's Next ? -------------------------

# HDBSCAN mst_linkage_core(...) takes n×n dense distance matrix as input.
#
# We have evaluated pdist(...)'s condensed distance vector of len n(n-1)/2 already.
# If we can patch the code to exploit the symmetry of (i, j) & (j, i),
# we can reduce peak memory by around 50% for this specific step..

# -------------------------------


## =======================================================




## =======================================================
# Density Separation of a Pair of Clusters (DSPC)
## =======================================================
def intercluster_analysis(
        N_clust: int,
        core_cluster_sort: npt.NDArray[np.floating],
        core_cluster_groups: List[npt.NDArray[np.floating]],
        core_cluster_bounds: npt.NDArray[np.integer],
        core_dists_arr: npt.NDArray[np.floating],
        **kwargs
    ) -> npt.NDArray[np.floating]:
    """
    Computes the separation value for each cluster according to the definitions in Moulavi et al.


    Args:
    -----
    N_clust : int
        Number of non-noise clusters to be analyzed.

    core_cluster_sort : npt.NDArray[np.floating]
        2-D master array containing coordinates for all core points, sorted by cluster label.

    core_cluster_groups : List[npt.NDArray[np.floating]]
        List of coordinates arrays containing the core points belonging to each non-noise cluster.

    core_cluster_bounds : npt.NDArray[np.integer]
        1-D array of cluster boundaries for slicing core points master array.

    core_dists_arr : npt.NDArray[np.floating]
        1-D array containing the all-points core distances for the core points, combined from all clusters.


    Returns:
    --------
    - npt.NDArray[np.floating]:
        Sparseness value for each cluster.


    WORKHORSE: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.html

    """

    separation = np.empty(N_clust, dtype=FLOAT_DTYPE)

    # all core-points
    Tree = cKDTree(core_cluster_sort)
    n_jobs = kwargs.get("n_jobs", 1)

    for id, cluster in enumerate(core_cluster_groups):
        # [start,end) corresponds to current label in core_cluster_sort
        start, end = core_cluster_bounds[id], core_cluster_bounds[id + 1]
        cluster_core_size = end - start 
        rows = np.arange(cluster_core_size)
        query_core_dists = core_dists_arr[start: end]

        # k is chosen such that each query must return atleast one neighbor with different cluster label.
        # We have preprocessed the data via _format_data() such that,
        #   no. of clusters >= 2
        #   each cluster having >= 3 nodes, i.e. atleast one core point
        # So, k = cluster_core_size + 1 works without fail.
        NN_dists, NN_indices = Tree.query(
            cluster, 
            k=cluster_core_size + 1,
            workers=n_jobs,
        )
        ## ----- Should we build separate tree per cluster ?? ------------------
        # N = no. of core points
        # p = no. of clusters
        # d = dimensions <-- fix for both approaches
        # N = sum(N_i)

        # TREE BUILDING COST : O(N * logN) 
        # --------------------
        #    per-cluster wins over global tree in terms of tree building.
        #        sum_i{N_i * logN_i} < N * logN 
        #
        # k-NN QUERY COST : O(logN + k) / query
        # -----------------
        #    Global approach : 
        #    [step-1] for cluster i, we have to choose k = N_i + 1 to ensure atleast one outside cluster neighbor.
        #        for all observations combined,
        #        sum_i{N_i * (logN + N_i)} 
        #    [step-2] out of the k neighbors per query, find the closest outside neighbor.
        #        sum_i{N_i * N_i}
        #
        #    per-cluster approach : 
        #    [step-1] for cluster i, we have loop over j != i, with k = 1 suffices.
        #        for all observations combined,
        #        sum_i{N_i * sum_j_neq_i{logN_j}} <---- can be pretty large as (logx + logy + ...) > log(x+y+...)
        #    [step-2] out of the p - 1 outside neighbors per query, find the closest one.
        #        sum_i{N_i * p} 

        # Unless p is very small compared to N_i (i.e. you have a handfew of very large clusters),
        # the additional development effort of per-cluster approach with python-loop overhead is not justified, 
        # over the optimized C++ based available global implementation.
        #
        # Note: query_ball_point(...) is effiectively applicable on a small fraction of points, hence not compared here.
        # ------------------------------------

        # Optimization: The nearest outside neighbor for each core point with respect to Euclidean distance is selected 
        # using a vectorized operation; no Python loop is required.
        # This utilizes the fact that cKDTree.query() returns neighbors in increasing order of distance for each query point,
        # hence the first point in a row outside [start,end) is the nearest outside neighbor for that query.
        NN_out_idx = np.argmax(
            ~(
                (NN_indices >= start) &
                (NN_indices < end)
            ), axis=1,
        )

        # Compute the mutual-reachability distance for each core point and its nearest (euclidean distance) 
        # outside-cluster neighbour.
        # Identify which component determines the MRD for each pair:
        #   0 = Euclidean distance, 1 = inner core distance, 2 = outer core distance.
        MRD_arr = np.stack(
            (
                # euclidean distance
                NN_dists[rows, NN_out_idx],
                # inner core distance
                query_core_dists, 
                # outer core distance
                core_dists_arr[
                    NN_indices[rows, NN_out_idx]
                ]
            ),
            axis=-1, dtype=FLOAT_DTYPE,
        )
        MRD_component = MRD_arr.argmax(axis=1)

        # Optimization: avoid computing max and argmax simultaneously; use argmax to get max.
        #
        # Use the minimum MRD among the nearest (euclidean distance) outside-cluster neighbours
        # as the initial estimate of cluster separation.
        init_separation = MRD_arr[rows, MRD_component].min()
        separation[id] = init_separation

        # Identify points requiring the radial check.
        # We shorlisted outer-core point j by euclidean distance d_ij, for each inner-core point i.
        #   MRD_ij = max(d_ij, c_i, c_j) 
        #
        # Case-1 : MRD_ij = c_i 
        #   We can-not reduce MRD_ij further by changing j for a fix i; always MRD_ik >= c_i.
        #
        # Case-2 : MRD_ij = d_ij
        #   By design, j is the nearest outside neighbour of i w.r.t. euclidean distance. 
        #   We can-not reduce MRD_ij further by changing j; always MRD_ik >= d_ik >= d_ij
        #
        # Case-3 : MRD_ij = c_j
        #   This is where we have scope of improvements.
        #   When (d_ij < init_separation) & (MRD_ij = c_j >= d_ij), 
        #   there can be potentially an outer-core point k having
        #   (d_ij <= d_ik < init_separation) but (MRD_ij = c_j >= c_k).
        #   So, we may achieve MRD_ik <= MRD_ij.
        check_radially = np.flatnonzero(
            (MRD_component == 2)
            & (MRD_arr[:, 0] < init_separation)
        )

        if not len(check_radially): 
            continue
        else:  
            radial_check = Tree.query_ball_point(
                cluster[check_radially],
                r=init_separation,
                return_sorted=False,
                workers=n_jobs,
            )
            
        # Optimization: Tree.query_ball_point() returns an object array of List[int].
        # Instead of doing same calculations in a loop once per query point,
        # flatten the arraay once and utilize numpy vectorized operations for the heavylifting.
        query = []
        neighbors_new = []
        for i, k in zip(check_radially, radial_check):
            l = len(k)
            if not l : continue 

            neighbors_new.extend(k)
            query.extend(l * [i])

        if neighbors_new:
            neighbors_new = np.array(neighbors_new, dtype=IDX_DTYPE)
            outer_core_pts_new = ~(
                (neighbors_new >= start) &
                (neighbors_new < end)
            )

            if outer_core_pts_new.any():
                neighbors_new = neighbors_new[outer_core_pts_new]  
                query = np.fromiter(
                    compress(query, outer_core_pts_new),
                    dtype=IDX_DTYPE,
                    count=outer_core_pts_new.sum(),
                ) # expecting only a small fraction of query to be selected

                MRD_NN_out_updated = np.maximum.reduce(
                    (
                        # euclidean distance
                        np.linalg.norm(
                            cluster[query] - core_cluster_sort[neighbors_new],
                            axis=1,
                        ),
                        # inner core distance
                        query_core_dists[query],
                        # outer core distance
                        core_dists_arr[neighbors_new]
                    )
                ).min()

                separation[id] = min(
                    separation[id],
                    MRD_NN_out_updated,
                )

    return separation



## ------------ What's Next ? -------------------------

# KD-Tree is not label-aware.
# So, to ensure we get neighbor from different cluster label,
# we use k = current_cluster_size + 1 as buffer in Tree.query(...), followed by immediate post-processing.

# When data is label-sorted, a cluster is an Index range [start, end).
# We can patch cKDTree(...) to do that Index filtering in C++ before returning the output,
# and set k = 1, our exact desired value.
# It can reduce the intermediate memory wastage, while the total complexity remains the same.

# -------------------------------


## =======================================================




## =======================================================
# Sort [Data | Label]   
## =======================================================
def _format_data(
        X: npt.NDArray[np.floating],
        labels: npt.NDArray[np.integer],
    ) -> Tuple[
        int,
        Optional[npt.NDArray[np.floating]],
        Optional[List[npt.NDArray[np.floating]]],
        Optional[npt.NDArray[np.integer]],
        int,
        int
    ]:
    """
    Formats observations into a label-sorted master array and per-cluster arrays for DBCV scoring.

    
    Args:
    -----
    X : npt.NDArray[np.floating]
        See DBCV_score() args.

    labels : npt.NDArray[np.integer]
        See DBCV_score() args.


    Returns:
    --------
    Tuple containing:
        - int:
            Status code: 0 = _SUCCESS, -1 = _ALL_NOISE, -2 = _NOT_ENOUGH_CLUSTERS.

        - Optional[npt.NDArray[np.floating]]:
            2-D master array containing coordinates for all non-noise observations, sorted by cluster label.
            Shape: (N, d), 
            where N is the number of non-noise observations and d is the dimensionality of the feature space.

        - Optional[List[npt.NDArray[np.floating]]]:
            List of coordinates arrays, containing the observations belonging to each non-noise cluster. 
            Clusters are ordered according to their sorted labels.

        - Optional[npt.NDArray[np.integer]]:
            1-D array of cluster boundaries for slicing the sorted master array. 
                [0, start_index_1, start_index_2, ..., start_index_last, N]
            Boundaries follow Python's [start, end) convention.
                
        - int:
            Total number of observations (including noise).

        - int:
            Number of non-noise clusters.

    """
    n_samp = X.shape[0]

    # count labels 
    unique, n_labels = np.unique(
        labels,
        return_counts=True, sorted=True,
    )

    # count noise
    if (unique[0] == -1):
        unique = unique[1:]
        n_noise, n_labels = n_labels[0], n_labels[1:]
    else: n_noise = 0

    if len(unique) == 0: # Check: if all data is noise.
        return _ALL_NOISE, None, None, None, 0, 0
    if len(unique) == 1: # Check: if fewer than two non-noise clusters.
        return _NOT_ENOUGH_CLUSTERS, None, None, None, 0, 0
        
    # Sort the arrays by cluster label.
    # Optimization: avoid hstack([X, labels]) as it allocates new memory, which is wasteful for large X.
    # Work on existing X and labels: WITH CAUTION - No Mutation.
    # The final label information is fully recoverable from cluster_bounds.
    sort_order = labels.argsort()
    X, labels = X[sort_order], labels[sort_order]

    # Sorting the labels does not change their counts.
    # if noise exists, it will be the very first cluster in the sorted output.
    #
    # If n_labels[0] = 5, n_labels[1] = 11, n_labels[2] = 4,
    # then in the sorted array X:
    #   unique[0] belongs to rows [0, 5) + n_noise
    #   unique[1] belongs to rows 5 + [0, 11) + n_noise
    #   unique[2] belongs to rows 5 + 11 + [0, 4) + n_noise
    cluster_bounds = np.empty(len(n_labels) + 1, dtype=IDX_DTYPE)
    cluster_bounds[0] = 0
    np.cumsum(n_labels, dtype=IDX_DTYPE, out=cluster_bounds[1:])

    # Compute the size of each contiguous label group [start, end).
    cluster_sizes = np.diff(cluster_bounds)

    # Following the R dbscan implementation, 
    # reassign non-noise clusters containing fewer than 3 points to noise, 
    # since such clusters cannot contain internal (degree > 1) MST vertices.
    small_clusters = (cluster_sizes < 3)

    if small_clusters.any():
        small_cluster_ids = np.flatnonzero(small_clusters)
        logger.warning(
            "⚠ %d cluster(s) %s have size < 3; reassigning to noise (see, https://rdrr.io/cran/dbscan/src/R/dbcv.R).",
            len(small_cluster_ids), unique[small_cluster_ids].tolist(),
        )

        for id in small_cluster_ids:
            cluster = slice(
                n_noise + cluster_bounds[id],
                n_noise + cluster_bounds[id + 1],
            )

            labels[cluster] = -1

        # re-count labels 
        unique, n_labels = np.unique(
            labels,
            return_counts=True, sorted=True,
        )

        # data must have noise after the readjustment of small_clusters
        unique = unique[1:]
        n_noise, n_labels = n_labels[0], n_labels[1:]
        if len(unique) == 0: # Check: if all data is noise.
            return _ALL_NOISE, None, None, None, 0, 0
        if len(unique) == 1: # Check: if fewer than two non-noise clusters.
            return _NOT_ENOUGH_CLUSTERS, None, None, None, 0, 0

        # Optimization: the non-noise rows are already label-sorted, with some noise rows in between.
        # Avoid sorting the data by label for a second time O(NlogN), when a boolean mask suffices O(N). 
        cluster_sort = np.asarray(
            X[(labels != -1)],
            dtype=X.dtype,
        )

        # updated cluster bounds
        cluster_bounds = np.empty(len(n_labels) + 1, dtype=IDX_DTYPE)
        cluster_bounds[0] = 0
        np.cumsum(n_labels, dtype=IDX_DTYPE, out=cluster_bounds[1:])

    else: 
    # Remove the noise group from the data.
        cluster_sort = X[n_noise:]

    # cluster_bounds is of len (N_clust + 1):
    #   [0, start_index_1, start_index_2, ..., start_index_last, N]
    # where first & last elements are first & last row id of cluster_sort.
    # Use the intermediate points to splits the data in N_clust groups.
    cluster_groups = np.vsplit(
        cluster_sort,
        cluster_bounds[1: -1], 
    )
    N_clust = len(cluster_groups)

    return (
        _SUCCESS, 
        cluster_sort, cluster_groups, cluster_bounds, 
        n_samp, N_clust
    )


def _format_core_points(
        cluster_sort: npt.NDArray[np.floating],
        cluster_bounds: npt.NDArray[np.integer],
        core_pts: List[npt.NDArray[np.integer]]
    ) -> Tuple[
        npt.NDArray[np.floating],
        List[npt.NDArray[np.floating]],
        npt.NDArray[np.integer]
    ]:
    """
    Formats core points for intercluster_analysis().

    [see, _format_data(...) for cluster_sort, cluster_groups, and cluster_bounds.]
    [see, intracluster_analysis(...) for core_pts.]

    """
    core_idx = []
    core_cluster_bounds = np.empty(len(core_pts) + 1, dtype=IDX_DTYPE)
    core_cluster_bounds[0] = 0

    # Convert cluster-level core_pts indices to global cluster_sort indices.
    # e.g.-
    #   Within cluster-3 the core points are at position array([2,5,8]),
    #   and the index-range for cluster-3 is [20,35).
    #   Then globally array([2,5,8]) + 20 = array([22,25,28]) rows are core points.
    for id, pts in enumerate(core_pts):
        core_cluster_bounds[id + 1] = len(pts) 
        core_idx.append(
            pts + cluster_bounds[id]
        )

    # Extract the core-point rows from the sorted master array.
    core_cluster_sort = cluster_sort[
        np.concat(core_idx, dtype=IDX_DTYPE)
    ]

    # [0, core_start_index_1, core_start_index_2, ..., core_start_index_last, core_end_index_last + 1]
    core_cluster_bounds = np.cumsum(
        core_cluster_bounds,
        dtype=IDX_DTYPE,
    )

    # Split core-point rows by cluster labels.
    core_cluster_groups = np.vsplit(
        core_cluster_sort,
        core_cluster_bounds[1: -1],
    )

    return (
        core_cluster_sort, 
        core_cluster_groups, core_cluster_bounds
    )


## =======================================================
