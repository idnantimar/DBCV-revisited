## the project skeleton is adapted form k-DBCV Copyright (c) 2024 Kaufman Lab Columbia
##
## Our own improvisation achieves 20x+ speedup using ~10% peak memory
## compared to available options in large data.
## license: MIT Copyright (c) 2026 idnantimar

import numpy as np
from joblib import Parallel, delayed

from .utils.dsc._apcd_mrd import APCD_condensedMRD
from .utils.dsc._mst import mst_linkage_core_condensed
from .utils.dspc._ckdtree import cKDTree

import numpy.typing as npt
from typing import List, Tuple, Optional

import os 
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


# Many downstream operations natively use Float64 and upcast Float32 inputs.
# Therefore, storing intermediate results as Float32 provides little benefit
# and may introduce unnecessary Float32 ⟷ Float64 round-trips.
FLOAT_DTYPE = np.float64
# For Index, use the system default. 
IDX_DTYPE = np.intp




__all__ = ['DBCV_score']


def DBCV_score(
        X: npt.NDArray[np.floating],
        labels: npt.NDArray[np.integer],
        *,
        per_cluster_scores: bool = False,
        n_jobs: int = 1,
    ) -> Optional[
        float | 
        Tuple[
            float | None,
            npt.NDArray[np.floating],
        ]
    ]:
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

    per_cluster_scores : bool, default=False
        If True, return the individual DBCV score for each cluster in addition to the aggregate DBCV score. 
        If False, only the aggregate DBCV score is returned.

    n_jobs : parameters to pass to the `joblib.Parallel(...)`.

    
    Returns
    -------
    float, tuple or None
        On success, returns:
            - float: Aggregate DBCV score if `per_cluster_scores=False`.
            - tuple: 
                Aggregate DBCV score and individual cluster scores if `per_cluster_scores=True`.
                The scores are sorted by cluster labels, not in original observed order.

        Returns None if the data cannot be scored, due to not having enough non-noise clusters.

        
    EDGE CASES
    ----------
    For a cluster i,
    separation > 0, sparseness = 0 : cluster_scores[i] = +1 ; heuristically the best case
    separation = 0, sparseness > 0 : cluster_scores[i] = -1 ; heuristically the worst case

    separation = 0, sparseness = 0 : cluster_scores[i] not defined in the original paper by Moulavi et al.
        The existing implementations also does not suggest any fallback.
        In such case, NumPy raises a RuntimeWarning and 
        we return None for the aggregate DBCV score and individual cluster scores are returned for downstream diagnostics.
              
    """
    ## Optimization: avoid redundant memory allocation if data is already in correct form
    X, labels = (
        np.asarray(X).astype(FLOAT_DTYPE, copy=False), 
        np.asarray(labels).astype(IDX_DTYPE, copy=False)
    )

    if X.ndim != 2:
        raise ValueError(
            "X must be a 2-D array of shape (n_samples, n_features)"
        )

    if labels.ndim != 1:
        raise ValueError(
            "labels must be a 1-D array of shape (n_samples,)"
        )

    if X.shape[0] != len(labels):
        raise ValueError(
            "X and labels must contain the same number of observations"
        )

    ## Shuffle the data to make clusters contiguous
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

    # Optimization: By ensuring that working data contains only finite values,
    # downstream C-routines can benefit from `-ffinite-math-only` whenever applicable.
    if not np.isfinite(cluster_sort).all():
        raise ValueError("data must be finite, check for nan or inf values.")
    
    ## Sparseness calculation and find core points
    (
        sparseness,
        core_pts, n_core_pts, core_dists_arr
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
        cluster_sort, cluster_bounds, N_clust,
        core_pts, n_core_pts,
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
        per_cluster_scores,
    )


def _weighted_score(
        n_samp: int,
        sparseness: npt.NDArray[np.floating],
        separation: npt.NDArray[np.floating],
        cluster_bounds: npt.NDArray[np.integer],
        per_cluster_scores: bool = False,
    ) -> float | Tuple[float, npt.NDArray[np.floating]] | None:
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
        per_cluster_scores = True
    else: 
        DBCV_val = np.sum(
            (np.diff(cluster_bounds) / n_samp) *
            cluster_scores
        ).item()

    return ( 
        DBCV_val
        if not per_cluster_scores 
        else (DBCV_val, cluster_scores)
    )




## =======================================================
# Density Sparseness of a Cluster (DSC)
## =======================================================
def intracluster_analysis(
        N_clust: int, 
        cluster_groups: List[npt.NDArray[np.floating]],
        **kwargs
    ) -> Tuple[
        npt.NDArray[np.floating],
        List[npt.NDArray[np.integer]],
        int,
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

        - int: 
            Total number of core points, combined from all clusters.

        - npt.NDArray[np.floating]:
            The all-points core distances for the core points, combined from all clusters.


    """

    sparseness = np.empty(N_clust, dtype=FLOAT_DTYPE)
    core_dists_arr = []
    core_pts = []

    n_jobs = kwargs.get("n_jobs", 1)
    if (not n_jobs) or (n_jobs == 1) or (not os.cpu_count()):
        results = (_dsc_base(cluster) for cluster in cluster_groups)
    else:    
        results = Parallel(
            n_jobs=n_jobs,
            return_as="generator",
            backend="threading",
        )(delayed(_dsc_base)(cluster) for cluster in cluster_groups)

    for id, val_pts_dists in enumerate(results):
        sparseness[id] = val_pts_dists[0]
        core_pts.append(val_pts_dists[1])
        core_dists_arr.append(val_pts_dists[2])

    # Combine the list of core-distance arrays for each cluster into a single array,
    # while maintaining the cluster order.
    core_dists_arr = np.concat(core_dists_arr, dtype=FLOAT_DTYPE)
    n_core_pts = len(core_dists_arr)

    return (
        sparseness,
        core_pts, n_core_pts, core_dists_arr
    )


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
    n_i = len(cluster)
    
    # Optimization: use allocated memory of mutual-reachability distances 
    # to hold ordinary distances in intermediate steps.
    all_pts_core_dists, intraclust_condensedMRD = APCD_condensedMRD(cluster)

    # Optimization: generic Kruskal-based MST builder is replaced with Prim's-based implementation.
    # MST builder is further customized in Cython to work on condensed MRD matrix.
    nodes, edges = mst_linkage_core_condensed(intraclust_condensedMRD)

    # Core points are defined as vertices with more than one incident edge in the MST.
    # Heuristically, these internal vertices better represent the cluster's internal structure 
    # than boundary points and are therefore used for sparseness/separation.
    core_pts = (np.bincount(nodes.ravel(), minlength=n_i) > 1) 
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

    return sparseness, core_pts, all_pts_core_dists[core_pts]


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

    """

    separation = np.empty(N_clust, dtype=FLOAT_DTYPE)

    # all core-points
    Tree = cKDTree(
        core_cluster_sort,
        copy_data=False,
        compact_nodes=True, balanced_tree=True,
    )

    n_jobs = kwargs.get("n_jobs", 1)
    for id, cluster in enumerate(core_cluster_groups):
        # [start,end) corresponds to current label in core_cluster_sort
        start, end = core_cluster_bounds[id], core_cluster_bounds[id + 1]
        cluster_core_size = end - start 
        query_core_dists = core_dists_arr[start: end]

        # Optimization: our custom C++ implementation to handle k-nn search with index filter.
        # no post-processing required. 
        NN_out_dists, NN_out_idx = (
            Tree.query_sqeuclidean_exact(
                cluster, 
                k=1,
                excludeIDx=(start, end),
                workers=n_jobs,
            )
        ) # each output has shape (cluster_core_size, 1)

        # Compute the mutual-reachability distance for each core point and its nearest (sqeuclidean distance) 
        # outside-cluster neighbour.
        # The intraclust_condensedMRD's are not useful here, as we need inter-cluster comparisons.
        # Identify which component determines the MRD for each pair:
        #   0 = square-Euclidean distance, 1 = inner core distance, 2 = outer core distance.
        MRD_arr = np.stack(
            (
                # sqeuclidean distance
                NN_out_dists.ravel(),
                # inner core distance
                query_core_dists, 
                # outer core distance
                core_dists_arr[
                    NN_out_idx.ravel()
                ]
            ),
            axis=-1, dtype=FLOAT_DTYPE,
        )
        MRD_component = MRD_arr.argmax(axis=1)

        # Use the minimum MRD among the nearest (sqeuclidean distance) outside-cluster neighbours
        # as the initial estimate of cluster separation.
        # Optimization: avoid computing max and argmax simultaneously; use argmax to get max.
        init_separation = MRD_arr[
            np.arange(cluster_core_size), MRD_component
        ].min()
        separation[id] = init_separation

        # Identify points requiring a validation.
        # We shorlisted outer-core point j by euclidean distance d_ij, for each inner-core point i.
        #   MRD_ij = max(d_ij, c_i, c_j) 
        #
        # Case-1 : MRD_ij = c_i 
        #   We can-not reduce MRD_ij further by changing j for a fix i; always MRD_ik >= c_i.
        #
        # Case-2 : MRD_ij = d_ij
        #   By design, j is the nearest outside neighbour of i w.r.t. sqeuclidean distance. 
        #   We can-not reduce MRD_ij further by changing j; always MRD_ik >= d_ik >= d_ij
        #
        # Case-3 : MRD_ij = c_j
        #   This is where we have scope of improvements.
        #   When (d_ij < init_separation) & (MRD_ij = c_j >= d_ij), 
        #   there can be potentially an outer-core point k having
        #   (d_ij <= d_ik < init_separation) but (c_k <= c_j = MRD_ij).
        #   So, we may achieve MRD_ik <= MRD_ij.
        check_radially = np.flatnonzero(
            (MRD_component == 2)
            & (NN_out_dists.ravel() < init_separation)
        )

        if not len(check_radially): 
            continue # closest sqeuclidean neighbor is already closest MRD neighbor
        else:  
            # Optimization: our custom C++ implementation to handle k-nn search with index filter.
            # no post-processing required. 
            neighbors_new_pooled, neighbors_new_offsets = (
                Tree.query_ball_point_sqeuclidean_exact(
                    cluster[check_radially],
                    r_sqeuclidean=init_separation,
                    excludeIDx=(start, end),
                    workers=n_jobs,
                ) 
            ) # outputs are in CSR adjacency-list representation

            if len(neighbors_new_pooled):
                query_indices_expanded = np.repeat(
                    check_radially, 
                    np.diff(neighbors_new_offsets),
                ) # expanded query indices, row-by-row correspondence with neighbors_new_pooled
                diff = cluster[query_indices_expanded] - core_cluster_sort[neighbors_new_pooled]
                MRD_NN_out_updated = np.maximum.reduce(
                    (
                        # sqeuclidean distance directly, not euclidean distance;
                        # avoid the cost of redundant sqrt(...) ---> **2 and intermediate peak memory
                        np.einsum('ij,ij->i', diff, diff),
                        # inner core distance
                        query_core_dists[query_indices_expanded],
                        # outer core distance
                        core_dists_arr[neighbors_new_pooled]
                    )
                ).min()

                separation[id] = min(
                    separation[id],
                    MRD_NN_out_updated,
                )

    return separation



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
    sort_order = labels.argsort(stable=True)
    X, labels = X[sort_order], labels[sort_order] # stable sort to avoid run on run fluctuations

    # Sorting the labels does not change their counts.
    # if noise exists, it will be the very first cluster in the sorted output.
    #
    # If n_labels[0] = 5, n_labels[1] = 11, n_labels[2] = 4,
    # then in the sorted array X:
    #   unique[0] belongs to rows [0, 5) + n_noise
    #   unique[1] belongs to rows 5 + [0, 11) + n_noise
    #   unique[2] belongs to rows 5 + 11 + [0, 4) + n_noise
    cluster_bounds = np.empty(len(unique) + 1, dtype=IDX_DTYPE)
    cluster_bounds[0] = 0
    np.cumsum(n_labels, dtype=IDX_DTYPE, out=cluster_bounds[1:])

    # Remove the noise group from the data.
    cluster_sort = X[n_noise:]

    # Following the R dbscan implementation, 
    # reassign non-noise clusters containing fewer than 3 points to noise, 
    # since such clusters contain no internal (degree > 1) MST vertices.
    cluster_sizes = np.diff(cluster_bounds)
    small_clusters = (cluster_sizes < 3)

    if small_clusters.any():
        small_cluster_ids = np.flatnonzero(small_clusters)
        logger.warning(
            "⚠ %d cluster(s) %s have size < 3; reassigning to noise (see, https://rdrr.io/cran/dbscan/src/R/dbcv.R).",
            len(small_cluster_ids), unique[small_cluster_ids].tolist(),
        )

        # re-count labels
        unique = unique[~small_clusters]
        n_noise += n_labels.sum(where=small_clusters)
        n_labels = n_labels[~small_clusters]
        if len(unique) == 0: # Check: if all data is noise.
            return _ALL_NOISE, None, None, None, 0, 0
        if len(unique) == 1: # Check: if fewer than two non-noise clusters.
            return _NOT_ENOUGH_CLUSTERS, None, None, None, 0, 0

        # Optimization: the non-noise rows are already label-sorted, with some noise rows in between.
        # Avoid sorting the data by label for a second time O(NlogN), when a XOR toggle suffices O(N). 
        start_old = cluster_bounds[:-1]
        mask = np.zeros(cluster_bounds[-1] + 1, dtype=bool) # placeholder for positions [0, 1, 2, ..., N].
        start_new = start_old[~small_clusters]
        mask[start_new] = True # assign all start id True
        mask[start_new + n_labels] ^= True # assign end id as True, overwrite overlapping start id as False
        mask = np.logical_xor.accumulate(mask)[:-1] # slice [0, N)

        cluster_sort = np.asarray(
            cluster_sort[mask],
            dtype=X.dtype,
        )

        # updated cluster bounds
        cluster_bounds = np.empty(len(unique) + 1, dtype=IDX_DTYPE)
        cluster_bounds[0] = 0
        np.cumsum(n_labels, dtype=IDX_DTYPE, out=cluster_bounds[1:])

    # cluster_bounds is of len (N_clust + 1):
    #   [0, start_index_1, start_index_2, ..., start_index_last, N]
    # where first & last elements are first & last row id of cluster_sort.
    # Use the intermediate points to split the data in N_clust groups.
    cluster_groups = np.vsplit(
        cluster_sort,
        cluster_bounds[1: -1], 
    )
    N_clust = len(unique)

    return (
        _SUCCESS, 
        cluster_sort, cluster_groups, cluster_bounds, 
        n_samp, N_clust
    )


def _format_core_points(
        cluster_sort: npt.NDArray[np.floating],
        cluster_bounds: npt.NDArray[np.integer],
        N_clust: int,
        core_pts: List[npt.NDArray[np.integer]],
        n_core_pts: int,
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
    core_idx = np.empty(n_core_pts, dtype=IDX_DTYPE)
    core_cluster_bounds = np.empty(N_clust + 1, dtype=IDX_DTYPE)
    core_cluster_bounds[0] = 0

    # Convert cluster-level core_pts indices to global cluster_sort indices.
    # e.g.-
    #   Within cluster-3 the core points are at position array([2,5,8]),
    #   and the index-range for cluster-3 is [20,35).
    #   Then globally array([2,5,8]) + 20 = array([22,25,28]) rows are core points.
    for id, pts in enumerate(core_pts):
        start = core_cluster_bounds[id]
        end = start + len(pts) 

        core_idx[start:end] =  pts + cluster_bounds[id]
        core_cluster_bounds[id + 1] = end

    # Extract the core-point rows from the sorted master array.
    core_cluster_sort = cluster_sort[core_idx]

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
