'''
Computes APCD & MRD in a single pairwise pass over input array.

OUR CONTRIBUTION: license: MIT Copyright (c) 2026 idnantimar
-----------------
    [1.] Execute the whole routine in Scipy's `pdist(...)` style condensed form, without ever materializing `squareform(...)` format.
        Some Index manipulation is utilized to exploit distance-matrix symmetry.
        This way, the complexity remains O(n^2),
        but peak memory reduces by ~50% by storing only the upper-triangular block of the n * n distance matrix.

    [2.] Fuse the pipeline : 
            pairwise distance ---> core distance ---> mutual reachability distance
        This way, the allocated mutual-reachability distance block can be reused as a temporary buffer 
        for ordinary distances, eliminating the requirement or additional O(nC2) memory in intermediate steps.

'''


import numpy as np
cimport numpy as np


# avoid generic pow(...) call for an integer exponent 
cdef inline np.double_t int_pow(np.double_t x, np.intp_t p) noexcept nogil:
    cdef np.double_t result = 1.0
    cdef np.double_t base = x
    cdef np.intp_t e = p
    while e > 0:
        if e & 1:
            result *= base
        base *= base
        e >>= 1
    return result

# avoid fmax(...) overhead; NaN handling is unnecessary for valid distances
cdef extern from *:
    """
    #define MAX_DOUBLE(a, b) ((a) > (b) ? (a) : (b))
    """
    np.double_t MAX_DOUBLE(
        np.double_t a,
        np.double_t b
    ) noexcept nogil



cpdef tuple APCD_condensedMRD(np.ndarray[np.double_t, ndim=2] X):
    """
    Computes APCD & MRD in a single pairwise pass over input array.

    The all-points core distance of point i is
        APCD(i) = [
            mean_{j != i} dist(i,j)^(-d)
        ]^(-1/d)

    and

    The mutual-reachability distances between points (i, j) is
        MRD(i,j) = max(
            dist(i,j),
            core(i), core(j)
        )

    where dist(i,j) is the squared-Euclidean distance.
    The paper by Moulavi et al. (2014) suggests, and the original MATLAB implementation subsequently uses, 
    squared-Euclidean distance rather than Euclidean distance
    (squared-Euclidean distance assigns greater weight to closer neighbors in the APCD calculation).

    NOTE: duplicate points (i, j) → APCD(i) = 0 = APCD(j) as intended.


    OUR CONTRIBUTION:
    -----------------
        [1.] Execute the whole routine in Scipy's `pdist(...)` style condensed form, without ever materializing `squareform(...)` format.
            Some Index manipulation is utilized to exploit distance-matrix symmetry.
            This way, the complexity remains O(n^2),
            but peak memory reduces by ~50% by storing only the upper-triangular block of the n * n distance matrix.

        [2.] Fuse the pipeline : 
                pairwise distance ---> core distance ---> mutual reachability distance
            This way, the allocated mutual-reachability distance block can be reused as a temporary buffer 
            for ordinary distances, eliminating the requirement or additional O(nC2) memory in intermediate steps.


    Returns
    -------
    apcd : ndarray of shape (n,)
        The all-points core distances.

    mrd : ndarray of shape (nC2,)
        The upper-triangular block of MRD matrix, stored in a row-major style:
            (0,1), (0,2), ..., (0,n-1), (1,2), ..., (1,n-1), ..., (n-2,n-1).

    """
    cdef np.intp_t n = X.shape[0]
    cdef np.intp_t d = X.shape[1]
    cdef np.intp_t m = n * (n - 1) // 2

    if n < 2: return None, None

    if d > 25: # a guard-rail for underflow; overflow is validated explicitly at the end
        raise ValueError(
            """
            This float64 implementation supports only d <= 25.  
            For higher dimensions, reimplement the same APCD accumulation logic using float128.
            """
        )

    cdef np.double_t n1_inv = 1.0 / (n - 1)
    cdef np.double_t p = -1.0 / d

    # Initialize core distance = 0
    cdef np.ndarray[np.double_t, ndim=1] apcd_arr = np.zeros(n, dtype=np.float64)
    cdef np.double_t *apcd = <np.double_t *> apcd_arr.data

    # raw pointer is preferred over memoryview in O(n^2) hot loop
    X = np.ascontiguousarray(X, dtype=np.float64)
    cdef np.ndarray[np.double_t, ndim=1] mrd_arr = np.empty(m, dtype=np.float64)
    cdef np.double_t *X_ptr = <np.double_t *> X.data
    cdef np.double_t *mrd = <np.double_t *> mrd_arr.data

    cdef np.intp_t i, j, k, idx
    cdef np.double_t dist_sq, diff, w
    cdef np.intp_t bad_idx = -1

    cdef np.double_t *X_i
    cdef np.double_t *X_j
    cdef np.double_t apcd_i

    with nogil:

        # First pass: compute squared-Euclidean distances and accumulate APCD.
        idx = 0
        X_i = X_ptr
        for i in range(n - 1): # row id i

            X_j = X_i + d
            for j in range(i + 1, n): # column id j ; i < j
                dist_sq = 0.0
                for k in range(d): # sqeuclidean
                    diff = X_i[k] - X_j[k]
                    dist_sq += diff * diff

                # due to symmetry d(i, j) contributes to both APCD(i) & APCD(j)
                w = n1_inv / int_pow(dist_sq, d) # avoid sum overflow for large n 
                apcd[i] += w
                apcd[j] += w

                # temporary
                mrd[idx] = dist_sq

                idx += 1
                X_j += d

            X_i += d

        # HOW apcd ACCUMULATION WORKS HERE: np.float64 works in a scale of 4.94e-324 to 1.80e+308
        # ---------------------------------------------------------------------------------------
            #   [case-1] : dist_sq either exact zero or extremely close to zero 
            #
            #       dist_sq**d = 0. --> w = inf --> apcd ** -1/d = 0.
            #
            #       i.e. when two points have no practically meaningful distance,
            #       APCD collapses to zero for both of them.
            #   
            #   range to collapse :-
            #       dist_sq < 3.42e-07 if d = 50
            #       dist_sq < 1.17e-13 if d = 25
            #       dist_sq < 4.67e-33 if d = 10   

            #   [case-2] : dist_sq extremely large
            #
            #       dist_sq**d = inf --> w = 0 --> apcd ** -1/d = inf
            #
            #       i.e. when there is an extreme outlier,
            #       APCD becomes infinity for that observation. 
            #
            #   range to collapse :-
            #       dist_sq > 1.46e+06 if d = 50
            #       dist_sq > 2.14e+12 if d = 25
            #       dist_sq > 6.69e+30 if d = 10

        # Optimization: instead of validating O(n^2) loop, it is much cheaper to validate O(n) loop.
        for i in range(n):
            if apcd[i]:
                apcd[i] = (apcd[i]) ** p
            else:
                bad_idx = i
                break

        # Second pass: construct MRD.
        if bad_idx == -1:
            idx = 0
            for i in range(n - 1): # row id i
                apcd_i = apcd[i]

                for j in range(i + 1, n): # column id j ; i < j
                    mrd[idx] = MAX_DOUBLE(
                        mrd[idx],
                        MAX_DOUBLE(apcd_i, apcd[j])
                    )

                    idx += 1

    # Overflow detection is necessary,
    # because an overflow makes the corresponding APCD value infinite, 
    # leaving the MRD graph disconnected 
    # (at least one node remains at infinite distance from all others); 
    # so an MST cannot be constructed (you will get spanning forest).
    if bad_idx != -1:
        raise ValueError(
            f"""
                APCD accumulation overflowed for point {bad_idx}.
                An outlier may be present, or try scaling down the data.
            """
        )

    return apcd_arr, mrd_arr


