/*
Adapted from SciPy
[see https://github.com/scipy/scipy/tree/main/scipy/spatial]

Copyright (c) 2001-2002 Enthought, Inc. 2003, SciPy Developers
License: 3-clause BSD


[[THIS FILE IS THE MATHEMATICAL BACKBONE FOR QUERY]]

*/


#ifndef CKDTREE_CPP_DISTANCE
#define CKDTREE_CPP_DISTANCE


#include <cmath>
#include <vector>
#include <algorithm>



/* 
* =======================
* Constructing Geometry
* ====================
*/


/* m - dimensional Hyperbox
 * ========================
 *
 * buf: [ 
 *     maxes[0], maxes[1], ..., maxes[m-1],
 *     mins[0],  mins[1],  ..., mins[m-1] 
 * ]
*/

struct Rectangle {
 
    const ckdtree_intp_t m;
 
    double * const maxes() const { return &buf[0]; }
    double * const mins() const { return &buf[0] + m; }
 
    Rectangle(
        const ckdtree_intp_t _m,
        const double *_mins,
        const double *_maxes
    ) : m(_m), buf(2 * m) {
        std::copy(_mins, _mins + m, mins());
        std::copy(_maxes, _maxes + m, maxes());
    };
 
    Rectangle(const Rectangle& rect) : m(rect.m), buf(rect.buf) {};
 
    private:
        mutable std::vector<double> buf;
};



/* Euclidean Geometry
 * ==================
 *
 * "shortest distance between two points is a straight line."
*/

struct PlainDist1D {

    static inline const double 
    side_distance_from_min_max(
        /*
         * absolute distance along dimension k between a point and a hyperrectangle.
        */
        const double *x,
        const double *mins, const double *maxes,
        const ckdtree_intp_t k
    )
    {
        return fmax(0., fmax(x[k] - maxes[k], mins[k] - x[k]));
        // Optimization: branchless single instruction fmin / fmax 
        // (given, data is finite)
    }

    static inline void
    interval_interval(
        /*
         * minimum & maximum distance along dimension k between two hyperrectangles.
        */     
        const Rectangle& rect1, const Rectangle& rect2,
        const ckdtree_intp_t k,
        double *min, double *max
    )
    {
        const double b2_a1 = rect2.maxes()[k] - rect1.mins()[k];
        const double b1_a2 = rect1.maxes()[k] - rect2.mins()[k];
        // [a1, b1] vs [a2, b2]
        *max = fmax(b2_a1, b1_a2);
        *min = fmax(0., -fmin(b2_a1, b1_a2));
        // Optimization: avoid redundant recalculation 
    }

    static inline double
    point_point_signed(
        /*
         * distance along dimension k between two points.
        */
        const double *x, const double *y,
        const ckdtree_intp_t k
    ) 
    {
        return x[k] - y[k];
        // Optimization: fabs(...) will be called only when necessary
    }
    
};



/* 
* =======================
* Measuring Distances
* ====================
*/


struct MinkowskiDistP2sq {
    // sqeuclidean distance;
    // sqrt(...) is a monotone function, 
    // hence the ordering of distances remains the same as euclidean,
    // while extra computation is avoided.

    static inline void
    interval_interval_p2(
        const Rectangle& rect1, const Rectangle& rect2,
        const ckdtree_intp_t k,
        double *min, double *max
    )
    {
        PlainDist1D::interval_interval(rect1, rect2, k, min, max);
        *min *= *min;
        *max *= *max;
    }

    static inline void
    rect_rect_p2(
        const Rectangle& rect1, const Rectangle& rect2,
        double *min, double *max
    )
    {
        *min = 0.;
        *max = 0.;
        for(ckdtree_intp_t i=0; i<rect1.m; ++i) {
            double min_, max_;

            PlainDist1D::interval_interval(rect1, rect2, i, &min_, &max_);
            *min += min_ * min_;
            *max += max_ * max_;
        }
    }

    static inline double
    point_point_p2(
        const double *x, const double *y,
        const ckdtree_intp_t m,
        const double upperbound_sqeuclidean
    )
    {
        double r;
        double r0, r1, r2, r3;
        ckdtree_intp_t i = 0;
        // Manually unrolled loop; exposes potential SIMD/ILP
        double acc[4] = {0., 0., 0., 0.};
        for (; i + 4 <= m; ++i) {
            r0 = PlainDist1D::point_point_signed(x, y, i);
            r1 = PlainDist1D::point_point_signed(x, y, ++i);
            r2 = PlainDist1D::point_point_signed(x, y, ++i);
            r3 = PlainDist1D::point_point_signed(x, y, ++i);
            acc[0] += r0 * r0;
            acc[1] += r1 * r1;
            acc[2] += r2 * r2;
            acc[3] += r3 * r3;
            r = acc[0] + acc[1] + acc[2] + acc[3];
            if (r > upperbound_sqeuclidean)
                return r;
        }
        r = acc[0] + acc[1] + acc[2] + acc[3];
        if (i < m) {
            for(; i<m; ++i) {
                r1 = PlainDist1D::point_point_signed(x, y, i);
                r += r1 * r1;
                if (r > upperbound_sqeuclidean)
                    return r;
            }
        }
        return r;
    }

    static inline double
    distance_p2(const double s)
    {
        return s * s;
    }
    
};



#endif