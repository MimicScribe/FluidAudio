#ifndef FASTCLUSTER_WRAPPER_H
#define FASTCLUSTER_WRAPPER_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Error codes returned by fastcluster wrapper.
typedef enum {
    FASTCLUSTER_WRAPPER_SUCCESS = 0,
    FASTCLUSTER_WRAPPER_INVALID_ARGUMENT = 1,
    FASTCLUSTER_WRAPPER_INDEX_OVERFLOW = 2,
    FASTCLUSTER_WRAPPER_OUTPUT_TOO_SMALL = 3,
    FASTCLUSTER_WRAPPER_ALLOCATION_FAILURE = 4,
    FASTCLUSTER_WRAPPER_RUNTIME_ERROR = 5,
    FASTCLUSTER_WRAPPER_UNKNOWN_ERROR = 255
} fastcluster_wrapper_status;

/// Compute centroid linkage dendrogram for the provided feature matrix.
///
/// - Parameters:
///   - data: Pointer to `pointCount * dimension` doubles laid out row-major.
///   - pointCount: Number of vectors (>= 1).
///   - dimension: Feature dimension (> 0).
///   - dendrogramOut: Output buffer receiving `(pointCount - 1) * 4` doubles in SciPy
///     linkage format (columns: left, right, distance, sample_count).
///   - dendrogramLength: Length of `dendrogramOut` in elements.
///
/// - Returns:
///   - `FASTCLUSTER_WRAPPER_SUCCESS` on success.
///   - One of the error codes above otherwise.
fastcluster_wrapper_status fastcluster_compute_centroid_linkage(
    const double *data,
    size_t pointCount,
    size_t dimension,
    double *dendrogramOut,
    size_t dendrogramLength
);

/// Compute average-linkage dendrogram from a precomputed condensed
/// distance matrix. Suitable for custom metrics (e.g., PLDA-LLR
/// distance) where the data is not in a Euclidean vector space.
///
/// - Parameters:
///   - condensedDistances: Pointer to `pointCount * (pointCount - 1) / 2`
///     doubles, upper-triangular packed in row-major order:
///     D[i,j] (i < j) is at index `i*(2*N - i - 1)/2 + (j - i - 1)`.
///     Must be symmetric pairwise distances (non-negative; smaller =
///     more similar). The buffer IS MUTATED by NN_chain — caller
///     should treat it as consumed.
///   - pointCount: Number of original points (>= 1).
///   - dendrogramOut: Output buffer receiving `(pointCount - 1) * 4`
///     doubles in SciPy linkage format (left, right, distance, count).
///   - dendrogramLength: Length of `dendrogramOut` in elements.
///
/// - Returns:
///   - `FASTCLUSTER_WRAPPER_SUCCESS` on success.
///   - One of the error codes above otherwise.
fastcluster_wrapper_status fastcluster_compute_average_linkage_from_distances(
    double *condensedDistances,
    size_t pointCount,
    double *dendrogramOut,
    size_t dendrogramLength
);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // FASTCLUSTER_WRAPPER_H
