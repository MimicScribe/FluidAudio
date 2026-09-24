#pragma once
// Parallel variant of fastcluster's generic_linkage_vector_alternative for
// METHOD_VECTOR_CENTROID (MimicScribe fork). It produces the same dendrogram as
// the serial algorithm with the same distance function: every argmin scan keeps
// the serial semantics (the FIRST index reaching the minimum wins, comparisons
// strictly `<`); blocks reduce first-min locally and are combined in index
// order, so the result never depends on thread scheduling. Included by
// FastClusterWrapper.cpp after fastcluster_internal.hpp.
#include <dispatch/dispatch.h>
#include <algorithm>
#include <atomic>
#include <unistd.h>

namespace {

constexpr size_t kParallelMinItems = 4096;
constexpr size_t kBlockItems = 1024;
// The initial pass does rows²/2 distances, so it pays off far earlier than a
// single-row scan does.
constexpr size_t kParallelMinInitialRows = 1024;

// Half the online cores (at least 1): the scans run at the caller's QoS, and a
// meeting may start while a finished one is still clustering — live capture
// keeps the other half.
size_t parallelWidth() {
    static const size_t width = [] {
        const long n = sysconf(_SC_NPROCESSORS_ONLN);
        return static_cast<size_t>(n > 1 ? n / 2 : 1);
    }();
    return width;
}

// Runs body(item) for item in [0, count) on at most parallelWidth() threads.
// Each item writes only its own output slot, so results do not depend on the
// schedule.
void boundedApply(size_t count, void (^body)(size_t)) {
    const size_t workers = std::min(count, parallelWidth());
    if (workers <= 1) {
        for (size_t item = 0; item < count; ++item) { body(item); }
        return;
    }
    dispatch_apply(workers, DISPATCH_APPLY_AUTO, ^(size_t w) {
        for (size_t item = w; item < count; item += workers) { body(item); }
    });
}

struct ArgMin {
    t_float value;
    t_index index;
};

// First-min of f(k) over k in [0, m), f returning the distance for slot k.
template <typename F>
ArgMin parallelFirstMin(size_t m, F f, std::atomic<bool> &sawNaN) {
    if (m < kParallelMinItems) {
        ArgMin best{std::numeric_limits<t_float>::infinity(), -1};
        for (size_t k = 0; k < m; ++k) {
            const t_float d = f(k);
            if (d != d) { sawNaN.store(true, std::memory_order_relaxed); }
            if (best.index < 0 || d < best.value) { best = {d, static_cast<t_index>(k)}; }
        }
        return best;
    }
    const size_t blocks = (m + kBlockItems - 1) / kBlockItems;
    std::vector<ArgMin> partial(blocks);
    ArgMin *out = partial.data();
    boundedApply(blocks, ^(size_t b) {
        const size_t lo = b * kBlockItems;
        const size_t hi = std::min(m, lo + kBlockItems);
        ArgMin best{std::numeric_limits<t_float>::infinity(), -1};
        for (size_t k = lo; k < hi; ++k) {
            const t_float d = f(k);
            if (d != d) { sawNaN.store(true, std::memory_order_relaxed); }
            if (best.index < 0 || d < best.value) { best = {d, static_cast<t_index>(k)}; }
        }
        out[b] = best;
    });
    ArgMin best = partial[0];
    for (size_t b = 1; b < blocks; ++b) {
        if (partial[b].value < best.value) { best = partial[b]; }
    }
    return best;
}

template <typename Dist>
void centroidLinkageParallel(const t_index N, Dist &dist, cluster_result &Z2) {
    const t_index N_1 = N - 1;
    auto_array_ptr<t_index> n_nghbr(2 * N - 2);
    auto_array_ptr<t_float> mindist(2 * N - 2);
    doubly_linked_list active_nodes(N + N_1);
    binary_min_heap nn_distances(&*mindist, N_1, 2 * N - 2, 1);
    std::atomic<bool> sawNaN(false);

    // Initial nearest neighbours: row i scans j < i. Rows are independent.
    {
        t_index *nn = &*n_nghbr;
        t_float *md = &*mindist;
        Dist *dp = &dist;
        std::atomic<bool> *nanFlag = &sawNaN;
        const size_t rows = static_cast<size_t>(N - 1);
        void (^row)(size_t) = ^(size_t r) {
            // Interleave long and short rows for load balance.
            const t_index i = static_cast<t_index>(
                (r % 2 == 0) ? (N - 1 - r / 2) : (1 + r / 2));
            t_float min = std::numeric_limits<t_float>::infinity();
            t_index idx = 0;
            for (t_index j = 0; j < i; ++j) {
                const t_float tmp = dp->template sqeuclidean<false>(i, j);
                if (tmp != tmp) { nanFlag->store(true, std::memory_order_relaxed); }
                if (tmp < min) { min = tmp; idx = j; }
            }
            md[i] = min;
            nn[i] = idx;
        };
        // Small inputs stay on the calling thread: waking workers costs more
        // than the whole pass.
        if (rows < kParallelMinInitialRows) {
            for (size_t r = 0; r < rows; ++r) { row(r); }
        } else {
            boundedApply(rows, row);
        }
    }
    if (sawNaN.load()) { throw nan_error(); }
    nn_distances.heapify();

    // Active nodes below the node being created, ascending (mirrors the
    // linked list's order, which is what the serial scans walk).
    std::vector<t_index> act(static_cast<size_t>(N));
    for (t_index k = 0; k < N; ++k) { act[static_cast<size_t>(k)] = k; }
    auto eraseActive = [&act](t_index v) {
        auto it = std::lower_bound(act.begin(), act.end(), v);
        if (it == act.end() || *it != v) { throw std::runtime_error("active set out of sync"); }
        act.erase(it);
    };

    for (t_index i = N; i < N + N_1; ++i) {
        t_index idx1 = nn_distances.argmin();
        while (active_nodes.is_inactive(n_nghbr[idx1])) {
            const size_t m = static_cast<size_t>(
                std::lower_bound(act.begin(), act.end(), idx1) - act.begin());
            const t_index *a = act.data();
            const t_index row = idx1;
            ArgMin best = parallelFirstMin(
                m, [&dist, a, row](size_t k) { return dist.sqeuclidean_extended_nothrow(row, a[k]); },
                sawNaN);
            if (sawNaN.load()) { throw nan_error(); }
            if (best.index < 0) { throw std::runtime_error("no active neighbour"); }
            n_nghbr[idx1] = act[static_cast<size_t>(best.index)];
            nn_distances.update_geq(idx1, best.value);
            idx1 = nn_distances.argmin();
        }

        const t_index idx2 = n_nghbr[idx1];
        active_nodes.remove(idx1);
        active_nodes.remove(idx2);
        eraseActive(idx1);
        eraseActive(idx2);

        Z2.append(idx1, idx2, mindist[idx1]);

        if (i < 2 * N_1) {
            dist.merge(idx1, idx2, i);
            const t_index *a = act.data();
            const t_index node = i;
            ArgMin best = parallelFirstMin(
                act.size(), [&dist, a, node](size_t k) { return dist.sqeuclidean_extended_nothrow(a[k], node); },
                sawNaN);
            if (sawNaN.load()) { throw nan_error(); }
            if (best.index < 0) { throw std::runtime_error("no active neighbour"); }
            n_nghbr[i] = act[static_cast<size_t>(best.index)];
            if (idx2 < active_nodes.start) {
                nn_distances.remove(active_nodes.start);
            } else {
                nn_distances.remove(idx2);
            }
            nn_distances.replace(idx1, i, best.value);
            act.push_back(i);
        }
    }
}

} // namespace
