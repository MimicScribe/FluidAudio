import Accelerate
import Foundation
import OSLog
import os.signpost

#if canImport(FastClusterWrapper)
import FastClusterWrapper
#elseif canImport(FluidAudio_FastClusterWrapper)
import FluidAudio_FastClusterWrapper
#endif

public struct AHCClustering {
    private let logger = AppLogger(category: "OfflineAHC")
    private let signposter = OSSignposter(
        subsystem: "com.fluidaudio.diarization",
        category: .pointsOfInterest
    )

    public init() {}

    // MARK: - Agglomerative Hierarchical Clustering
    /// Centroid-linkage AHC over `embeddingFeatures`, cut at `threshold`
    /// (Euclidean distance on unit vectors).
    ///
    /// `collapseExactDuplicates` (MimicScribe fork, default on): rows that are
    /// bit-for-bit identical are clustered once, as a single point weighted by
    /// its copy count. Identical rows sit at distance 0 and merge before
    /// anything else, and centroid linkage only reads a node's member count, so
    /// the partition matches clustering every copy up to float ties: a merge
    /// can only change where two distances agree to ~1e-16 (none did on a
    /// 57-recording corpus at eight cuts). Speaker-chunk embeddings carry many
    /// such copies (the extractor's mask-similarity skip reuses an embedding
    /// across up to 32 windows), and AHC is O(n²), so this is where most of its
    /// time goes. Rows that merely look equal (-0.0 vs 0.0, NaN payloads, a
    /// scaled copy) are left as separate points.
    public func cluster(
        embeddingFeatures: [[Double]],
        threshold: Double,
        collapseExactDuplicates: Bool = true
    ) -> [Int] {
        let count = embeddingFeatures.count
        guard count > 0 else { return [] }
        guard let dimension = embeddingFeatures.first?.count, dimension > 0 else {
            return Array(repeating: 0, count: count)
        }
        if count == 1 {
            return [0]
        }

        let ahcState = signposter.beginInterval("Agglomerative Hierarchical Clustering")
        defer { signposter.endInterval("Agglomerative Hierarchical Clustering", ahcState) }

        // Every row maps to a point; without collapsing, point i is row i.
        let groups =
            collapseExactDuplicates
            ? Self.exactDuplicateGroups(embeddingFeatures)
            : DuplicateGroups(
                rowToPoint: Array(0..<count), pointRows: Array(0..<count),
                weights: [Int32](repeating: 1, count: count))
        let pointCount = groups.pointRows.count
        if pointCount == 1 {
            return Array(repeating: 0, count: count)
        }

        let normalized = normalizeFeatures(
            groups.pointRows.map { embeddingFeatures[$0] }, dimension: dimension)
        let dendrogramLength = (pointCount - 1) * 4
        var dendrogram = [Double](repeating: 0, count: dendrogramLength)

        // MARK: - Fastcluster FFI Boundary
        let status = normalized.withUnsafeBufferPointer { normalizedPointer in
            groups.weights.withUnsafeBufferPointer { weightsPointer in
                dendrogram.withUnsafeMutableBufferPointer { dendrogramPointer in
                    fastcluster_compute_centroid_linkage_weighted(
                        normalizedPointer.baseAddress,
                        pointCount,
                        dimension,
                        weightsPointer.baseAddress,
                        dendrogramPointer.baseAddress,
                        dendrogramLength
                    )
                }
            }
        }

        guard status == FASTCLUSTER_WRAPPER_SUCCESS else {
            logger.error("fastcluster failed with status \(status.rawValue)")
            return Array(0..<count)
        }

        let distanceThreshold = clampDistanceThreshold(threshold)
        let pointAssignments = assignmentsFromDendrogram(
            dendrogram,
            count: pointCount,
            distanceThreshold: distanceThreshold
        )

        // Labels are renumbered by first appearance in ROW order, so they
        // depend only on the partition, not on how points were indexed.
        return remapClusterIds(groups.rowToPoint.map { pointAssignments[$0] })
    }

    // MARK: - Incremental (savepoint) clustering

    /// Centroid-linkage AHC over points that may already be GROUPS (MimicScribe
    /// fork, live incremental clustering). Each point is used as given — a group's
    /// true centroid (the weighted mean of its unit rows, NOT re-normalized), or a
    /// new unit row — weighted by its member count, so centroid linkage treats a
    /// group exactly as the merged node of its members. Cut at `threshold` (the
    /// partition) and at `savepointThreshold` (the tight groups a caller keeps as
    /// the next run's savepoint; ≤ `threshold`). Rows are not deduplicated here.
    ///
    /// A savepoint run is not identical to clustering every row from scratch: a
    /// merge frozen in the savepoint is never undone. Live sim 2026-10-20 (22
    /// recordings, savepoint at 0.7 × the cut, full refresh every 10 min): 99.99 %
    /// median / 98.9 % worst-file row agreement with a full run, ~6× fewer points.
    /// The after-Stop pipeline keeps `cluster(embeddingFeatures:threshold:)`.
    public func clusterWeightedPoints(
        points: [[Double]],
        weights: [Int32],
        threshold: Double,
        savepointThreshold: Double
    ) -> (labels: [Int], savepointLabels: [Int]) {
        let count = points.count
        guard count > 0, weights.count == count else { return ([], []) }
        guard let dimension = points.first?.count, dimension > 0, count > 1 else {
            return (Array(repeating: 0, count: count), Array(repeating: 0, count: count))
        }
        var flat = [Double]()
        flat.reserveCapacity(count * dimension)
        for point in points {
            precondition(point.count == dimension, "All points must share the same dimension")
            flat.append(contentsOf: point)
        }
        let dendrogramLength = (count - 1) * 4
        var dendrogram = [Double](repeating: 0, count: dendrogramLength)
        let status = flat.withUnsafeBufferPointer { flatPointer in
            weights.withUnsafeBufferPointer { weightsPointer in
                dendrogram.withUnsafeMutableBufferPointer { dendrogramPointer in
                    fastcluster_compute_centroid_linkage_weighted(
                        flatPointer.baseAddress,
                        count,
                        dimension,
                        weightsPointer.baseAddress,
                        dendrogramPointer.baseAddress,
                        dendrogramLength
                    )
                }
            }
        }
        guard status == FASTCLUSTER_WRAPPER_SUCCESS else {
            logger.error("fastcluster (weighted points) failed with status \(status.rawValue)")
            return (Array(0..<count), Array(0..<count))
        }
        let cut = clampDistanceThreshold(threshold)
        let tight = min(cut, clampDistanceThreshold(savepointThreshold))
        return (
            remapClusterIds(assignmentsFromDendrogram(dendrogram, count: count, distanceThreshold: cut)),
            remapClusterIds(assignmentsFromDendrogram(dendrogram, count: count, distanceThreshold: tight))
        )
    }

    // MARK: - Exact-duplicate collapsing

    struct DuplicateGroups {
        /// Point index of every input row.
        let rowToPoint: [Int]
        /// Representative (first) row of every point, in first-appearance order.
        let pointRows: [Int]
        /// Copies per point.
        let weights: [Int32]
    }

    /// Groups bit-identical rows. Hash buckets are confirmed element by element
    /// on bit patterns, so a hash collision never merges two different rows.
    static func exactDuplicateGroups(_ rows: [[Double]]) -> DuplicateGroups {
        var rowToPoint = [Int](repeating: 0, count: rows.count)
        var pointRows: [Int] = []
        var weights: [Int32] = []
        var buckets: [Int: [Int]] = [:]
        buckets.reserveCapacity(rows.count)
        for (row, values) in rows.enumerated() {
            var hasher = Hasher()
            hasher.combine(values.count)
            for value in values { hasher.combine(value.bitPattern) }
            let key = hasher.finalize()
            if let match = buckets[key]?.first(where: { bitIdentical(rows[pointRows[$0]], values) }) {
                rowToPoint[row] = match
                weights[match] += 1
            } else {
                let point = pointRows.count
                pointRows.append(row)
                weights.append(1)
                buckets[key, default: []].append(point)
                rowToPoint[row] = point
            }
        }
        return DuplicateGroups(rowToPoint: rowToPoint, pointRows: pointRows, weights: weights)
    }

    private static func bitIdentical(_ a: [Double], _ b: [Double]) -> Bool {
        guard a.count == b.count else { return false }
        for i in 0..<a.count where a[i].bitPattern != b[i].bitPattern {
            return false
        }
        return true
    }

    // MARK: - L2 Feature Normalization
    private func normalizeFeatures(_ features: [[Double]], dimension: Int) -> [Double] {
        var normalized = [Double](repeating: 0, count: features.count * dimension)

        for (rowIndex, vector) in features.enumerated() {
            precondition(vector.count == dimension, "All feature vectors must share the same dimension")

            var norm: Double = 0
            vector.withUnsafeBufferPointer { pointer in
                vDSP_dotprD(
                    pointer.baseAddress!,
                    1,
                    pointer.baseAddress!,
                    1,
                    &norm,
                    vDSP_Length(dimension)
                )
            }

            let scale = norm > 0 ? 1.0 / sqrt(norm) : 0
            var mutableScale = scale
            vector.withUnsafeBufferPointer { source in
                normalized.withUnsafeMutableBufferPointer { destination in
                    vDSP_vsmulD(
                        source.baseAddress!,
                        1,
                        &mutableScale,
                        destination.baseAddress!.advanced(by: rowIndex * dimension),
                        1,
                        vDSP_Length(dimension)
                    )
                }
            }
        }

        return normalized
    }

    // MARK: - Distance Threshold Validation
    // The threshold is a Euclidean cut distance on unit-normalized embeddings,
    // applied directly to the dendrogram like pyannote's
    // `fcluster(dendrogram, threshold, criterion="distance")`. Valid range is
    // [0, 2]; larger values merge more aggressively (fewer clusters).
    private func clampDistanceThreshold(_ threshold: Double) -> Double {
        guard !threshold.isNaN else {
            logger.warning("Clustering threshold is NaN; disabling AHC merging")
            return 0
        }
        if threshold < 0 || threshold > 2.0 {
            logger.warning("Clustering threshold \(threshold) outside [0, 2]; clamping")
        }
        return max(0, min(2.0, threshold))
    }

    /// Cluster from a precomputed condensed distance matrix using
    /// average linkage (NN-chain algorithm). Distances must be
    /// non-negative; smaller = more similar. Distance threshold is
    /// applied directly (no similarity-to-distance conversion — the
    /// caller is responsible for choosing a threshold in the same
    /// units as `condensedDistances`).
    ///
    /// - Parameters:
    ///   - condensedDistances: Upper-triangular packed pairwise
    ///     distances, row-major. Length must be `n * (n-1) / 2`
    ///     where `n` is the number of points. The buffer IS MUTATED
    ///     by NN-chain — callers should treat it as consumed.
    ///   - pointCount: Number of original points.
    ///   - distanceThreshold: Merge node iff its merge distance is
    ///     ≤ this threshold.
    public func clusterFromCondensedDistances(
        condensedDistances: inout [Double],
        pointCount: Int,
        distanceThreshold: Double
    ) -> [Int] {
        guard pointCount > 0 else { return [] }
        if pointCount == 1 { return [0] }

        let ahcState = signposter.beginInterval("AHC from precomputed distances")
        let expectedLength = pointCount * (pointCount - 1) / 2
        guard condensedDistances.count == expectedLength else {
            logger.error(
                "condensedDistances has \(condensedDistances.count) entries; expected \(expectedLength)"
            )
            return Array(0..<pointCount)
        }

        let dendrogramLength = (pointCount - 1) * 4
        var dendrogram = [Double](repeating: 0, count: dendrogramLength)

        let status = condensedDistances.withUnsafeMutableBufferPointer { distPtr in
            dendrogram.withUnsafeMutableBufferPointer { dendPtr in
                fastcluster_compute_average_linkage_from_distances(
                    distPtr.baseAddress,
                    pointCount,
                    dendPtr.baseAddress,
                    dendrogramLength
                )
            }
        }

        guard status == FASTCLUSTER_WRAPPER_SUCCESS else {
            logger.error("fastcluster (average linkage) failed with status \(status.rawValue)")
            return Array(0..<pointCount)
        }

        let assignments = assignmentsFromDendrogram(
            dendrogram,
            count: pointCount,
            distanceThreshold: distanceThreshold
        )
        let result = remapClusterIds(assignments)
        signposter.endInterval("AHC from precomputed distances", ahcState)
        return result
    }

    // MARK: - Dendrogram Parsing & Threshold-Based Cluster Assignment
    private func assignmentsFromDendrogram(
        _ dendrogram: [Double],
        count: Int,
        distanceThreshold: Double
    ) -> [Int] {
        guard count > 0 else { return [] }
        if count == 1 {
            return [0]
        }

        let totalNodes = count * 2 - 1
        var leftChild = [Int](repeating: -1, count: totalNodes)
        var rightChild = [Int](repeating: -1, count: totalNodes)
        var nodeDistance = [Double](repeating: 0, count: totalNodes)

        for mergeIndex in 0..<(count - 1) {
            let base = mergeIndex * 4
            let left = Int(dendrogram[base])
            let right = Int(dendrogram[base + 1])
            let dist = dendrogram[base + 2]
            let newNode = count + mergeIndex
            leftChild[newNode] = left
            rightChild[newNode] = right
            nodeDistance[newNode] = dist
        }

        let root = totalNodes - 1
        var assignments = [Int](repeating: -1, count: count)
        var stack = [root]
        var nextLabel = 0

        while let node = stack.popLast() {
            if node < 0 {
                continue
            }

            if node < count {
                if assignments[node] == -1 {
                    assignments[node] = nextLabel
                    nextLabel += 1
                }
                continue
            }

            let distance = nodeDistance[node]
            if distance <= distanceThreshold {
                let label = nextLabel
                nextLabel += 1
                var queue = [node]
                while let current = queue.popLast() {
                    if current < count {
                        assignments[current] = label
                    } else {
                        let left = leftChild[current]
                        let right = rightChild[current]
                        if left >= 0 { queue.append(left) }
                        if right >= 0 { queue.append(right) }
                    }
                }
            } else {
                let left = leftChild[node]
                let right = rightChild[node]
                if left >= 0 { stack.append(left) }
                if right >= 0 { stack.append(right) }
            }
        }

        for index in 0..<assignments.count where assignments[index] == -1 {
            assignments[index] = nextLabel
            nextLabel += 1
        }

        return assignments
    }

    // MARK: - Cluster ID Remapping
    private func remapClusterIds(_ assignments: [Int]) -> [Int] {
        var mapping: [Int: Int] = [:]
        var nextId = 0
        return assignments.map { original in
            if mapping[original] == nil {
                mapping[original] = nextId
                nextId += 1
            }
            return mapping[original]!
        }
    }
}
