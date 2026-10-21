import Foundation
import XCTest

@testable import FluidAudio

final class AHCClusteringTests: XCTestCase {

    private let clustering = AHCClustering()

    // MARK: - Weighted points (incremental savepoints)

    /// Deterministic unit rows jittered around `centers` (LCG; no audio involved).
    private func jitteredRows(centers: [[Double]], perCenter: Int, jitter: Double) -> [[Double]] {
        var state: UInt64 = 0x9E37_79B9_7F4A_7C15
        func next() -> Double {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Double(state >> 11) / Double(1 << 53) - 0.5
        }
        var rows: [[Double]] = []
        for index in 0..<(centers.count * perCenter) {
            let center = centers[index % centers.count]
            let row = center.map { $0 + jitter * next() }
            let norm = sqrt(row.reduce(0) { $0 + $1 * $1 })
            rows.append(row.map { $0 / norm })
        }
        return rows
    }

    private func samePartition(_ a: [Int], _ b: [Int]) -> Bool {
        guard a.count == b.count else { return false }
        var forward: [Int: Int] = [:]
        var backward: [Int: Int] = [:]
        for (x, y) in zip(a, b) {
            if forward[x, default: y] != y || backward[y, default: x] != x { return false }
            forward[x] = y
            backward[y] = x
        }
        return true
    }

    func testWeightedPointMatchesItsCopies() {
        let rows = jitteredRows(centers: [[1, 0, 0, 0], [0, 1, 0, 0]], perCenter: 6, jitter: 0.3)
        // Row 0 three times vs row 0 once with weight 3.
        let copies = [rows[0], rows[0]] + rows
        let full = clustering.cluster(embeddingFeatures: copies, threshold: 0.6, collapseExactDuplicates: false)
        let weighted = clustering.clusterWeightedPoints(
            points: rows, weights: [3] + [Int32](repeating: 1, count: rows.count - 1), threshold: 0.6,
            savepointThreshold: 0.6)
        XCTAssertTrue(samePartition(Array(full.dropFirst(2)), weighted.labels))
    }

    func testSavepointRunMatchesAFullRunOnSeparatedVoices() {
        let centers: [[Double]] = [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0]]
        let rows = jitteredRows(centers: centers, perCenter: 20, jitter: 0.4)
        let threshold = 0.6325
        let full = clustering.cluster(embeddingFeatures: rows, threshold: threshold)

        // First half: rows as weight-1 points; keep the tight groups as the savepoint.
        let firstHalf = Array(rows.prefix(30))
        let first = clustering.clusterWeightedPoints(
            points: firstHalf, weights: [Int32](repeating: 1, count: 30), threshold: threshold,
            savepointThreshold: 0.7 * threshold)
        let groupIds = Array(Set(first.savepointLabels)).sorted()
        var centroids: [[Double]] = []
        var counts: [Int32] = []
        for id in groupIds {
            let members = firstHalf.indices.filter { first.savepointLabels[$0] == id }.map { firstHalf[$0] }
            centroids.append((0..<5).map { d in members.reduce(0) { $0 + $1[d] } / Double(members.count) })
            counts.append(Int32(members.count))
        }
        XCTAssertLessThan(centroids.count, 30, "the savepoint must hold fewer points than its rows")

        // Second run: the groups (true centroids, weighted) plus the new rows.
        let second = clustering.clusterWeightedPoints(
            points: centroids + Array(rows.suffix(30)), weights: counts + [Int32](repeating: 1, count: 30),
            threshold: threshold, savepointThreshold: 0.7 * threshold)
        let groupOfRow = first.savepointLabels.map { groupIds.firstIndex(of: $0) ?? 0 }
        let incremental = groupOfRow.map { second.labels[$0] } + Array(second.labels.suffix(30))
        XCTAssertTrue(samePartition(full, incremental))
    }

    // MARK: - Empty & Single

    func testEmptyEmbeddingsReturnsEmptyAssignments() {
        let result = clustering.cluster(embeddingFeatures: [], threshold: 0.7)
        XCTAssertTrue(result.isEmpty)
    }

    func testSingleEmbeddingReturnsSingleCluster() {
        let result = clustering.cluster(embeddingFeatures: [[1.0, 0.0, 0.0]], threshold: 0.7)
        XCTAssertEqual(result.count, 1)
        XCTAssertEqual(result[0], 0)
    }

    // MARK: - Identical Embeddings

    func testIdenticalEmbeddingsClusterTogether() {
        let embedding = [1.0, 2.0, 3.0]
        let embeddings = Array(repeating: embedding, count: 5)

        let result = clustering.cluster(embeddingFeatures: embeddings, threshold: 0.7)
        XCTAssertEqual(result.count, 5)

        // All should be in the same cluster
        let uniqueClusters = Set(result)
        XCTAssertEqual(uniqueClusters.count, 1, "Identical embeddings should all be in one cluster")
    }

    // MARK: - Orthogonal Embeddings

    func testOrthogonalGroupsSeparateWhenThresholdBelowGroupDistance() {
        // Two groups: pointing in very different directions. Unit-normalized,
        // the inter-group Euclidean distance is ~sqrt(2) ≈ 1.414, so a cut at
        // 0.8 merges within groups but not across them.
        let group1: [[Double]] = [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.95, 0.05, 0.0],
        ]
        let group2: [[Double]] = [
            [0.0, 1.0, 0.0],
            [0.0, 0.9, 0.1],
            [0.0, 0.95, 0.05],
        ]

        let embeddings = group1 + group2
        let result = clustering.cluster(embeddingFeatures: embeddings, threshold: 0.8)
        XCTAssertEqual(result.count, 6)

        // Group 1 should share a cluster, group 2 should share a different cluster
        let group1Clusters = Set(result[0..<3])
        let group2Clusters = Set(result[3..<6])

        XCTAssertEqual(group1Clusters.count, 1, "Group 1 should be in one cluster")
        XCTAssertEqual(group2Clusters.count, 1, "Group 2 should be in one cluster")
        XCTAssertNotEqual(
            group1Clusters.first, group2Clusters.first,
            "Groups should be in different clusters"
        )
    }

    func testLargerThresholdMergesMore() {
        // pyannote semantics: the threshold is the dendrogram cut distance,
        // so raising it can only reduce the number of clusters.
        let embeddings: [[Double]] = [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.9, 0.1],
        ]

        let tight = Set(clustering.cluster(embeddingFeatures: embeddings, threshold: 0.5)).count
        let loose = Set(clustering.cluster(embeddingFeatures: embeddings, threshold: 1.5)).count

        XCTAssertEqual(tight, 2, "Cut below the inter-group distance should keep two clusters")
        XCTAssertEqual(loose, 1, "Cut above the inter-group distance should merge everything")
        XCTAssertLessThanOrEqual(loose, tight)
    }

    // MARK: - Cluster ID Properties

    func testClusterIdsAreContiguousFromZero() {
        let embeddings: [[Double]] = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]

        let result = clustering.cluster(embeddingFeatures: embeddings, threshold: 0.5)
        XCTAssertEqual(result.count, 3)

        let uniqueIds = Set(result).sorted()
        // IDs should be contiguous starting from 0
        for (i, id) in uniqueIds.enumerated() {
            XCTAssertEqual(id, i, "Cluster IDs should be contiguous from 0")
        }
    }

    // MARK: - Threshold Extremes

    func testMaxThresholdMergesAll() {
        // 2.0 is the largest possible distance between unit vectors, so
        // everything should merge into one cluster.
        let embeddings: [[Double]] = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]

        let result = clustering.cluster(embeddingFeatures: embeddings, threshold: 2.0)
        let uniqueClusters = Set(result)
        XCTAssertEqual(uniqueClusters.count, 1, "Maximum threshold should merge all into one cluster")
    }

    func testZeroThresholdKeepsDistinctEmbeddingsApart() {
        let embeddings: [[Double]] = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]

        let result = clustering.cluster(embeddingFeatures: embeddings, threshold: 0.0)
        let uniqueClusters = Set(result)
        XCTAssertEqual(uniqueClusters.count, 3, "Zero threshold should not merge distinct embeddings")
    }

    // MARK: - Zero Vector

    func testZeroDimensionEmbeddings() {
        // Zero-dimension vectors should all get cluster 0
        let embeddings: [[Double]] = [[], [], []]
        let result = clustering.cluster(embeddingFeatures: embeddings, threshold: 0.7)
        XCTAssertEqual(result.count, 3)
        for id in result {
            XCTAssertEqual(id, 0)
        }
    }
}
