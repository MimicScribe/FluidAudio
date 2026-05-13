import Accelerate
import CoreML
import Foundation
import OSLog

@available(macOS 14.0, iOS 17.0, *)
public struct PLDATransform {
    private let pldaRhoModel: MLModel
    private let psi: [Double]
    private let memoryOptimizer = ANEMemoryOptimizer()
    private let logger = AppLogger(category: "OfflinePLDA")

    private let embeddingDimension = 256
    private let rhoDimension = 128
    private let maxBatchSize = 32

    public init(pldaRhoModel: MLModel, psi: [Double]) {
        self.pldaRhoModel = pldaRhoModel
        self.psi = psi
    }

    public var phiParameters: [Double] { psi }

    /// Transform a sequence of 256-dimensional embeddings into 128-dimensional
    /// PLDA-space rho features using the Core ML model exported from Pyannote.
    public func transform(_ embeddings: [[Float]]) async throws -> [[Double]] {
        guard !embeddings.isEmpty else { return [] }

        for embedding in embeddings {
            guard embedding.count == embeddingDimension else {
                throw OfflineDiarizationError.invalidConfiguration(
                    "Expected \(embeddingDimension)-dim embeddings, got \(embedding.count)"
                )
            }
        }

        var results: [[Double]] = []
        results.reserveCapacity(embeddings.count)

        do {
            try await performWarmup()
        } catch {
            logger.debug("PLDA warmup skipped due to error: \(error.localizedDescription)")
        }

        var startIndex = 0
        while startIndex < embeddings.count {
            try Task.checkCancellation()
            let endIndex = min(startIndex + maxBatchSize, embeddings.count)
            let batch = embeddings[startIndex..<endIndex]
            let batchResults = try await transformBatch(batch)
            results.append(contentsOf: batchResults)
            startIndex = endIndex
        }

        return results
    }

    /// Convenience wrapper for single-embedding transform.
    public func transform(_ embedding: [Float]) async throws -> [Double] {
        guard embedding.count == embeddingDimension else {
            throw OfflineDiarizationError.invalidConfiguration(
                "Expected \(embeddingDimension)-dim embedding, got \(embedding.count)"
            )
        }
        let transformed = try await transform([embedding])
        return transformed.first ?? []
    }

    /// Decision-relevant two-covariance PLDA log-likelihood-ratio score
    /// between a query and a candidate, both in PLDA-projected rho space.
    ///
    /// Returns the terms of the LLR that vary with the *candidate* given
    /// a fixed query: the bilinear "psi-weighted dot product" minus the
    /// candidate's PLDA-weighted self-energy. The query's own self-energy
    /// and the log-determinant constant drop out of argmax-over-candidates
    /// comparisons, so this score gives the correct ranking when picking
    /// the most-likely cluster for a query.
    ///
    /// Score formula (in the eigendecomposed basis where Σ_w = I and
    /// Σ_b = diag(psi)):
    ///
    ///   score(q, c) = Σ_k [ psi_k / (1 + 2·psi_k) ] · q_k · c_k
    ///              − Σ_k [ psi_k² / (2·(1+psi_k)·(1+2·psi_k)) ] · c_k²
    ///
    /// Reference: Brummer & van Leeuwen (2010), "The speaker partitioning
    /// problem" — standard two-covariance PLDA scoring formulation.
    /// Distinct from `score(_:_:)` which is plain cosine on rho.
    public func llrScore(_ query: [Double], _ candidate: [Double]) -> Double {
        guard query.count == rhoDimension, candidate.count == rhoDimension,
            psi.count == rhoDimension
        else { return 0 }
        var bilinear: Double = 0
        var candidateSelfEnergy: Double = 0
        for k in 0..<rhoDimension {
            let p = psi[k]
            let onePlusPsi = 1.0 + p
            let onePlusTwoPsi = 1.0 + 2.0 * p
            let a = p / onePlusTwoPsi
            let b = (p * p) / (2.0 * onePlusPsi * onePlusTwoPsi)
            bilinear += a * query[k] * candidate[k]
            candidateSelfEnergy += b * candidate[k] * candidate[k]
        }
        return bilinear - candidateSelfEnergy
    }

    /// Cosine similarity score between two rho vectors.
    public func score(_ lhs: [Double], _ rhs: [Double]) -> Double {
        guard lhs.count == rhoDimension, rhs.count == rhoDimension else {
            return 0
        }

        var dot: Double = 0
        var normLhs: Double = 0
        var normRhs: Double = 0

        lhs.withUnsafeBufferPointer { lhsPointer in
            rhs.withUnsafeBufferPointer { rhsPointer in
                vDSP_dotprD(
                    lhsPointer.baseAddress!,
                    1,
                    rhsPointer.baseAddress!,
                    1,
                    &dot,
                    vDSP_Length(rhoDimension)
                )
                vDSP_dotprD(
                    lhsPointer.baseAddress!,
                    1,
                    lhsPointer.baseAddress!,
                    1,
                    &normLhs,
                    vDSP_Length(rhoDimension)
                )
                vDSP_dotprD(
                    rhsPointer.baseAddress!,
                    1,
                    rhsPointer.baseAddress!,
                    1,
                    &normRhs,
                    vDSP_Length(rhoDimension)
                )
            }
        }

        let magnitude = sqrt(normLhs) * sqrt(normRhs)
        if magnitude <= 0 {
            return 0
        }

        return dot / magnitude
    }

    private func transformBatch(_ embeddings: ArraySlice<[Float]>) async throws -> [[Double]] {
        guard !embeddings.isEmpty else { return [] }
        guard embeddings.count <= maxBatchSize else {
            throw OfflineDiarizationError.invalidBatchSize(
                "PldaRho batch size must be <= \(maxBatchSize), got \(embeddings.count)"
            )
        }

        let shape: [NSNumber] = [NSNumber(value: embeddings.count), NSNumber(value: embeddingDimension)]
        let inputArray = try memoryOptimizer.createAlignedArray(shape: shape, dataType: .float32)
        let pointer = inputArray.dataPointer.assumingMemoryBound(to: Float.self)

        for (batchIndex, embedding) in embeddings.enumerated() {
            let base = batchIndex * embeddingDimension
            embedding.withUnsafeBufferPointer { buffer in
                vDSP_mmov(
                    buffer.baseAddress!,
                    pointer.advanced(by: base),
                    vDSP_Length(embeddingDimension),
                    1,
                    vDSP_Length(embeddingDimension),
                    1
                )
            }
        }

        let provider = ZeroCopyDiarizerFeatureProvider(
            features: ["embeddings": MLFeatureValue(multiArray: inputArray)]
        )
        let options = MLPredictionOptions()
        inputArray.prefetchToNeuralEngine()

        let output = try await pldaRhoModel.prediction(from: provider, options: options)

        guard let rhoArray = output.featureValue(for: "rho")?.multiArrayValue else {
            throw OfflineDiarizationError.processingFailed("PldaRho model did not produce rho output")
        }

        let rhoPointer = rhoArray.dataPointer.assumingMemoryBound(to: Float.self)
        var results: [[Double]] = []
        results.reserveCapacity(embeddings.count)

        let totalRhoCount = embeddings.count * rhoDimension
        var rhoScratch = [Double](repeating: 0, count: totalRhoCount)

        let floatPointer = UnsafePointer<Float>(rhoPointer)
        let sourceBuffer = UnsafeBufferPointer(start: floatPointer, count: totalRhoCount)
        rhoScratch.withUnsafeMutableBufferPointer { dest in
            guard let destBase = dest.baseAddress else { return }
            var destinationBuffer = UnsafeMutableBufferPointer(start: destBase, count: totalRhoCount)
            vDSP.convertElements(of: sourceBuffer, to: &destinationBuffer)
        }

        for batchIndex in 0..<embeddings.count {
            let start = batchIndex * rhoDimension
            let end = start + rhoDimension
            let rhoSlice = Array(rhoScratch[start..<end])
            results.append(rhoSlice)
        }

        return results
    }

    private func performWarmup() async throws {
        let warmupShape: [NSNumber] = [1, NSNumber(value: embeddingDimension)]
        let warmupKey = "offline_plda_warmup_embedding_\(embeddingDimension)"
        let warmupArray = try memoryOptimizer.getPooledBuffer(
            key: warmupKey,
            shape: warmupShape,
            dataType: .float32
        )
        let pointer = warmupArray.dataPointer.assumingMemoryBound(to: Float.self)
        vDSP_vclr(pointer, 1, vDSP_Length(warmupArray.count))

        let provider = ZeroCopyDiarizerFeatureProvider(
            features: ["embeddings": MLFeatureValue(multiArray: warmupArray)]
        )
        let options = MLPredictionOptions()
        warmupArray.prefetchToNeuralEngine()
        _ = try await pldaRhoModel.prediction(from: provider, options: options)
    }
}
