import Foundation

/// The `maskSimilarity` embedding-skip cache, carried from one extraction call
/// to the next so a long recording can be extracted in slices that start
/// MID-batch and still reproduce a single full pass bit for bit.
///
/// Within one FBANK batch (32 windows) a window may reuse the embedding of an
/// EARLIER window of the same batch; the cache clears when a batch completes.
/// Segmentation and FBANK are per window, so the only state that crosses a
/// window boundary is this cache plus the position inside the current batch.
/// A slice that starts at window `w` of a full pass therefore matches it
/// exactly when it starts with the cache the full pass held after window
/// `w - 1` and with `batchPhase` = how many windows of the current batch came
/// before `w`.
///
/// Usage: extract slice k with `captureAfterChunk` = its last committed
/// (slice-local) chunk index; pass the returned carry as `carryIn` of slice
/// k+1, whose first window is the one after.
public struct EmbeddingSkipCarry: Sendable, Equatable {
    public struct Entry: Sendable, Equatable {
        /// The mask that GENERATED `embedding` (the skip compares against it).
        public let mask: [Float]
        public let embedding: [Float]

        public init(mask: [Float], embedding: [Float]) {
            self.mask = mask
            self.embedding = embedding
        }
    }

    /// Keyed by the chunk-local speaker slot (0..<3).
    public var entries: [Int: Entry]
    /// Windows of the current FBANK batch already processed before the next
    /// call's first window (0 = the next window starts a new batch).
    public var batchPhase: Int

    public init(entries: [Int: Entry], batchPhase: Int) {
        self.entries = entries
        self.batchPhase = batchPhase
    }

    /// A fresh batch with an empty cache — what a call without `carryIn` uses.
    public static let empty = EmbeddingSkipCarry(entries: [:], batchPhase: 0)
}
