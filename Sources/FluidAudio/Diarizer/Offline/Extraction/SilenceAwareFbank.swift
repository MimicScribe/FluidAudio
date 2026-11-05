import CoreML
import Foundation
import os

/// Re-centers FBANK features on the frames that hold signal (#981).
///
/// The bundled `FBank.mlmodelc` ends with `features = mel - reduce_mean(mel, frames)`.
/// A frame of exact digital silence sits at the log floor, so in a window that is
/// mostly silence (a muted call track, end-of-file padding) that mean is dominated by
/// the floor and every speech frame is shifted by it: the window's embedding loses the
/// voice. Because the model only subtracts a per-band constant, subtracting the mean of
/// its output over the non-silent frames gives exactly the features of a model whose
/// mean excluded the silent frames. Windows without a silent frame are left untouched.
///
/// Public so an app running FBank itself (a masked per-sentence extractor) applies the
/// same correction. Kill switch: `MIMICSCRIBE_SILENCE_FBANK=0`.
public enum SilenceAwareFbank {
    /// FBANK frame geometry: 25 ms frames every 10 ms at 16 kHz.
    public static let frameLength = 400
    public static let frameShift = 160

    /// Off only with `MIMICSCRIBE_SILENCE_FBANK=0`, for a same-binary A/B.
    public static let isEnabled: Bool =
        ProcessInfo.processInfo.environment["MIMICSCRIBE_SILENCE_FBANK"] != "0"

    private static let counts = OSAllocatedUnfairLock(initialState: (checked: 0, recentered: 0))

    /// Windows checked and windows re-centered since launch, so a run can prove the
    /// correction fired on audio that has digital silence.
    public static var counters: (checked: Int, recentered: Int) {
        counts.withLock { $0 }
    }

    /// Marks each frame whose samples are all exactly zero.
    public static func silentFrames(audio: UnsafeBufferPointer<Float>, frameCount: Int) -> [Bool] {
        guard audio.contains(0) else { return [Bool](repeating: false, count: frameCount) }
        var nonZeroBefore = [Int](repeating: 0, count: audio.count + 1)
        for index in 0..<audio.count {
            nonZeroBefore[index + 1] = nonZeroBefore[index] + (audio[index] != 0 ? 1 : 0)
        }
        return (0..<frameCount).map { frame in
            let start = min(frame * frameShift, audio.count)
            let end = min(start + frameLength, audio.count)
            return nonZeroBefore[end] == nonZeroBefore[start]
        }
    }

    /// Subtracts from each band its mean over the non-silent frames.
    ///
    /// - Returns: `false`, leaving `features` untouched, when no frame or every frame is silent.
    @discardableResult
    public static func recenter(
        features: UnsafeMutablePointer<Float>,
        bandCount: Int,
        bandStride: Int,
        frameStride: Int,
        silent: [Bool]
    ) -> Bool {
        let voiced = silent.indices.filter { !silent[$0] }
        guard !voiced.isEmpty, voiced.count < silent.count else { return false }
        for band in 0..<bandCount {
            let row = features + band * bandStride
            var sum: Float = 0
            for frame in voiced {
                sum += row[frame * frameStride]
            }
            let mean = sum / Float(voiced.count)
            for frame in silent.indices {
                row[frame * frameStride] -= mean
            }
        }
        return true
    }

    /// Applies the correction in place to one `[1, 1, bands, frames]` float32 FBANK output,
    /// given the exact audio fed to the model. Returns whether it changed the features.
    @discardableResult
    public static func recenterInPlace(_ features: MLMultiArray, audio: UnsafeBufferPointer<Float>) -> Bool {
        guard isEnabled, features.dataType == .float32, features.shape.count == 4 else { return false }
        let silent = silentFrames(audio: audio, frameCount: features.shape[3].intValue)
        let changed = recenter(
            features: features.dataPointer.assumingMemoryBound(to: Float.self),
            bandCount: features.shape[2].intValue,
            bandStride: features.strides[2].intValue,
            frameStride: features.strides[3].intValue,
            silent: silent
        )
        counts.withLock {
            $0.checked += 1
            if changed { $0.recentered += 1 }
        }
        return changed
    }
}
