import Accelerate
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
    ///
    /// vDSP, no per-window allocation beyond the result: one min-magnitude pass
    /// answers "any zero at all?" (most windows), then one max-magnitude per
    /// frame. vDSP runs optimised even in a debug build, where the earlier
    /// scalar prefix-sum scan cost ~34 ms per window with zeros. Same result:
    /// a frame is silent iff every sample in it is 0 (or -0); a frame that
    /// starts at or past the end of `audio` is silent.
    public static func silentFrames(audio: UnsafeBufferPointer<Float>, frameCount: Int) -> [Bool] {
        guard let base = audio.baseAddress, !audio.isEmpty else {
            return [Bool](repeating: true, count: frameCount)
        }
        var minMagnitude: Float = 0
        vDSP_minmgv(base, 1, &minMagnitude, vDSP_Length(audio.count))
        guard minMagnitude == 0 else { return [Bool](repeating: false, count: frameCount) }
        return (0..<frameCount).map { frame in
            let start = min(frame * frameShift, audio.count)
            let end = min(start + frameLength, audio.count)
            guard end > start else { return true }
            var maxMagnitude: Float = 0
            vDSP_maxmgv(base + start, 1, &maxMagnitude, vDSP_Length(end - start))
            return maxMagnitude == 0
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
        let frameCount = silent.count
        let voicedMask = silent.map { $0 ? Float(0) : Float(1) }
        let voicedCount = frameCount - silent.lazy.filter { $0 }.count
        guard voicedCount > 0, voicedCount < frameCount else { return false }
        // vDSP for the per-band mean (masked dot product) and the shift, so a
        // debug build does not pay ~10 ms of scalar loops per window. Only the
        // summation order differs from a scalar loop (float rounding).
        let scale = 1 / Float(voicedCount)
        for band in 0..<bandCount {
            let row = features + band * bandStride
            var sum: Float = 0
            vDSP_dotpr(row, vDSP_Stride(frameStride), voicedMask, 1, &sum, vDSP_Length(frameCount))
            var shift = -sum * scale
            vDSP_vsadd(row, vDSP_Stride(frameStride), &shift, row, vDSP_Stride(frameStride), vDSP_Length(frameCount))
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
