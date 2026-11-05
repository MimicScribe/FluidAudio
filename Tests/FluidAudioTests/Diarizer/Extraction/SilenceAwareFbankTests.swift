import CoreML
import XCTest

@testable import FluidAudio

final class SilenceAwareFbankTests: XCTestCase {

    func testFramesOfExactZerosAreSilent() {
        // Frames start every 160 samples and span 400: speech in [0, 800), zeros after.
        let audio = [Float](repeating: 0.1, count: 800) + [Float](repeating: 0, count: 1_600)

        let silent = audio.withUnsafeBufferPointer { SilenceAwareFbank.silentFrames(audio: $0, frameCount: 13) }

        // Frame 2 covers [320, 720) and frame 3 covers [480, 880): only frames from 5 on are all zero.
        XCTAssertEqual(silent, [false, false, false, false, false, true, true, true, true, true, true, true, true])
    }

    func testRecenterMatchesAMeanThatExcludesSilentFrames() {
        // Two bands, four frames (band-major); frames 2 and 3 are silent at the log floor.
        let raw: [Float] = [1, 3, -13.8, -13.8, 2, 6, -13.8, -13.8]
        let modelMean: [Float] = [(1 + 3 - 27.6) / 4, (2 + 6 - 27.6) / 4]
        var modelOutput = (0..<8).map { raw[$0] - modelMean[$0 / 4] }

        let changed = modelOutput.withUnsafeMutableBufferPointer {
            SilenceAwareFbank.recenter(
                features: $0.baseAddress!, bandCount: 2, bandStride: 4, frameStride: 1,
                silent: [false, false, true, true])
        }

        XCTAssertTrue(changed)
        XCTAssertEqual(modelOutput[0], 1 - 2, accuracy: 1e-5)
        XCTAssertEqual(modelOutput[1], 3 - 2, accuracy: 1e-5)
        XCTAssertEqual(modelOutput[4], 2 - 4, accuracy: 1e-5)
        XCTAssertEqual(modelOutput[5], 6 - 4, accuracy: 1e-5)
    }

    func testWindowsWithoutSilenceOrWithOnlySilenceAreUntouched() {
        var features: [Float] = [1, 2, 3, 4]

        let withoutSilence = features.withUnsafeMutableBufferPointer {
            SilenceAwareFbank.recenter(
                features: $0.baseAddress!, bandCount: 1, bandStride: 4, frameStride: 1,
                silent: [false, false, false, false])
        }
        let allSilent = features.withUnsafeMutableBufferPointer {
            SilenceAwareFbank.recenter(
                features: $0.baseAddress!, bandCount: 1, bandStride: 4, frameStride: 1,
                silent: [true, true, true, true])
        }

        XCTAssertFalse(withoutSilence)
        XCTAssertFalse(allSilent)
        XCTAssertEqual(features, [1, 2, 3, 4])
    }

    func testRecenterInPlaceOnAModelShapedArrayCountsAndMatchesTheFormula() throws {
        // [1, 1, bands=2, frames=4].
        let features = try MLMultiArray(shape: [1, 1, 2, 4], dataType: .float32)
        let values: [Float] = [1, 3, -13.8, -13.8, 2, 6, -13.8, -13.8]
        for (i, v) in values.enumerated() { features[i] = NSNumber(value: v) }
        let audio = [Float](repeating: 0.1, count: 300) + [Float](repeating: 0, count: 580)
        let before = SilenceAwareFbank.counters

        let changed = audio.withUnsafeBufferPointer { SilenceAwareFbank.recenterInPlace(features, audio: $0) }

        XCTAssertEqual(changed, SilenceAwareFbank.isEnabled)
        guard SilenceAwareFbank.isEnabled else { return }
        // Signal in [0, 300): frames 0 and 1 hold it (frame 1 covers [160, 560)); frame 2 starts at 320, all zero.
        XCTAssertEqual(features[0].floatValue, 1 - 2, accuracy: 1e-5)
        XCTAssertEqual(features[5].floatValue, 6 - 4, accuracy: 1e-5)
        XCTAssertEqual(SilenceAwareFbank.counters.checked, before.checked + 1)
        XCTAssertEqual(SilenceAwareFbank.counters.recentered, before.recentered + 1)
    }

    func testAudioWithoutAnyZeroSkipsTheFrameScan() {
        let audio = [Float](repeating: 0.1, count: 2_400)
        let silent = audio.withUnsafeBufferPointer { SilenceAwareFbank.silentFrames(audio: $0, frameCount: 13) }
        XCTAssertEqual(silent, [Bool](repeating: false, count: 13))
    }

    /// The vDSP scan must agree exactly with the original prefix-sum definition
    /// (upstream #988), including -0, NaN-free noise, short tails and windows
    /// with no zero.
    func testVDSPScanMatchesThePrefixSumDefinition() {
        func reference(_ audio: [Float], _ frameCount: Int) -> [Bool] {
            var nonZeroBefore = [Int](repeating: 0, count: audio.count + 1)
            for i in 0..<audio.count { nonZeroBefore[i + 1] = nonZeroBefore[i] + (audio[i] != 0 ? 1 : 0) }
            return (0..<frameCount).map { f in
                let start = min(f * 160, audio.count)
                let end = min(start + 400, audio.count)
                return nonZeroBefore[end] == nonZeroBefore[start]
            }
        }
        var state: UInt32 = 7
        func next() -> UInt32 { state = state &* 1_664_525 &+ 1_013_904_223; return state }
        for trial in 0..<40 {
            let count = 2_000 + Int(next() % 6_000)
            var audio = (0..<count).map { _ in Float(Int32(bitPattern: next())) / Float(Int32.max) }
            for _ in 0..<Int(next() % 6) {
                let a = Int(next() % UInt32(count)); let b = min(count, a + Int(next() % 1_500))
                for i in a..<b { audio[i] = (trial % 3 == 0 && i % 7 == 0) ? -0.0 : 0 }
            }
            let frameCount = (count - 400) / 160 + 1 + Int(next() % 4)
            let got = audio.withUnsafeBufferPointer { SilenceAwareFbank.silentFrames(audio: $0, frameCount: frameCount) }
            XCTAssertEqual(got, reference(audio, frameCount), "trial \(trial)")
        }
    }
}
