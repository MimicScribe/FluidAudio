import XCTest

@testable import FluidAudio

final class Nemotron3TailChunkTests: XCTestCase {

    /// Simulate `Nemotron3FeatureLoader` + `Nemotron3StateUpdater.update` across tail lengths
    /// covering exact boundary, short (`1..160`), dup+drop (`161..312`), `>= rc`, and sub-frame
    /// edge clips (`0, 1, 7`), asserting per-chunk core lengths, exact total emitted length,
    /// VAD-skip parity, and zero duplicate right-context embeddings pushed into FIFO/spkcache.
    func testOfflineTailEmitsExactCoreFramesWithoutFifoDuplication() throws {
        let config = Nemotron3Config.offline  // chunkLen=340 (27.2s), rc=40 (3.2s)
        let sub = config.subsamplingFactor
        let up = config.upsampleFactor
        let s = config.numSpeakers
        let d = config.preEncoderDims
        let updater = Nemotron3StateUpdater(
            config: config, silenceEmbedding: [Float](repeating: 0, count: d))

        let lengths: [Int] =
            [0, 1, 7]
            + [0, 1, 80, 160, 161, 209, 220, 312, 319, 320, 400].map { config.chunkLen * sub + $0 }

        for featLength in lengths {
            var state = Nemotron3StreamingState(config: config)
            var loader = Nemotron3FeatureLoader(
                config: config,
                featSeq: [Float](repeating: 0, count: featLength * config.melFeatures),
                featLength: featLength,
                featSeqLength: featLength
            )

            var emittedFrameIndices: [Int] = []
            var vadSkippedTotalFrames = 0
            var coreStart = 0
            while let chunk = loader.next() {
                let lc = (chunk.leftOffset + sub / 2) / sub
                let rc = (chunk.rightOffset + sub - 1) / sub
                let chunkEncLength = (chunk.length + sub - 1) / sub
                let scLen = state.spkcacheLength
                let fifoLen = state.fifoLength

                let expectedCoreMel = min(coreStart + config.chunkLen * sub, featLength) - coreStart
                let expectedCoreEnc = (expectedCoreMel + sub - 1) / sub

                var hiRes = [Float](repeating: 0, count: config.packedFrames * up * s)
                var chunkEmbs = [Float](repeating: 0, count: config.chunkEncFrames * d)
                for enc in 0..<chunkEncLength {
                    let absEnc = (coreStart - chunk.leftOffset) / sub + enc
                    // Stamp unique marker (absEnc + 1) into dim 0 of chunkEmbeddings so FIFO/spkcache
                    // duplication of right-context frames is detected immediately.
                    chunkEmbs[enc * d] = Float(absEnc + 1)
                    let absMelBase = coreStart - chunk.leftOffset + enc * sub
                    for u in 0..<up {
                        let packedFrame = (scLen + fifoLen + enc) * up + u
                        hiRes[packedFrame * s] = Float(absMelBase + u)
                    }
                }

                let res = try updater.update(
                    state: &state,
                    chunkEmbeddings: chunkEmbs,
                    chunkEncLength: chunkEncLength,
                    predictions: [Float](repeating: 0, count: config.packedFrames * s),
                    highResPredictions: hiRes,
                    lc: lc,
                    rc: rc
                )
                XCTAssertEqual(
                    res.frameCount, expectedCoreEnc * up,
                    "Unexpected chunk frameCount at featLength=\(featLength), coreStart=\(coreStart)")
                let vadSkipEnc = Nemotron3Diarizer.coreEncFrames(
                    chunkLength: chunk.length,
                    leftOffset: chunk.leftOffset,
                    rightOffset: chunk.rightOffset,
                    config: config)
                XCTAssertEqual(
                    vadSkipEnc, expectedCoreEnc,
                    "VAD-skip enc count diverged at featLength=\(featLength), coreStart=\(coreStart)")
                vadSkippedTotalFrames += vadSkipEnc * up

                for f in 0..<res.frameCount {
                    emittedFrameIndices.append(Int(res.probabilities[f * s].rounded()))
                }
                coreStart = min(coreStart + config.chunkLen * sub, featLength)
            }

            let expectedTotalEmitted = ((featLength + sub - 1) / sub) * up
            XCTAssertEqual(emittedFrameIndices.count, expectedTotalEmitted, "featLength=\(featLength)")
            XCTAssertEqual(vadSkippedTotalFrames, expectedTotalEmitted, "featLength=\(featLength)")
            XCTAssertEqual(
                Array(emittedFrameIndices.prefix(featLength)), Array(0..<featLength),
                "Tail frames duplicated or dropped for offline featLength=\(featLength)")

            // Verify FIFO contains no duplicate encoder-frame markers (old code pushed right-context
            // frames into FIFO on chunk 0 right before chunk 1 pushed the same frames again).
            var seenFifoMarkers: Set<Int> = []
            for idx in 0..<state.fifoLength {
                let marker = Int(state.fifo[idx * d].rounded())
                XCTAssertGreaterThan(marker, 0, "Expected stamped FIFO marker at featLength=\(featLength)")
                XCTAssertTrue(
                    seenFifoMarkers.insert(marker).inserted,
                    "Duplicate encoder frame \(marker) in FIFO at featLength=\(featLength)")
            }
        }
    }

    /// Exercise `chunkLeftContext > 0` (e.g., offline short-step `chunkLen=128, lc=248, rc=4`),
    /// where the old `chunkEncFrames - lc - rc` over-emitted on early ramp-up chunks too.
    func testNonZeroLeftContextRampUpAndTailEmitExactFrames() throws {
        var config = Nemotron3Config.offline
        config.chunkLen = 128
        config.chunkLeftContext = 248
        config.chunkRightContext = 4
        let sub = config.subsamplingFactor
        let up = config.upsampleFactor
        let s = config.numSpeakers
        let d = config.preEncoderDims
        let updater = Nemotron3StateUpdater(
            config: config, silenceEmbedding: [Float](repeating: 0, count: d))

        let featLength = 3 * config.chunkLen * sub + 20  // 3 full chunks + 0.20s tail (< rc=0.32s)
        var state = Nemotron3StreamingState(config: config)
        var loader = Nemotron3FeatureLoader(
            config: config,
            featSeq: [Float](repeating: 0, count: featLength * config.melFeatures),
            featLength: featLength,
            featSeqLength: featLength
        )

        var emitted: [Int] = []
        var coreStart = 0
        while let chunk = loader.next() {
            let lc = (chunk.leftOffset + sub / 2) / sub
            let rc = (chunk.rightOffset + sub - 1) / sub
            let chunkEncLength = (chunk.length + sub - 1) / sub
            let scLen = state.spkcacheLength
            let fifoLen = state.fifoLength

            let expectedCoreMel = min(coreStart + config.chunkLen * sub, featLength) - coreStart
            let expectedCoreEnc = (expectedCoreMel + sub - 1) / sub

            var hiRes = [Float](repeating: 0, count: config.packedFrames * up * s)
            for enc in 0..<chunkEncLength {
                let absMelBase = coreStart - chunk.leftOffset + enc * sub
                for u in 0..<up {
                    let packedFrame = (scLen + fifoLen + enc) * up + u
                    hiRes[packedFrame * s] = Float(absMelBase + u)
                }
            }

            let res = try updater.update(
                state: &state,
                chunkEmbeddings: [Float](repeating: 0, count: config.chunkEncFrames * d),
                chunkEncLength: chunkEncLength,
                predictions: [Float](repeating: 0, count: config.packedFrames * s),
                highResPredictions: hiRes,
                lc: lc,
                rc: rc
            )
            XCTAssertEqual(
                res.frameCount, expectedCoreEnc * up, "lc>0 chunk frameCount mismatch at coreStart=\(coreStart)")
            for f in 0..<res.frameCount {
                emitted.append(Int(res.probabilities[f * s].rounded()))
            }
            coreStart = min(coreStart + config.chunkLen * sub, featLength)
        }

        XCTAssertEqual(Array(emitted.prefix(featLength)), Array(0..<featLength))
    }

    /// Exercise `Nemotron3StreamingFrontend` (`nextChunk(final: false)` then `nextChunk(final: true)`)
    /// through `Nemotron3StateUpdater.update` and `runChunks`'s final trim. Advances ground-truth
    /// `coreStart` by the frontend's true stride (never the code-under-test's output), covering
    /// both offline and shared-capacity `fast128` bug windows (including `10.44 s` → `rem = 21`
    /// where `rcEnc = 3 < 4` triggered dup+drop on the old formula).
    func testStreamingFinishPathEmitsExactFramesWithoutDuplicationOrDrop() throws {
        var fastShared = Nemotron3Config.fast128
        fastShared.modelChunkEncFrames = Nemotron3Config.offline.chunkEncFrames

        let cases: [(name: String, config: Nemotron3Config, durationSeconds: Double)] = [
            ("offline_29.4s_tail_2.2s", .offline, 29.4),
            ("offline_28.0s_tail_0.8s", .offline, 28.0),
            ("offline_1.5s_short_clip", .offline, 1.5),
            ("fast128_10.44s_rem21_dup_drop", fastShared, 10.44),
            ("fast128_10.32s_rem9_spill", fastShared, 10.32),
        ]

        for tc in cases {
            let config = tc.config
            let sub = config.subsamplingFactor
            let up = config.upsampleFactor
            let s = config.numSpeakers
            let d = config.preEncoderDims
            let totalSamples = Int((tc.durationSeconds * 16_000.0).rounded())

            var frontend = Nemotron3StreamingFrontend(config: config)
            var state = Nemotron3StreamingState(config: config)
            let updater = Nemotron3StateUpdater(
                config: config, silenceEmbedding: [Float](repeating: 0, count: d))

            var emitted: [Int] = []
            var coreStart = 0
            var streamedFrameCount = 0

            func drain(final: Bool) throws {
                while let chunk = frontend.nextChunk(final: final) {
                    let lc = (chunk.leftOffset + sub / 2) / sub
                    let rc = (chunk.rightOffset + sub - 1) / sub
                    let chunkEncLength = (chunk.length + sub - 1) / sub
                    let scLen = state.spkcacheLength
                    let fifoLen = state.fifoLength

                    let nextCoreEnd = min(coreStart + config.chunkLen * sub, frontend.melFramesComputed)
                    let expectedCoreEnc = (nextCoreEnd - coreStart + sub - 1) / sub

                    var hiRes = [Float](repeating: 0, count: config.inputPackedFrames * up * s)
                    for enc in 0..<chunkEncLength {
                        let absMelBase = coreStart - chunk.leftOffset + enc * sub
                        for u in 0..<up {
                            let packedFrame = (scLen + fifoLen + enc) * up + u
                            hiRes[packedFrame * s] = Float(absMelBase + u)
                        }
                    }

                    let res = try updater.update(
                        state: &state,
                        chunkEmbeddings: [Float](repeating: 0, count: config.inputChunkEncFrames * d),
                        chunkEncLength: chunkEncLength,
                        predictions: [Float](repeating: 0, count: config.inputPackedFrames * s),
                        highResPredictions: hiRes,
                        lc: lc,
                        rc: rc
                    )
                    XCTAssertEqual(
                        res.frameCount, expectedCoreEnc * up,
                        "\(tc.name) unexpected streaming chunk frameCount at coreStart=\(coreStart)")
                    let vadSkipEnc = Nemotron3Diarizer.coreEncFrames(
                        chunkLength: chunk.length,
                        leftOffset: chunk.leftOffset,
                        rightOffset: chunk.rightOffset,
                        config: config)
                    XCTAssertEqual(vadSkipEnc, expectedCoreEnc, "\(tc.name) VAD-skip mismatch")

                    var keep = res.frameCount
                    let remaining = frontend.melFramesComputed - streamedFrameCount
                    if final, keep > remaining {
                        keep = max(remaining, 0)
                    }
                    for f in 0..<keep {
                        emitted.append(Int(res.probabilities[f * s].rounded()))
                    }
                    streamedFrameCount += keep
                    coreStart = nextCoreEnd
                }
            }

            let pushSize = 25_600
            var offset = 0
            while offset < totalSamples {
                let n = min(pushSize, totalSamples - offset)
                frontend.append([Float](repeating: 0.01, count: n))
                offset += n
                try drain(final: false)
            }
            try drain(final: true)

            let expectedFrames = frontend.melFramesComputed
            XCTAssertEqual(
                emitted, Array(0..<expectedFrames),
                "Streaming finish frames duplicated or dropped for \(tc.name)")
        }
    }
}
