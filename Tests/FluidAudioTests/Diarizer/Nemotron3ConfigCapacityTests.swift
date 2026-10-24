import XCTest

@testable import FluidAudio

final class Nemotron3ConfigCapacityTests: XCTestCase {
    func testDefaultCapacityIsTheConfigsOwn() {
        let config = Nemotron3Config.fast128
        XCTAssertEqual(config.inputChunkEncFrames, config.chunkEncFrames)
        XCTAssertEqual(config.inputChunkMelFrames, config.chunkMelFrames)
        XCTAssertEqual(config.inputPackedFrames, config.packedFrames)
    }

    func testShortStepOnTheOfflineModelUsesItsCapacity() {
        var config = Nemotron3Config.fast128
        config.modelChunkEncFrames = Nemotron3Config.offline.chunkEncFrames
        // Stream logic keeps fast128's step; only the model buffers grow.
        XCTAssertEqual(config.chunkEncFrames, 132)
        XCTAssertEqual(config.inputChunkEncFrames, 380)
        XCTAssertEqual(config.inputChunkMelFrames, 3040)
        XCTAssertEqual(config.inputPackedFrames, config.spkcacheLen + config.fifoLen + 380)
    }

    func testCapacityNeverShrinksBelowTheConfig() {
        var config = Nemotron3Config.offline
        config.modelChunkEncFrames = 10
        XCTAssertEqual(config.inputChunkEncFrames, config.chunkEncFrames)
    }
}
