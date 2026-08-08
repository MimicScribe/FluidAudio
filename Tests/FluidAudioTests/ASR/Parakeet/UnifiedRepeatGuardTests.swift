import Testing

@testable import FluidAudio

/// Tests for `CrossFrameRepeatGuard`, the pure tracker behind
/// `UnifiedRnntDecoder`'s cross-frame repeat guard (WP0, 2026-08-07
/// MimicScribe field evidence: a 31x consecutive-token loop observed at
/// IS1009a t≈770s under 1120ms streaming). The guard is pure — no CoreML
/// model needed — so these tests drive `shouldSuppress`/`reset` directly in
/// decode order, exactly as `UnifiedRnntDecoder.decode` calls it once per
/// accepted (non-blank) token proposal.
struct UnifiedRepeatGuardTests {

    private let token: Int32 = 7
    private let otherToken: Int32 = 9

    /// A genuine short stammer — the same token repeated a handful of times
    /// in a row — must never trip the guard. Default limit is 5, so up to 5
    /// consecutive proposals of the same token are all accepted; this test
    /// exercises a 3x stammer specifically since that's the largest genuine
    /// repeat observed in the corpus.
    @Test
    func genuineStammerUpToThreeTimesPassesUntouched() {
        var guardState = CrossFrameRepeatGuard()
        for _ in 0..<3 {
            #expect(guardState.shouldSuppress(token) == false)
        }
    }

    /// The 6th consecutive proposal of the same token is the first one the
    /// guard suppresses (limit 5: proposals 1-5 accepted, proposal 6 is the
    /// (limit + 1)-th in a row), and every further same-token proposal after
    /// that must stay suppressed rather than flipping back.
    @Test
    func sixthConsecutiveRepeatIsSuppressedAndStaysSuppressed() {
        var guardState = CrossFrameRepeatGuard()
        for i in 1...5 {
            #expect(guardState.shouldSuppress(token) == false, "proposal \(i) should be accepted")
        }
        #expect(guardState.shouldSuppress(token) == true, "6th consecutive proposal should be suppressed")
        for i in 7...10 {
            #expect(guardState.shouldSuppress(token) == true, "proposal \(i) should stay suppressed")
        }
    }

    /// A different token proposed immediately after a suppressed streak must
    /// pass through (it is not itself a repeat) and resets the streak, so it
    /// gets its own full run of accepted proposals before it can trip the
    /// guard.
    @Test
    func differentTokenAfterSuppressionPassesAndResetsCounter() {
        var guardState = CrossFrameRepeatGuard()
        for _ in 1...6 {
            _ = guardState.shouldSuppress(token)
        }
        #expect(guardState.shouldSuppress(otherToken) == false, "new token must not inherit the old streak")

        // The new token gets a fresh budget: proposals 2-5 of otherToken are
        // still accepted, and its own 6th consecutive proposal trips fresh.
        for i in 2...5 {
            #expect(guardState.shouldSuppress(otherToken) == false, "otherToken proposal \(i) should be accepted")
        }
        #expect(guardState.shouldSuppress(otherToken) == true, "otherToken's 6th consecutive proposal should trip")
    }

    /// `reset()` must clear the streak entirely — after a reset, even the
    /// token that had just tripped the guard starts a brand new streak from
    /// scratch instead of resuming past the limit.
    @Test
    func resetClearsState() {
        var guardState = CrossFrameRepeatGuard()
        for _ in 1...6 {
            _ = guardState.shouldSuppress(token)
        }
        #expect(guardState.shouldSuppress(token) == true, "sanity: streak is tripped before reset")

        guardState.reset()

        for i in 1...5 {
            #expect(guardState.shouldSuppress(token) == false, "post-reset proposal \(i) should be accepted")
        }
    }

    /// `UnifiedRnntDecoder` never calls `shouldSuppress` for a blank token —
    /// its inner loop breaks on blank before consulting the guard — so a
    /// blank-separated gap between calls must not disturb an in-progress
    /// streak. This mirrors production exactly: the guard has no notion of
    /// "blank" at all, it just counts whatever it's given, so a frame that
    /// decoded blank (and therefore never called `shouldSuppress`) is
    /// invisible to the streak, and the same token proposed again after that
    /// gap continues the SAME count rather than restarting at 1.
    @Test
    func blankGapsDoNotDisturbTheConsecutiveCount() {
        var guardState = CrossFrameRepeatGuard()
        // Frame 1-3: token proposed and accepted (repeatCount reaches 3).
        for _ in 1...3 {
            #expect(guardState.shouldSuppress(token) == false)
        }
        // Frame 4: decoder proposes blank — its loop breaks before ever
        // calling shouldSuppress, so nothing happens here; the streak is
        // untouched at 3.
        // Frame 5-6: token proposed again; count continues 4, 5 (still
        // accepted) rather than restarting at 1, 2.
        #expect(guardState.shouldSuppress(token) == false, "4th consecutive proposal (post-gap) should be accepted")
        #expect(guardState.shouldSuppress(token) == false, "5th consecutive proposal (post-gap) should be accepted")
        // The 6th proposal overall — still consecutive across the blank gap
        // — is the one that trips, proving the gap never reset the count.
        #expect(guardState.shouldSuppress(token) == true, "6th consecutive proposal (post-gap) should trip")
    }
}
