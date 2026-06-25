package com.katka.engine.smoothing

import org.junit.Assert.assertEquals
import org.junit.Test

class OptimalAlphaTest {

    private val kf = doubleArrayOf(0.0, 0.0)
    private val sg = doubleArrayOf(2.0, 0.0)

    @Test
    fun pseudoTruthAtKalman_givesZero() {
        // x* == x_KF → no correction needed → α = 0
        assertEquals(0.0, OptimalAlpha.solve(kf, sg, doubleArrayOf(0.0, 0.0), 0.0), 1e-9)
    }

    @Test
    fun pseudoTruthAtSavitzkyGolay_givesOne() {
        // x* == x_SG → fully trust SG → α = 1
        assertEquals(1.0, OptimalAlpha.solve(kf, sg, doubleArrayOf(2.0, 0.0), 0.0), 1e-9)
    }

    @Test
    fun pseudoTruthMidway_givesHalf() {
        assertEquals(0.5, OptimalAlpha.solve(kf, sg, doubleArrayOf(1.0, 0.0), 0.0), 1e-9)
    }

    @Test
    fun degenerateDirection_defaultsToHalf() {
        // x_SG == x_KF → direction undefined → α = 0.5
        assertEquals(0.5, OptimalAlpha.solve(kf, doubleArrayOf(0.0, 0.0), doubleArrayOf(3.0, 1.0), 0.0), 1e-9)
    }

    @Test
    fun sharpTurn_suppressesAlpha() {
        // raw α = 1, but φ = 60° → suppression factor max(0.05, 1 - 60/60) = 0.05
        assertEquals(0.05, OptimalAlpha.solve(kf, sg, doubleArrayOf(2.0, 0.0), 60.0), 1e-9)
        // φ = 30° → factor 0.5
        assertEquals(0.5, OptimalAlpha.solve(kf, sg, doubleArrayOf(2.0, 0.0), 30.0), 1e-9)
    }
}
