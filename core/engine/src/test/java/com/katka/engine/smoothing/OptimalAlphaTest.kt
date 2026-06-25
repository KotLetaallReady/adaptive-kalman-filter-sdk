package com.katka.engine.smoothing

import org.junit.Assert.assertEquals
import org.junit.Test

class OptimalAlphaTest {

    private val kf = doubleArrayOf(0.0, 0.0)
    private val sg = doubleArrayOf(2.0, 0.0)

    @Test
    fun pseudoTruthAtKalman_givesZero() {
        assertEquals(0.0, OptimalAlpha.solve(kf, sg, doubleArrayOf(0.0, 0.0), 0.0), 1e-9)
    }

    @Test
    fun pseudoTruthAtSavitzkyGolay_givesOne() {
        assertEquals(1.0, OptimalAlpha.solve(kf, sg, doubleArrayOf(2.0, 0.0), 0.0), 1e-9)
    }

    @Test
    fun pseudoTruthMidway_givesHalf() {
        assertEquals(0.5, OptimalAlpha.solve(kf, sg, doubleArrayOf(1.0, 0.0), 0.0), 1e-9)
    }

    @Test
    fun degenerateDirection_defaultsToHalf() {
        assertEquals(0.5, OptimalAlpha.solve(kf, doubleArrayOf(0.0, 0.0), doubleArrayOf(3.0, 1.0), 0.0), 1e-9)
    }

    @Test
    fun sharpTurn_suppressesAlphaWithDefaultConfig() {
        assertEquals(0.05, OptimalAlpha.solve(kf, sg, doubleArrayOf(2.0, 0.0), 60.0), 1e-9)
        assertEquals(0.5, OptimalAlpha.solve(kf, sg, doubleArrayOf(2.0, 0.0), 30.0), 1e-9)
    }

    @Test
    fun sharpTurn_usesCustomSuppressionConfig() {
        val config = SmootherConfig(
            alphaSuppressionTurnDeg = 100.0,
            minTurnSuppression = 0.2
        )
        assertEquals(0.5, OptimalAlpha.solve(kf, sg, doubleArrayOf(2.0, 0.0), 50.0, config), 1e-9)
        assertEquals(0.2, OptimalAlpha.solve(kf, sg, doubleArrayOf(2.0, 0.0), 100.0, config), 1e-9)
    }
}
