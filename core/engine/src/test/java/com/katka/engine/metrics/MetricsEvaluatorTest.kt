package com.katka.engine.metrics

import org.junit.Assert.assertEquals
import org.junit.Test

class MetricsEvaluatorTest {

    private val reference = (0 until 60).map { TimedPosition(it * 1000L, 1.4 * it, 0.0) }

    @Test
    fun delayedTrack_isPairedByTimestamp() {
        // A fixed-lag smoother emits its first point 5 s after the filter.
        val delayed = reference.drop(5)

        val metrics = MetricsEvaluator.compute(delayed, reference)

        assertEquals(0.0, metrics.rmse, 1e-12)
        assertEquals(55, metrics.sampleCount)
    }

    @Test
    fun tolerance_pairsTheNearestReferencePoint() {
        val shifted = reference.map { it.copy(timestamp = it.timestamp + 40L, y = 3.0) }

        assertEquals(0, MetricsEvaluator.compute(shifted, reference).sampleCount)

        val metrics = MetricsEvaluator.compute(shifted, reference, toleranceMs = 50L)
        assertEquals(60, metrics.sampleCount)
        assertEquals(3.0, metrics.rmse, 1e-12)
    }
}
