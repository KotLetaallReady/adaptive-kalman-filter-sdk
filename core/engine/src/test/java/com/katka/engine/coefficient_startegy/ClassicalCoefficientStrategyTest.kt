package com.katka.engine.coefficient_startegy

import com.katka.engine.MatrixOps
import com.katka.model.Observation
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ClassicalCoefficientStrategyTest {

    private val H = arrayOf(
        doubleArrayOf(1.0, 0.0, 0.0, 0.0),
        doubleArrayOf(0.0, 1.0, 0.0, 0.0)
    )

    private fun makeObservation(accM: Float = 4.0f) = Observation(
        timestamp = 1000L,
        latitude = 55.75,
        longitude = 37.61,
        accuracy = accM,
        altitude = 0.0,
        speed = 0.0f,
        bearing = 0.0f,
        hasSpeed = false,
        hasBearing = false,
        ax = 0.0,
        ay = 0.0,
        az = 0.0,
        hasImu = false,
        provider = "gps"
    )

    @Test
    fun smallInnovation_doesNotInflateR() {
        val strategy = ClassicalCoefficientStrategy(robustUpdate = true, huberThreshold = 2.45)
        val pPred = MatrixOps.zeros(4, 4)
        val obs = makeObservation(accM = 4.0f) // R0 = 16.0 m², sigma_S = 4.0 m

        // Innovation = 4.0m -> u = 4.0 / 4.0 = 1.0 < 2.45
        val innovation = doubleArrayOf(4.0, 0.0)
        val result = strategy.computeGain(pPred, H, obs, innovation)

        assertEquals("R[0][0] should not be inflated when u <= 2.45", 16.0, result.R[0][0], 1e-4)
        assertEquals("R[1][1] should not be inflated when u <= 2.45", 16.0, result.R[1][1], 1e-4)
    }

    @Test
    fun largeInnovation_inflatesR_proportionally() {
        val strategy = ClassicalCoefficientStrategy(robustUpdate = true, huberThreshold = 2.45)
        val pPred = MatrixOps.zeros(4, 4)
        val obs = makeObservation(accM = 4.0f) // R0 = 16.0 m², sigma_S = 4.0 m

        // Innovation = 19.6m -> u = 19.6 / 4.0 = 4.9 = 2.0 * huberThreshold
        val innovation = doubleArrayOf(19.6, 0.0)
        val result = strategy.computeGain(pPred, H, obs, innovation)

        // Expected inflation factor = 2.0 -> R = 32.0 m²
        assertEquals("R[0][0] should be scaled by u/c = 2.0", 32.0, result.R[0][0], 1e-4)
        assertEquals("R[1][1] should be scaled by u/c = 2.0", 32.0, result.R[1][1], 1e-4)
    }

    @Test
    fun robustUpdateDisabled_doesNotInflateR() {
        val strategy = ClassicalCoefficientStrategy(robustUpdate = false, huberThreshold = 2.45)
        val pPred = MatrixOps.zeros(4, 4)
        val obs = makeObservation(accM = 4.0f)

        // Even with huge innovation, R stays at 16.0 when robustUpdate is disabled
        val innovation = doubleArrayOf(50.0, 0.0)
        val result = strategy.computeGain(pPred, H, obs, innovation)

        assertEquals("R[0][0] must stay at baseline 16.0 when robustUpdate is false", 16.0, result.R[0][0], 1e-4)
    }

    @Test
    fun huber_decreasesKalmanGain_forLargeInnovation() {
        val pPred = MatrixOps.diagonal(doubleArrayOf(25.0, 25.0, 5.0, 5.0))
        val obs = makeObservation(accM = 4.0f)
        val innovation = doubleArrayOf(20.0, 0.0)

        val nonRobustStrategy = ClassicalCoefficientStrategy(robustUpdate = false)
        val robustStrategy = ClassicalCoefficientStrategy(robustUpdate = true, huberThreshold = 2.45)

        val nonRobustGain = nonRobustStrategy.computeGain(pPred, H, obs, innovation)
        val robustGain = robustStrategy.computeGain(pPred, H, obs, innovation)

        assertTrue(
            "Robust Kalman gain should be smaller than non-robust gain (${robustGain.K[0][0]} < ${nonRobustGain.K[0][0]})",
            robustGain.K[0][0] < nonRobustGain.K[0][0]
        )
    }
}
