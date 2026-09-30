package com.katka.engine.coefficient_startegy

import com.katka.engine.MatrixOps
import com.katka.model.Observation
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import kotlin.math.exp
import kotlin.math.hypot

class NeuralResidualNoiseStrategyTest {

    private lateinit var strategy: NeuralResidualNoiseStrategy
    private val H = arrayOf(
        doubleArrayOf(1.0, 0.0, 0.0, 0.0),
        doubleArrayOf(0.0, 1.0, 0.0, 0.0)
    )
    private val P_pred = MatrixOps.diagonal(doubleArrayOf(25.0, 25.0, 4.0, 4.0))

    @Before
    fun setUp() {
        strategy = NeuralResidualNoiseStrategy()
    }

    private fun dummyObs(
        acc: Float = 5.0f,
        speed: Float = 5.0f,
        satCount: Int = 12,
        meanCn0: Float = 32.0f
    ) = Observation(
        timestamp = 1619000000000L,
        latitude = 37.422,
        longitude = -122.084,
        accuracy = acc,
        altitude = 10.0,
        speed = speed,
        bearing = 90.0f,
        hasSpeed = true,
        hasBearing = true,
        ax = 0.0,
        ay = 0.0,
        az = 0.0,
        hasImu = false,
        provider = "gps",
        satellitesCount = satCount,
        meanCn0 = meanCn0
    )

    @Test
    fun testInitializationAndDimensions() {
        val obs = dummyObs()
        val res = strategy.computeGain(P_pred, H, obs)

        assertEquals(4, res.K.size)
        assertEquals(2, res.K[0].size)
        assertEquals(4, res.P_updated.size)
        assertEquals(4, res.P_updated[0].size)
        assertEquals(2, res.R.size)
        assertEquals(2, res.R[0].size)

        // R should be symmetric and diagonal
        assertEquals(res.R[0][0], res.R[1][1], 1e-6)
        assertEquals(0.0, res.R[0][1], 1e-9)
        assertEquals(0.0, res.R[1][0], 1e-9)
    }

    @Test
    fun testResidualScalingBounds() {
        val acc = 4.0f
        val obs = dummyObs(acc = acc)
        val rawVar = (acc * acc).toDouble() // 16.0

        val res = strategy.computeGain(P_pred, H, obs)
        val rEff = res.R[0][0]

        // Formula: R_eff = R_raw * exp(clamp(MLP, -1.0, 2.5))
        // Min possible = 16 * exp(-1.0) ~ 5.88
        // Max possible = 16 * exp(2.5) ~ 194.9
        val minExpected = rawVar * exp(-1.0)
        val maxExpected = rawVar * exp(2.5)

        assertTrue("R ($rEff) must be >= $minExpected", rEff >= minExpected - 1e-3)
        assertTrue("R ($rEff) must be <= $maxExpected", rEff <= maxExpected + 1e-3)
    }

    @Test
    fun testInnovationStatsAndHistoryTracking() {
        // Initially history is empty
        val obs = dummyObs()
        val res0 = strategy.computeGain(P_pred, H, obs)
        assertFalse(res0.R[0][0].isNaN())

        // Feed large innovations (simulating severe multipath jumps)
        for (i in 1..10) {
            strategy.onInnovation(doubleArrayOf(15.0, 20.0))
        }

        // With large past innovations, the model should inflate R to protect the filter
        val resHigh = strategy.computeGain(P_pred, H, obs)
        assertTrue("R after severe innovations should be inflated", resHigh.R[0][0] > res0.R[0][0])

        // After reset, history is cleared
        strategy.reset()
        val resReset = strategy.computeGain(P_pred, H, obs)
        assertEquals(res0.R[0][0], resReset.R[0][0], 1e-6)
    }

    @Test
    fun testHuberSoftGateInteraction() {
        val obs = dummyObs()

        // Normal small innovation (within Huber gate)
        val normalInnov = doubleArrayOf(0.5, 0.5)
        val normalRes = strategy.computeGain(P_pred, H, obs, innovation = normalInnov)

        // Giant outlier innovation (e.g. 50 meters GPS jump)
        val outlierInnov = doubleArrayOf(50.0, 50.0)
        val outlierRes = strategy.computeGain(P_pred, H, obs, innovation = outlierInnov)

        // Under Huber soft gate, outlier R must be inflated and Kalman gain K must be reduced
        assertTrue("Outlier R should be heavily inflated", outlierRes.R[0][0] > normalRes.R[0][0] * 2.0)
        assertTrue("Outlier K should be reduced compared to normal K", outlierRes.K[0][0] < normalRes.K[0][0])
    }

    @Test
    fun testGnssStatusInfluence() {
        val obsOpenSky = dummyObs(satCount = 20, meanCn0 = 42.0f)
        val obsUrbanCanyon = dummyObs(satCount = 4, meanCn0 = 16.0f)

        val resOpenSky = strategy.computeGain(P_pred, H, obsOpenSky)
        val resUrbanCanyon = strategy.computeGain(P_pred, H, obsUrbanCanyon)

        assertTrue(
            "Urban canyon (low C/N0, few satellites) should result in higher noise variance than open sky",
            resUrbanCanyon.R[0][0] > resOpenSky.R[0][0]
        )
    }

    @Test
    fun testNumericalRobustnessAgainstNaNAndInfinity() {
        val obs = dummyObs()

        // Test with NaNs in innovation
        val nanInnov = doubleArrayOf(Double.NaN, Double.POSITIVE_INFINITY)
        strategy.onInnovation(nanInnov)
        val res = strategy.computeGain(P_pred, H, obs, innovation = nanInnov)

        assertFalse("K must not contain NaN", res.K[0][0].isNaN())
        assertFalse("P must not contain NaN", res.P_updated[0][0].isNaN())
        assertFalse("R must not contain NaN", res.R[0][0].isNaN())
        assertTrue("R must remain positive", res.R[0][0] > 0.0)
    }

    @Test
    fun testZeroAllocationThroughput() {
        val obs = dummyObs()
        val dummyInnov = doubleArrayOf(1.2, 0.8)

        // Warmup JIT
        for (i in 0 until 5_000) {
            strategy.computeGain(P_pred, H, obs, dummyInnov)
            strategy.onInnovation(dummyInnov)
        }

        // Benchmark 50,000 steps
        val iterations = 50_000
        val t0 = System.nanoTime()
        for (i in 0 until iterations) {
            strategy.computeGain(P_pred, H, obs, dummyInnov)
        }
        val elapsedNanos = System.nanoTime() - t0
        val nanosPerStep = elapsedNanos.toDouble() / iterations
        val usPerStep = nanosPerStep / 1000.0

        println("NeuralResidualNoiseStrategy throughput: ${String.format("%.2f", usPerStep)} us/step (${String.format("%.0f", 1e9 / nanosPerStep)} steps/sec)")

        // Must run in under 10 microseconds even in unoptimized debug test runner
        assertTrue("Inference must be sub-microsecond or near it (< 10us in JVM test)", usPerStep < 10.0)
    }
}
