package com.katka.engine.coefficient_startegy

import com.katka.engine.MatrixOps
import com.katka.model.Observation
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.Random
import kotlin.math.exp
import kotlin.math.max

class NeuralResidualNoiseStrategyTest {

    private val H = arrayOf(
        doubleArrayOf(1.0, 0.0, 0.0, 0.0),
        doubleArrayOf(0.0, 1.0, 0.0, 0.0)
    )
    private val P_pred = MatrixOps.diagonal(doubleArrayOf(25.0, 25.0, 4.0, 4.0))

    private fun obs(acc: Float = 5.0f, speed: Float = 5.0f) = Observation(
        timestamp = 1L, latitude = 55.75, longitude = 37.62, accuracy = acc, altitude = 0.0,
        speed = speed, bearing = 90.0f, hasSpeed = true, hasBearing = true,
        ax = 0.0, ay = 0.0, az = 0.0, hasImu = false, provider = "gps"
    )

    private fun NeuralResidualNoiseStrategy.feed(innovation: DoubleArray, times: Int = 10) =
        repeat(times) { onInnovation(innovation, H, P_pred) }

    @Test
    fun noiseStaysInsideTheClampCorridor() {
        for (acc in listOf(2f, 5f, 20f)) {
            val s = NeuralResidualNoiseStrategy(robustUpdate = false)
            s.feed(doubleArrayOf(30.0, 30.0))
            val r = s.computeGain(P_pred, H, obs(acc)).R
            val raw = (acc * acc).toDouble()
            assertTrue(r[0][0] >= raw * exp(-1.0) - 1e-9 && r[0][0] <= raw * exp(2.5) + 1e-9)
            assertEquals(r[0][0], r[1][1], 1e-12)
            assertEquals(0.0, r[0][1], 1e-12)
        }
    }

    @Test
    fun largePastInnovations_inflateR_andResetForgetsThem() {
        val s = NeuralResidualNoiseStrategy(robustUpdate = false)
        s.feed(doubleArrayOf(1.0, 1.0))
        val quiet = s.computeGain(P_pred, H, obs()).R[0][0]
        s.feed(doubleArrayOf(20.0, 20.0))
        val noisy = s.computeGain(P_pred, H, obs()).R[0][0]
        assertTrue("noisy $noisy vs quiet $quiet", noisy > quiet)

        s.reset()
        val fresh = NeuralResidualNoiseStrategy(robustUpdate = false).computeGain(P_pred, H, obs()).R[0][0]
        assertEquals(fresh, s.computeGain(P_pred, H, obs()).R[0][0], 1e-12)
    }

    @Test
    fun huberGate_inflatesROnlyForTheOutlier() {
        val s = NeuralResidualNoiseStrategy()
        val normal = s.computeGain(P_pred, H, obs(), innovation = doubleArrayOf(0.5, 0.5))
        val outlier = s.computeGain(P_pred, H, obs(), innovation = doubleArrayOf(50.0, 50.0))
        assertTrue(outlier.R[0][0] > normal.R[0][0] * 2.0)
        assertTrue(outlier.K[0][0] < normal.K[0][0])
    }

    @Test
    fun nonFiniteInnovations_areIgnored() {
        val s = NeuralResidualNoiseStrategy()
        s.feed(doubleArrayOf(Double.NaN, Double.POSITIVE_INFINITY))
        val res = s.computeGain(P_pred, H, obs())
        assertFalse(res.K[0][0].isNaN() || res.P_updated[0][0].isNaN() || res.R[0][0].isNaN())
        assertEquals(NeuralResidualNoiseStrategy().computeGain(P_pred, H, obs()).R[0][0], res.R[0][0], 1e-12)
    }

    @Test
    fun predict_matchesAPlainForwardPass() {
        val rng = Random(3)
        fun g() = rng.nextGaussian() * 0.5
        val m = PretrainedResidualModel(
            mean = DoubleArray(5) { g() }, std = DoubleArray(5) { 1.0 + rng.nextDouble() },
            w1 = Array(16) { DoubleArray(5) { g() } }, b1 = DoubleArray(16) { g() },
            w2 = Array(8) { DoubleArray(16) { g() } }, b2 = DoubleArray(8) { g() },
            w3 = DoubleArray(8) { g() }, b3 = g()
        )
        repeat(100) {
            val x = DoubleArray(5) { rng.nextGaussian() * 2 }
            val xn = DoubleArray(5) { (x[it] - m.mean[it]) / m.std[it] }
            val h1 = DoubleArray(16) { i -> max(0.0, m.b1[i] + (0 until 5).sumOf { j -> m.w1[i][j] * xn[j] }) }
            val h2 = DoubleArray(8) { i -> max(0.0, m.b2[i] + (0 until 16).sumOf { j -> m.w2[i][j] * h1[j] }) }
            assertEquals(m.b3 + (0 until 8).sumOf { m.w3[it] * h2[it] }, m.predict(x), 1e-12)
        }
    }

    @Test
    fun defaultModel_dependsOnItsInputs() {
        val m = PretrainedResidualModel.DEFAULT
        val quiet = m.predict(doubleArrayOf(kotlin.math.ln(4.0), kotlin.math.ln(3.0), kotlin.math.ln(2.0), 1.4, kotlin.math.ln(3.0)))
        val noisy = m.predict(doubleArrayOf(kotlin.math.ln(4.0), kotlin.math.ln(16.0), kotlin.math.ln(9.0), 1.4, kotlin.math.ln(3.0)))
        assertNotEquals(quiet, noisy, 1e-3)
    }
}
