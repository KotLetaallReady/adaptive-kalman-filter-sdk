package com.katka.engine.smoothing

import com.katka.engine.neural.NetworkConfig
import com.katka.engine.neural.NeuralNetwork
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class SmootherPipelineTest {

    private fun input(i: Int, kfx: Double, kfy: Double, rawx: Double, rawy: Double) =
        SmootherInput(
            timestamp = i.toLong(),
            kfX = kfx, kfY = kfy,
            rawX = rawx, rawY = rawy,
            vx = 1.0, vy = 0.0,
            speed = 1.0, accuracy = 5.0,
            innovationMag = 0.5, sigmaPos = 2.0
        )

    private fun moving(i: Int, kfx: Double, kfy: Double, vx: Double, vy: Double) =
        SmootherInput(
            timestamp = i.toLong(),
            kfX = kfx, kfY = kfy,
            rawX = kfx, rawY = kfy,
            vx = vx, vy = vy,
            speed = kotlin.math.hypot(vx, vy), accuracy = 5.0,
            innovationMag = 0.5, sigmaPos = 2.0
        )

    private fun fill(config: SmootherConfig, step: (Int) -> SmootherInput): SmootherWindow =
        SmootherWindow(config).apply { for (i in 0 until config.windowLength) push(step(i)) }

    @Test
    fun walkingJitter_isATurnForPointDiff_butNotForVelocity() {
        // Walking straight east at 1.4 m/s; the filtered points wobble ±0.8 m sideways.
        val step = { i: Int -> moving(i, 1.4 * i, if (i % 2 == 0) 0.8 else -0.8, vx = 1.4, vy = 0.0) }

        val legacy = fill(SmootherConfig(turnDetection = TurnDetection.POINT_DIFF), step)
        val fixed = fill(SmootherConfig(), step)

        assertTrue("point differences see phantom turns", legacy.turnSuppressionPhiDeg() > 60.0)
        assertEquals(0.0, fixed.turnSuppressionPhiDeg(), 1e-9)
        assertEquals(0.0, fixed.rawFeatures()[2], 1e-9)
    }

    @Test
    fun velocityHeading_detectsARealTurn() {
        // Heading swings from east to north in 45° steps mid-window.
        val config = SmootherConfig()
        val w = fill(config) { i ->
            val deg = ((i - 4).coerceIn(0, 2)) * 45.0
            val rad = Math.toRadians(deg)
            moving(i, i.toDouble(), 0.0, vx = 2.0 * kotlin.math.cos(rad), vy = 2.0 * kotlin.math.sin(rad))
        }
        assertEquals(90.0, w.turnSuppressionPhiDeg(), 1e-6)
    }

    @Test
    fun velocityHeading_ignoresSamplesBelowMinTurnSpeed() {
        // Standing still: the velocity estimate is noise pointing anywhere.
        val config = SmootherConfig()
        val w = fill(config) { i ->
            val rad = Math.toRadians(i * 137.0)
            moving(i, 0.0, 0.0, vx = 0.2 * kotlin.math.cos(rad), vy = 0.2 * kotlin.math.sin(rad))
        }
        assertEquals(0.0, w.turnSuppressionPhiDeg(), 1e-9)
        assertEquals(0.0, w.rawFeatures()[2], 1e-9)
    }

    @Test
    fun window_fillsAtConfiguredLength_andCentreIsMiddle() {
        val config = SmootherConfig()
        val w = SmootherWindow(config)
        for (i in 0 until config.windowLength - 1) {
            w.push(input(i, i.toDouble(), 0.0, i.toDouble(), 0.0))
            assertTrue("window must not be full before $i+1 pushes", !w.isFull)
        }
        w.push(input(config.windowLength - 1, (config.windowLength - 1).toDouble(), 0.0, 0.0, 0.0))
        assertTrue(w.isFull)
        assertEquals(config.halfWindow.toDouble(), w.centralInput().kfX, 1e-9)
    }

    @Test
    fun window_usesCustomLength() {
        val config = SmootherConfig(windowLength = 7)
        val w = SmootherWindow(config)
        for (i in 0 until config.windowLength) {
            w.push(input(i, i.toDouble(), 0.0, i.toDouble(), 0.0))
        }
        assertTrue(w.isFull)
        assertEquals(config.halfWindow.toDouble(), w.centralInput().kfX, 1e-9)
    }

    @Test
    fun features_haveSixComponents_andStraightLineHasNoTurn() {
        val config = SmootherConfig()
        val w = SmootherWindow(config)
        for (i in 0 until config.windowLength) {
            w.push(input(i, i.toDouble(), 0.0, i.toDouble(), 0.0))
        }
        val f = w.rawFeatures()
        assertEquals(SmootherConfig.FEATURE_COUNT, f.size)
        assertEquals(0.0, f[2], 1e-6)
        assertEquals(0.0, w.turnSuppressionPhiDeg(), 1e-6)
    }

    @Test
    fun collector_producesExpectedSampleCount_andLabelsInRange() {
        val config = SmootherConfig()
        val collector = SmoothingTrainingCollector(config)
        val n = 30
        for (i in 0 until n) {
            val kfy = (i - 15) * (i - 15) * 0.05
            collector.addStep(input(i, i.toDouble(), kfy, i.toDouble(), kfy + 0.3))
        }
        assertEquals(n - config.windowLength + 1, collector.sampleCount)

        val ds = collector.buildDataset()
        assertEquals(SmootherConfig.FEATURE_COUNT, ds.normalizer.mean.size)
        assertEquals(collector.sampleCount, ds.samples.size)
        ds.samples.forEach {
            val a = it.labels[0]
            assertTrue("alpha* $a out of [0,1]", a in 0.0..1.0)
        }
    }

    @Test
    fun smoother_emitsOnFullWindow_andOutputIsOnSegment() {
        val config = SmootherConfig()
        val net = NeuralNetwork(NetworkConfig.default())
        val smoother = NeuralTrajectorySmoother(net, FeatureNormalizer.identity(), config)

        var last: SmoothedSample? = null
        for (i in 0 until config.windowLength) {
            val kfy = (i - config.halfWindow) * (i - config.halfWindow) * 0.2
            val s = smoother.push(input(i, i.toDouble(), kfy, i.toDouble(), kfy + 0.1))
            if (i < config.windowLength - 1) assertNull("no sample before window is full at step $i", s)
            else last = s
        }

        val sample = last
        assertNotNull("a sample must be emitted once the window is full", sample)
        sample!!

        assertTrue("alpha ${sample.alpha} in [0,1]", sample.alpha in 0.0..1.0)

        val loX = minOf(sample.kfX, sample.sgX) - 1e-9
        val hiX = maxOf(sample.kfX, sample.sgX) + 1e-9
        val loY = minOf(sample.kfY, sample.sgY) - 1e-9
        val hiY = maxOf(sample.kfY, sample.sgY) + 1e-9
        assertTrue(sample.outX in loX..hiX)
        assertTrue(sample.outY in loY..hiY)
    }
}
