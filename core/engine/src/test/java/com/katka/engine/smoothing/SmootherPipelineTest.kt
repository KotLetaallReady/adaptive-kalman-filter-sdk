package com.katka.engine.smoothing

import com.katka.engine.neural.NeuralNetwork
import com.katka.engine.neural.NetworkConfig
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

    @Test
    fun window_fillsAtL_andCentreIsMiddle() {
        val w = SmootherWindow()
        for (i in 0 until SmootherFeatures.L - 1) {
            w.push(input(i, i.toDouble(), 0.0, i.toDouble(), 0.0))
            assertTrue("window must not be full before $i+1 pushes", !w.isFull)
        }
        w.push(input(SmootherFeatures.L - 1, (SmootherFeatures.L - 1).toDouble(), 0.0, 0.0, 0.0))
        assertTrue(w.isFull)
        // central index = HALF → kfX should equal HALF
        assertEquals(SmootherFeatures.HALF.toDouble(), w.centralInput().kfX, 1e-9)
    }

    @Test
    fun features_haveSixComponents_andStraightLineHasNoTurn() {
        val w = SmootherWindow()
        for (i in 0 until SmootherFeatures.L) {
            w.push(input(i, i.toDouble(), 0.0, i.toDouble(), 0.0))
        }
        val f = w.rawFeatures()
        assertEquals(SmootherFeatures.COUNT, f.size)
        // f3 = total turn angle of a straight line ≈ 0
        assertEquals(0.0, f[2], 1e-6)
        // φ for suppression also ≈ 0 on a straight line
        assertEquals(0.0, w.turnSuppressionPhiDeg(), 1e-6)
    }

    @Test
    fun collector_producesExpectedSampleCount_andLabelsInRange() {
        val collector = SmoothingTrainingCollector()
        val n = 30
        for (i in 0 until n) {
            // parabola so x_KF and x_SG differ, raw = KF + tiny offset
            val kfy = (i - 15) * (i - 15) * 0.05
            collector.addStep(input(i, i.toDouble(), kfy, i.toDouble(), kfy + 0.3))
        }
        // samples appear once the window is full: n - L + 1
        assertEquals(n - SmootherFeatures.L + 1, collector.sampleCount)

        val ds = collector.buildDataset()
        assertEquals(SmootherFeatures.COUNT, ds.normalizer.mean.size)
        assertEquals(collector.sampleCount, ds.samples.size)
        ds.samples.forEach {
            val a = it.labels[0]
            assertTrue("alpha* $a out of [0,1]", a in 0.0..1.0)
        }
    }

    @Test
    fun smoother_emitsOnFullWindow_andOutputIsOnSegment() {
        val net = NeuralNetwork(NetworkConfig.default())
        val smoother = NeuralTrajectorySmoother(net, FeatureNormalizer.identity())

        var last: SmoothedSample? = null
        for (i in 0 until SmootherFeatures.L) {
            val kfy = (i - 5) * (i - 5) * 0.2   // curved → SG differs from KF at centre
            val s = smoother.push(input(i, i.toDouble(), kfy, i.toDouble(), kfy + 0.1))
            if (i < SmootherFeatures.L - 1) assertNull("no sample before window is full at step $i", s)
            else last = s
        }

        val sample = last
        assertNotNull("a sample must be emitted once the window is full", sample)
        sample!!

        assertTrue("alpha ${sample.alpha} in [0,1]", sample.alpha in 0.0..1.0)

        // x_out = (1-α)·x_KF + α·x_SG must lie on the segment [x_KF, x_SG]
        val loX = minOf(sample.kfX, sample.sgX) - 1e-9
        val hiX = maxOf(sample.kfX, sample.sgX) + 1e-9
        val loY = minOf(sample.kfY, sample.sgY) - 1e-9
        val hiY = maxOf(sample.kfY, sample.sgY) + 1e-9
        assertTrue(sample.outX in loX..hiX)
        assertTrue(sample.outY in loY..hiY)
    }
}
