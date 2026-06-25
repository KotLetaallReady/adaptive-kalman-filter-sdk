package com.katka.engine.smoothing

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.hypot
import kotlin.math.sqrt

/** Internal feature extractor used by training and inference for the neural smoother. */
internal object SmootherFeatures {

    const val COUNT = SmootherConfig.FEATURE_COUNT

    /** Extracts the raw, un-normalised feature vector from a full window. */
    fun extract(
        window: List<SmootherInput>,
        config: SmootherConfig = SmootherConfig()
    ): DoubleArray {
        require(window.size == config.windowLength) {
            "SmootherFeatures.extract expects a full ${config.windowLength}-point window."
        }

        val innov = window.map { it.innovationMag }
        val f1 = innov.average()
        val f2 = std(innov, f1)
        val f3 = turnAnglesDeg(window).sum()
        val f4 = window.map { it.speed }.average()
        val f5 = window.map { it.accuracy }.average()
        val steps = stepLengths(window)
        val f6 = if (steps.isEmpty()) 0.0 else std(steps, steps.average())
        return doubleArrayOf(f1, f2, f3, f4, f5, f6)
    }

    /** Total turn angle in degrees over segments steeper than the configured threshold. */
    fun suppressionAngleDeg(
        window: List<SmootherInput>,
        config: SmootherConfig = SmootherConfig()
    ): Double =
        turnAnglesDeg(window).filter { it > config.turnThresholdDeg }.sum()

    /** Absolute turn angles in degrees between consecutive Kalman-track segments. */
    private fun turnAnglesDeg(window: List<SmootherInput>): List<Double> {
        if (window.size < 3) return emptyList()

        val bearings = ArrayList<Double>(window.size - 1)
        for (i in 0 until window.size - 1) {
            val dx = window[i + 1].kfX - window[i].kfX
            val dy = window[i + 1].kfY - window[i].kfY
            bearings.add(atan2(dy, dx))
        }

        val turns = ArrayList<Double>(bearings.size - 1)
        for (i in 1 until bearings.size) {
            var d = bearings[i] - bearings[i - 1]
            while (d > PI) d -= 2 * PI
            while (d < -PI) d += 2 * PI
            turns.add(abs(Math.toDegrees(d)))
        }
        return turns
    }

    /** Lengths of consecutive Kalman-track steps. */
    private fun stepLengths(window: List<SmootherInput>): List<Double> {
        if (window.size < 2) return emptyList()
        val steps = ArrayList<Double>(window.size - 1)
        for (i in 0 until window.size - 1) {
            val dx = window[i + 1].kfX - window[i].kfX
            val dy = window[i + 1].kfY - window[i].kfY
            steps.add(hypot(dx, dy))
        }
        return steps
    }

    /** Population standard deviation. */
    private fun std(xs: List<Double>, mean: Double): Double {
        if (xs.size < 2) return 0.0
        val v = xs.sumOf { (it - mean) * (it - mean) } / xs.size
        return sqrt(v)
    }
}
