package com.katka.engine.smoothing

import kotlin.math.abs

/** Internal Savitzky-Golay smoother used by the neural trajectory smoother. */
internal object SavitzkyGolaySmoother {

    /** Returns the quadratic-fit value at the centre of a symmetric 1-D window. */
    fun smoothCentre(values: DoubleArray): Double {
        val n = values.size
        require(n > 0) { "smoothCentre: empty window" }
        val half = (n - 1) / 2

        var sumX = 0.0
        var sumT2X = 0.0
        var s2 = 0.0
        var s4 = 0.0
        for (i in 0 until n) {
            val t = (i - half).toDouble()
            val t2 = t * t
            sumX += values[i]
            sumT2X += t2 * values[i]
            s2 += t2
            s4 += t2 * t2
        }

        val nN = n.toDouble()
        val denom = nN * s4 - s2 * s2
        if (abs(denom) < 1e-12) return sumX / nN
        return (s4 * sumX - s2 * sumT2X) / denom
    }

    /** Applies [smoothCentre] independently to the x and y axes of a point window. */
    fun smoothCentre2D(points: List<DoubleArray>): DoubleArray {
        val xs = DoubleArray(points.size) { points[it][0] }
        val ys = DoubleArray(points.size) { points[it][1] }
        return doubleArrayOf(smoothCentre(xs), smoothCentre(ys))
    }
}
