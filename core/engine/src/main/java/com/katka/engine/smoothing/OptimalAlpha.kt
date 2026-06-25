package com.katka.engine.smoothing

import kotlin.math.sqrt

/** Internal label builder for neural-smoother training. */
internal object OptimalAlpha {

    private const val DEGENERATE_EPS = 1e-9

    /**
     * Returns alpha* = clip(d*e/|d|^2, 0, 1) with configurable turn-angle
     * suppression; returns 0.5 when the correction direction is undefined.
     */
    fun solve(
        xKf: DoubleArray,
        xSg: DoubleArray,
        xStar: DoubleArray,
        turnAngleDeg: Double,
        config: SmootherConfig = SmootherConfig()
    ): Double {
        val dx = xSg[0] - xKf[0]
        val dy = xSg[1] - xKf[1]
        val ex = xStar[0] - xKf[0]
        val ey = xStar[1] - xKf[1]

        val dNorm2 = dx * dx + dy * dy
        var alpha = if (dNorm2 < DEGENERATE_EPS) {
            0.5
        } else {
            ((dx * ex + dy * ey) / dNorm2).coerceIn(0.0, 1.0)
        }

        val suppression = (1.0 - turnAngleDeg / config.alphaSuppressionTurnDeg)
            .coerceAtLeast(config.minTurnSuppression)
        alpha *= suppression
        return alpha.coerceIn(0.0, 1.0)
    }

    /** Euclidean norm of a 2-vector. */
    fun norm(v: DoubleArray): Double = sqrt(v[0] * v[0] + v[1] * v[1])
}
