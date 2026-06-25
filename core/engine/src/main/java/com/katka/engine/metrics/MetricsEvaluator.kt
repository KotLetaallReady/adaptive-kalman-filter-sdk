package com.katka.engine.metrics

import com.katka.model.AccuracyMetrics
import kotlin.math.sqrt

/**
 * Utility for comparing estimated tracks against a reference track.
 *
 * Points are expected in the same local metre coordinate system. The object is
 * stateless, so it can be used from tests, offline analysis tools and apps.
 */
object MetricsEvaluator {

    /**
     * Scores [estimated] against [reference] over their common prefix.
     *
     * Returns [AccuracyMetrics.EMPTY] when fewer than two paired points are
     * available.
     */
    fun compute(
        estimated: List<Pair<Double, Double>>,
        reference: List<Pair<Double, Double>>
    ): AccuracyMetrics {
        val n = minOf(estimated.size, reference.size)
        if (n < 2) return AccuracyMetrics.EMPTY

        val est = estimated.subList(0, n)
        val ref = reference.subList(0, n)

        val errors = est.zip(ref).map { (e, r) ->
            val dx = e.first - r.first
            val dy = e.second - r.second
            sqrt(dx * dx + dy * dy)
        }
        val rmse = sqrt(errors.sumOf { it * it } / n)
        val mae = errors.sumOf { it } / n
        val maxError = errors.max()

        val deltas = est.zipWithNext { a, b ->
            val dx = b.first - a.first
            val dy = b.second - a.second
            sqrt(dx * dx + dy * dy)
        }
        val meanStep = deltas.sum() / deltas.size
        val stability = sqrt(deltas.sumOf { (it - meanStep) * (it - meanStep) } / deltas.size)

        val lag = estimateLag(est.map { it.first }, ref.map { it.first })

        return AccuracyMetrics(rmse, mae, maxError, lag, stability, n)
    }

    /** Scores two estimated tracks against the same reference track. */
    fun compareTracks(
        trackA: List<Pair<Double, Double>>,
        trackB: List<Pair<Double, Double>>,
        reference: List<Pair<Double, Double>>
    ): Pair<AccuracyMetrics, AccuracyMetrics> =
        compute(trackA, reference) to compute(trackB, reference)

    /** Returns the lag (samples) maximising the cross-correlation of [signal] and [reference]. */
    private fun estimateLag(
        signal: List<Double>,
        reference: List<Double>,
        maxLag: Int = minOf(signal.size / 4, 20)
    ): Double {
        if (signal.size < 4 || maxLag < 1) return 0.0

        val n = signal.size
        val sDemean = signal.map { it - signal.average() }
        val rDemean = reference.map { it - reference.average() }

        var bestLag = 0
        var bestCorr = Double.NEGATIVE_INFINITY
        for (lag in -maxLag..maxLag) {
            var corr = 0.0
            var count = 0
            for (i in 0 until n) {
                val j = i + lag
                if (j < 0 || j >= n) continue
                corr += sDemean[i] * rDemean[j]
                count++
            }
            if (count > 0 && corr / count > bestCorr) {
                bestCorr = corr / count
                bestLag = lag
            }
        }
        return bestLag.toDouble()
    }
}
