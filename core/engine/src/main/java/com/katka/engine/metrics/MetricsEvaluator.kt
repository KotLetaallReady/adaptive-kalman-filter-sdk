package com.katka.engine.metrics

import com.katka.model.AccuracyMetrics
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * Position of a track point at a moment in time.
 *
 * @property timestamp Timestamp in milliseconds.
 * @property x Local X position in metres.
 * @property y Local Y position in metres.
 */
data class TimedPosition(
    val timestamp: Long,
    val x: Double,
    val y: Double
)

/** Compares estimated tracks with a reference track in the same local metre frame. */
object MetricsEvaluator {

    /**
     * Scores [estimated] against [reference], pairing each point with the nearest reference point
     * within [toleranceMs]; unpaired points are skipped, so a delayed smoother is scored correctly.
     * Returns [AccuracyMetrics.EMPTY] when fewer than two points are paired.
     */
    fun compute(
        estimated: List<TimedPosition>,
        reference: List<TimedPosition>,
        toleranceMs: Long = 0L
    ): AccuracyMetrics {
        require(toleranceMs >= 0L) { "toleranceMs must be non-negative." }
        val ref = reference.sortedBy { it.timestamp }
        val refTimes = LongArray(ref.size) { ref[it].timestamp }

        val pairedEst = ArrayList<Pair<Double, Double>>()
        val pairedRef = ArrayList<Pair<Double, Double>>()
        for (e in estimated.sortedBy { it.timestamp }) {
            val r = nearest(ref, refTimes, e.timestamp) ?: continue
            if (abs(r.timestamp - e.timestamp) > toleranceMs) continue
            pairedEst.add(e.x to e.y)
            pairedRef.add(r.x to r.y)
        }
        return score(pairedEst, pairedRef)
    }

    /** Scores two estimated tracks against the same reference track, pairing points by timestamp. */
    fun compareTracks(
        trackA: List<TimedPosition>,
        trackB: List<TimedPosition>,
        reference: List<TimedPosition>,
        toleranceMs: Long = 0L
    ): Pair<AccuracyMetrics, AccuracyMetrics> =
        compute(trackA, reference, toleranceMs) to compute(trackB, reference, toleranceMs)

    @Deprecated(
        "Pairs points by index, so a delayed track is compared with the wrong reference points. " +
            "Use the TimedPosition overload."
    )
    fun compute(
        estimated: List<Pair<Double, Double>>,
        reference: List<Pair<Double, Double>>
    ): AccuracyMetrics {
        val n = minOf(estimated.size, reference.size)
        return score(estimated.subList(0, n), reference.subList(0, n))
    }

    @Deprecated(
        "Pairs points by index, so a delayed track is compared with the wrong reference points. " +
            "Use the TimedPosition overload."
    )
    fun compareTracks(
        trackA: List<Pair<Double, Double>>,
        trackB: List<Pair<Double, Double>>,
        reference: List<Pair<Double, Double>>
    ): Pair<AccuracyMetrics, AccuracyMetrics> {
        val nA = minOf(trackA.size, reference.size)
        val nB = minOf(trackB.size, reference.size)
        return score(trackA.subList(0, nA), reference.subList(0, nA)) to
            score(trackB.subList(0, nB), reference.subList(0, nB))
    }

    private fun score(
        est: List<Pair<Double, Double>>,
        ref: List<Pair<Double, Double>>
    ): AccuracyMetrics {
        val n = est.size
        if (n < 2) return AccuracyMetrics.EMPTY

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

    private fun nearest(ref: List<TimedPosition>, times: LongArray, timestamp: Long): TimedPosition? {
        if (ref.isEmpty()) return null
        val i = times.binarySearch(timestamp)
        if (i >= 0) return ref[i]
        val after = -i - 1
        val before = after - 1
        return when {
            before < 0 -> ref[after]
            after >= ref.size -> ref[before]
            timestamp - times[before] <= times[after] - timestamp -> ref[before]
            else -> ref[after]
        }
    }

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
