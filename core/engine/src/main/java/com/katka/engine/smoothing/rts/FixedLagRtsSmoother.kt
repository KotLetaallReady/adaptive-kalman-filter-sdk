package com.katka.engine.smoothing.rts

import com.katka.engine.KalmanFilter
import com.katka.engine.MatrixOps
import com.katka.engine.model.FilterResult
import com.katka.engine.model.KalmanState
import com.katka.model.Observation
import kotlin.math.atan2
import kotlin.math.hypot

/**
 * Online fixed-lag Rauch–Tung–Striebel (RTS) trajectory smoother.
 *
 * Buffers incoming [FilterResult] steps in a sliding window of size `[lag] + 1`
 * and applies RTS backward recursion to refine the state at time `k - lag` using
 * the future observations up to time `k`.
 *
 * This delivers the optimal linear smoother accuracy for live streams at the cost of
 * a fixed time delay (typically 5 seconds / 5 samples).
 *
 * @param lag Number of future samples used to smooth each point. When `0`, acts as a pass-through.
 */
class FixedLagRtsSmoother(val lag: Int = 5) {

    init {
        require(lag >= 0) { "lag must be non-negative, got $lag" }
    }

    private val window = ArrayDeque<Pair<FilterResult, Observation?>>()

    /** Number of samples currently held in the fixed-lag window. */
    val bufferedCount: Int get() = window.size

    /** Clears the internal window buffer. */
    fun reset() {
        window.clear()
    }

    /**
     * Pushes a new filter result and returns a smoothed point once the lag window is filled.
     *
     * @param result Forward Kalman filter output for the current step.
     * @param localToGeo Function converting local metres (x, y) back to WGS-84 (lat, lon).
     * @param obs Optional original raw observation.
     * @return Smoothed sample delayed by [lag] steps, or `null` while buffering.
     */
    fun push(
        result: FilterResult,
        localToGeo: (Double, Double) -> Pair<Double, Double>,
        obs: Observation? = null
    ): SmoothedResult? {
        window.addLast(result to obs)

        if (window.size <= lag) {
            return null
        }

        return smoothOldestInWindow(localToGeo)
    }

    /**
     * Overload delegating coordinate conversion to [KalmanFilter.localToGeo].
     */
    fun push(
        result: FilterResult,
        filter: KalmanFilter,
        obs: Observation? = null
    ): SmoothedResult? = push(result, filter::localToGeo, obs)

    /**
     * Flushes all remaining buffered samples by running backward smoothing over the remaining window.
     * Call this when a tracking session finishes to retrieve the final [lag] points.
     */
    fun flush(localToGeo: (Double, Double) -> Pair<Double, Double>): List<SmoothedResult> {
        val out = ArrayList<SmoothedResult>(window.size)
        while (window.isNotEmpty()) {
            out.add(smoothOldestInWindow(localToGeo))
        }
        return out
    }

    /**
     * Overload delegating coordinate conversion to [KalmanFilter.localToGeo].
     */
    fun flush(filter: KalmanFilter): List<SmoothedResult> = flush(filter::localToGeo)

    private fun smoothOldestInWindow(localToGeo: (Double, Double) -> Pair<Double, Double>): SmoothedResult {
        val n = window.size
        var xs = window[n - 1].first.state.toVector()
        var Ps = window[n - 1].first.state.P

        for (i in n - 2 downTo 0) {
            val cur = window[i].first
            val nx = window[i + 1].first
            val dt = nx.dt

            val F = arrayOf(
                doubleArrayOf(1.0, 0.0, dt, 0.0),
                doubleArrayOf(0.0, 1.0, 0.0, dt),
                doubleArrayOf(0.0, 0.0, 1.0, 0.0),
                doubleArrayOf(0.0, 0.0, 0.0, 1.0)
            )

            val Pf = cur.state.P
            val Pp = nx.predicted.P
            val C = MatrixOps.mul(MatrixOps.mul(Pf, MatrixOps.transpose(F)), MatrixOps.inverse(Pp))
            val xp = nx.predicted.toVector()
            val xf = cur.state.toVector()
            val cd = MatrixOps.mulVec(C, DoubleArray(KalmanState.DIM) { xs[it] - xp[it] })
            xs = DoubleArray(KalmanState.DIM) { xf[it] + cd[it] }
            Ps = MatrixOps.symmetrise(
                MatrixOps.add(
                    Pf,
                    MatrixOps.mul(MatrixOps.mul(C, MatrixOps.sub(Ps, Pp)), MatrixOps.transpose(C))
                )
            )
        }

        val oldest = window.removeFirst()
        val oldestRes = oldest.first
        val oldestObs = oldest.second

        val (lat, lon) = localToGeo(xs[0], xs[1])
        val speed = hypot(xs[2], xs[3])
        var bearingDeg = Math.toDegrees(atan2(xs[2], xs[3]))
        if (bearingDeg < 0.0) bearingDeg += 360.0

        val smoothedState = KalmanState(
            x = xs[0],
            y = xs[1],
            vx = xs[2],
            vy = xs[3],
            P = Ps
        )

        return SmoothedResult(
            timestamp = oldestRes.timestamp,
            latitude = lat,
            longitude = lon,
            altitude = oldestObs?.altitude ?: 0.0,
            speed = speed,
            bearing = bearingDeg.toFloat(),
            accuracy = smoothedState.positionUncertaintyMeters.toFloat(),
            state = smoothedState,
            rawObservation = oldestObs
        )
    }
}
