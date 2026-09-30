package com.katka.engine.smoothing.rts

import com.katka.engine.KalmanFilter
import com.katka.engine.MatrixOps
import com.katka.engine.model.FilterResult
import com.katka.engine.model.KalmanState
import com.katka.model.Observation
import kotlin.math.atan2
import kotlin.math.hypot

/**
 * Robust batch trajectory smoother for recorded tracks.
 *
 * For a linear Gaussian state-space model, factor-graph M-estimation is solved exactly
 * by running RTS inside Iteratively Reweighted Least Squares (IRLS).
 *
 * On each iteration, the forward Kalman filter is re-run with measurement variances
 * inflated inversely by the Cauchy weight:
 *
 * `R_i = R_raw / w_i`, where `w_i = 1 / (1 + (u_i / c)^2)` and `u_i = ||z_i - x_i^s|| / sigma_i`.
 *
 * This provides extreme robustness against multipath spikes, GPS jumps, and false accuracy reports
 * without requiring non-linear solvers (GTSAM/Ceres).
 *
 * @param processNoiseStd Process noise std Q (typically 0.2 for smoothed tracks).
 * @param iterations Number of IRLS reweighting passes (4 is optimal).
 * @param cauchyScale Cauchy scale factor c (2.45 corresponds to sqrt(chi2_2 at 95%)).
 * @param useDoppler Whether to fuse Doppler velocity measurements in the forward pass.
 */
class OfflineTrackSmoother(
    val processNoiseStd: Double = 0.2,
    val iterations: Int = 4,
    val cauchyScale: Double = 2.45,
    val useDoppler: Boolean = true
) {

    /**
     * Smooths a full sequence of recorded GNSS observations.
     *
     * @param observations Time-ordered list of GPS fixes.
     * @return List of smoothed points matching the input size.
     */
    fun smooth(observations: List<Observation>): List<SmoothedResult> {
        if (observations.isEmpty()) return emptyList()
        if (observations.size == 1) {
            val obs = observations[0]
            val initP = MatrixOps.diagonal(doubleArrayOf(obs.accuracy.toDouble() * obs.accuracy, obs.accuracy.toDouble() * obs.accuracy, 10.0, 10.0))
            val initState = KalmanState(0.0, 0.0, obs.speed.toDouble(), 0.0, initP)
            return listOf(
                SmoothedResult(
                    timestamp = obs.timestamp,
                    latitude = obs.latitude,
                    longitude = obs.longitude,
                    altitude = obs.altitude,
                    speed = obs.speed.toDouble(),
                    bearing = obs.bearing,
                    accuracy = obs.accuracy,
                    state = initState,
                    rawObservation = obs
                )
            )
        }

        val weights = HashMap<Long, Double>()
        var lastXs: Array<DoubleArray>? = null
        var lastPs: Array<Array<DoubleArray>>? = null
        var lastFilter: KalmanFilter? = null

        val totalPasses = (iterations + 1).coerceAtLeast(1)
        for (pass in 0 until totalPasses) {
            val filter = KalmanFilter(
                processNoiseStd = processNoiseStd,
                useWallClockTime = false,
                useDoppler = useDoppler
            )
            val strategy = WeightedCoefficientStrategy(weights)
            filter.reset(strategy)

            val results = ArrayList<FilterResult>(observations.size)
            for (obs in observations) {
                results.add(filter.process(obs, strategy))
            }

            val (xs, ps) = rtsFull(results)
            lastXs = xs
            lastPs = ps
            lastFilter = filter

            // Update Cauchy weights for next iteration
            if (pass < totalPasses - 1) {
                for (i in observations.indices) {
                    val obs = observations[i]
                    val (zx, zy) = filter.geoToLocal(obs.latitude, obs.longitude)
                    val sigma = obs.accuracy.toDouble().coerceIn(1.0, 50.0)
                    val u = hypot(zx - xs[i][0], zy - xs[i][1]) / sigma
                    val w = (1.0 / (1.0 + (u / cauchyScale) * (u / cauchyScale))).coerceAtLeast(0.01)
                    weights[obs.timestamp] = w
                }
            }
        }

        val xs = lastXs!!
        val ps = lastPs!!
        val filter = lastFilter!!

        val out = ArrayList<SmoothedResult>(observations.size)
        for (i in observations.indices) {
            val obs = observations[i]
            val (lat, lon) = filter.localToGeo(xs[i][0], xs[i][1])
            val vx = xs[i][2]
            val vy = xs[i][3]
            val speed = hypot(vx, vy)
            var bearingDeg = Math.toDegrees(atan2(vx, vy))
            if (bearingDeg < 0.0) bearingDeg += 360.0

            val state = KalmanState(
                x = xs[i][0],
                y = xs[i][1],
                vx = vx,
                vy = vy,
                P = ps[i]
            )

            out.add(
                SmoothedResult(
                    timestamp = obs.timestamp,
                    latitude = lat,
                    longitude = lon,
                    altitude = obs.altitude,
                    speed = speed,
                    bearing = bearingDeg.toFloat(),
                    accuracy = state.positionUncertaintyMeters.toFloat(),
                    state = state,
                    rawObservation = obs
                )
            )
        }

        return out
    }

    companion object {
        /**
         * Backward RTS recursion across all forward filter results.
         *
         * @return Pair of smoothed state vectors and smoothed covariance matrices.
         */
        fun rtsFull(results: List<FilterResult>): Pair<Array<DoubleArray>, Array<Array<DoubleArray>>> {
            val n = results.size
            val xs = Array(n) { DoubleArray(KalmanState.DIM) }
            val ps = Array(n) { MatrixOps.zeros(KalmanState.DIM, KalmanState.DIM) }
            if (n == 0) return xs to ps

            xs[n - 1] = results[n - 1].state.toVector()
            ps[n - 1] = results[n - 1].state.P

            for (i in n - 2 downTo 0) {
                val nx = results[i + 1]
                val dt = nx.dt
                val F = arrayOf(
                    doubleArrayOf(1.0, 0.0, dt, 0.0),
                    doubleArrayOf(0.0, 1.0, 0.0, dt),
                    doubleArrayOf(0.0, 0.0, 1.0, 0.0),
                    doubleArrayOf(0.0, 0.0, 0.0, 1.0)
                )

                val Pf = results[i].state.P
                val Pp = nx.predicted.P
                val C = MatrixOps.mul(MatrixOps.mul(Pf, MatrixOps.transpose(F)), MatrixOps.inverse(Pp))
                val xp = nx.predicted.toVector()
                val xf = results[i].state.toVector()
                val cd = MatrixOps.mulVec(C, DoubleArray(KalmanState.DIM) { xs[i + 1][it] - xp[it] })
                xs[i] = DoubleArray(KalmanState.DIM) { xf[it] + cd[it] }
                ps[i] = MatrixOps.symmetrise(
                    MatrixOps.add(
                        Pf,
                        MatrixOps.mul(MatrixOps.mul(C, MatrixOps.sub(ps[i + 1], Pp)), MatrixOps.transpose(C))
                    )
                )
            }
            return xs to ps
        }
    }
}
