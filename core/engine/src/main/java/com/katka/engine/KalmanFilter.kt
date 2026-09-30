package com.katka.engine

import com.katka.engine.coefficient_startegy.CoefficientStrategy
import com.katka.engine.model.FilterMode
import com.katka.engine.model.FilterResult
import com.katka.engine.model.KalmanState
import com.katka.model.Observation
import kotlin.math.*

/**
 * Kalman filter for 2D position tracking with a constant-velocity model.
 *
 * It works in local metres around the first accepted fix. A fix outside the outlier gate is
 * skipped, but after [GATE_RECOVERY_REJECTIONS] skips in a row the next one is accepted, so the
 * filter cannot lose the track for good.
 *
 * @param processNoiseStd Process noise std; larger values follow manoeuvres faster but smooth less.
 * @param useWallClockTime `true` cross-checks the time step against the system clock (live
 * stream); `false` takes it only from [Observation.timestamp], which makes replays deterministic.
 * @param useDoppler `true` fuses GNSS Doppler speed and bearing when reported in observations.
 */
class KalmanFilter(
    val processNoiseStd: Double = 0.5,
    private val useWallClockTime: Boolean = true,
    val useDoppler: Boolean = false
) {
    private var state: KalmanState? = null
    private var lastTimestamp: Long = -1L
    private var lastWallClockMs: Long = -1L
    private var consecutiveRejections = 0

    /** Latitude of the first accepted GPS fix, used as the local coordinate origin. */
    var refLat: Double = 0.0
        private set

    /** Longitude of the first accepted GPS fix, used as the local coordinate origin. */
    var refLon: Double = 0.0
        private set

    /** Whether the filter already has an initial state and reference point. */
    val isInitialised: Boolean get() = state != null

    private val H: Array<DoubleArray> = arrayOf(
        doubleArrayOf(1.0, 0.0, 0.0, 0.0),
        doubleArrayOf(0.0, 1.0, 0.0, 0.0)
    )

    /** Clears all state and the reference point, and resets the strategy, for a new session. */
    fun reset(strategy: CoefficientStrategy? = null) {
        state = null
        lastTimestamp   = -1L
        lastWallClockMs = -1L
        consecutiveRejections = 0
        refLat = 0.0
        refLon = 0.0
        strategy?.reset()
    }

    /**
     * Processes one observation and returns the posterior state with diagnostics. The first call
     * only sets up the local frame; each accepted fix is passed to [CoefficientStrategy.onInnovation].
     */
    fun process(obs: Observation, strategy: CoefficientStrategy): FilterResult {
        if (!isInitialised) {
            return initialise(obs)
        }

        val nowMs = if (useWallClockTime) System.currentTimeMillis() else -1L

        val dtFromGps  = (obs.timestamp - lastTimestamp) / 1000.0
        val dt = if (!useWallClockTime) {
            dtFromGps.coerceIn(0.001, 30.0)
        } else {
            val dtFromWall = if (lastWallClockMs > 0)
                (nowMs - lastWallClockMs) / 1000.0
            else
                dtFromGps

            when {
                dtFromGps <= 0 ->
                    dtFromWall
                dtFromWall <= 0 ->
                    dtFromGps.coerceIn(0.001, 30.0)
                dtFromGps / dtFromWall > 3.0 ->
                    dtFromWall
                dtFromWall / dtFromGps > 3.0 ->
                    dtFromWall
                else ->
                    dtFromGps
            }.coerceIn(0.001, 30.0)
        }

        lastTimestamp   = obs.timestamp
        lastWallClockMs = nowMs

        val currentState = state!!

        val (xPred, PPred) = predict(currentState, dt, obs)

        val (gpsX, gpsY) = geoToLocal(obs.latitude, obs.longitude)
        val z = doubleArrayOf(gpsX, gpsY)

        val hxPred     = MatrixOps.mulVec(H, xPred.toVector())
        val innovation = DoubleArray(H.size) { i -> z[i] - hxPred[i] }

        val innovationMag = kotlin.math.sqrt(
            innovation[0] * innovation[0] + innovation[1] * innovation[1]
        )

        val gateThreshold = (xPred.positionUncertaintyMeters * 5.0).coerceIn(15.0, 80.0)

        if (innovationMag > gateThreshold && consecutiveRejections < GATE_RECOVERY_REJECTIONS) {
            consecutiveRejections++
            var rejectedState = xPred
            if (useDoppler && obs.hasSpeed) {
                rejectedState = updateVelocity(rejectedState, obs)
            }
            state = rejectedState
            val zeroK = MatrixOps.zeros(KalmanState.DIM, 2)
            val Rgate = MatrixOps.diagonal(doubleArrayOf(
                (obs.accuracy * obs.accuracy).toDouble().coerceAtLeast(1.0),
                (obs.accuracy * obs.accuracy).toDouble().coerceAtLeast(1.0)
            ))
            val HP   = MatrixOps.mul(H, PPred)
            val HPHt = MatrixOps.mul(HP, MatrixOps.transpose(H))
            val Sgate = MatrixOps.add(HPHt, Rgate)
            return FilterResult(
                timestamp         = obs.timestamp,
                state             = rejectedState,
                predicted         = xPred,
                innovation        = innovation,
                innovationCovS    = Sgate,
                kalmanGain        = zeroK,
                measurementNoiseR = Rgate,
                filterMode        = FilterMode.CLASSICAL,
                dt                = dt
            )
        }

        consecutiveRejections = 0

        val gainResult = strategy.computeGain(PPred, H, obs, innovation)
        val K = gainResult.K
        val R = gainResult.R

        val HP = MatrixOps.mul(H, PPred)
        val S  = MatrixOps.add(MatrixOps.mul(HP, MatrixOps.transpose(H)), R)

        val Ky       = MatrixOps.mulVec(K, innovation)
        val xPredVec = xPred.toVector()
        val xNew     = DoubleArray(KalmanState.DIM) { i -> xPredVec[i] + Ky[i] }

        val maxSpeedMs = 15.0
        xNew[2] = xNew[2].coerceIn(-maxSpeedMs, maxSpeedMs)
        xNew[3] = xNew[3].coerceIn(-maxSpeedMs, maxSpeedMs)

        var posterior = KalmanState(
            x  = xNew[0], y  = xNew[1],
            vx = xNew[2], vy = xNew[3],
            P  = gainResult.P_updated
        )
        if (useDoppler && obs.hasSpeed) {
            posterior = updateVelocity(posterior, obs)
        }
        state = posterior
        strategy.onInnovation(innovation, H, PPred)

        return FilterResult(
            timestamp         = obs.timestamp,
            state             = posterior,
            predicted         = xPred,
            innovation        = innovation,
            innovationCovS    = S,
            kalmanGain        = K,
            measurementNoiseR = R,
            filterMode        = FilterMode.CLASSICAL,
            dt                = dt
        )
    }

    /** Returns the current posterior estimate without processing a new observation. */
    fun getCurrentState(): KalmanState? = state

    /** Converts WGS-84 degrees to local metres around [refLat], [refLon]; valid after the first [process] call. */
    fun geoToLocal(lat: Double, lon: Double): Pair<Double, Double> {
        val R = EARTH_RADIUS_M
        val dLat = Math.toRadians(lat - refLat)
        val dLon = Math.toRadians(lon - refLon)
        val x = R * dLon * cos(Math.toRadians(refLat))
        val y = R * dLat
        return x to y
    }

    /** Converts local metre coordinates back to WGS-84 degrees. */
    fun localToGeo(x: Double, y: Double): Pair<Double, Double> {
        val R = EARTH_RADIUS_M
        val lat = refLat + Math.toDegrees(y / R)
        val lon = refLon + Math.toDegrees(x / (R * cos(Math.toRadians(refLat))))
        return lat to lon
    }

    private fun initialise(obs: Observation): FilterResult {
        refLat = obs.latitude
        refLon = obs.longitude
        lastTimestamp = obs.timestamp
        consecutiveRejections = 0

        val posVariance = (obs.accuracy * obs.accuracy).toDouble().coerceAtLeast(1.0)
        var initState = KalmanState.initial(x = 0.0, y = 0.0, posVariance = posVariance)
        if (useDoppler && obs.hasSpeed) {
            initState = updateVelocity(initState, obs)
        }
        state = initState

        val zeroR = MatrixOps.zeros(2, 2)
        return FilterResult(
            timestamp = obs.timestamp,
            state = initState,
            predicted = initState,
            innovation = DoubleArray(2),
            innovationCovS = zeroR,
            kalmanGain = MatrixOps.zeros(KalmanState.DIM, 2),
            measurementNoiseR = zeroR,
            filterMode = FilterMode.CLASSICAL,
            dt = 0.0
        )
    }

    private fun predict(
        s: KalmanState,
        dt: Double,
        obs: Observation
    ): Pair<KalmanState, Array<DoubleArray>> {

        val dt2 = dt * dt
        val dt3 = dt2 * dt
        val dt4 = dt3 * dt
        val halfDt2 = 0.5 * dt2

        val F = arrayOf(
            doubleArrayOf(1.0, 0.0, dt,  0.0),
            doubleArrayOf(0.0, 1.0, 0.0, dt ),
            doubleArrayOf(0.0, 0.0, 1.0, 0.0),
            doubleArrayOf(0.0, 0.0, 0.0, 1.0)
        )

        val B = arrayOf(
            doubleArrayOf(halfDt2, 0.0    ),
            doubleArrayOf(0.0,     halfDt2),
            doubleArrayOf(dt,      0.0    ),
            doubleArrayOf(0.0,     dt     )
        )

        val u = if (obs.hasImu && obs.hasRotation)
            doubleArrayOf(obs.axGeo, obs.ayGeo)
        else
            doubleArrayOf(0.0, 0.0)

        val Fx  = MatrixOps.mulVec(F, s.toVector())
        val Bu  = MatrixOps.mulVec(B, u)
        val xPredVec = DoubleArray(KalmanState.DIM) { i -> Fx[i] + Bu[i] }

        val q = processNoiseStd * processNoiseStd
        val Q = arrayOf(
            doubleArrayOf(q * dt4 / 4, 0.0,        q * dt3 / 2, 0.0       ),
            doubleArrayOf(0.0,         q * dt4 / 4, 0.0,        q * dt3 / 2),
            doubleArrayOf(q * dt3 / 2, 0.0,        q * dt2,    0.0        ),
            doubleArrayOf(0.0,         q * dt3 / 2, 0.0,        q * dt2   )
        )

        val FP    = MatrixOps.mul(F, s.P)
        val FPFt  = MatrixOps.mul(FP, MatrixOps.transpose(F))
        val PPred = MatrixOps.symmetrise(MatrixOps.add(FPFt, Q))

        val xPred = KalmanState(
            x  = xPredVec[0], y  = xPredVec[1],
            vx = xPredVec[2], vy = xPredVec[3],
            P  = PPred
        )
        return xPred to PPred
    }

    private fun updateVelocity(
        currentState: KalmanState,
        obs: Observation
    ): KalmanState {
        val s = obs.speed.toDouble()
        val sa = if (obs.speedAccuracy > 0.05f) obs.speedAccuracy.toDouble() else 0.25

        val zv: DoubleArray
        val Rv: Array<DoubleArray>

        if (obs.hasBearing && s > 0.5) {
            val psi = wrapAngle(Math.toRadians(90.0 - obs.bearing.toDouble()))
            val cosPsi = cos(psi)
            val sinPsi = sin(psi)

            val bRad = if (obs.bearingAccuracy > 0.05f) {
                Math.toRadians(obs.bearingAccuracy.toDouble())
            } else {
                (sa / s).coerceIn(0.02, PI / 2)
            }

            zv = doubleArrayOf(s * cosPsi, s * sinPsi)

            val j00 = cosPsi
            val j01 = -s * sinPsi
            val j10 = sinPsi
            val j11 = s * cosPsi

            val sVar = sa * sa
            val bVar = bRad * bRad

            val r00 = j00 * j00 * sVar + j01 * j01 * bVar + 1e-6
            val r01 = j00 * j10 * sVar + j01 * j11 * bVar
            val r11 = j10 * j10 * sVar + j11 * j11 * bVar + 1e-6

            Rv = arrayOf(
                doubleArrayOf(r00, r01),
                doubleArrayOf(r01, r11)
            )
        } else {
            val vVar = (s * s + sa * sa).coerceAtLeast(0.01)
            zv = doubleArrayOf(0.0, 0.0)
            Rv = arrayOf(
                doubleArrayOf(vVar, 0.0),
                doubleArrayOf(0.0, vVar)
            )
        }

        val x = currentState.toVector()
        val P = currentState.P

        val y0 = zv[0] - x[2]
        val y1 = zv[1] - x[3]

        val s00 = P[2][2] + Rv[0][0]
        val s01 = P[2][3] + Rv[0][1]
        val s10 = P[3][2] + Rv[1][0]
        val s11 = P[3][3] + Rv[1][1]

        val det = s00 * s11 - s01 * s10
        if (abs(det) < 1e-12) return currentState

        val invS00 = s11 / det
        val invS01 = -s01 / det
        val invS10 = -s10 / det
        val invS11 = s00 / det

        val K = Array(KalmanState.DIM) { r ->
            val pR2 = P[r][2]
            val pR3 = P[r][3]
            doubleArrayOf(
                pR2 * invS00 + pR3 * invS10,
                pR2 * invS01 + pR3 * invS11
            )
        }

        val xNew = DoubleArray(KalmanState.DIM) { r ->
            x[r] + K[r][0] * y0 + K[r][1] * y1
        }

        val maxSpeedMs = 15.0
        xNew[2] = xNew[2].coerceIn(-maxSpeedMs, maxSpeedMs)
        xNew[3] = xNew[3].coerceIn(-maxSpeedMs, maxSpeedMs)

        val IKH = MatrixOps.identity(KalmanState.DIM)
        for (r in 0 until KalmanState.DIM) {
            IKH[r][2] -= K[r][0]
            IKH[r][3] -= K[r][1]
        }

        val term1 = MatrixOps.mul(MatrixOps.mul(IKH, P), MatrixOps.transpose(IKH))
        val term2 = MatrixOps.mul(MatrixOps.mul(K, Rv), MatrixOps.transpose(K))
        val Pnew = MatrixOps.symmetrise(MatrixOps.add(term1, term2))

        return KalmanState(
            x = xNew[0],
            y = xNew[1],
            vx = xNew[2],
            vy = xNew[3],
            P = Pnew
        )
    }

    private fun wrapAngle(rad: Double): Double =
        atan2(sin(rad), cos(rad))

    companion object {
        private const val EARTH_RADIUS_M = 6_371_000.0

        /** Gated fixes skipped in a row before the next one is accepted anyway. */
        const val GATE_RECOVERY_REJECTIONS = 3
    }
}
