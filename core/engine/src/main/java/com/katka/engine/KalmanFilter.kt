package com.katka.engine

import com.katka.engine.coefficient_startegy.CoefficientStrategy
import com.katka.engine.model.FilterMode
import com.katka.engine.model.FilterResult
import com.katka.engine.model.KalmanState
import com.katka.model.Observation
import kotlin.math.cos

/**
 * Discrete-time Kalman filter for two-dimensional position tracking.
 *
 * The filter works in a local metre-based coordinate system whose origin is
 * created from the first accepted GPS fix. Incoming WGS-84 coordinates are
 * projected into that local frame, processed with a constant-velocity model
 * and can then be converted back to latitude/longitude.
 *
 * @param processNoiseStd Standard deviation of the process noise used in the
 * prediction step. Larger values make the filter adapt faster but smooth less.
 */
class KalmanFilter(
    private val processNoiseStd: Double = 0.5
) {
    private var state: KalmanState? = null
    private var lastTimestamp: Long = -1L
    private var lastWallClockMs: Long = -1L

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
        refLat = 0.0
        refLon = 0.0
        strategy?.reset()
    }

    /**
     * Processes one observation and returns the posterior state with diagnostics.
     *
     * The first accepted observation initializes the local coordinate system and
     * returns an initial [FilterResult]. Later calls run the prediction and
     * correction steps using the supplied [strategy].
     */
    fun process(obs: Observation, strategy: CoefficientStrategy): FilterResult {
        if (!isInitialised) {
            return initialise(obs)
        }

        val nowMs = System.currentTimeMillis()

        val dtFromGps  = (obs.timestamp - lastTimestamp) / 1000.0
        val dtFromWall = if (lastWallClockMs > 0)
            (nowMs - lastWallClockMs) / 1000.0
        else
            dtFromGps

        val dt = when {
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

        if (innovationMag > gateThreshold) {
            state = xPred
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
                state             = xPred,
                predicted         = xPred,
                innovation        = innovation,
                innovationCovS    = Sgate,
                kalmanGain        = zeroK,
                measurementNoiseR = Rgate,
                filterMode        = FilterMode.CLASSICAL,
                dt                = dt
            )
        }

        val gainResult = strategy.computeGain(PPred, H, obs)
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

        val posterior = KalmanState(
            x  = xNew[0], y  = xNew[1],
            vx = xNew[2], vy = xNew[3],
            P  = gainResult.P_updated
        )
        state = posterior

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

    /**
     * Converts WGS-84 degrees into local metres relative to [refLat] and [refLon].
     *
     * Call this after the filter has been initialized by [process], otherwise
     * the reference point is still `(0.0, 0.0)`.
     */
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

    /** Initialises the filter and reference point from the first GPS fix. */
    private fun initialise(obs: Observation): FilterResult {
        refLat = obs.latitude
        refLon = obs.longitude
        lastTimestamp = obs.timestamp

        val posVariance = (obs.accuracy * obs.accuracy).toDouble().coerceAtLeast(1.0)
        val initState = KalmanState.initial(x = 0.0, y = 0.0, posVariance = posVariance)
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

    /** Prediction step: extrapolates state and covariance with the constant-velocity model plus the IMU control input. */
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

    private companion object {
        const val EARTH_RADIUS_M = 6_371_000.0
    }
}
