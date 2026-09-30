package com.katka.engine.smoothing.rts

import com.katka.engine.model.KalmanState
import com.katka.model.Observation

/**
 * Output of the Rauch–Tung–Striebel (RTS) trajectory smoother.
 *
 * Contains both geographic (WGS-84) coordinates ready for map rendering
 * and the internal state-space estimates (local position, velocity, and covariance).
 *
 * @property timestamp Observation timestamp in milliseconds.
 * @property latitude Smoothed WGS-84 latitude in degrees.
 * @property longitude Smoothed WGS-84 longitude in degrees.
 * @property altitude Altitude in metres above sea level.
 * @property speed Smoothed horizontal speed in metres per second.
 * @property bearing Smoothed bearing in degrees clockwise from True North [0, 360).
 * @property accuracy Estimated 1-σ horizontal position uncertainty in metres.
 * @property state Complete 4D state vector (x, y, vx, vy) and smoothed covariance P.
 * @property rawObservation Original raw GNSS observation if available.
 */
data class SmoothedResult(
    val timestamp: Long,
    val latitude: Double,
    val longitude: Double,
    val altitude: Double = 0.0,
    val speed: Double = 0.0,
    val bearing: Float = 0f,
    val accuracy: Float = 0f,
    val state: KalmanState,
    val rawObservation: Observation? = null
)
