package com.katka.engine.smoothing

import com.katka.engine.KalmanFilter

/**
 * One filter step used as input for the fixed-lag trajectory smoother.
 *
 * Coordinates are local metres in the same coordinate system as [KalmanFilter].
 *
 * @property timestamp Observation timestamp in milliseconds.
 * @property kfX Kalman-filtered local X position in metres.
 * @property kfY Kalman-filtered local Y position in metres.
 * @property rawX Raw GPS local X position in metres.
 * @property rawY Raw GPS local Y position in metres.
 * @property vx Estimated local X velocity in metres per second.
 * @property vy Estimated local Y velocity in metres per second.
 * @property speed Reported or estimated speed in metres per second.
 * @property accuracy GPS accuracy in metres.
 * @property innovationMag Kalman innovation magnitude in metres.
 * @property sigmaPos One-sigma position uncertainty in metres.
 */
data class SmootherInput(
    val timestamp: Long,
    val kfX: Double,
    val kfY: Double,
    val rawX: Double,
    val rawY: Double,
    val vx: Double,
    val vy: Double,
    val speed: Double,
    val accuracy: Double,
    val innovationMag: Double,
    val sigmaPos: Double
)

/**
 * Smoothed output for the central point of a full fixed-lag window.
 *
 * The smoother cannot emit a result until enough future points are available,
 * so every sample represents a slightly delayed central point.
 *
 * @property timestamp Timestamp of the central point.
 * @property kfX Kalman-filtered local X position.
 * @property kfY Kalman-filtered local Y position.
 * @property sgX Savitzky-Golay local X estimate for the same point.
 * @property sgY Savitzky-Golay local Y estimate for the same point.
 * @property outX Final blended local X position.
 * @property outY Final blended local Y position.
 * @property alpha Trust weight in `[0, 1]`; `0` means Kalman, `1` means Savitzky-Golay.
 * @property rawX Raw GPS local X position for diagnostics.
 * @property rawY Raw GPS local Y position for diagnostics.
 * @property vx Local X velocity in metres per second.
 * @property vy Local Y velocity in metres per second.
 * @property speed Speed in metres per second.
 * @property innovationMag Kalman innovation magnitude in metres.
 * @property sigmaPos One-sigma position uncertainty in metres.
 * @property accuracy GPS accuracy in metres.
 */
data class SmoothedSample(
    val timestamp: Long,
    val kfX: Double,
    val kfY: Double,
    val sgX: Double,
    val sgY: Double,
    val outX: Double,
    val outY: Double,
    val alpha: Double,
    val rawX: Double,
    val rawY: Double,
    val vx: Double,
    val vy: Double,
    val speed: Double,
    val innovationMag: Double,
    val sigmaPos: Double,
    val accuracy: Double
)
