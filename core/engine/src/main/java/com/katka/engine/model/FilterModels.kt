package com.katka.engine.model

import com.katka.engine.MatrixOps

/** Filter mode label stored in [FilterResult] diagnostics. */
enum class FilterMode {
    /** Classical Kalman gain computed from covariance and measurement noise. */
    CLASSICAL,

    /** Neural or learned mode used by consumers that extend the engine. */
    NEURAL
}

/**
 * Output of a Kalman-gain strategy for one correction step.
 *
 * @property K Kalman gain matrix.
 * @property P_updated Posterior covariance after the measurement update.
 * @property R Measurement-noise covariance used for this observation.
 */
data class GainResult(
    val K: Array<DoubleArray>,
    val P_updated: Array<DoubleArray>,
    val R: Array<DoubleArray>
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is GainResult) return false
        return K.contentDeepEquals(other.K) &&
                P_updated.contentDeepEquals(other.P_updated) &&
                R.contentDeepEquals(other.R)
    }

    override fun hashCode(): Int {
        var result = K.contentDeepHashCode()
        result = 31 * result + P_updated.contentDeepHashCode()
        result = 31 * result + R.contentDeepHashCode()
        return result
    }
}

/**
 * Complete result of one filter step.
 *
 * @property timestamp Observation timestamp in milliseconds.
 * @property state Posterior state after the correction step.
 * @property predicted Prior state produced by the prediction step.
 * @property innovation Measurement residual vector `z - Hx`.
 * @property innovationCovS Innovation covariance matrix `S = HPH^T + R`.
 * @property kalmanGain Kalman gain matrix used for the correction.
 * @property measurementNoiseR Measurement-noise covariance used for the step.
 * @property filterMode Diagnostic label of the active filter mode.
 * @property dt Time delta used by the prediction step, in seconds.
 */
data class FilterResult(
    val timestamp: Long,
    val state: KalmanState,
    val predicted: KalmanState,
    val innovation: DoubleArray,
    val innovationCovS: Array<DoubleArray>,
    val kalmanGain: Array<DoubleArray>,
    val measurementNoiseR: Array<DoubleArray>,
    val filterMode: FilterMode,
    val dt: Double
) {
    /** Normalised Innovation Squared, a chi-square consistency statistic for the filter step. */
    val nis: Double by lazy {
        try {
            val SInv = MatrixOps.inverse(innovationCovS)
            val SInvY = MatrixOps.mulVec(SInv, innovation)
            innovation.indices.sumOf { i -> innovation[i] * SInvY[i] }
        } catch (_: Exception) {
            Double.NaN
        }
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is FilterResult) return false
        return timestamp == other.timestamp && state == other.state
    }

    override fun hashCode(): Int = timestamp.hashCode() * 31 + state.hashCode()
}
