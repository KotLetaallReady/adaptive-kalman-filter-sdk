package com.katka.engine.smoothing

/** How the smoother measures turn angles inside a window. */
enum class TurnDetection {
    /**
     * Bearing between consecutive Kalman points. At walking speed it reports phantom turns;
     * kept only so models saved before format `v4` keep their features.
     */
    POINT_DIFF,

    /** Bearing of the Kalman velocity; samples slower than [SmootherConfig.minTurnSpeedMs] are skipped. */
    VELOCITY
}

/**
 * Configuration of the fixed-lag neural smoother. Use the same config for
 * [SmoothingTrainingCollector] and [NeuralTrajectorySmoother].
 *
 * @property windowLength Window length `L`; a longer window smooths more but delays the output
 * by `L / 2` samples.
 * @property turnThresholdDeg Turn angles below this value are ignored by turn suppression.
 * @property alphaSuppressionTurnDeg Turn angle that lowers the target alpha to
 * [minTurnSuppression] during training.
 * @property minTurnSuppression Lower bound for the alpha suppression factor on sharp turns.
 * @property turnDetection How turn angles are measured.
 * @property minTurnSpeedMs With [TurnDetection.VELOCITY], samples slower than this speed (m/s)
 * are ignored.
 */
data class SmootherConfig(
    val windowLength: Int = DEFAULT_WINDOW_LENGTH,
    val turnThresholdDeg: Double = DEFAULT_TURN_THRESHOLD_DEG,
    val alphaSuppressionTurnDeg: Double = DEFAULT_ALPHA_SUPPRESSION_TURN_DEG,
    val minTurnSuppression: Double = DEFAULT_MIN_TURN_SUPPRESSION,
    val turnDetection: TurnDetection = TurnDetection.VELOCITY,
    val minTurnSpeedMs: Double = DEFAULT_MIN_TURN_SPEED_MS
) {
    init {
        require(windowLength >= 3) { "windowLength must be at least 3." }
        require(windowLength % 2 == 1) { "windowLength must be odd so the smoother has a centre point." }
        require(turnThresholdDeg >= 0.0) { "turnThresholdDeg must be non-negative." }
        require(alphaSuppressionTurnDeg > 0.0) { "alphaSuppressionTurnDeg must be positive." }
        require(minTurnSuppression in 0.0..1.0) { "minTurnSuppression must be in [0, 1]." }
        require(minTurnSpeedMs >= 0.0) { "minTurnSpeedMs must be non-negative." }
    }

    /** Index of the point emitted when the fixed-lag window is full. */
    val halfWindow: Int get() = windowLength / 2

    companion object {
        /** Original fixed-lag window length used by the smoother. */
        const val DEFAULT_WINDOW_LENGTH = 11

        /** The smoother always extracts six trajectory features. */
        const val FEATURE_COUNT = 6

        /** Original turn threshold used before alpha suppression is applied. */
        const val DEFAULT_TURN_THRESHOLD_DEG = 30.0

        /** Original turn angle that drives suppression to the minimum. */
        const val DEFAULT_ALPHA_SUPPRESSION_TURN_DEG = 60.0

        /** Original minimum alpha suppression factor for sharp turns. */
        const val DEFAULT_MIN_TURN_SUPPRESSION = 0.05

        /** Below this speed the velocity heading is dominated by filter noise. */
        const val DEFAULT_MIN_TURN_SPEED_MS = 0.5
    }
}
