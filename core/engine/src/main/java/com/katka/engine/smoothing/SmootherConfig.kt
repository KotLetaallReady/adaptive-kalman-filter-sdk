package com.katka.engine.smoothing

/**
 * Configuration for the fixed-lag neural trajectory smoother.
 *
 * The defaults match the original library behaviour. Pass the same config to
 * [SmoothingTrainingCollector] and [NeuralTrajectorySmoother] when training and
 * inference should use custom smoothing behaviour.
 *
 * @property windowLength Fixed-lag window length, also known as `L`. Larger
 * windows give Savitzky-Golay and the neural model more context, at the cost of
 * more output delay.
 * @property turnThresholdDeg Turn angles below this value are ignored when
 * computing turn suppression.
 * @property alphaSuppressionTurnDeg Turn angle that reduces the target alpha to
 * [minTurnSuppression] during training.
 * @property minTurnSuppression Lower bound for the alpha suppression factor on
 * sharp turns.
 */
data class SmootherConfig(
    val windowLength: Int = DEFAULT_WINDOW_LENGTH,
    val turnThresholdDeg: Double = DEFAULT_TURN_THRESHOLD_DEG,
    val alphaSuppressionTurnDeg: Double = DEFAULT_ALPHA_SUPPRESSION_TURN_DEG,
    val minTurnSuppression: Double = DEFAULT_MIN_TURN_SUPPRESSION
) {
    init {
        require(windowLength >= 3) { "windowLength must be at least 3." }
        require(windowLength % 2 == 1) { "windowLength must be odd so the smoother has a centre point." }
        require(turnThresholdDeg >= 0.0) { "turnThresholdDeg must be non-negative." }
        require(alphaSuppressionTurnDeg > 0.0) { "alphaSuppressionTurnDeg must be positive." }
        require(minTurnSuppression in 0.0..1.0) { "minTurnSuppression must be in [0, 1]." }
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
    }
}
