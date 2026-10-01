package com.katka.engine.coefficient_startegy

import kotlin.math.max

/**
 * Small MLP (5 -> 16 -> 8 -> 1, ReLU) that predicts the log-correction Δ = ln(R_true / accuracy²)
 * of the GNSS measurement noise from causal features, in this order:
 * ln(accuracy), ln(1 + mean |innovation|), ln(1 + std |innovation|) over the last 10 accepted fixes,
 * reported speed (m/s) and ln σ_pos of the predicted state.
 *
 * Weights are stored row-major: `w1[i][j]` links input `j` to hidden neuron `i`.
 *
 * @property mean Feature means used for normalisation.
 * @property std Feature standard deviations used for normalisation.
 */
class PretrainedResidualModel(
    val mean: DoubleArray,
    val std: DoubleArray,
    val w1: Array<DoubleArray>,
    val b1: DoubleArray,
    val w2: Array<DoubleArray>,
    val b2: DoubleArray,
    val w3: DoubleArray,
    val b3: Double
) {
    init {
        require(mean.size == FEATURES && std.size == FEATURES) { "Feature normalisers must have size $FEATURES" }
        require(w1.size == b1.size && w1.all { it.size == FEATURES }) { "Layer 1 must take $FEATURES inputs" }
        require(w2.size == b2.size && w2.all { it.size == w1.size }) { "Layer 2 must take ${w1.size} inputs" }
        require(w3.size == w2.size) { "Output layer must take ${w2.size} inputs" }
    }

    /**
     * Forward pass for one feature vector of size [FEATURES]; returns the unclamped Δ ln R. Normalised
     * features are clipped to ±[MAX_Z] so inputs outside the training data (e.g. car speeds) are not extrapolated.
     */
    fun predict(features: DoubleArray): Double {
        val z = DoubleArray(FEATURES) { j -> ((features[j] - mean[j]) / std[j]).coerceIn(-MAX_Z, MAX_Z) }
        val h1 = DoubleArray(w1.size) { i ->
            var s = b1[i]
            for (j in 0 until FEATURES) s += w1[i][j] * z[j]
            max(0.0, s)
        }
        var out = b3
        for (i in w2.indices) {
            var s = b2[i]
            for (j in h1.indices) s += w2[i][j] * h1[j]
            out += w3[i] * max(0.0, s)
        }
        return out
    }

    companion object {
        /** Number of input features. */
        const val FEATURES = 5

        /** Normalised features are clipped to ±MAX_Z standard deviations of the training data. */
        const val MAX_Z = 3.0

        /**
         * Model trained by `./gradlew :benchmarks:run --args="noise-train"`: self-training (3 rounds) on the
         * train routes of all five synthetic scenarios, teacher = robust offline RTS (Cauchy) of this SDK.
         * Those are walking and cycling routes with ~3-5 m accuracy; other conditions (cars, poor fixes) are
         * extrapolation, bounded only by the clamp in [NeuralResidualNoiseStrategy]. Not yet validated on real recordings.
         */
        val DEFAULT: PretrainedResidualModel by lazy { PretrainedWeights.create() }
    }
}
