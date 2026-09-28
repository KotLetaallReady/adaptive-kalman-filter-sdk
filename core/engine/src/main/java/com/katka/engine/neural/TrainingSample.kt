package com.katka.engine.neural

/**
 * One training pair consumed by [NeuralNetworkTrainer].
 *
 * @property features Input feature vector.
 * @property labels Expected output vector.
 * @property weight Multiplier for this sample's loss and gradient. Use it to
 * balance rare situations against common ones; `1.0` is plain mean squared error.
 */
data class TrainingSample(
    val features: DoubleArray,
    val labels: DoubleArray,
    val weight: Double = 1.0
) {
    init {
        require(weight >= 0.0) { "weight must be non-negative." }
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is TrainingSample) return false
        return features.contentEquals(other.features) && labels.contentEquals(other.labels) &&
            weight == other.weight
    }

    override fun hashCode(): Int =
        31 * (31 * features.contentHashCode() + labels.contentHashCode()) + weight.hashCode()
}
