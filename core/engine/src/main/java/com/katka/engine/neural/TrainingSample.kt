package com.katka.engine.neural

/**
 * One training pair consumed by [NeuralNetworkTrainer].
 *
 * @property features Input feature vector.
 * @property labels Expected output vector.
 */
data class TrainingSample(
    val features: DoubleArray,
    val labels: DoubleArray
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is TrainingSample) return false
        return features.contentEquals(other.features) && labels.contentEquals(other.labels)
    }

    override fun hashCode(): Int = 31 * features.contentHashCode() + labels.contentHashCode()
}
