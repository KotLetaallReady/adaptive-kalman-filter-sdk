package com.katka.engine.neural

import com.katka.engine.smoothing.FeatureNormalizer
import com.katka.engine.smoothing.NeuralTrajectorySmoother
import com.katka.engine.smoothing.SmootherConfig

/**
 * Restored smoother model loaded from [SmootherRepository].
 *
 * @property network Trained network used by [NeuralTrajectorySmoother].
 * @property featureMean Feature means that must be passed to [FeatureNormalizer].
 * @property featureStd Feature standard deviations that must be passed to [FeatureNormalizer].
 * @property smootherConfig Smoother config used when the model was trained.
 */
data class LoadedSmoother(
    val network: NeuralNetwork,
    val featureMean: DoubleArray,
    val featureStd: DoubleArray,
    val smootherConfig: SmootherConfig = SmootherConfig()
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is LoadedSmoother) return false
        return network === other.network &&
                featureMean.contentEquals(other.featureMean) &&
                featureStd.contentEquals(other.featureStd) &&
                smootherConfig == other.smootherConfig
    }

    override fun hashCode(): Int {
        var r = network.hashCode()
        r = 31 * r + featureMean.contentHashCode()
        r = 31 * r + featureStd.contentHashCode()
        r = 31 * r + smootherConfig.hashCode()
        return r
    }
}
