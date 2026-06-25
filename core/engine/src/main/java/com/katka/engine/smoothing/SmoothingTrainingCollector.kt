package com.katka.engine.smoothing

import com.katka.engine.neural.NeuralNetworkTrainer
import com.katka.engine.neural.TrainingSample

/**
 * Training dataset produced from a recorded smoothing session.
 *
 * @property normalizer Fitted feature normalizer that must be stored with the model.
 * @property samples Normalized feature/label pairs for [NeuralNetworkTrainer].
 */
data class SmoothingDataset(
    val normalizer: FeatureNormalizer,
    val samples: List<TrainingSample>
)

/**
 * Builds neural-smoother training samples from a filter session.
 *
 * Feed every completed [SmootherInput] into [addStep]. Once enough points are
 * available, the collector estimates the target blend weight and stores the raw
 * feature row. Call [buildDataset] after recording to fit the normalizer and
 * produce samples ready for [NeuralNetworkTrainer].
 *
 * @param config Smoother parameters used to build windows and target alpha
 * labels. Use the same config during inference.
 */
class SmoothingTrainingCollector(
    val config: SmootherConfig = SmootherConfig()
) {

    private val window = SmootherWindow(config)
    private val rawFeatureRows = mutableListOf<DoubleArray>()
    private val alphaStars = mutableListOf<Double>()

    /** Number of training samples collected so far. */
    val sampleCount: Int get() = rawFeatureRows.size

    /** Feeds one completed filter step, emitting a training pair once the window is full. */
    fun addStep(input: SmootherInput) {
        window.push(input)
        if (!window.isFull) return

        val centre = window.centralInput()
        val xKf = doubleArrayOf(centre.kfX, centre.kfY)
        val xSg = window.sgKf()
        val xStar = window.sgRaw()
        val phi = window.turnSuppressionPhiDeg()

        rawFeatureRows.add(window.rawFeatures())
        alphaStars.add(OptimalAlpha.solve(xKf, xSg, xStar, phi, config))
    }

    /** Clears collected windows and training rows. */
    fun reset() {
        window.clear()
        rawFeatureRows.clear()
        alphaStars.clear()
    }

    /** Fits the normalizer over all collected rows and returns the normalized dataset. */
    fun buildDataset(): SmoothingDataset {
        require(rawFeatureRows.isNotEmpty()) { "buildDataset: no samples collected" }
        val normalizer = FeatureNormalizer.fit(rawFeatureRows)
        val samples = rawFeatureRows.indices.map { i ->
            TrainingSample(
                features = normalizer.normalize(rawFeatureRows[i]),
                labels = doubleArrayOf(alphaStars[i])
            )
        }
        return SmoothingDataset(normalizer, samples)
    }
}
