package com.katka.engine.smoothing

import com.katka.engine.neural.NeuralNetwork

/**
 * Inference-time fixed-lag smoother for a Kalman trajectory.
 *
 * Each call to [push] adds one [SmootherInput]. Once the internal window is
 * full, the smoother predicts an `alpha` value with [network] and blends the
 * Kalman point with a Savitzky-Golay estimate for the window centre.
 *
 * @param network Trained neural network that predicts the blend weight.
 * @param normalizer Feature normalizer fitted on the same data as [network].
 * @param config Smoother parameters. Use the same config that was used while
 * collecting the training dataset for [network].
 */
class NeuralTrajectorySmoother(
    private val network: NeuralNetwork,
    private val normalizer: FeatureNormalizer,
    val config: SmootherConfig = SmootherConfig()
) {
    private val window = SmootherWindow(config)

    /** Clears the internal window so the next [push] starts a new sequence. */
    fun reset() = window.clear()

    /**
     * Adds one filter step and returns a smoothed sample when the window is full.
     *
     * Returns `null` during the warm-up period before enough points have been
     * collected.
     */
    fun push(input: SmootherInput): SmoothedSample? {
        window.push(input)
        if (!window.isFull) return null

        val centre = window.centralInput()
        val xKf = doubleArrayOf(centre.kfX, centre.kfY)
        val xSg = window.sgKf()

        val features = normalizer.normalize(window.rawFeatures())
        val alpha = network.predict(features)[0].coerceIn(0.0, 1.0)

        val outX = (1.0 - alpha) * xKf[0] + alpha * xSg[0]
        val outY = (1.0 - alpha) * xKf[1] + alpha * xSg[1]

        return SmoothedSample(
            timestamp = centre.timestamp,
            kfX = centre.kfX, kfY = centre.kfY,
            sgX = xSg[0], sgY = xSg[1],
            outX = outX, outY = outY,
            alpha = alpha,
            rawX = centre.rawX, rawY = centre.rawY,
            vx = centre.vx, vy = centre.vy,
            speed = centre.speed,
            innovationMag = centre.innovationMag,
            sigmaPos = centre.sigmaPos,
            accuracy = centre.accuracy
        )
    }
}
