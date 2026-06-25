package com.katka.engine.neural

import kotlin.math.sqrt

/**
 * Trains a [NeuralNetwork] in place with mini-batch Adam and mean squared error.
 *
 * This class is intentionally platform-free, so it can be used in JVM tests,
 * desktop tools or Android host applications.
 *
 * @param network Network instance to update.
 * @param learningRate Adam learning rate.
 * @param batchSize Number of samples per mini-batch.
 * @param beta1 Adam first-moment decay.
 * @param beta2 Adam second-moment decay.
 * @param eps Small denominator offset for numerical stability.
 */
class NeuralNetworkTrainer(
    private val network: NeuralNetwork,
    private val learningRate: Double = 1e-3,
    private val batchSize: Int = 32,
    private val beta1: Double = 0.9,
    private val beta2: Double = 0.999,
    private val eps: Double = 1e-8
) {
    private val mW: Array<Array<DoubleArray>> = zeroLike(network.weights)
    private val vW: Array<Array<DoubleArray>> = zeroLike(network.weights)
    private val mB: Array<DoubleArray> = zeroLike(network.biases)
    private val vB: Array<DoubleArray> = zeroLike(network.biases)

    private var t = 0

    /** Trains for [epochs] passes over [dataset]; returns the mean MSE loss per epoch. */
    fun train(dataset: List<TrainingSample>, epochs: Int): List<Double> {
        require(dataset.isNotEmpty()) { "Cannot train on an empty dataset" }
        val lossPerEpoch = ArrayList<Double>(epochs)

        repeat(epochs) {
            val shuffled = dataset.shuffled()
            var epochLoss = 0.0
            var nBatches = 0
            for (batch in shuffled.chunked(batchSize)) {
                epochLoss += trainBatch(batch)
                nBatches++
            }
            lossPerEpoch.add(epochLoss / nBatches)
        }
        return lossPerEpoch
    }

    /** Runs one mini-batch (forward, backprop, Adam update) and returns its mean MSE loss. */
    fun trainBatch(batch: List<TrainingSample>): Double {
        val L = network.weights.size
        val dW = zeroLike(network.weights)
        val dB = zeroLike(network.biases)
        var batchLoss = 0.0

        for (sample in batch) {
            val cache = network.forwardWithCache(sample.features)
            val output = cache.activations[L]
            val target = sample.labels

            val m = output.size.toDouble()
            batchLoss += output.indices.sumOf { i ->
                val e = output[i] - target[i]; e * e
            } / m

            var delta = DoubleArray(output.size) { i -> (2.0 / m) * (output[i] - target[i]) }

            for (l in L - 1 downTo 0) {
                val pre = cache.preActivations[l]
                val prevAct = cache.activations[l]
                val isLast = l == L - 1

                val actDelta = if (isLast) {
                    when (network.config.outputActivation) {
                        OutputActivation.LINEAR -> delta
                        OutputActivation.SIGMOID ->
                            DoubleArray(output.size) { i -> delta[i] * output[i] * (1.0 - output[i]) }
                    }
                } else {
                    DoubleArray(pre.size) { i -> if (pre[i] > 0.0) delta[i] else 0.0 }
                }

                val bs = batch.size.toDouble()
                for (i in actDelta.indices) {
                    dB[l][i] += actDelta[i] / bs
                    for (j in prevAct.indices) dW[l][i][j] += actDelta[i] * prevAct[j] / bs
                }

                delta = DoubleArray(prevAct.size) { j ->
                    network.weights[l].indices.sumOf { i -> network.weights[l][i][j] * actDelta[i] }
                }
            }
        }

        t++
        val lrCorrected = learningRate *
                sqrt(1.0 - beta2.pow(t)) / (1.0 - beta1.pow(t))

        for (l in 0 until L) {
            for (i in network.weights[l].indices) {
                mB[l][i] = beta1 * mB[l][i] + (1.0 - beta1) * dB[l][i]
                vB[l][i] = beta2 * vB[l][i] + (1.0 - beta2) * dB[l][i] * dB[l][i]
                network.biases[l][i] -= lrCorrected * mB[l][i] / (sqrt(vB[l][i]) + eps)

                for (j in network.weights[l][i].indices) {
                    mW[l][i][j] = beta1 * mW[l][i][j] + (1.0 - beta1) * dW[l][i][j]
                    vW[l][i][j] = beta2 * vW[l][i][j] + (1.0 - beta2) * dW[l][i][j] * dW[l][i][j]
                    network.weights[l][i][j] -= lrCorrected * mW[l][i][j] / (sqrt(vW[l][i][j]) + eps)
                }
            }
        }

        return batchLoss / batch.size
    }

    /** Resets the Adam moment estimates and step counter. */
    fun resetOptimiserState() {
        for (l in mW.indices) {
            for (i in mW[l].indices) {
                mW[l][i].fill(0.0); vW[l][i].fill(0.0)
            }
            mB[l].fill(0.0); vB[l].fill(0.0)
        }
        t = 0
    }

    private fun zeroLike(src: Array<Array<DoubleArray>>) =
        Array(src.size) { l -> Array(src[l].size) { i -> DoubleArray(src[l][i].size) } }

    private fun zeroLike(src: Array<DoubleArray>) =
        Array(src.size) { l -> DoubleArray(src[l].size) }

    private fun Double.pow(n: Int): Double {
        var r = 1.0; repeat(n) { r *= this }; return r
    }
}
