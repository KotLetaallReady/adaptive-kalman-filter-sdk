package com.katka.engine.neural

import com.katka.engine.smoothing.NeuralTrajectorySmoother
import kotlin.math.exp
import kotlin.math.sqrt
import kotlin.random.Random

/**
 * Small feed-forward neural network implemented in pure Kotlin.
 *
 * The network is used by [NeuralTrajectorySmoother] to predict a single blend
 * weight, but the topology is configurable for experiments and tests.
 *
 * @param config Network topology and output activation.
 * @param seed Random seed used for deterministic weight initialization.
 */
class NeuralNetwork(val config: NetworkConfig, seed: Long = 42L) {

    /** Learnable weights per connection, row-major: weights[l][i][j] links neuron j in layer l to neuron i in l+1. */
    val weights: Array<Array<DoubleArray>>

    /** Learnable bias per neuron, excluding the input layer. */
    val biases: Array<DoubleArray>

    init {
        val rng = Random(seed)
        val sizes = config.allSizes
        val nLayers = sizes.size - 1

        weights = Array(nLayers) { l ->
            val fanIn = sizes[l]
            val std = sqrt(2.0 / fanIn)
            Array(sizes[l + 1]) {
                DoubleArray(fanIn) { rng.nextDouble(-std, std) }
            }
        }
        biases = Array(nLayers) { l -> DoubleArray(sizes[l + 1]) { 0.0 } }
    }

    /** Forward pass returning the output vector. */
    fun predict(input: DoubleArray): DoubleArray {
        var act = input
        for (l in weights.indices) {
            val W = weights[l]
            val b = biases[l]
            val isLast = l == weights.lastIndex
            act = DoubleArray(W.size) { i ->
                var s = b[i]
                for (j in act.indices) s += W[i][j] * act[j]
                if (isLast) outputActivate(s) else relu(s)
            }
        }
        return act
    }

    /** Forward pass returning per-layer pre- and post-activations for backpropagation. */
    internal fun forwardWithCache(input: DoubleArray): ForwardCache {
        val nLayers = weights.size
        val preActs = arrayOfNulls<DoubleArray>(nLayers)
        val acts = arrayOfNulls<DoubleArray>(nLayers + 1)
        acts[0] = input

        for (l in weights.indices) {
            val W = weights[l]
            val b = biases[l]
            val isLast = l == nLayers - 1
            val prevAct = acts[l]!!

            val pre = DoubleArray(W.size) { i ->
                var s = b[i]
                for (j in prevAct.indices) s += W[i][j] * prevAct[j]
                s
            }
            preActs[l] = pre
            acts[l + 1] = DoubleArray(pre.size) { if (isLast) outputActivate(pre[it]) else relu(pre[it]) }
        }

        @Suppress("UNCHECKED_CAST")
        return ForwardCache(
            preActivations = preActs as Array<DoubleArray>,
            activations = acts as Array<DoubleArray>
        )
    }

    private fun relu(x: Double) = if (x > 0.0) x else 0.0

    /** Output-layer activation selected by [NetworkConfig.outputActivation]. */
    private fun outputActivate(x: Double): Double = when (config.outputActivation) {
        OutputActivation.LINEAR -> x
        OutputActivation.SIGMOID -> 1.0 / (1.0 + exp(-x))
    }

    /** Per-layer activations cached during a forward pass for use in backpropagation. */
    internal data class ForwardCache(
        val preActivations: Array<DoubleArray>,
        val activations: Array<DoubleArray>
    )
}

/** Activation function applied to the output layer. Hidden layers always use ReLU. */
enum class OutputActivation {
    /** Identity, for unbounded regression targets. */
    LINEAR,
    /** Sigmoid, for targets bounded to (0,1) such as the trust weight alpha. */
    SIGMOID
}

/**
 * Immutable descriptor of a [NeuralNetwork] topology.
 *
 * @property inputSize Number of input features.
 * @property hiddenSizes Hidden layer sizes in order.
 * @property outputSize Number of output neurons.
 * @property outputActivation Activation applied to the output layer.
 */
data class NetworkConfig(
    val inputSize: Int,
    val hiddenSizes: List<Int>,
    val outputSize: Int,
    val outputActivation: OutputActivation = OutputActivation.SIGMOID
) {
    /** Flat array of all layer sizes: input, hidden layers and output. */
    val allSizes: IntArray
        get() = (listOf(inputSize) + hiddenSizes + listOf(outputSize)).toIntArray()

    companion object {
        /** Default smoother topology: 6 -> 8 -> 4 -> 1 with a sigmoid output. */
        fun default() = NetworkConfig(
            inputSize = 6,
            hiddenSizes = listOf(8, 4),
            outputSize = 1,
            outputActivation = OutputActivation.SIGMOID
        )
    }
}
