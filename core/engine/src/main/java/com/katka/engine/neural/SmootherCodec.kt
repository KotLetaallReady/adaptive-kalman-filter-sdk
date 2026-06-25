package com.katka.engine.neural

/**
 * Dependency-free serialization of a trained smoother model to a text blob.
 *
 * Most users should access this through [SmootherRepository]. Use the codec
 * directly only when you need to move the model text between custom systems.
 */
object SmootherCodec {

    private const val VERSION = "v2"

    /** Serialises the network and its feature normalisation to a text blob. */
    fun encode(network: NeuralNetwork, featureMean: DoubleArray, featureStd: DoubleArray): String {
        val cfg = network.config
        val sb = StringBuilder()
        sb.appendLine(VERSION)
        sb.appendLine(cfg.inputSize)
        sb.appendLine(cfg.hiddenSizes.joinToString(","))
        sb.appendLine(cfg.outputSize)
        sb.appendLine(cfg.outputActivation.name)
        sb.appendLine(featureMean.joinToString(" "))
        sb.appendLine(featureStd.joinToString(" "))
        sb.appendLine(buildString {
            for (layer in network.weights) for (neuron in layer) for (v in neuron) { append(v); append(' ') }
        }.trim())
        sb.appendLine(buildString {
            for (layer in network.biases) for (v in layer) { append(v); append(' ') }
        }.trim())
        return sb.toString()
    }

    /** Restores the model from a text blob, or null if it is missing or corrupt. */
    fun decode(text: String): LoadedSmoother? = runCatching {
        val lines = text.trim().lines()
        require(lines[0].trim() == VERSION) { "unknown smoother format: ${lines[0]}" }

        val inputSize = lines[1].trim().toInt()
        val hiddenSizes = lines[2].trim()
            .let { if (it.isEmpty()) emptyList() else it.split(",").map { s -> s.trim().toInt() } }
        val outputSize = lines[3].trim().toInt()
        val activation = OutputActivation.valueOf(lines[4].trim())
        val featureMean = parseDoubles(lines[5])
        val featureStd = parseDoubles(lines[6])
        val weightValues = parseDoubles(lines[7])
        val biasValues = parseDoubles(lines[8])

        val network = NeuralNetwork(NetworkConfig(inputSize, hiddenSizes, outputSize, activation))

        var wi = 0
        for (layer in network.weights) for (neuron in layer) for (j in neuron.indices) neuron[j] = weightValues[wi++]
        var bi = 0
        for (layer in network.biases) for (i in layer.indices) layer[i] = biasValues[bi++]

        LoadedSmoother(network, featureMean, featureStd)
    }.getOrNull()

    private fun parseDoubles(line: String): DoubleArray {
        val t = line.trim()
        return if (t.isEmpty()) DoubleArray(0)
        else t.split(" ").map { it.toDouble() }.toDoubleArray()
    }
}
