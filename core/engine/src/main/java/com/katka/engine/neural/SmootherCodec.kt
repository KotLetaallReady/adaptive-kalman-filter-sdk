package com.katka.engine.neural

import com.katka.engine.smoothing.SmootherConfig

/**
 * Dependency-free serialization of a trained smoother model to a text blob.
 *
 * Most users should access this through [SmootherRepository]. Use the codec
 * directly only when you need to move the model text between custom systems.
 */
object SmootherCodec {

    private const val VERSION = "v3"
    private const val LEGACY_V2 = "v2"

    /** Serialises the network, feature normalisation and smoother config to a text blob. */
    fun encode(
        network: NeuralNetwork,
        featureMean: DoubleArray,
        featureStd: DoubleArray,
        smootherConfig: SmootherConfig = SmootherConfig()
    ): String {
        val cfg = network.config
        val sb = StringBuilder()
        sb.appendLine(VERSION)
        sb.appendLine(cfg.inputSize)
        sb.appendLine(cfg.hiddenSizes.joinToString(","))
        sb.appendLine(cfg.outputSize)
        sb.appendLine(cfg.outputActivation.name)
        sb.appendLine(smootherConfig.toCodecLine())
        sb.appendLine(featureMean.joinToString(" "))
        sb.appendLine(featureStd.joinToString(" "))
        sb.appendLine(buildString {
            for (layer in network.weights) for (neuron in layer) for (v in neuron) {
                append(v)
                append(' ')
            }
        }.trim())
        sb.appendLine(buildString {
            for (layer in network.biases) for (v in layer) {
                append(v)
                append(' ')
            }
        }.trim())
        return sb.toString()
    }

    /** Restores the model from a text blob, or null if it is missing or corrupt. */
    fun decode(text: String): LoadedSmoother? = runCatching {
        val lines = text.trim().lines()
        val version = lines[0].trim()
        require(version == VERSION || version == LEGACY_V2) { "unknown smoother format: ${lines[0]}" }

        val inputSize = lines[1].trim().toInt()
        val hiddenSizes = lines[2].trim()
            .let { if (it.isEmpty()) emptyList() else it.split(",").map { s -> s.trim().toInt() } }
        val outputSize = lines[3].trim().toInt()
        val activation = OutputActivation.valueOf(lines[4].trim())

        val smootherConfig = if (version == VERSION) parseConfig(lines[5]) else SmootherConfig()
        val featureOffset = if (version == VERSION) 6 else 5
        val featureMean = parseDoubles(lines[featureOffset])
        val featureStd = parseDoubles(lines[featureOffset + 1])
        val weightValues = parseDoubles(lines[featureOffset + 2])
        val biasValues = parseDoubles(lines[featureOffset + 3])

        val network = NeuralNetwork(NetworkConfig(inputSize, hiddenSizes, outputSize, activation))

        var wi = 0
        for (layer in network.weights) {
            for (neuron in layer) {
                for (j in neuron.indices) neuron[j] = weightValues[wi++]
            }
        }
        var bi = 0
        for (layer in network.biases) {
            for (i in layer.indices) layer[i] = biasValues[bi++]
        }

        LoadedSmoother(network, featureMean, featureStd, smootherConfig)
    }.getOrNull()

    private fun SmootherConfig.toCodecLine(): String =
        listOf(windowLength, turnThresholdDeg, alphaSuppressionTurnDeg, minTurnSuppression)
            .joinToString(" ")

    private fun parseConfig(line: String): SmootherConfig {
        val values = parseDoubles(line)
        require(values.size == 4) { "invalid smoother config line" }
        return SmootherConfig(
            windowLength = values[0].toInt(),
            turnThresholdDeg = values[1],
            alphaSuppressionTurnDeg = values[2],
            minTurnSuppression = values[3]
        )
    }

    private fun parseDoubles(line: String): DoubleArray {
        val t = line.trim()
        return if (t.isEmpty()) DoubleArray(0)
        else t.split(" ").map { it.toDouble() }.toDoubleArray()
    }
}
