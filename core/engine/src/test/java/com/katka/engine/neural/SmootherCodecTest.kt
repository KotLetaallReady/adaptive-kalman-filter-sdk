package com.katka.engine.neural

import com.katka.engine.smoothing.SmootherConfig
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Test
import java.util.Random

class SmootherCodecTest {

    @Test
    fun roundTrip_preservesWeightsBiasesAndNormalizer() {
        val net = NeuralNetwork(NetworkConfig.default(), seed = 5)
        val rnd = Random(3)
        for (layer in net.weights) for (neuron in layer) for (j in neuron.indices) neuron[j] = rnd.nextGaussian()
        for (layer in net.biases) for (i in layer.indices) layer[i] = rnd.nextGaussian()
        val mean = DoubleArray(SmootherConfig.FEATURE_COUNT) { it.toDouble() + 0.5 }
        val std = DoubleArray(SmootherConfig.FEATURE_COUNT) { (it + 1).toDouble() }
        val config = SmootherConfig(
            windowLength = 7,
            turnThresholdDeg = 20.0,
            alphaSuppressionTurnDeg = 90.0,
            minTurnSuppression = 0.1
        )

        val text = SmootherCodec.encode(net, mean, std, config)
        val loaded = SmootherCodec.decode(text)
        assertNotNull(loaded); loaded!!

        assertEquals(net.config.allSizes.toList(), loaded.network.config.allSizes.toList())
        assertEquals(net.config.outputActivation, loaded.network.config.outputActivation)
        assertArrayEquals(mean, loaded.featureMean, 0.0)
        assertArrayEquals(std, loaded.featureStd, 0.0)
        assertEquals(config, loaded.smootherConfig)

        // Identical predictions ⇒ weights restored exactly.
        val input = DoubleArray(SmootherConfig.FEATURE_COUNT) { 0.1 * it }
        assertArrayEquals(net.predict(input), loaded.network.predict(input), 1e-12)
    }

    @Test
    fun decode_garbage_returnsNull() {
        assertNull(SmootherCodec.decode("not a model"))
        assertNull(SmootherCodec.decode(""))
    }

    @Test
    fun decodeLegacyV2_usesDefaultSmootherConfig() {
        val net = NeuralNetwork(NetworkConfig.default(), seed = 7)
        val mean = DoubleArray(SmootherConfig.FEATURE_COUNT) { it.toDouble() }
        val std = DoubleArray(SmootherConfig.FEATURE_COUNT) { 1.0 }
        val lines = SmootherCodec.encode(net, mean, std).trim().lines().toMutableList()
        lines[0] = "v2"
        lines.removeAt(5)

        val loaded = SmootherCodec.decode(lines.joinToString("\n"))
        assertNotNull(loaded); loaded!!
        assertEquals(SmootherConfig(), loaded.smootherConfig)
    }
}
