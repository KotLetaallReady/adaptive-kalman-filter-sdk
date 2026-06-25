package com.katka.engine.neural

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
        val mean = DoubleArray(6) { it.toDouble() + 0.5 }
        val std = DoubleArray(6) { (it + 1).toDouble() }

        val text = SmootherCodec.encode(net, mean, std)
        val loaded = SmootherCodec.decode(text)
        assertNotNull(loaded); loaded!!

        assertEquals(net.config.allSizes.toList(), loaded.network.config.allSizes.toList())
        assertEquals(net.config.outputActivation, loaded.network.config.outputActivation)
        assertArrayEquals(mean, loaded.featureMean, 0.0)
        assertArrayEquals(std, loaded.featureStd, 0.0)

        // Identical predictions ⇒ weights restored exactly.
        val input = DoubleArray(6) { 0.1 * it }
        assertArrayEquals(net.predict(input), loaded.network.predict(input), 1e-12)
    }

    @Test
    fun decode_garbage_returnsNull() {
        assertNull(SmootherCodec.decode("not a model"))
        assertNull(SmootherCodec.decode(""))
    }
}
