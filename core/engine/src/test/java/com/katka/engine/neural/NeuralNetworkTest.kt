package com.katka.engine.neural

import com.katka.engine.smoothing.SmootherConfig
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.Random
import kotlin.math.exp

class NeuralNetworkTest {

    @Test
    fun defaultTopology_isSixToOne() {
        val net = NeuralNetwork(NetworkConfig.default())
        assertEquals(listOf(6, 8, 4, 1), net.config.allSizes.toList())
        assertEquals(OutputActivation.SIGMOID, net.config.outputActivation)
    }

    @Test
    fun sigmoidOutput_isStrictlyInUnitInterval() {
        val net = NeuralNetwork(NetworkConfig.default())
        val rnd = Random(7)
        repeat(50) {
            val f = DoubleArray(SmootherConfig.FEATURE_COUNT) { rnd.nextGaussian() * 100.0 } // deliberately large
            val y = net.predict(f)
            assertEquals(1, y.size)
            assertTrue("output ${y[0]} must be > 0", y[0] > 0.0)
            assertTrue("output ${y[0]} must be < 1", y[0] < 1.0)
        }
    }

    @Test
    fun training_reducesLossOnLearnableTarget() {
        // Target α* = σ(2·f0 − 1.5·f1): a smooth function the MLP can fit.
        val rnd = Random(11)
        val samples = (0 until 200).map {
            val f = DoubleArray(SmootherConfig.FEATURE_COUNT) { rnd.nextGaussian() }
            val target = 1.0 / (1.0 + exp(-(2.0 * f[0] - 1.5 * f[1])))
            TrainingSample(f, doubleArrayOf(target))
        }

        val net = NeuralNetwork(NetworkConfig.default())
        val trainer = NeuralNetworkTrainer(net, learningRate = 1e-2, batchSize = 16)
        val losses = trainer.train(samples, epochs = 300)

        assertTrue("loss should decrease: first=${losses.first()} last=${losses.last()}",
            losses.last() < losses.first() * 0.5)
    }

    @Test
    fun sampleWeights_pullTheFitTowardsTheWeightedMean() {
        // Same input, conflicting labels: plain MSE settles at 0.5, weights 9:1 at 0.26.
        val f = DoubleArray(3)
        val samples = listOf(
            TrainingSample(f, doubleArrayOf(0.2), weight = 9.0),
            TrainingSample(f, doubleArrayOf(0.8), weight = 1.0)
        )

        val net = NeuralNetwork(NetworkConfig(3, listOf(4), 1, OutputActivation.LINEAR), seed = 3)
        NeuralNetworkTrainer(net, learningRate = 1e-2, batchSize = samples.size).train(samples, epochs = 2_000)

        assertEquals(0.26, net.predict(f)[0], 0.02)
    }

    @Test
    fun seededShuffle_makesTrainingReproducible() {
        val rnd = Random(5)
        val samples = (0 until 100).map {
            val f = DoubleArray(SmootherConfig.FEATURE_COUNT) { rnd.nextGaussian() }
            TrainingSample(f, doubleArrayOf(if (f[0] > 0) 0.9 else 0.1))
        }

        fun trainedWeights(): List<Double> {
            val net = NeuralNetwork(NetworkConfig.default(), seed = 1)
            NeuralNetworkTrainer(net, batchSize = 16, random = kotlin.random.Random(9)).train(samples, epochs = 5)
            return net.weights.flatMap { layer -> layer.flatMap { it.toList() } }
        }

        assertEquals(trainedWeights(), trainedWeights())
    }
}
