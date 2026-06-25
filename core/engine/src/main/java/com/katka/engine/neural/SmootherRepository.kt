package com.katka.engine.neural

/**
 * Repository for a persisted neural smoother model.
 *
 * The repository owns serialization and delegates the actual storage mechanism
 * to [SmootherStore], allowing Android files, memory stores or custom backends.
 */
class SmootherRepository(private val store: SmootherStore) {

    /** Whether a trained model is available. */
    fun exists(): Boolean = store.exists()

    /** Restores the trained model, or null if absent or corrupt. */
    fun load(): LoadedSmoother? = store.load()?.let { SmootherCodec.decode(it) }

    /** Serialises and persists the network together with its feature normalisation. */
    fun save(network: NeuralNetwork, featureMean: DoubleArray, featureStd: DoubleArray) {
        store.save(SmootherCodec.encode(network, featureMean, featureStd))
    }

    /** Deletes the persisted model. */
    fun delete(): Boolean = store.delete()
}
