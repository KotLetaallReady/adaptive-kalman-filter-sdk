package com.katka.data

import com.katka.model.Observation
import kotlinx.coroutines.flow.Flow

/**
 * Platform-independent source of [Observation] values.
 *
 * Implement this interface when the library should read measurements from a
 * phone, a file, a simulator or any other source. The engine only depends on
 * this abstraction and does not require Android classes.
 */
interface SensorDataSource {

    /** Stream of sensor observations used to drive the filter pipeline. */
    val observations: Flow<Observation>

    /** Starts acquisition. Implementations should make repeated calls safe. */
    fun start()

    /** Stops acquisition and releases resources. Implementations should make repeated calls safe. */
    fun stop()

    /** Whether the source is currently active. */
    val isRunning: Boolean
}

/**
 * In-memory [SensorDataSource] that emits a fixed list of observations.
 *
 * This is useful for unit tests, recorded routes and simple demos where no
 * real sensors are available.
 */
class ReplaySensorDataSource(
    private val items: List<Observation>
) : SensorDataSource {

    private val _flow = kotlinx.coroutines.flow.flow {
        items.forEach { emit(it) }
    }

    override val observations: Flow<Observation> = _flow
    override var isRunning: Boolean = false
        private set

    override fun start()  { isRunning = true  }
    override fun stop()   { isRunning = false }
}
