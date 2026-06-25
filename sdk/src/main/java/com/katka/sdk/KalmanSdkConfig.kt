package com.katka.sdk

/**
 * Runtime configuration for the default Android Kalman SDK setup.
 */
data class KalmanSdkConfig(
    val gpsIntervalMs: Long = 1_000L,
    val minDisplacementM: Float = 0f,
    val processNoiseStd: Double = 0.5,
    val minAccuracyM: Float = 1.0f,
    val maxAccuracyM: Float = 50.0f,
    val adaptiveR: Boolean = false,
    val adaptiveWindow: Int = 20,
    val forgettingB: Double = 0.97,
    val smootherModelFileName: String = "neural_smoother.model",
    val logcatTag: String = "KalmanLog"
) {
    init {
        require(gpsIntervalMs > 0L) { "gpsIntervalMs must be positive." }
        require(minDisplacementM >= 0f) { "minDisplacementM must be non-negative." }
        require(processNoiseStd > 0.0) { "processNoiseStd must be positive." }
        require(minAccuracyM > 0f) { "minAccuracyM must be positive." }
        require(maxAccuracyM >= minAccuracyM) { "maxAccuracyM must be >= minAccuracyM." }
        require(adaptiveWindow >= 3) { "adaptiveWindow must be at least 3." }
        require(forgettingB in 0.0..1.0) { "forgettingB must be in [0, 1]." }
        require(smootherModelFileName.isNotBlank()) { "smootherModelFileName must not be blank." }
        require(logcatTag.isNotBlank()) { "logcatTag must not be blank." }
    }
}
