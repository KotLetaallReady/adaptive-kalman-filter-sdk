package com.katka.model

/**
 * Quality summary for a filtered or smoothed trajectory.
 *
 * The metrics are expressed in metres except [lag], which is expressed in samples.
 *
 * @property rmse Root mean squared error against the reference track.
 * @property mae Mean absolute error against the reference track.
 * @property maxError Largest point-wise error in the compared segment.
 * @property lag Estimated sample lag of the evaluated track relative to the reference.
 * @property stability Standard deviation of step lengths; lower values mean less jitter.
 * @property sampleCount Number of points used to compute the metrics.
 */
data class AccuracyMetrics(
    val rmse: Double,
    val mae: Double,
    val maxError: Double,
    val lag: Double,
    val stability: Double,
    val sampleCount: Int
) {
    companion object {
        /** Empty metrics value returned when there are not enough samples to compare tracks. */
        val EMPTY = AccuracyMetrics(
            rmse = Double.NaN,
            mae = Double.NaN,
            maxError = Double.NaN,
            lag = Double.NaN,
            stability = Double.NaN,
            sampleCount = 0
        )
    }
}
