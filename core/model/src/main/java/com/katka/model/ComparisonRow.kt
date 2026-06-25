package com.katka.model

/**
 * One row of a comparison session exported by a host application or a custom tool.
 *
 * Each row describes the same timestamp across raw GPS, Kalman output,
 * Savitzky-Golay output and the final neural smoother output.
 *
 * @property stepIndex Sequential index of the processed sample.
 * @property timestampMs Observation timestamp in milliseconds.
 * @property rawLat Raw GPS latitude in WGS-84 degrees.
 * @property rawLon Raw GPS longitude in WGS-84 degrees.
 * @property gpsAccuracyM Reported GPS horizontal accuracy in metres.
 * @property gpsSpeedMs Reported GPS speed in metres per second.
 * @property kfLat Kalman-filtered latitude in WGS-84 degrees.
 * @property kfLon Kalman-filtered longitude in WGS-84 degrees.
 * @property kfVx Estimated local X velocity in metres per second.
 * @property kfVy Estimated local Y velocity in metres per second.
 * @property kfSigmaPos Estimated one-sigma horizontal position uncertainty in metres.
 * @property kfInnov Kalman innovation magnitude in metres.
 * @property sgLat Savitzky-Golay latitude in WGS-84 degrees.
 * @property sgLon Savitzky-Golay longitude in WGS-84 degrees.
 * @property smoothedLat Final smoothed latitude in WGS-84 degrees.
 * @property smoothedLon Final smoothed longitude in WGS-84 degrees.
 * @property alpha Trust weight in `[0, 1]` predicted by the smoother network.
 */
data class ComparisonRow(
    val stepIndex:     Int,
    val timestampMs:   Long,
    val rawLat:        Double,
    val rawLon:        Double,
    val gpsAccuracyM:  Float,
    val gpsSpeedMs:    Float,
    val kfLat:         Double,
    val kfLon:         Double,
    val kfVx:          Double,
    val kfVy:          Double,
    val kfSigmaPos:    Double,
    val kfInnov:       Double,
    val sgLat:         Double,
    val sgLon:         Double,
    val smoothedLat:   Double,
    val smoothedLon:   Double,
    val alpha:         Double
)
