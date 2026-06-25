package com.katka.model

/**
 * Single measurement consumed by the filter pipeline.
 *
 * The observation always contains a GPS fix and may also contain IMU data.
 * Coordinates are WGS-84 degrees; acceleration values are metres per second squared.
 *
 * @property timestamp GPS timestamp in milliseconds.
 * @property latitude Latitude in WGS-84 degrees.
 * @property longitude Longitude in WGS-84 degrees.
 * @property accuracy Horizontal GPS accuracy in metres.
 * @property altitude Altitude in metres, or `0.0` when altitude is unavailable.
 * @property speed Ground speed in metres per second.
 * @property bearing Direction of movement in degrees clockwise from true north.
 * @property hasSpeed Whether [speed] was reported by the location provider.
 * @property hasBearing Whether [bearing] was reported by the location provider.
 * @property ax Device-frame linear acceleration on the X axis.
 * @property ay Device-frame linear acceleration on the Y axis.
 * @property az Device-frame linear acceleration on the Z axis.
 * @property axGeo East-facing acceleration after rotation into the geographic frame.
 * @property ayGeo North-facing acceleration after rotation into the geographic frame.
 * @property hasImu Whether acceleration data is available for this observation.
 * @property hasRotation Whether device rotation was available for geographic acceleration.
 * @property provider Name of the location provider that produced the GPS fix.
 */
data class Observation(
    val timestamp: Long,
    val latitude: Double,
    val longitude: Double,
    val accuracy: Float,
    val altitude: Double,
    val speed: Float,
    val bearing: Float,
    val hasSpeed: Boolean,
    val hasBearing: Boolean,
    val ax: Double,
    val ay: Double,
    val az: Double,
    val axGeo: Double = 0.0,
    val ayGeo: Double = 0.0,
    val hasImu: Boolean,
    val hasRotation: Boolean = false,
    val provider: String
)
