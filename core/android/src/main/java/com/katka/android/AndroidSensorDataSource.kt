package com.katka.android

import android.annotation.SuppressLint
import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.location.LocationListener
import android.location.LocationManager
import android.os.Looper
import android.os.PowerManager
import com.google.android.gms.common.ConnectionResult
import com.google.android.gms.common.GoogleApiAvailability
import com.google.android.gms.location.FusedLocationProviderClient
import com.google.android.gms.location.LocationCallback
import com.google.android.gms.location.LocationRequest
import com.google.android.gms.location.LocationResult
import com.google.android.gms.location.LocationServices
import com.google.android.gms.location.Priority
import com.katka.data.SensorDataSource
import com.katka.model.Observation
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.asSharedFlow

/**
 * Android implementation of [SensorDataSource].
 *
 * The source requests high-accuracy GPS updates through Google Play Services
 * when available and falls back to [LocationManager] otherwise. It also listens
 * to linear acceleration and game-rotation sensors, then rotates acceleration
 * into the geographic frame for use by the Kalman prediction step.
 *
 * The caller is responsible for location permissions before [start] is called.
 *
 * @param context Android context used to access system services.
 * @param gpsIntervalMs Requested interval between GPS fixes, in milliseconds.
 * @param minDisplacementM Minimum displacement between location updates, in metres.
 */
class AndroidSensorDataSource(
    private val context: Context,
    private val gpsIntervalMs: Long = 1_000L,
    private val minDisplacementM: Float = 0f
) : SensorDataSource, SensorEventListener {

    private val powerManager: PowerManager =
        context.getSystemService(Context.POWER_SERVICE) as PowerManager

    private val wakeLock: PowerManager.WakeLock by lazy {
        powerManager.newWakeLock(
            PowerManager.PARTIAL_WAKE_LOCK,
            "KalmanFilter::GpsWakeLock"
        )
    }

    private val _observations = MutableSharedFlow<Observation>(
        replay = 0,
        extraBufferCapacity = 8,
        onBufferOverflow = BufferOverflow.DROP_OLDEST
    )
    override val observations: Flow<Observation> = _observations.asSharedFlow()

    private val fusedClient: FusedLocationProviderClient =
        LocationServices.getFusedLocationProviderClient(context)
    private val sensorManager: SensorManager =
        context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val locationManager: LocationManager =
        context.getSystemService(Context.LOCATION_SERVICE) as LocationManager

    private val hasGms: Boolean by lazy {
        try {
            GoogleApiAvailability.getInstance()
                .isGooglePlayServicesAvailable(context) == ConnectionResult.SUCCESS
        } catch (_: Exception) {
            false
        }
    }

    @Volatile private var latestAx: Double = 0.0
    @Volatile private var latestAy: Double = 0.0
    @Volatile private var latestAz: Double = 0.0
    @Volatile private var hasImu: Boolean = false

    @Volatile private var latestRotationMatrix: FloatArray? = null
    @Volatile private var hasRotation: Boolean = false

    @Volatile private var yawCorrectionCos: Double = 1.0
    @Volatile private var yawCorrectionSin: Double = 0.0
    @Volatile private var isYawAligned: Boolean = false

    override var isRunning: Boolean = false
        private set

    private val locationCallback = object : LocationCallback() {
        override fun onLocationResult(result: LocationResult) {
            val location = result.lastLocation ?: return

            if (location.hasBearing() && location.hasSpeed()) {
                tryAlignYaw(location.bearing, location.speed)
            }

            val (axGeo, ayGeo, rotValid) = computeGeoAcceleration()

            val observation = Observation(
                timestamp   = location.time,
                latitude    = location.latitude,
                longitude   = location.longitude,
                accuracy    = location.accuracy,
                altitude    = if (location.hasAltitude()) location.altitude else 0.0,
                speed       = if (location.hasSpeed())    location.speed    else 0f,
                bearing     = if (location.hasBearing())  location.bearing  else 0f,
                hasSpeed    = location.hasSpeed(),
                hasBearing  = location.hasBearing(),
                ax          = latestAx,
                ay          = latestAy,
                az          = latestAz,
                axGeo       = axGeo,
                ayGeo       = ayGeo,
                hasImu      = hasImu,
                hasRotation = rotValid,
                provider    = location.provider ?: "fused"
            )
            _observations.tryEmit(observation)
        }
    }

    private val fallbackListener = object : LocationListener {
        override fun onLocationChanged(location: android.location.Location) {

            if (location.hasBearing() && location.hasSpeed()) {
                tryAlignYaw(location.bearing, location.speed)
            }

            val (axGeo, ayGeo, rotValid) = computeGeoAcceleration()

            val observation = Observation(
                timestamp   = location.time,
                latitude    = location.latitude,
                longitude   = location.longitude,
                accuracy    = location.accuracy,
                altitude    = if (location.hasAltitude()) location.altitude else 0.0,
                speed       = if (location.hasSpeed())    location.speed    else 0f,
                bearing     = if (location.hasBearing())  location.bearing  else 0f,
                hasSpeed    = location.hasSpeed(),
                hasBearing  = location.hasBearing(),
                ax          = latestAx,
                ay          = latestAy,
                az          = latestAz,
                axGeo       = axGeo,
                ayGeo       = ayGeo,
                hasImu      = hasImu,
                hasRotation = rotValid,
                provider    = location.provider ?: "gps"
            )
            _observations.tryEmit(observation)
        }
    }

    /** Snapshots linear acceleration and the device rotation matrix as IMU events arrive. */
    override fun onSensorChanged(event: SensorEvent) {
        when (event.sensor.type) {

            Sensor.TYPE_LINEAR_ACCELERATION -> {
                latestAx = event.values[0].toDouble()
                latestAy = event.values[1].toDouble()
                latestAz = event.values[2].toDouble()
                hasImu   = true
            }

            Sensor.TYPE_GAME_ROTATION_VECTOR -> {
                val rotMat = FloatArray(9)
                SensorManager.getRotationMatrixFromVector(rotMat, event.values)
                latestRotationMatrix = rotMat
                hasRotation = true
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor, accuracy: Int) {}

    /** Computes the yaw correction (game frame → true North) once from the first reliable GPS bearing (speed > 1 m/s). */
    private fun tryAlignYaw(bearing: Float, speed: Float) {
        if (isYawAligned || speed < 1.0f) return
        val rotMat = latestRotationMatrix ?: return

        val orientationAngles = FloatArray(3)
        SensorManager.getOrientation(rotMat, orientationAngles)
        val gameAzimuthRad = orientationAngles[0].toDouble()

        val gpsBearingRad = Math.toRadians(bearing.toDouble())

        val alpha = gpsBearingRad - gameAzimuthRad
        yawCorrectionCos = kotlin.math.cos(alpha)
        yawCorrectionSin = kotlin.math.sin(alpha)
        isYawAligned = true
    }

    /** Rotates the latest device-frame acceleration into geographic (East/North); returns (axGeo, ayGeo, valid). */
    private fun computeGeoAcceleration(): Triple<Double, Double, Boolean> {
        val rotMat = latestRotationMatrix
        if (!hasImu || !hasRotation || rotMat == null) return Triple(0.0, 0.0, false)

        val ax = latestAx; val ay = latestAy; val az = latestAz

        val axRaw = rotMat[0] * ax + rotMat[1] * ay + rotMat[2] * az
        val ayRaw = rotMat[3] * ax + rotMat[4] * ay + rotMat[5] * az

        val axGeo = axRaw * yawCorrectionCos - ayRaw * yawCorrectionSin
        val ayGeo = axRaw * yawCorrectionSin + ayRaw * yawCorrectionCos

        return Triple(axGeo, ayGeo, true)
    }

    /**
     * Acquires a wake lock and registers GPS and IMU listeners for a new session.
     *
     * This method expects location permission to have already been granted by
     * the host application.
     */
    @SuppressLint("MissingPermission")
    override fun start() {
        if (isRunning) return
        if (!wakeLock.isHeld) wakeLock.acquire(30 * 60 * 1000L)

        if (hasGms) startWithFused() else startWithLocationManager()

        sensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION)?.let { sensor ->
            sensorManager.registerListener(this, sensor, SensorManager.SENSOR_DELAY_GAME)
        }

        sensorManager.getDefaultSensor(Sensor.TYPE_GAME_ROTATION_VECTOR)?.let { sensor ->
            sensorManager.registerListener(this, sensor, SensorManager.SENSOR_DELAY_GAME)
        }

        isYawAligned     = false
        yawCorrectionCos = 1.0
        yawCorrectionSin = 0.0

        isRunning = true
    }

    /** Requests high-accuracy updates from FusedLocationProvider. */
    @SuppressLint("MissingPermission")
    private fun startWithFused() {
        val locationRequest = LocationRequest.Builder(
            Priority.PRIORITY_HIGH_ACCURACY,
            gpsIntervalMs
        )
            .setMinUpdateDistanceMeters(minDisplacementM)
            .setWaitForAccurateLocation(false)
            .build()

        fusedClient.requestLocationUpdates(
            locationRequest,
            locationCallback,
            Looper.getMainLooper()
        )
    }

    /** Falls back to LocationManager (GPS or network provider) when Play Services are absent. */
    @SuppressLint("MissingPermission")
    private fun startWithLocationManager() {
        val provider = when {
            locationManager.isProviderEnabled(LocationManager.GPS_PROVIDER) ->
                LocationManager.GPS_PROVIDER
            locationManager.isProviderEnabled(LocationManager.NETWORK_PROVIDER) ->
                LocationManager.NETWORK_PROVIDER
            else -> return
        }
        locationManager.requestLocationUpdates(
            provider, gpsIntervalMs, minDisplacementM, fallbackListener, Looper.getMainLooper()
        )
    }

    /** Releases the wake lock and unregisters all GPS and IMU listeners. */
    override fun stop() {
        if (!isRunning) return
        if (wakeLock.isHeld) wakeLock.release()

        fusedClient.removeLocationUpdates(locationCallback)
        locationManager.removeUpdates(fallbackListener)
        sensorManager.unregisterListener(this)

        hasImu               = false
        hasRotation          = false
        latestRotationMatrix = null
        isYawAligned         = false
        yawCorrectionCos     = 1.0
        yawCorrectionSin     = 0.0
        isRunning            = false
    }
}
