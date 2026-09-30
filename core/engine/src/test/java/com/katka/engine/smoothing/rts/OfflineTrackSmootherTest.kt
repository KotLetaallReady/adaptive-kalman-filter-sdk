package com.katka.engine.smoothing.rts

import com.katka.model.Observation
import org.junit.Assert.*
import org.junit.Test
import kotlin.math.abs
import kotlin.math.cos

class OfflineTrackSmootherTest {

    private val refLat = 55.75
    private val refLon = 37.61
    private val earthRadiusM = 6_371_000.0

    private fun makeObservation(
        tMs: Long,
        xM: Double,
        yM: Double,
        accM: Float = 4.0f,
        speedMs: Float = 5.0f,
        bearingDeg: Float = 90.0f
    ): Observation {
        val lat = refLat + Math.toDegrees(yM / earthRadiusM)
        val lon = refLon + Math.toDegrees(xM / (earthRadiusM * cos(Math.toRadians(refLat))))
        return Observation(
            timestamp = tMs,
            latitude = lat,
            longitude = lon,
            accuracy = accM,
            altitude = 150.0,
            speed = speedMs,
            bearing = bearingDeg,
            hasSpeed = true,
            hasBearing = true,
            ax = 0.0,
            ay = 0.0,
            az = 0.0,
            hasImu = false,
            speedAccuracy = 0.2f,
            bearingAccuracy = 2.0f,
            provider = "gps"
        )
    }

    @Test
    fun emptyAndSinglePoint_handledGracefully() {
        val smoother = OfflineTrackSmoother()
        val emptyResult = smoother.smooth(emptyList())
        assertTrue(emptyResult.isEmpty())

        val singleObs = makeObservation(1000L, 0.0, 0.0)
        val singleResult = smoother.smooth(listOf(singleObs))
        assertEquals(1, singleResult.size)
        assertEquals(singleObs.latitude, singleResult[0].latitude, 1e-7)
        assertEquals(singleObs.longitude, singleResult[0].longitude, 1e-7)
    }

    @Test
    fun smoothsStraightLineTrack_withAccurateVelocityAndBearing() {
        val smoother = OfflineTrackSmoother(processNoiseStd = 0.2, iterations = 2, useDoppler = true)
        val n = 20
        // Moving East at 5 m/s (bearing 90 deg)
        val track = (0 until n).map { i ->
            makeObservation(tMs = i * 1000L, xM = i * 5.0, yM = 0.0, speedMs = 5.0f, bearingDeg = 90.0f)
        }

        val smoothed = smoother.smooth(track)
        assertEquals(n, smoothed.size)

        for (i in 2 until (n - 2)) {
            val pt = smoothed[i]
            assertEquals("X should advance at ~5m/s", i * 5.0, pt.state.x, 1.0)
            assertEquals("Y should stay near 0", 0.0, pt.state.y, 0.5)
            assertEquals("vx should be ~5 m/s", 5.0, pt.state.vx, 0.5)
            assertEquals("vy should be ~0 m/s", 0.0, pt.state.vy, 0.5)
            assertEquals("Speed should be ~5 m/s", 5.0, pt.speed, 0.5)
            assertEquals("Bearing should be ~90 deg", 90.0, pt.bearing.toDouble(), 5.0)
        }
    }

    @Test
    fun robustCauchy_suppressesMultipathJump() {
        // Track moving East along Y = 0
        val n = 25
        val track = (0 until n).map { i ->
            if (i == 12) {
                // Outlier multipath jump: 60m North, but phone claims 4m accuracy
                makeObservation(tMs = i * 1000L, xM = i * 5.0, yM = 60.0, accM = 4.0f)
            } else {
                makeObservation(tMs = i * 1000L, xM = i * 5.0, yM = 0.0, accM = 4.0f)
            }
        }

        val robustSmoother = OfflineTrackSmoother(processNoiseStd = 0.2, iterations = 4, cauchyScale = 2.45)
        val smoothed = robustSmoother.smooth(track)

        assertEquals(n, smoothed.size)

        // At index 12, raw GPS jumped to 60m North.
        // Cauchy IRLS reweights the outlier down to ~0.01, so the smoothed point stays close to 0m!
        val outlierSmoothedY = abs(smoothed[12].state.y)
        assertTrue("Smoothed Y at outlier should be close to 0m, got $outlierSmoothedY (raw was 60m)", outlierSmoothedY < 12.0)
    }

    @Test
    fun deterministic_output() {
        val smoother = OfflineTrackSmoother(processNoiseStd = 0.2, iterations = 3)
        val track = (0 until 15).map { i ->
            makeObservation(tMs = i * 1000L, xM = i * 5.0, yM = if (i % 2 == 0) 2.0 else -2.0)
        }

        val run1 = smoother.smooth(track)
        val run2 = smoother.smooth(track)

        for (i in track.indices) {
            assertEquals(run1[i].latitude, run2[i].latitude, 1e-12)
            assertEquals(run1[i].longitude, run2[i].longitude, 1e-12)
            assertEquals(run1[i].state.x, run2[i].state.x, 1e-12)
            assertEquals(run1[i].state.y, run2[i].state.y, 1e-12)
        }
    }
}
