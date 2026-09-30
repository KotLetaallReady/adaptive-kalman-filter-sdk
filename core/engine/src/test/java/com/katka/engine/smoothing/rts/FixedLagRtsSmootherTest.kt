package com.katka.engine.smoothing.rts

import com.katka.engine.KalmanFilter
import com.katka.engine.coefficient_startegy.ClassicalCoefficientStrategy
import com.katka.model.Observation
import org.junit.Assert.*
import org.junit.Test
import kotlin.math.cos
import kotlin.math.sin

class FixedLagRtsSmootherTest {

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
    fun buffersUntilLagIsReached() {
        val lag = 5
        val smoother = FixedLagRtsSmoother(lag = lag)
        val filter = KalmanFilter(processNoiseStd = 0.5, useWallClockTime = false, useDoppler = true)
        val strategy = ClassicalCoefficientStrategy()
        filter.reset(strategy)

        val observations = (0..10).map { i ->
            makeObservation(tMs = i * 1000L, xM = i * 5.0, yM = 0.0)
        }

        val emitted = mutableListOf<SmoothedResult>()

        for (i in 0 until lag) {
            val res = filter.process(observations[i], strategy)
            val smoothed = smoother.push(res, filter, observations[i])
            assertNull("Expected null while buffer is filling ($i < $lag)", smoothed)
        }

        // 6th point (index 5) should trigger first output for index 0
        val res5 = filter.process(observations[lag], strategy)
        val smoothed5 = smoother.push(res5, filter, observations[lag])
        assertNotNull("Expected smoothed output at step $lag", smoothed5)
        assertEquals(observations[0].timestamp, smoothed5!!.timestamp)
        emitted.add(smoothed5)

        // Subsequent points each trigger one smoothed point
        for (i in (lag + 1)..10) {
            val res = filter.process(observations[i], strategy)
            val smoothed = smoother.push(res, filter, observations[i])
            assertNotNull(smoothed)
            emitted.add(smoothed!!)
        }

        assertEquals(6, emitted.size) // points 0 to 5
        assertEquals(observations[0].timestamp, emitted[0].timestamp)
        assertEquals(observations[5].timestamp, emitted[5].timestamp)
    }

    @Test
    fun lagZero_emitsImmediately() {
        val smoother = FixedLagRtsSmoother(lag = 0)
        val filter = KalmanFilter(processNoiseStd = 0.5, useWallClockTime = false, useDoppler = true)
        val strategy = ClassicalCoefficientStrategy()
        filter.reset(strategy)

        val obs = makeObservation(1000L, 10.0, 0.0)
        val res = filter.process(obs, strategy)
        val smoothed = smoother.push(res, filter, obs)

        assertNotNull(smoothed)
        assertEquals(1000L, smoothed!!.timestamp)
    }

    @Test
    fun flush_emitsRemainingPoints() {
        val lag = 4
        val smoother = FixedLagRtsSmoother(lag = lag)
        val filter = KalmanFilter(processNoiseStd = 0.5, useWallClockTime = false, useDoppler = true)
        val strategy = ClassicalCoefficientStrategy()
        filter.reset(strategy)

        val observations = (0..5).map { i ->
            makeObservation(tMs = i * 1000L, xM = i * 5.0, yM = 0.0)
        }

        val emitted = mutableListOf<SmoothedResult>()
        for (obs in observations) {
            val res = filter.process(obs, strategy)
            smoother.push(res, filter, obs)?.let { emitted.add(it) }
        }

        // Total 6 points pushed with lag=4: points 0 and 1 were emitted during push
        assertEquals(2, emitted.size)

        // Flush should yield the remaining 4 points (2, 3, 4, 5)
        val flushed = smoother.flush(filter)
        assertEquals(4, flushed.size)
        assertEquals(0, smoother.bufferedCount)

        val allTimestamps = (emitted + flushed).map { it.timestamp }
        val expectedTimestamps = observations.map { it.timestamp }
        assertEquals(expectedTimestamps, allTimestamps)
    }

    @Test
    fun reset_clearsBuffer() {
        val lag = 3
        val smoother = FixedLagRtsSmoother(lag = lag)
        val filter = KalmanFilter(processNoiseStd = 0.5, useWallClockTime = false, useDoppler = true)
        val strategy = ClassicalCoefficientStrategy()
        filter.reset(strategy)

        val obs1 = makeObservation(1000L, 0.0, 0.0)
        val obs2 = makeObservation(2000L, 5.0, 0.0)
        smoother.push(filter.process(obs1, strategy), filter, obs1)
        smoother.push(filter.process(obs2, strategy), filter, obs2)
        assertEquals(2, smoother.bufferedCount)

        smoother.reset()
        assertEquals(0, smoother.bufferedCount)
    }

    @Test
    fun rts_reducesJitter_andSmoothsTrajectory() {
        val lag = 5
        val smoother = FixedLagRtsSmoother(lag = lag)
        val filter = KalmanFilter(processNoiseStd = 0.2, useWallClockTime = false, useDoppler = false)
        val strategy = ClassicalCoefficientStrategy()
        filter.reset(strategy)

        // Generate track moving East with alternating +/- 3m lateral noise on Y
        val n = 30
        val observations = (0 until n).map { i ->
            val noisyY = if (i == 0) 0.0 else if (i % 2 == 0) 3.0 else -3.0
            makeObservation(tMs = i * 1000L, xM = i * 5.0, yM = noisyY, accM = 4.0f)
        }

        val smoothedPoints = mutableListOf<SmoothedResult>()
        val filteredPoints = mutableListOf<Double>() // Y values

        for (obs in observations) {
            val res = filter.process(obs, strategy)
            filteredPoints.add(res.state.y)
            smoother.push(res, filter, obs)?.let { smoothedPoints.add(it) }
        }
        smoothedPoints.addAll(smoother.flush(filter))

        assertEquals(n, smoothedPoints.size)

        // Raw lateral noise std is 3.0m. RTS should smooth Y close to 0.0
        val rtsMaxLateral = smoothedPoints.drop(3).take(20).maxOf { Math.abs(it.state.y) }
        assertTrue("RTS lateral error should be smaller than raw 3.0m noise, got $rtsMaxLateral", rtsMaxLateral < 2.0)
    }
}
