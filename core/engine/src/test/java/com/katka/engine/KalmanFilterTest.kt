package com.katka.engine

import com.katka.engine.coefficient_startegy.ClassicalCoefficientStrategy
import com.katka.engine.coefficient_startegy.CoefficientStrategy
import com.katka.engine.model.FilterResult
import com.katka.engine.model.GainResult
import com.katka.model.Observation
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.Random
import kotlin.math.cos

class KalmanFilterTest {

    private val refLat = 55.75
    private val refLon = 37.62

    private fun fix(tMs: Long, x: Double, y: Double, accuracy: Float = 4f) = Observation(
        timestamp = tMs,
        latitude = refLat + Math.toDegrees(y / EARTH_RADIUS_M),
        longitude = refLon + Math.toDegrees(x / (EARTH_RADIUS_M * cos(Math.toRadians(refLat)))),
        accuracy = accuracy,
        altitude = 0.0,
        speed = 0f,
        bearing = 0f,
        hasSpeed = false,
        hasBearing = false,
        ax = 0.0,
        ay = 0.0,
        az = 0.0,
        hasImu = false,
        provider = "test"
    )

    /** Walking east at 1.4 m/s with GPS noise; the first fix is exact, so the filter origin is (0, 0). */
    private fun noisyWalk(n: Int, seed: Long, sigma: Double = 3.0): List<Observation> {
        val rng = Random(seed)
        return (0 until n).map { i ->
            if (i == 0) fix(0L, 0.0, 0.0)
            else fix(i * 1000L, 1.4 * i + sigma * rng.nextGaussian(), sigma * rng.nextGaussian())
        }
    }

    private fun standing(range: IntRange, x: Double = 0.0) = range.map { fix(it * 1000L, x, 0.0) }

    private fun KalmanFilter.run(
        track: List<Observation>,
        strategy: CoefficientStrategy = ClassicalCoefficientStrategy(),
        beforeStep: (Int) -> Unit = {}
    ): List<FilterResult> = track.mapIndexed { i, obs ->
        beforeStep(i)
        process(obs, strategy)
    }

    private val FilterResult.isRejected: Boolean
        get() = kalmanGain.all { row -> row.all { it == 0.0 } } && state == predicted

    @Test
    fun timestampOnlyMode_replaysDeterministically() {
        val track = noisyWalk(120, seed = 1)

        val first = KalmanFilter(useWallClockTime = false).run(track)
        // Wall-clock pauses between steps must not change anything.
        val second = KalmanFilter(useWallClockTime = false).run(track) { i -> if (i in 1..5) Thread.sleep(3) }

        assertEquals(first, second)
        first.drop(1).forEach { assertEquals(1.0, it.dt, 0.0) }
    }

    @Test
    fun timestampOnlyMode_takesDtFromIrregularTimestamps() {
        val times = longArrayOf(0L, 1_000L, 3_000L, 3_500L, 8_500L)
        val track = times.map { fix(it, 0.0, 0.0) }

        val dts = KalmanFilter(useWallClockTime = false).run(track).drop(1).map { it.dt }

        assertEquals(listOf(1.0, 2.0, 0.5, 5.0), dts)
    }

    @Test
    fun gate_stillRejectsASingleOutlier() {
        val track = standing(0..29) + fix(30_000L, 100.0, 0.0) + standing(31..39)

        val results = KalmanFilter(useWallClockTime = false).run(track)

        assertTrue(results[30].isRejected)
        assertFalse(results[31].isRejected)
        assertEquals(0.0, results.last().state.x, 0.5)
    }

    @Test
    fun gate_acceptsAFixAfterThreeRejections_andCatchesUp() {
        // The position jumps 100 m, past the 80 m gate cap, and stays there.
        val track = standing(0..29) + standing(30..59, x = 100.0)

        val results = KalmanFilter(useWallClockTime = false).run(track)

        for (i in 30 until 30 + KalmanFilter.GATE_RECOVERY_REJECTIONS) {
            assertTrue("fix $i should be gated", results[i].isRejected)
        }
        assertFalse(results[30 + KalmanFilter.GATE_RECOVERY_REJECTIONS].isRejected)
        assertEquals(100.0, results.last().state.x, 1.0)
    }

    @Test
    fun strategy_receivesInnovationOfEveryAcceptedFix() {
        val strategy = RecordingStrategy()
        val track = standing(0..29) + fix(30_000L, 100.0, 0.0) + standing(31..39)

        val results = KalmanFilter(useWallClockTime = false).run(track, strategy)

        // Neither the initialising fix nor the gated outlier is reported.
        val accepted = results.drop(1).filterNot { it.isRejected }
        assertEquals(track.size - 2, strategy.innovations.size)
        accepted.zip(strategy.innovations).forEach { (result, innovation) ->
            assertArrayEquals(result.innovation, innovation, 0.0)
        }
    }

    @Test
    fun adaptiveR_learnsFromInnovations_whenFilterIsUsedDirectly() {
        // The phone reports 2 m accuracy, but the real noise is 6 m per axis.
        val rng = Random(3)
        val track = (0 until 90).map { i ->
            if (i == 0) fix(0L, 0.0, 0.0, accuracy = 2f)
            else fix(i * 1000L, 6.0 * rng.nextGaussian(), 6.0 * rng.nextGaussian(), accuracy = 2f)
        }
        val strategy = ClassicalCoefficientStrategy(adaptiveR = true)

        val r = KalmanFilter(useWallClockTime = false).run(track, strategy).last().measurementNoiseR[0][0]

        assertTrue("R should grow above accuracy² = 4 m², got $r", r > 8.0)
    }

    @Test
    fun doppler_correctsVelocity_whenMovingEast() {
        // Moving East at 5 m/s (bearing = 90 deg)
        val track = (0..10).map { i ->
            fixWithDoppler(
                tMs = i * 1000L,
                x = 5.0 * i,
                y = 0.0,
                speed = 5.0f,
                bearing = 90.0f,
                speedAccuracy = 0.2f,
                bearingAccuracy = 1.0f
            )
        }
        val results = KalmanFilter(useWallClockTime = false, useDoppler = true).run(track)
        val lastState = results.last().state

        // In our local frame, East is positive X. vx should be close to 5.0, vy close to 0.0
        assertEquals(5.0, lastState.vx, 0.5)
        assertEquals(0.0, lastState.vy, 0.5)
    }

    @Test
    fun doppler_dampsZeroVelocity_whenStanding() {
        val track = (0..10).map { i ->
            fixWithDoppler(
                tMs = i * 1000L,
                x = 0.0,
                y = 0.0,
                speed = 0.0f,
                bearing = 0.0f,
                speedAccuracy = 0.1f,
                bearingAccuracy = 0.0f
            )
        }
        val results = KalmanFilter(useWallClockTime = false, useDoppler = true).run(track)
        val lastState = results.last().state

        assertEquals(0.0, lastState.vx, 0.1)
        assertEquals(0.0, lastState.vy, 0.1)
    }

    @Test
    fun doppler_canBeDisabled_viaConstructor() {
        val track = (0..10).map { i ->
            fixWithDoppler(
                tMs = i * 1000L,
                x = 5.0 * i,
                y = 0.0,
                speed = 0.0f, // false zero speed report
                bearing = 0.0f,
                speedAccuracy = 0.1f,
                bearingAccuracy = 0.0f
            )
        }
        val withDoppler = KalmanFilter(useWallClockTime = false, useDoppler = true).run(track).last().state
        val withoutDoppler = KalmanFilter(useWallClockTime = false, useDoppler = false).run(track).last().state

        // Without Doppler, the filter infers ~5 m/s from positions alone
        assertTrue("Without Doppler vx should be around 5 m/s, got ${withoutDoppler.vx}", withoutDoppler.vx > 4.0)
        // With Doppler forced to 0, vx is damped towards 0
        assertTrue("With Doppler vx should be damped towards 0, got ${withDoppler.vx}", withDoppler.vx < withoutDoppler.vx)
    }

    @Test
    fun doppler_updatesVelocity_whenPositionOutlierIsGated() {
        val track = (0..5).map { i ->
            fixWithDoppler(i * 1000L, 5.0 * i, 0.0, speed = 5.0f, bearing = 90.0f)
        }.toMutableList()

        // 6th fix is an outlier in position (jump of 100m north), but Doppler speed is still 5 m/s East
        track.add(fixWithDoppler(6000L, 30.0, 100.0, speed = 5.0f, bearing = 90.0f))

        val results = KalmanFilter(useWallClockTime = false, useDoppler = true).run(track)
        val outlierResult = results.last()

        // Position was rejected by gate: position gain is zero and coordinates did not jump to the 100m outlier
        assertTrue("Position Kalman gain should be zero when gated", outlierResult.kalmanGain.all { r -> r.all { it == 0.0 } })
        assertEquals("X coordinate should stay near 30m", 30.0, outlierResult.state.x, 1.0)
        assertEquals("Y coordinate should not jump to 100m outlier", 0.0, outlierResult.state.y, 1.0)

        // Velocity should still be updated by Doppler to ~5 m/s East instead of exploding North
        assertEquals(5.0, outlierResult.state.vx, 1.0)
        assertEquals(0.0, outlierResult.state.vy, 1.0)
    }

    @Test
    fun robustUpdate_boundsLateralError_onMultipathSpike() {
        val track = (0..10).map { i ->
            fixWithDoppler(i * 1000L, 5.0 * i, 0.0, speed = 5.0f, bearing = 90.0f)
        }.toMutableList()

        // 11th fix has a 12m lateral spike (below the 15m hard gate threshold, so it enters the update step)
        track.add(fixWithDoppler(11000L, 55.0, 12.0, speed = 5.0f, bearing = 90.0f))

        val nonRobustStrategy = ClassicalCoefficientStrategy(robustUpdate = false)
        val robustStrategy = ClassicalCoefficientStrategy(robustUpdate = true, huberThreshold = 2.45)

        val nonRobustFilter = KalmanFilter(processNoiseStd = 0.5, useWallClockTime = false, useDoppler = true)
        val robustFilter = KalmanFilter(processNoiseStd = 0.5, useWallClockTime = false, useDoppler = true)

        nonRobustFilter.reset(nonRobustStrategy)
        robustFilter.reset(robustStrategy)

        track.forEach { nonRobustFilter.process(it, nonRobustStrategy) }
        track.forEach { robustFilter.process(it, robustStrategy) }

        val nonRobustY = nonRobustFilter.getCurrentState()!!.y
        val robustY = robustFilter.getCurrentState()!!.y

        assertTrue(
            "Robust filter lateral deviation ($robustY) should be strictly smaller than non-robust ($nonRobustY)",
            robustY < nonRobustY
        )
    }

    private fun fixWithDoppler(
        tMs: Long,
        x: Double,
        y: Double,
        speed: Float,
        bearing: Float,
        speedAccuracy: Float = 0.2f,
        bearingAccuracy: Float = 2.0f,
        accuracy: Float = 4f
    ) = Observation(
        timestamp = tMs,
        latitude = refLat + Math.toDegrees(y / EARTH_RADIUS_M),
        longitude = refLon + Math.toDegrees(x / (EARTH_RADIUS_M * cos(Math.toRadians(refLat)))),
        accuracy = accuracy,
        altitude = 0.0,
        speed = speed,
        bearing = bearing,
        hasSpeed = true,
        hasBearing = speed > 0.5f,
        ax = 0.0,
        ay = 0.0,
        az = 0.0,
        hasImu = false,
        speedAccuracy = speedAccuracy,
        bearingAccuracy = bearingAccuracy,
        provider = "gps"
    )

    private class RecordingStrategy : CoefficientStrategy {
        private val delegate = ClassicalCoefficientStrategy()
        val innovations = mutableListOf<DoubleArray>()

        override fun computeGain(P_pred: Array<DoubleArray>, H: Array<DoubleArray>, obs: Observation): GainResult =
            delegate.computeGain(P_pred, H, obs)

        override fun computeGain(P_pred: Array<DoubleArray>, H: Array<DoubleArray>, obs: Observation, innovation: DoubleArray?): GainResult =
            delegate.computeGain(P_pred, H, obs, innovation)

        override fun onInnovation(innovation: DoubleArray, H: Array<DoubleArray>, P_pred: Array<DoubleArray>) {
            innovations.add(innovation.copyOf())
        }
    }

    private companion object {
        const val EARTH_RADIUS_M = 6_371_000.0
    }
}

