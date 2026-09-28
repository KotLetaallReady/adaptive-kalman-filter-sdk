package com.katka.benchmarks

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class EngineParityTest {

    private fun maxDiff(a: Track, b: Track): Double {
        val common = a.keys intersect b.keys
        assertTrue("tracks share no timestamps", common.isNotEmpty())
        return common.maxOf { (a.getValue(it) - b.getValue(it)).norm() }
    }

    @Test
    fun benchmarkKfCv_matchesTheSdkEngine_pointByPoint() {
        // WALK/REALISTIC: multipath bursts and dropouts exercise the gate and irregular dt.
        val (train, _) = makeData(SCENARIOS[1])
        for ((_, ms) in train.take(3)) {
            val sdk = runKf(ms.map { it.obs }, 0.2, false)
            val steps = GFilter(Motion.CV, false, 0.2, 0.0, doppler = false, gyro = false).run(ms)

            val filtered = maxDiff(sdk.track(), filtTrack(steps))
            val smoothed = maxDiff(rtsTrack(sdk, rtsEngine(sdk, LAG)), fixedLag(steps, LAG, -1))

            assertTrue("KF differs by $filtered m", filtered < TOLERANCE_M)
            assertTrue("fixed-lag RTS differs by $smoothed m", smoothed < TOLERANCE_M)
        }
    }

    @Test
    fun simulator_isDeterministicForASeed() {
        fun fingerprint() = makeData(SCENARIOS[4]).first.take(2).flatMap { (_, ms) ->
            ms.map { listOf(it.obs.timestamp.toDouble(), it.obs.latitude, it.obs.longitude, it.obs.speed.toDouble(), it.gyroZ) }
        }
        assertEquals(fingerprint(), fingerprint())
    }

    private companion object {
        const val TOLERANCE_M = 0.01
    }
}
