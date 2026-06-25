package com.katka.engine.smoothing

import org.junit.Assert.assertEquals
import org.junit.Test
import kotlin.math.sqrt

class FeatureNormalizerTest {

    @Test
    fun fitThenNormalize_givesZeroMeanUnitStd() {
        val rows = listOf(
            doubleArrayOf(1.0, 100.0),
            doubleArrayOf(3.0, 100.0),
            doubleArrayOf(5.0, 100.0)
        )
        val nrm = FeatureNormalizer.fit(rows)
        val normalized = rows.map { nrm.normalize(it) }

        // Column 0: mean ≈ 0
        val mean0 = normalized.sumOf { it[0] } / normalized.size
        assertEquals(0.0, mean0, 1e-9)

        // Column 0: population std ≈ 1
        val std0 = sqrt(normalized.sumOf { it[0] * it[0] } / normalized.size)
        assertEquals(1.0, std0, 1e-9)

        // Column 1 has zero variance → std treated as 1 → all values 0
        normalized.forEach { assertEquals(0.0, it[1], 1e-9) }
    }

    @Test
    fun identity_isNoOp() {
        val nrm = FeatureNormalizer.identity(3)
        val v = doubleArrayOf(7.0, -2.0, 0.5)
        val out = nrm.normalize(v)
        for (i in v.indices) assertEquals(v[i], out[i], 1e-12)
    }
}
