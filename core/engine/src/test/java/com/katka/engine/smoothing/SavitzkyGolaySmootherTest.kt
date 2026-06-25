package com.katka.engine.smoothing

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs

class SavitzkyGolaySmootherTest {

    @Test
    fun constantWindow_returnsConstant() {
        val values = DoubleArray(11) { 5.0 }
        assertEquals(5.0, SavitzkyGolaySmoother.smoothCentre(values), 1e-9)
    }

    @Test
    fun linearWindow_returnsCentreOfLine() {
        // x_i = 2*i + 3 ; centre index = 5 → 13. A quadratic fits a line exactly.
        val values = DoubleArray(11) { i -> 2.0 * i + 3.0 }
        assertEquals(13.0, SavitzkyGolaySmoother.smoothCentre(values), 1e-9)
    }

    @Test
    fun parabolaWindow_isFittedExactly() {
        // x_i = (i-5)^2 + 4 ; centre value (τ=0) = 4. Degree-2 fit is exact.
        val values = DoubleArray(11) { i -> val t = (i - 5).toDouble(); t * t + 4.0 }
        assertEquals(4.0, SavitzkyGolaySmoother.smoothCentre(values), 1e-9)
    }

    @Test
    fun alternatingNoise_isReducedAtCentre() {
        // Straight line at 10 with ±1 alternating noise. Raw centre = 11; SG should be near 10.
        val values = DoubleArray(11) { i -> 10.0 + if (i % 2 == 0) -1.0 else 1.0 }
        val sg = SavitzkyGolaySmoother.smoothCentre(values)
        assertTrue("SG centre $sg should be closer to 10 than the raw 11",
            abs(sg - 10.0) < abs(values[5] - 10.0))
    }

    @Test
    fun twoDimensional_smoothsEachAxis() {
        val pts = (0 until 11).map { i -> doubleArrayOf(i.toDouble(), 0.0) }
        val c = SavitzkyGolaySmoother.smoothCentre2D(pts)
        assertEquals(5.0, c[0], 1e-9)
        assertEquals(0.0, c[1], 1e-9)
    }
}
