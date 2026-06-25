package com.katka.engine.coefficient_startegy

import com.katka.engine.MatrixOps
import com.katka.engine.model.GainResult
import com.katka.model.Observation
import kotlin.math.ln

/**
 * Classical Kalman-gain strategy.
 *
 * The strategy builds the measurement-noise matrix `R` from GPS accuracy,
 * clamps unrealistic accuracy values and optionally blends this value with a
 * Sage-Husa-style adaptive estimate based on recent innovations.
 *
 * @param minAccuracyM Lower bound for reported GPS accuracy, in metres.
 * @param maxAccuracyM Upper bound for reported GPS accuracy, in metres.
 * @param adaptiveR Whether to adapt `R` from recent innovation statistics.
 * @param adaptiveWindow Number of recent innovations used by the adaptive estimate.
 * @param forgettingB Forgetting factor for the adaptive `R` estimate.
 */
class ClassicalCoefficientStrategy(
    private val minAccuracyM: Float = 1.0f,
    private val maxAccuracyM: Float = 50.0f,
    private val adaptiveR: Boolean = false,
    private val adaptiveWindow: Int = 20,
    private val forgettingB: Double = 0.97
) : CoefficientStrategy {

    private val innovationWindow = ArrayDeque<DoubleArray>(adaptiveWindow)
    private var adaptiveREstimate: Array<DoubleArray>? = null
    private var stepCount = 0

    /** Feeds the innovation back into the Sage-Husa adaptive-R estimator (no-op unless [adaptiveR]). */
    fun updateInnovation(
        innovation: DoubleArray,
        H: Array<DoubleArray>,
        P_pred: Array<DoubleArray>
    ) {
        if (!adaptiveR) return

        if (innovationWindow.size >= adaptiveWindow) {
            innovationWindow.removeFirst()
            hphtWindow.removeFirst()
        }
        innovationWindow.addLast(innovation)

        val HP   = MatrixOps.mul(H, P_pred)
        val HPHt = MatrixOps.mul(HP, MatrixOps.transpose(H))
        hphtWindow.addLast(HPHt)

        stepCount++
        if (innovationWindow.size < 3) return

        val m = innovation.size

        val Syy = MatrixOps.zeros(m, m)
        for (y in innovationWindow) {
            val yyT = MatrixOps.outerProduct(y, y)
            for (i in 0 until m) for (j in 0 until m) Syy[i][j] += yyT[i][j]
        }
        for (i in 0 until m) for (j in 0 until m) Syy[i][j] /= innovationWindow.size

        val HPHt_mean = MatrixOps.zeros(m, m)
        for (hpht in hphtWindow) {
            for (i in 0 until m) for (j in 0 until m) HPHt_mean[i][j] += hpht[i][j]
        }
        for (i in 0 until m) for (j in 0 until m) HPHt_mean[i][j] /= hphtWindow.size

        val Rraw = MatrixOps.sub(Syy, HPHt_mean)

        val minVariance = (minAccuracyM * minAccuracyM).toDouble()
        val Rclamped = MatrixOps.copy(Rraw)
        for (i in 0 until m) {
            Rclamped[i][i] = Rclamped[i][i].coerceAtLeast(minVariance)
        }
        for (i in 0 until m) for (j in 0 until m) {
            if (i != j) {
                val sym = (Rclamped[i][j] + Rclamped[j][i]) / 2.0
                val maxOffDiag = 0.99 * kotlin.math.sqrt(Rclamped[i][i] * Rclamped[j][j])
                Rclamped[i][j] = sym.coerceIn(-maxOffDiag, maxOffDiag)
                Rclamped[j][i] = Rclamped[i][j]
            }
        }

        val dk = (1.0 - forgettingB) / (1.0 - Math.pow(forgettingB, stepCount.toDouble() + 1))
        val prev = adaptiveREstimate ?: Rclamped
        val blended = MatrixOps.zeros(m, m)
        for (i in 0 until m) for (j in 0 until m) {
            blended[i][j] = (1.0 - dk) * prev[i][j] + dk * Rclamped[i][j]
        }
        adaptiveREstimate = blended
    }

    /** Computes R, the Kalman gain K and the Joseph-form posterior covariance for one update step. */
    override fun computeGain(
        P_pred: Array<DoubleArray>,
        H: Array<DoubleArray>,
        obs: Observation
    ): GainResult {
        val R = computeR(obs, H, P_pred)

        val HP   = MatrixOps.mul(H, P_pred)
        val HPHt = MatrixOps.mul(HP, MatrixOps.transpose(H))
        val S    = MatrixOps.add(HPHt, R)

        val sScale = (S[0][0] + S[1][1]) / 2.0
        val Sreg = MatrixOps.addDiagEps(S, eps = sScale * 1e-6 + 1e-6)

        val PHt  = MatrixOps.mul(P_pred, MatrixOps.transpose(H))
        val SInv = MatrixOps.inverse(Sreg)
        val K    = MatrixOps.mul(PHt, SInv)
        val kDiagSum = K.indices.sumOf { i -> K.getOrNull(i)?.getOrNull(i) ?: 0.0 }
        if (kDiagSum.isNaN() || kDiagSum.isInfinite()) {
            return GainResult(
                K         = MatrixOps.zeros(P_pred.size, H.size),
                P_updated = P_pred,
                R         = R
            )
        }
        val n        = P_pred.size
        val I        = MatrixOps.identity(n)
        val KH       = MatrixOps.mul(K, H)
        val IminusKH = MatrixOps.sub(I, KH)

        val left  = MatrixOps.mul(MatrixOps.mul(IminusKH, P_pred), MatrixOps.transpose(IminusKH))
        val KRKt  = MatrixOps.mul(MatrixOps.mul(K, R), MatrixOps.transpose(K))
        val PJoseph = MatrixOps.symmetrise(MatrixOps.add(left, KRKt))

        return GainResult(K = K, P_updated = PJoseph, R = R)
    }

    private val hphtWindow = ArrayDeque<Array<DoubleArray>>(adaptiveWindow)

    override fun reset() {
        innovationWindow.clear()
        hphtWindow.clear()
        adaptiveREstimate = null
        stepCount = 0
    }

    /** Builds R from clamped GPS accuracy, blended with the adaptive estimate when available. */
    private fun computeR(
        obs: Observation,
        H: Array<DoubleArray>,
        P_pred: Array<DoubleArray>
    ): Array<DoubleArray> {
        val clampedAccuracy = obs.accuracy.coerceIn(minAccuracyM, maxAccuracyM).toDouble()
        val variance = clampedAccuracy * clampedAccuracy
        val Rbase = MatrixOps.diagonal(doubleArrayOf(variance, variance))

        if (!adaptiveR) return Rbase

        val Radaptive = adaptiveREstimate ?: return Rbase

        val alpha = chipsetTrustWeight(obs.accuracy)
        val m = Rbase.size
        return Array(m) { i ->
            DoubleArray(m) { j ->
                alpha * Rbase[i][j] + (1.0 - alpha) * Radaptive[i][j]
            }
        }
    }

    /** Maps GPS accuracy (log-space) to the chipset trust weight in [0.10, 0.95]. */
    private fun chipsetTrustWeight(accuracy: Float): Double {
        val lo = ln(minAccuracyM.toDouble().coerceAtLeast(0.5))
        val hi = ln(maxAccuracyM.toDouble())
        val v  = ln(accuracy.toDouble().coerceIn(minAccuracyM.toDouble(), maxAccuracyM.toDouble()))
        val t  = ((v - lo) / (hi - lo)).coerceIn(0.0, 1.0)
        return 0.95 - t * 0.85
    }
}
