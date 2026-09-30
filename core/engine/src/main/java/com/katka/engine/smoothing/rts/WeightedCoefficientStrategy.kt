package com.katka.engine.smoothing.rts

import com.katka.engine.MatrixOps
import com.katka.engine.coefficient_startegy.CoefficientStrategy
import com.katka.engine.model.GainResult
import com.katka.engine.model.KalmanState
import com.katka.model.Observation

/**
 * Coefficient strategy that scales measurement covariance by sample weights `R_i = R_raw / w_i`.
 *
 * Used by [OfflineTrackSmoother] to perform Iteratively Reweighted Least Squares (IRLS).
 *
 * @param weights Map of observation timestamps to weights in `(0, 1]`. Missing timestamps default to 1.0.
 * @param minAccuracyM Lower clamp on reported GPS accuracy.
 * @param maxAccuracyM Upper clamp on reported GPS accuracy.
 */
class WeightedCoefficientStrategy(
    private val weights: Map<Long, Double> = emptyMap(),
    private val minAccuracyM: Float = 1.0f,
    private val maxAccuracyM: Float = 50.0f
) : CoefficientStrategy {

    override fun computeGain(
        P_pred: Array<DoubleArray>,
        H: Array<DoubleArray>,
        obs: Observation
    ): GainResult {
        val w = (weights[obs.timestamp] ?: 1.0).coerceIn(0.01, 1.0)
        val clampedAcc = obs.accuracy.coerceIn(minAccuracyM, maxAccuracyM).toDouble()
        val variance = (clampedAcc * clampedAcc) / w
        val R = MatrixOps.diagonal(doubleArrayOf(variance, variance))

        val HP = MatrixOps.mul(H, P_pred)
        val HPHt = MatrixOps.mul(HP, MatrixOps.transpose(H))
        val S = MatrixOps.add(HPHt, R)

        val sScale = (S[0][0] + S[1][1]) / 2.0
        val Sreg = MatrixOps.addDiagEps(S, eps = sScale * 1e-6 + 1e-6)

        val PHt = MatrixOps.mul(P_pred, MatrixOps.transpose(H))
        val SInv = MatrixOps.inverse(Sreg)
        val K = MatrixOps.mul(PHt, SInv)

        val I = MatrixOps.identity(KalmanState.DIM)
        val KH = MatrixOps.mul(K, H)
        val IminusKH = MatrixOps.sub(I, KH)

        val left = MatrixOps.mul(MatrixOps.mul(IminusKH, P_pred), MatrixOps.transpose(IminusKH))
        val KRKt = MatrixOps.mul(MatrixOps.mul(K, R), MatrixOps.transpose(K))
        val PJoseph = MatrixOps.symmetrise(MatrixOps.add(left, KRKt))

        return GainResult(K = K, P_updated = PJoseph, R = R)
    }
}
